# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# 重采样器模块
# 用于处理视觉特征并将其与语言特征对齐

from collections import OrderedDict
import math
import requests
from io import BytesIO
from functools import partial
from PIL import Image
from typing import Callable, Optional, Sequence, Tuple, List, Union
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def get_abs_pos(abs_pos, tgt_size):
    """
    获取绝对位置编码，根据目标大小调整
    
    参数:
        abs_pos: 原始位置编码，形状为 (L, C)
        tgt_size: 目标大小 M
        
    返回:
        调整后的位置编码，形状为 (M, C)
    """
    # 步骤1：计算原始位置编码的空间尺寸
    # 假设原始位置编码是二维展平的，通过开方得到边长
    # 例如：如果abs_pos.size(0)=64，则src_size=8，表示8x8的网格
    src_size = int(math.sqrt(abs_pos.size(0)))
    
    # 步骤2：计算目标位置编码的空间尺寸
    # 同样通过开方得到目标网格的边长
    # 例如：如果tgt_size=576，则tgt_size=24，表示24x24的网格
    tgt_size = int(math.sqrt(tgt_size))
    
    # 步骤3：保存原始数据类型，用于最后的类型转换
    dtype = abs_pos.dtype

    # 步骤4：判断是否需要调整尺寸
    if src_size != tgt_size:
        # 步骤5：使用双三次插值调整位置编码尺寸
        return F.interpolate(
            # 步骤5a：将一维展平的位置编码重塑为二维网格格式
            # abs_pos形状从(L, C)变为(1, src_size, src_size, C)
            # 其中1是batch维度，最后一维是特征维度
            abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
            # 步骤5b：设置目标尺寸为(tgt_size, tgt_size)
            size=(tgt_size, tgt_size),
            # 步骤5c：使用双三次插值模式，提供更平滑的插值结果
            mode="bicubic",
            # 步骤5d：不对齐角点，避免边界效应
            align_corners=False,
        # 步骤5e：将插值后的结果重新排列维度并展平
        # 从(1, C, tgt_size, tgt_size)变回(1, tgt_size, tgt_size, C)
        # 然后展平为(tgt_size*tgt_size, C)格式
        ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)
    else:
        # 步骤6：如果尺寸相同，直接返回原始位置编码
        return abs_pos


# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    获取二维正弦余弦位置编码
    
    参数:
        embed_dim: 嵌入维度
        grid_size: 网格的高度和宽度
        cls_token: 是否包含类别标记
        
    返回:
        pos_embed: 形状为[grid_size*grid_size, embed_dim]的位置编码
                  如果cls_token为True，则形状为[1+grid_size*grid_size, embed_dim]
    """
    # 步骤1：创建网格坐标
    # 生成高度方向的坐标序列，从0到grid_size-1
    # 例如：如果grid_size=8，则grid_h=[0, 1, 2, 3, 4, 5, 6, 7]
    grid_h = np.arange(grid_size, dtype=np.float32)
    
    # 生成宽度方向的坐标序列，从0到grid_size-1
    # 例如：如果grid_size=8，则grid_w=[0, 1, 2, 3, 4, 5, 6, 7]
    grid_w = np.arange(grid_size, dtype=np.float32)
    
    # 步骤2：创建二维网格坐标矩阵
    # 使用meshgrid创建坐标网格，注意这里w（宽度）在前，h（高度）在后
    # 这样做是为了符合图像处理中的坐标约定（x轴对应宽度，y轴对应高度）
    # 返回两个矩阵：第一个是x坐标矩阵，第二个是y坐标矩阵
    grid = np.meshgrid(grid_w, grid_h)  # 注意w先行
    
    # 步骤3：堆叠坐标矩阵
    # 将x坐标矩阵和y坐标矩阵沿着新的轴（axis=0）堆叠
    # 结果形状为(2, grid_size, grid_size)
    # 其中第0维表示坐标类型（0为x坐标，1为y坐标）
    grid = np.stack(grid, axis=0)

    # 步骤4：重塑网格形状以适配后续处理
    # 将形状从(2, grid_size, grid_size)重塑为(2, 1, grid_size, grid_size)
    # 添加的维度1是为了与后续函数的输入格式保持一致
    # 这种格式便于批量处理多个网格
    grid = grid.reshape([2, 1, grid_size, grid_size])
    
    # 步骤5：生成位置编码
    # 调用专门的函数从网格坐标生成二维正弦余弦位置编码
    # 返回形状为(grid_size*grid_size, embed_dim)的位置编码矩阵
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    # 步骤6：处理类别标记（可选）
    if cls_token:
        # 如果需要包含类别标记，在位置编码前面添加一个全零向量
        # 这个全零向量作为类别标记的位置编码
        # 最终形状变为(1+grid_size*grid_size, embed_dim)
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0)
    
    # 步骤7：返回最终的位置编码
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    从网格获取二维正弦余弦位置编码
    
    参数:
        embed_dim: 嵌入维度
        grid: 位置网格
        
    返回:
        二维位置编码
    """
    assert embed_dim % 2 == 0

    # 使用一半维度编码grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    从网格获取一维正弦余弦位置编码
    
    参数:
        embed_dim: 每个位置的输出维度
        pos: 要编码的位置列表，大小为(M,)
        
    返回:
        形状为(M, D)的位置编码
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2)，外积

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Resampler(nn.Module):
    """
    二维感知重采样网络，具有一个交叉注意力层
    使用(grid_size**2)个可学习查询和二维正弦余弦位置编码
    
    输出:
        形状为(grid_size**2, embed_dim)的张量
    """

    def __init__(
            self,
            grid_size,
            embed_dim,
            num_heads,
            kv_dim=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
    ):
        """
        初始化重采样器
        
        参数:
            grid_size: 网格大小
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            kv_dim: 键值对维度，默认与embed_dim相同
            norm_layer: 归一化层，默认为LayerNorm
        """
        super().__init__()
        self.num_queries = grid_size ** 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 初始化位置编码，不参与梯度更新
        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(
                embed_dim, grid_size)).float()
        ).requires_grad_(False)

        # 初始化可学习查询
        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query, std=.02)

        # 键值对投影层
        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        # 多头注意力层和归一化层
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)

        self.ln_post = norm_layer(embed_dim)
        self.proj = nn.Parameter(
            (embed_dim ** -0.5) * torch.randn(embed_dim, embed_dim))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        初始化模型权重
        
        参数:
            m: 模型层
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, attn_mask=None):
        """
        前向传播
        功能: 通过双线性插值将固定的8×8位置编码调整到匹配实际输入大小
             支持任意分辨率图像，无论是224×224还是1344×1344都能正确处理
        
        参数:
            x: 输入特征
            attn_mask: 注意力掩码，默认为None
            
        返回:
            重采样后的特征
        """
        # 获取适应当前输入大小的位置编码
        pos_embed = get_abs_pos(self.pos_embed, x.size(1))

        # 投影和归一化
        x = self.kv_proj(x) # 确保维度匹配
        x = self.ln_kv(x).permute(1, 0, 2) # Key/Value 归一化

        # 维度对齐投影
        N = x.shape[1]
         # 归一化+维度变换
        q = self.ln_q(self.query)
        
        # 执行注意力计算
        # 4. 交叉注意力 - 核心机制 ⭐
    
        out = self.attn(
            self._repeat(q, N) + self.pos_embed.unsqueeze(1), # Query: 查询 + 位置编码
            x + pos_embed.unsqueeze(1),  # Key: 视觉特征 + 位置编码  
            x,  # Value: 视觉特征
            attn_mask=attn_mask)[0] 
        x = out.permute(1, 0, 2)

        # 最终处理
        x = self.ln_post(x)
        x = x @ self.proj
        return x

    def _repeat(self, query, N: int):
        """
        重复查询向量
        
        参数:
            query: 查询向量
            N: 重复次数
            
        返回:
            重复后的查询向量
        """
        return query.unsqueeze(1).repeat(1, N, 1)
