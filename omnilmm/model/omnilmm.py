# OmniLMM模型实现文件
# 实现MiniCPM-o多模态大语言模型的核心结构与功能
# 该文件是MiniCPM-o架构的核心实现，包含视觉-语言特征融合、多模态对齐等关键技术

import gc
import math
import timm
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import List, Optional, Tuple, Union

from transformers import AutoConfig, AutoModelForCausalLM
from transformers import MistralForCausalLM, MistralModel, MistralConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from omnilmm.model.utils import build_transform
from omnilmm.model.resampler import Resampler

# 多模态特殊Token定义 - 用于标记图像在文本序列中的位置
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"  # 图像patch标记，每个图像patch对应一个token
DEFAULT_IM_START_TOKEN = "<im_start>"     # 图像开始标记，标识图像内容的起始位置
DEFAULT_IM_END_TOKEN = "<im_end>"         # 图像结束标记，标识图像内容的结束位置


class OmniLMMConfig(MistralConfig):
    """
    OmniLMM模型配置类

    继承自MistralConfig，扩展了多模态相关的配置参数

    功能：
    - 继承Mistral语言模型的所有配置参数
    - 添加视觉模块相关的配置选项
    - 支持多模态训练和推理的参数设置

    在MiniCPM-o架构中的作用：
    - 作为模型配置的统一入口
    - 管理视觉编码器、重采样器等组件的参数
    - 控制多模态特征融合的行为
    """
    model_type = "omnilmm"


class Identity(torch.nn.Identity):
    """
    恒等映射层 - 用于替换不需要的网络层

    功能：
    - 保持输入张量不变，直接返回输入
    - 用于替换EVA02模型中不需要的层（如最后一层注意力层）

    在MiniCPM-o架构中的作用：
    - 替换EVA02视觉编码器的最后一层，使用倒数第二层的特征
    - 移除注意力池化层，保留更丰富的空间特征信息
    """
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return super().forward(input)


def create_vision_module(config):
    """
    创建视觉模块 - MiniCPM-o多模态架构的核心组件

    该函数实现了MiniCPM-o架构分析中提到的视觉编码器创建过程，
    包括EVA02-enormous视觉编码器和Resampler重采样器的初始化。

    参数:
        config: OmniLMMConfig配置对象，包含模型的所有配置参数

    返回:
        tuple: (vision_tower, resampler)
            - vision_tower: EVA02视觉编码器，用于提取图像特征
            - resampler: 重采样器，用于将视觉特征对齐到语言模型空间

    技术细节:
        1. 使用EVA02-enormous-patch14作为视觉backbone
        2. 支持动态图像尺寸和填充，最高支持1.8M像素
        3. 移除最后一层，使用倒数第二层特征，保留更丰富的语义信息
        4. Resampler实现视觉特征到固定长度token序列的转换
    """
    # 创建EVA02-enormous模型作为视觉编码器
    # 这是MiniCPM-o使用的强大视觉backbone，支持高分辨率图像处理
    vision_tower = timm.create_model(
        'eva02_enormous_patch14_clip_224.laion2b_plus',  # EVA02-enormous预训练模型
        pretrained=False,        # 不加载预训练权重，后续会单独加载
        num_classes=0,          # 不需要分类头
        dynamic_img_size=True,  # 支持动态图像尺寸，关键特性
        dynamic_img_pad=True    # 支持动态填充，处理任意宽高比图像
    )

    # 如果是Vision Transformer模型，移除注意力池化层
    # 保留完整的patch特征，避免信息损失
    if isinstance(vision_tower, timm.models.VisionTransformer):
        if vision_tower.attn_pool is not None:
            vision_tower.attn_pool = Identity()  # 用恒等映射替换注意力池化

    # 关键技术：使用倒数第二层的输出而非最后一层
    # 这样可以保留更丰富的视觉语义信息，避免过度抽象
    vision_tower.blocks[-1] = Identity()

    # 创建Resampler重采样器 - 视觉特征到语言空间的桥梁
    embed_dim = config.hidden_size  # 语言模型的隐藏维度，通常是4096
    resampler = Resampler(
        grid_size=int(math.sqrt(config.num_query)),  # 查询网格大小，通常8x8=64
        embed_dim=embed_dim,                         # 输出维度，与语言模型对齐
        num_heads=embed_dim // 128,                  # 注意力头数，通常32个
        kv_dim=vision_tower.embed_dim,               # 输入维度，EVA02的特征维度
    )
    return vision_tower, resampler


class OmniLMMModel(MistralModel):
    """
    OmniLMM多模态模型核心类 - MiniCPM-o架构的主要实现

    继承关系：
    - 继承自MistralModel，获得强大的语言理解能力
    - 扩展多模态处理功能，实现视觉-语言特征融合

    核心功能：
    1. 多模态特征统一：将视觉特征和文本特征融合到统一空间
    2. 视觉编码器集成：集成EVA02视觉编码器进行图像理解
    3. 特征重采样：通过Resampler实现视觉特征到固定长度token的转换
    4. 梯度连通性：确保纯文本和多模态样本的梯度连通性

    在MiniCPM-o架构中的作用：
    - 实现notebook中描述的多模态特征统一机制
    - 支持时分复用（TDM）的实时多模态处理
    - 提供高效的token密度（2822 pixels/token）
    """
    config_class = OmniLMMConfig

    def __init__(self, config: OmniLMMConfig, mm_vision_tower=None, mm_hidden_size=None, tune_clip=True):
        """
        初始化OmniLMM模型

        参数:
            config: OmniLMMConfig配置对象
            mm_vision_tower: 多模态视觉塔（保留参数，实际未使用）
            mm_hidden_size: 多模态隐藏层大小（保留参数，实际未使用）
            tune_clip: 是否微调CLIP模型，控制视觉编码器的训练模式

        初始化过程：
        1. 调用父类MistralModel的初始化
        2. 创建视觉模块（EVA02 + Resampler）
        3. 设置FSDP兼容的模型结构
        4. 配置视觉相关的参数
        """
        # 调用父类初始化，获得Mistral语言模型的完整功能
        super(OmniLMMModel, self).__init__(config)

        # 检查配置中是否包含视觉模块设置
        if hasattr(config, "mm_vision_tower"):
            # 创建视觉模块：EVA02编码器 + Resampler重采样器
            vision_tower, resampler = create_vision_module(config)

            # HACK: 为FSDP（Fully Sharded Data Parallel）兼容性处理
            # FSDP要求模型参数以特定方式组织，这里将vision_tower包装为列表
            self.vision_tower = [vision_tower]
            self.resampler = resampler

            # 根据是否微调CLIP决定vision_tower的存储方式
            if tune_clip:
                # 微调模式：直接使用vision_tower对象，允许梯度更新
                self.vision_tower = self.vision_tower[0]
            # 否则保持列表形式，用于冻结参数的推理模式

        # 初始化视觉配置对象（占位符，后续会被具体配置替换）
        self.vision_config = lambda x: None

    def initialize_vision_modules(self, vision_tower, no_randaug, num_query, image_size, tune_clip=False):
        """
        初始化视觉模块 - 设置MiniCPM-o的视觉处理组件

        该方法实现了MiniCPM-o架构中视觉模块的完整初始化过程，
        包括EVA02权重加载、图像预处理器配置等关键步骤。

        参数:
            vision_tower: 视觉塔模型路径或标识符
            no_randaug: 是否禁用随机数据增强（训练时通常为False）
            num_query: Resampler查询数量，决定视觉特征的token数量（通常64）
            image_size: 输入图像尺寸，支持动态尺寸处理
            tune_clip: 是否微调CLIP模型参数

        返回:
            dict: 包含以下组件的字典
                - image_processor: (训练变换, 评估变换) 图像预处理器元组
                - image_token_len: 图像token长度，用于序列长度计算
                - vision_config: 视觉配置对象

        技术细节:
        1. 配置模型的多模态参数
        2. 加载预训练的EVA02权重
        3. 设置FSDP兼容的模型结构
        4. 构建OPENAI_CLIP标准的图像预处理流水线
        """
        # 设置模型配置中的多模态相关参数
        self.config.mm_vision_tower = vision_tower      # 视觉塔标识符
        self.config.use_mm_proj = True                  # 启用多模态投影
        self.config.num_query = num_query               # Resampler查询数量
        self.config.image_size = image_size             # 图像输入尺寸

        # 检查是否已经初始化了视觉模块
        if not hasattr(self, 'vision_tower'):
            # 首次初始化：创建视觉模块并加载预训练权重
            vision_tower, resampler = create_vision_module(self.config)

            # 加载EVA02-enormous预训练权重
            # 这是MiniCPM-o强大视觉理解能力的基础
            state_dict = torch.load(
                '/tt/data/public/multimodal/multimodal_model_ckpts/timm/eva02_enormous_patch14_clip_224.laion2b_plus.pt')
            vision_tower.load_state_dict(state_dict, strict=False)

            # 清理内存，释放state_dict占用的空间
            del state_dict
            gc.collect()
        else:
            # 已存在视觉模块：从现有结构中提取组件
            if isinstance(self.vision_tower, list):
                # FSDP模式：从列表中提取vision_tower
                vision_tower = self.vision_tower[0]
            else:
                # 标准模式：直接使用vision_tower
                vision_tower = self.vision_tower
            resampler = self.resampler

        # 根据微调设置配置vision_tower的存储方式
        # 这影响参数的梯度计算和FSDP的分片策略
        self.vision_tower = vision_tower if tune_clip else [vision_tower]
        self.resampler = resampler

        # 构建图像预处理流水线
        # 使用OPENAI_CLIP标准，确保与EVA02预训练权重兼容
        train_img_transform = build_transform(
            is_train=True,                              # 训练模式：包含数据增强
            randaug=not no_randaug,                     # 随机增强设置
            input_size=self.config.image_size,          # 输入图像尺寸
            std_mode='OPENAI_CLIP'                      # 使用CLIP标准化参数
        )
        eval_img_transform = build_transform(
            is_train=False,                             # 评估模式：无数据增强
            input_size=self.config.image_size,          # 输入图像尺寸
            std_mode='OPENAI_CLIP'                      # 使用CLIP标准化参数
        )

        # 返回初始化完成的组件
        return dict(
            image_processor=(train_img_transform, eval_img_transform),  # 图像预处理器
            image_token_len=num_query,                                  # 图像token长度
            vision_config=self.vision_config                            # 视觉配置对象
        )

    def get_vision_embedding(self, pixel_values):
        """
        获取视觉嵌入 - MiniCPM-o视觉特征提取的核心方法

        该方法实现了MiniCPM-o架构分析中描述的视觉特征提取流程：
        EVA02编码器 -> 特征提取 -> Resampler重采样 -> 统一特征空间

        参数:
            pixel_values: 图像像素值张量 [batch_size, channels, height, width]
                         支持动态尺寸，最高1.8M像素

        返回:
            Tensor: 重采样后的视觉嵌入 [batch_size, num_queries, hidden_size]
                   通常为 [batch_size, 64, 4096]，实现高效的token密度

        技术细节:
        1. FSDP兼容性处理：正确提取vision_tower对象
        2. 数据类型对齐：确保与位置编码的数据类型一致
        3. 前缀token移除：去除CLS等分类token，保留纯patch特征
        4. Resampler重采样：将可变长度特征转换为固定长度token序列

        性能特点:
        - 支持2822 pixels/token的高效编码密度
        - 1.8M像素图像仅产生64个token，减少75%的token使用
        """
        # FSDP兼容性处理：正确获取vision_tower对象
        if isinstance(self.vision_tower, list):
            # FSDP模式：vision_tower被包装在列表中
            vision_tower = self.vision_tower[0]  # HACK: for FSDP
        else:
            # 标准模式：直接使用vision_tower对象
            vision_tower = self.vision_tower

        # 数据类型对齐：确保输入数据类型与模型参数一致
        # 使用位置编码的数据类型作为参考，避免类型不匹配错误
        dtype = vision_tower.pos_embed.data.dtype

        # EVA02特征提取：使用forward_features方法获取patch特征
        # 这里使用倒数第二层的输出，保留更丰富的语义信息
        vision_embedding = vision_tower.forward_features(
            pixel_values.type(dtype))

        # 移除前缀token（如CLS token）
        # 只保留patch特征，确保特征的纯净性和空间一致性
        if hasattr(vision_tower, 'num_prefix_tokens') and vision_tower.num_prefix_tokens > 0:
            vision_embedding = vision_embedding[:,
                                                vision_tower.num_prefix_tokens:]

        # Resampler重采样：关键的特征对齐步骤
        # 将可变长度的视觉特征转换为固定长度的token序列
        # 实现视觉特征到语言模型空间的统一映射
        res = self.resampler(vision_embedding)
        return res

    def get_vllm_embedding(self, data):
        """
        获取vLLM推理优化的多模态嵌入 - 高效推理的关键方法

        该方法专门为vLLM推理框架优化，实现了高吞吐量的多模态特征融合。
        支持批量处理和内存高效的推理，是MiniCPM-o部署优化的重要组成部分。

        参数:
            data: 输入数据字典，包含以下字段：
                - input_ids: 文本token序列 [batch_size, seq_len]
                - pixel_values: 图像像素值列表，支持批量图像处理
                - vision_hidden_states: 可选的预计算视觉特征（用于缓存优化）

        返回:
            tuple: (inputs_embeds, vision_hidden_states)
                - inputs_embeds: 融合后的多模态嵌入 [batch_size, seq_len, hidden_size]
                - vision_hidden_states: 视觉隐藏状态列表，用于后续缓存

        技术细节:
        1. 支持视觉特征缓存，避免重复计算
        2. 批量处理多个图像，提高推理效率
        3. 数据类型自动对齐，确保计算稳定性
        4. 梯度连通性保证，支持端到端训练
        """
        # 步骤1: 获取或计算视觉隐藏状态
        if 'vision_hidden_states' not in data:
            # 首次计算：从像素值提取视觉特征
            pixel_values_list = data['pixel_values']
            vision_hidden_states = []

            # 批量处理每个图像
            for pixel_values in pixel_values_list:
                if len(pixel_values) > 0:
                    # 有图像：提取视觉特征
                    vision_hidden_states.append(self.get_vision_embedding(pixel_values.unsqueeze(0))[0])
                else:
                    # 无图像：添加空占位符
                    vision_hidden_states.append([])
        else:
            # 使用缓存的视觉特征，提高推理效率
            vision_hidden_states = data['vision_hidden_states']

        # 步骤2: 获取文本嵌入
        # 注释掉的代码是原始LLaVA的实现方式
        #vllm_embedding = self.llm.model.embed_tokens(data['input_ids']) * self.llm.config.scale_emb
        inputs_embeds = self.embed_tokens(data['input_ids'])

        # 步骤3: 数据类型对齐
        # 确保视觉特征与文本嵌入的数据类型一致，避免计算错误
        vision_hidden_states = [i.type(inputs_embeds.dtype)
            if isinstance(i, torch.Tensor) else i for i in vision_hidden_states
        ]

        # HACK: LLaVA预训练的兼容性处理
        # 支持原始嵌入参数的替换，用于特定的预训练场景
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)

        # 步骤4: 创建虚拟图像特征用于梯度连通性
        # 确保纯文本样本也能与视觉模块保持梯度连接，支持端到端训练
        dummy_image_features = torch.zeros(
            self.config.num_query,           # 查询数量，通常为64
            self.config.hidden_size,         # 隐藏层维度，通常为4096
            device=inputs_embeds.device,     # 与输入嵌入相同的设备
            dtype=inputs_embeds.dtype        # 与输入嵌入相同的数据类型
        )

        # 步骤5: 多模态特征融合 - MiniCPM-o的核心技术
        new_input_embeds = []
        cur_image_idx = 0

        # 逐个处理批次中的每个样本
        for cur_input_ids, cur_input_embeds in zip(data['input_ids'], inputs_embeds):
            # 检查当前样本是否包含图像
            if (cur_input_ids == self.vision_config.im_patch_token).sum() == 0:
                # 纯文本样本：添加零乘虚拟特征保持梯度连通性
                # 这是关键的技术细节，确保多模态模型能同时处理纯文本和多模态输入
                cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
                new_input_embeds.append(cur_input_embeds)
                continue

            # 多模态样本：执行视觉-文本特征融合
            if self.vision_config.use_im_start_end:
                # 获取当前图像的视觉特征
                cur_image_features = vision_hidden_states[cur_image_idx]
                num_patches = cur_image_features.shape[0]  # 图像patch数量，通常为64

                # 验证图像开始和结束token的配对正确性
                # 这是多模态对齐的关键检查，确保序列结构的完整性
                if (cur_input_ids == self.vision_config.im_start_token).sum() != (cur_input_ids == self.vision_config.im_end_token).sum():
                    raise ValueError(
                        "The number of image start tokens and image end tokens should be the same.")

                # 定位所有图像开始token的位置
                image_start_tokens = torch.where(
                    cur_input_ids == self.vision_config.im_start_token)[0]

                # 处理每个图像位置（支持多图像输入）
                for image_start_token_pos in image_start_tokens:
                    # 设备对齐：确保图像特征与文本嵌入在同一设备上
                    cur_image_features = vision_hidden_states[cur_image_idx].to(
                        device=cur_input_embeds.device)
                    num_patches = cur_image_features.shape[0]

                    # 验证图像结束token的位置正确性
                    # 确保图像特征插入位置的准确性：<im_start> + patches + <im_end>
                    if cur_input_ids[image_start_token_pos + num_patches + 1] != self.vision_config.im_end_token:
                        raise ValueError(
                            "The image end token should follow the image start token.")

                    # 特征拼接：实现视觉-文本的无缝融合
                    if orig_embeds_params is not None:
                        # LLaVA预训练模式：保留特定token的梯度
                        cur_new_input_embeds = torch.cat((
                            cur_input_embeds[:image_start_token_pos].detach(),           # 前文本（detach）
                            cur_input_embeds[image_start_token_pos:image_start_token_pos+1],  # <im_start>
                            cur_image_features,                                          # 图像特征
                            cur_input_embeds[image_start_token_pos + num_patches + 1:image_start_token_pos + num_patches + 2],  # <im_end>
                            cur_input_embeds[image_start_token_pos + num_patches + 2:].detach()  # 后文本（detach）
                        ), dim=0)
                    else:
                        # 标准模式：直接拼接所有特征
                        # 这是MiniCPM-o的标准多模态融合方式
                        cur_new_input_embeds = torch.cat((
                            cur_input_embeds[:image_start_token_pos+1],                  # 前文本 + <im_start>
                            cur_image_features,                                          # 图像特征（64个token）
                            cur_input_embeds[image_start_token_pos + num_patches + 1:]  # <im_end> + 后文本
                        ), dim=0)

                    cur_image_idx += 1  # 移动到下一个图像
                new_input_embeds.append(cur_new_input_embeds)
            else:
                # 不支持的配置：必须使用<im_start>和<im_end>标记
                raise NotImplementedError

        # 步骤6: 堆叠所有融合后的嵌入
        # 将列表转换为张量，形成最终的多模态输入嵌入
        inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return inputs_embeds, vision_hidden_states

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        OmniLMM模型前向传播 - MiniCPM-o多模态理解的核心实现

        该方法实现了MiniCPM-o架构分析中描述的多模态前向传播过程：
        1. 文本embedding初始化
        2. 视觉特征提取和处理
        3. 多模态特征融合
        4. 调用父类Mistral模型进行推理

        参数:
            input_ids: 输入token序列 [batch_size, seq_len]
            attention_mask: 注意力掩码，控制哪些token参与注意力计算
            past_key_values: 缓存的键值对，用于生成任务的增量解码
            inputs_embeds: 预计算的输入嵌入（可选，与input_ids二选一）
            use_cache: 是否使用KV缓存，提高生成效率
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出隐藏状态
            images: 图像张量 [batch_size, channels, height, width] 或图像列表
            return_dict: 是否返回字典格式的输出

        返回:
            BaseModelOutputWithPast: 包含last_hidden_state、past_key_values等的输出对象

        技术细节:
        - 支持纯文本和多模态输入的统一处理
        - 实现梯度连通性保证，确保端到端训练
        - 支持多图像输入和批量处理
        - 兼容vLLM等高效推理框架
        """

        # HACK: LLaVA预训练兼容性处理
        # 获取原始嵌入参数，用于特定的预训练场景
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)

        # 步骤1: 输入嵌入初始化和多模态处理条件检查
        if inputs_embeds is None and past_key_values is None:
            # 从token ID获取文本嵌入
            inputs_embeds = self.embed_tokens(input_ids)

            # 检查是否需要进行多模态处理
            vision_tower = getattr(self, 'vision_tower', None)
            # 多模态处理条件：
            # 1. 存在视觉模块
            # 2. 不是单token生成（input_ids.shape[1] != 1）或处于训练模式
            # 3. 提供了图像输入
            if vision_tower is not None and (input_ids.shape[1] != 1 or self.training) and images is not None:

                # 步骤2: 视觉特征提取
                if type(images) is list:
                    # 多图像输入：逐个处理每张图像
                    image_features = []
                    for image in images:
                        # 为单张图像添加batch维度并提取特征
                        image_forward_out = self.get_vision_embedding(image.unsqueeze(0))[0]
                        image_features.append(image_forward_out)
                else:
                    # 单图像或批量图像输入：直接处理
                    image_features = self.get_vision_embedding(images)

                # 步骤3: 创建虚拟图像特征用于梯度连通性
                # 确保纯文本样本也能与视觉模块保持梯度连接
                dummy_image_features = torch.zeros(
                    self.config.num_query,           # 查询数量，通常为64
                    self.config.hidden_size,         # 隐藏层维度，通常为4096
                    device=inputs_embeds.device,     # 与输入嵌入相同的设备
                    dtype=inputs_embeds.dtype        # 与输入嵌入相同的数据类型
                )

                # 步骤4: 多模态特征融合 - MiniCPM-o的核心技术
                # 这里实现了与get_vllm_embedding相同的融合逻辑
                new_input_embeds = []
                cur_image_idx = 0

                # 逐个处理批次中的每个样本
                for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                    # 检查当前样本是否包含图像patch token
                    if (cur_input_ids == self.vision_config.im_patch_token).sum() == 0:
                        # 纯文本样本：添加零乘虚拟特征保持梯度连通性
                        # 这是关键的技术细节，确保多模态模型能同时处理纯文本和多模态输入
                        cur_input_embeds = cur_input_embeds + \
                            (0. * dummy_image_features).sum()
                        new_input_embeds.append(cur_input_embeds)
                        continue

                    # 多模态样本：执行视觉-文本特征融合
                    if self.vision_config.use_im_start_end:
                        cur_image_features = image_features[cur_image_idx]
                        num_patches = cur_image_features.shape[0]  # 图像patch数量

                        # 验证图像开始和结束token的配对正确性
                        if (cur_input_ids == self.vision_config.im_start_token).sum() != (cur_input_ids == self.vision_config.im_end_token).sum():
                            raise ValueError(
                                "The number of image start tokens and image end tokens should be the same.")

                        # 定位所有图像开始token的位置
                        image_start_tokens = torch.where(
                            cur_input_ids == self.vision_config.im_start_token)[0]

                        # 处理每个图像位置（支持多图像输入）
                        for image_start_token_pos in image_start_tokens:
                            # 设备对齐：确保图像特征与文本嵌入在同一设备上
                            cur_image_features = image_features[cur_image_idx].to(
                                device=cur_input_embeds.device)
                            num_patches = cur_image_features.shape[0]

                            # 验证图像结束token的位置正确性
                            if cur_input_ids[image_start_token_pos + num_patches + 1] != self.vision_config.im_end_token:
                                raise ValueError(
                                    "The image end token should follow the image start token.")

                            # 特征拼接：实现视觉-文本的无缝融合
                            if orig_embeds_params is not None:
                                # LLaVA预训练模式：保留特定token的梯度
                                cur_new_input_embeds = torch.cat((
                                    cur_input_embeds[:image_start_token_pos].detach(),           # 前文本（detach）
                                    cur_input_embeds[image_start_token_pos:image_start_token_pos+1],  # <im_start>
                                    cur_image_features,                                          # 图像特征
                                    cur_input_embeds[image_start_token_pos + num_patches + 1:image_start_token_pos + num_patches + 2],  # <im_end>
                                    cur_input_embeds[image_start_token_pos + num_patches + 2:].detach()  # 后文本（detach）
                                ), dim=0)
                            else:
                                # 标准模式：直接拼接所有特征
                                # 这是MiniCPM-o的标准多模态融合方式
                                cur_new_input_embeds = torch.cat((
                                    cur_input_embeds[:image_start_token_pos+1],                  # 前文本 + <im_start>
                                    cur_image_features,                                          # 图像特征（64个token）
                                    cur_input_embeds[image_start_token_pos + num_patches + 1:]  # <im_end> + 后文本
                                ), dim=0)
                            cur_image_idx += 1  # 移动到下一个图像
                        new_input_embeds.append(cur_new_input_embeds)
                    else:
                        # 不支持的配置：必须使用<im_start>和<im_end>标记
                        raise NotImplementedError

                # 步骤5: 堆叠所有融合后的嵌入
                inputs_embeds = torch.stack(new_input_embeds, dim=0)
                input_ids = None  # 使用融合后的嵌入，不再需要原始token ID

        # 步骤6: 调用父类Mistral模型进行推理
        # 将融合后的多模态嵌入传递给Mistral语言模型进行处理
        # 这里实现了MiniCPM-o架构中"调用基础模型"的步骤
        return super(OmniLMMModel, self).forward(
            input_ids=input_ids,                                    # 原始token ID（多模态情况下为None）
            attention_mask=attention_mask,                          # 注意力掩码
            past_key_values=past_key_values,                        # KV缓存
            inputs_embeds=inputs_embeds,                            # 融合后的多模态嵌入
            use_cache=use_cache,                                    # 是否使用缓存
            output_attentions=output_attentions,                    # 是否输出注意力权重
            output_hidden_states=output_hidden_states,              # 是否输出隐藏状态
            return_dict=return_dict,                                # 返回格式
            **kwargs                                                # 其他参数
        )


class OmniLMMForCausalLM(MistralForCausalLM):
    """
    OmniLMM因果语言模型类 - MiniCPM-o的完整实现

    继承关系：
    - 继承自MistralForCausalLM，获得因果语言建模能力
    - 集成OmniLMMModel，实现多模态理解功能

    核心功能：
    1. 多模态因果语言建模：支持文本生成任务
    2. 损失函数计算：实现标准的交叉熵损失
    3. 生成接口：提供文本生成和vLLM优化推理
    4. Token管理：处理多模态特殊token的初始化

    在MiniCPM-o架构中的作用：
    - 作为对外的主要接口，封装完整的多模态语言模型
    - 支持训练和推理的统一接口
    - 实现高效的生成和推理优化
    """
    config_class = OmniLMMConfig

    def __init__(self, config, mm_vision_tower=None, tune_clip=True):
        """
        初始化OmniLMM因果语言模型

        参数:
            config: OmniLMMConfig配置对象
            mm_vision_tower: 多模态视觉塔（保留参数）
            tune_clip: 是否微调CLIP模型

        初始化过程:
        1. 调用父类初始化，获得因果语言建模基础
        2. 创建OmniLMMModel作为核心模型
        3. 初始化语言模型头，用于词汇表预测
        4. 执行权重初始化和后处理
        """
        # 调用父类初始化，获得MistralForCausalLM的完整功能
        super(MistralForCausalLM, self).__init__(config)

        # 创建核心的多模态模型
        self.model = OmniLMMModel(
            config, mm_vision_tower=mm_vision_tower, tune_clip=tune_clip)

        # 初始化语言模型头：隐藏状态 -> 词汇表概率分布
        # 无偏置设计，与现代语言模型的标准做法一致
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        # 执行权重初始化和最终处理
        # 确保模型参数的正确初始化
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        OmniLMM因果语言模型前向传播 - 完整的多模态语言建模流程

        该方法实现了MiniCPM-o的完整推理和训练流程：
        1. 调用OmniLMMModel进行多模态特征融合
        2. 通过语言模型头生成词汇表概率分布
        3. 计算因果语言建模损失（训练时）

        参数:
            input_ids: 输入token序列 [batch_size, seq_len]
            attention_mask: 注意力掩码
            past_key_values: KV缓存，用于生成任务的增量解码
            inputs_embeds: 预计算的输入嵌入
            labels: 训练标签 [batch_size, seq_len]，用于损失计算
            use_cache: 是否使用KV缓存
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出隐藏状态
            images: 图像输入
            return_dict: 是否返回字典格式输出

        返回:
            CausalLMOutputWithPast: 包含loss、logits、past_key_values等的输出对象
        """
        # 配置输出选项，使用默认配置或传入参数
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调试信息（已注释）
        # print(f'@@@ At forward, labels: {labels.shape}-{labels}', flush=True)
        # print(f'@@@ At forward, input_ids: {input_ids.shape}-{input_ids}', flush=True)
        # print(f'@@@ At forward, input_ids: {attention_mask.shape}-{attention_mask}', flush=True)

        # 步骤1: 调用OmniLMMModel进行多模态处理
        # 这里执行了完整的视觉-文本特征融合流程
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images,
            **kwargs
        )

        # 步骤2: 提取隐藏状态并生成logits
        hidden_states = outputs[0]  # 获取最后一层的隐藏状态
        logits = self.lm_head(hidden_states)  # 通过语言模型头生成词汇表概率分布

        # 步骤3: 计算因果语言建模损失（训练时）
        loss = None
        if labels is not None:
            # 因果语言建模的标准损失计算：
            # 使用位置i的token预测位置i+1的token

            # 移位操作：logits和labels都向左移动一位
            shift_logits = logits[..., :-1, :].contiguous()  # 去掉最后一个位置的logits
            shift_labels = labels[..., 1:].contiguous()      # 去掉第一个位置的labels

            # 展平张量以便计算损失
            loss_fct = CrossEntropyLoss()  # 标准交叉熵损失函数
            shift_logits = shift_logits.view(-1, self.config.vocab_size)  # [batch*seq, vocab_size]
            shift_labels = shift_labels.view(-1)                          # [batch*seq]

            # 设备对齐：确保labels与logits在同一设备上
            # 这对于模型并行和流水线并行很重要
            shift_labels = shift_labels.to(shift_logits.device)

            # 计算最终损失
            loss = loss_fct(shift_logits, shift_labels)

        # 步骤4: 返回结果
        if not return_dict:
            # 元组格式返回：兼容旧版本接口
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # 字典格式返回：标准的transformers输出格式
        return CausalLMOutputWithPast(
            loss=loss,                                    # 训练损失
            logits=logits,                               # 词汇表概率分布
            past_key_values=outputs.past_key_values,     # KV缓存
            hidden_states=outputs.hidden_states,         # 隐藏状态
            attentions=outputs.attentions,               # 注意力权重
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """
        为生成任务准备输入 - 支持增量解码和KV缓存

        该方法为Hugging Face的generate()方法提供输入准备逻辑，
        支持高效的文本生成和多模态对话。

        参数:
            input_ids: 输入token序列
            past_key_values: KV缓存，用于增量解码
            attention_mask: 注意力掩码
            inputs_embeds: 预计算的输入嵌入
            **kwargs: 其他参数，包括images等

        返回:
            dict: 准备好的模型输入字典

        技术细节:
        - 支持KV缓存的增量解码，只处理最后一个token
        - 优先使用inputs_embeds（多模态场景）
        - 保持图像输入的传递
        """
        # KV缓存优化：只处理最后一个token
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # 输入类型选择：优先使用嵌入（多模态场景）
        if inputs_embeds is not None and past_key_values is None:
            # 首次生成步骤：使用预计算的多模态嵌入
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # 后续生成步骤：使用token ID
            model_inputs = {"input_ids": input_ids}

        # 添加其他必要的输入参数
        model_inputs.update(
            {
                "past_key_values": past_key_values,           # KV缓存
                "use_cache": kwargs.get("use_cache"),         # 缓存控制
                "attention_mask": attention_mask,             # 注意力掩码
                "images": kwargs.get("images", None),         # 图像输入
            }
        )
        return model_inputs

    def generate_vllm(
        self,
        input_ids: torch.LongTensor = None,
        images: Optional[torch.FloatTensor] = None,
        vision_hidden_states=None,
        return_vision_hidden_states=False,
        **kwargs
    ):
        """
        vLLM优化的生成方法 - 高效推理的核心接口

        该方法专门为vLLM推理框架优化，实现了高吞吐量的多模态文本生成。
        支持视觉特征缓存和批量处理，是MiniCPM-o部署优化的重要组成部分。

        参数:
            input_ids: 输入token序列 [batch_size, seq_len]
            images: 图像输入 [batch_size, channels, height, width]
            vision_hidden_states: 预计算的视觉特征（可选，用于缓存优化）
            return_vision_hidden_states: 是否返回视觉特征（用于后续缓存）
            **kwargs: 传递给generate()的其他参数

        返回:
            torch.Tensor 或 tuple: 生成的token序列，可选返回视觉特征

        技术优势:
        1. 视觉特征缓存：避免重复计算视觉特征
        2. 推理模式优化：使用torch.inference_mode()提高效率
        3. 内存高效：支持大批量推理
        4. vLLM兼容：与vLLM推理引擎无缝集成
        """
        # 步骤1: 准备模型输入
        model_inputs = {'input_ids': input_ids}

        # 视觉输入处理：支持缓存优化
        if vision_hidden_states is None:
            # 首次推理：使用原始图像
            model_inputs['pixel_values'] = images
        else:
            # 缓存推理：使用预计算的视觉特征
            model_inputs['vision_hidden_states'] = vision_hidden_states

        # 步骤2: 推理模式下执行生成
        with torch.inference_mode():
            # 获取多模态嵌入：执行视觉-文本特征融合
            inputs_embeds, vision_hidden_states = self.model.get_vllm_embedding(model_inputs)

            # 执行文本生成：使用融合后的多模态嵌入
            result = self.generate(
                inputs_embeds=inputs_embeds,  # 使用多模态嵌入而非token ID
                **kwargs                      # 传递生成参数（max_length, temperature等）
            )

        # 步骤3: 返回结果
        if return_vision_hidden_states:
            # 返回生成结果和视觉特征（用于后续缓存）
            return result, vision_hidden_states

        # 只返回生成结果
        return result


    def initialize_vision_tokenizer(self, mm_use_im_start_end, tokenizer, device,
                                    tune_mm_mlp_adapter=False):
        """
        初始化视觉tokenizer - 设置多模态特殊token

        该方法是MiniCPM-o多模态对齐的关键步骤，负责：
        1. 添加多模态特殊token到词汇表
        2. 初始化新token的嵌入权重
        3. 配置训练参数的梯度设置

        参数:
            mm_use_im_start_end: 是否使用图像开始/结束标记
            tokenizer: 分词器对象，需要扩展多模态token
            device: 设备类型（CPU/GPU）
            tune_mm_mlp_adapter: 是否微调多模态MLP适配器

        技术细节:
        - 添加<im_patch>、<im_start>、<im_end>等核心多模态token
        - 使用平均初始化策略为新token设置合理的初始嵌入
        - 支持目标检测相关的特殊token（<box>、<ref>等）
        - 配置LLaVA预训练兼容的梯度设置
        """
        # 步骤1: 配置基础的多模态设置
        self.model.vision_config.use_im_start_end = mm_use_im_start_end

        # 步骤2: 添加图像patch token
        # <im_patch>是最基础的多模态token，标记图像特征的位置
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))  # 调整嵌入层大小

        # 步骤3: 添加图像开始/结束标记（如果启用）
        if mm_use_im_start_end:
            # 添加<im_start>和<im_end>标记
            # 这些标记用于精确定位图像特征在序列中的位置
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            # 获取新添加token的ID并保存到配置中
            self.model.vision_config.im_start_token, self.model.vision_config.im_end_token = tokenizer.convert_tokens_to_ids(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

            # 步骤4: 初始化新token的嵌入权重
            if num_new_tokens > 0:
                # 获取输入和输出嵌入层的权重
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                # 计算现有token嵌入的平均值作为新token的初始值
                # 这是一种常用的初始化策略，避免随机初始化带来的不稳定性
                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                # 将平均值赋给新添加的token嵌入
                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            # 步骤5: 添加目标检测和引用相关的特殊token
            # 这些token用于支持更高级的视觉理解任务，如目标检测、引用表达式理解等
            num_new_tokens = tokenizer.add_tokens(
                ['<box>', '</box>', '<ref>', '</ref>', '<quad>', '</quad>'], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            # 为新的SFT（监督微调）token初始化嵌入权重
            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                # 使用相同的平均初始化策略
                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            # 步骤6: 配置多模态MLP适配器的训练设置
            if tune_mm_mlp_adapter:
                # 保存原始嵌入参数，用于LLaVA预训练兼容性
                self.model.orig_embeds_params = [
                    self.get_input_embeddings().weight.data.clone().to(device=device)]

                # 设置梯度计算：只训练输入嵌入，冻结输出嵌入
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True   # 允许输入嵌入更新
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False  # 冻结输出嵌入

        # 步骤7: 保存图像patch token的ID到配置中
        # 这个ID在多模态特征融合时用于识别图像位置
        self.model.vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IMAGE_PATCH_TOKEN])[0]

        # 调试信息：输出tokenizer配置
        print(f'Tokenizer: {tokenizer}\n patch_token_id: {self.model.vision_config.im_patch_token}, visoin_config: {self.model.vision_config}', flush=True)


# ========== 模型注册 ==========
# 将OmniLMM模型注册到transformers库中，使其可以通过AutoConfig和AutoModelForCausalLM加载

# 注册配置类：使transformers能够识别"omnilmm"模型类型
AutoConfig.register("omnilmm", OmniLMMConfig)

# 注册模型类：将OmniLMMConfig与OmniLMMForCausalLM关联
# 这样就可以使用AutoModelForCausalLM.from_pretrained()加载MiniCPM-o模型
AutoModelForCausalLM.register(OmniLMMConfig, OmniLMMForCausalLM)
