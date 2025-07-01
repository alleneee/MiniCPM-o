# 图像处理和模型工具函数
from torchvision import transforms
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from transformers import AutoConfig
from PIL import Image
from io import BytesIO
import torch.distributed as dist
import numpy as np
import pickle
import base64
import cv2
import os
import torch
from transformers import AutoConfig, StoppingCriteria

try:
    from timm.data.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
except ImportError:
    # 如果导入失败，手动定义CLIP模型的均值和标准差
    OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def auto_upgrade(config):
    """
    自动升级LLaVA模型配置
    
    参数:
        config: 模型配置路径
    """
    cfg = AutoConfig.from_pretrained(config)
    if 'llava' in config and cfg.model_type != 'llava':
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input(
            "Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = 'LlavaLlamaForCausalLM'
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)


class KeywordsStoppingCriteria(StoppingCriteria):
    """
    基于关键词的生成停止条件
    
    当生成的文本中包含指定关键词时停止生成
    """
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(
                output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


def auto_upgrade(config):
    """
    自动升级LLaVA模型配置
    
    参数:
        config: 模型配置路径
    """
    cfg = AutoConfig.from_pretrained(config)
    if 'llava' in config and cfg.model_type != 'llava':
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input(
            "Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = 'LlavaLlamaForCausalLM'
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)

# aug functions


def identity_func(img):
    """
    图像恒等函数
    
    返回原图
    """
    return img


def autocontrast_func(img, cutoff=0):
    '''
    自动对比度函数，与PIL.ImageOps.autocontrast功能相同
    
    参数:
        img: 输入图像
        cutoff: 剪裁百分比
        
    返回:
        应用自动对比度后的图像
    '''
    n_bins = 256

    def tune_channel(ch):
        n = ch.size
        cut = cutoff * n // 100
        if cut == 0:
            high, low = ch.max(), ch.min()
        else:
            hist = cv2.calcHist([ch], [0], None, [n_bins], [0, n_bins])
            low = np.argwhere(np.cumsum(hist) > cut)
            low = 0 if low.shape[0] == 0 else low[0]
            high = np.argwhere(np.cumsum(hist[::-1]) > cut)
            high = n_bins - 1 if high.shape[0] == 0 else n_bins - 1 - high[0]
        if high <= low:
            table = np.arange(n_bins)
        else:
            scale = (n_bins - 1) / (high - low)
            table = np.arange(n_bins) * scale - low * scale
            table[table < 0] = 0
            table[table > n_bins - 1] = n_bins - 1
        table = table.clip(0, 255).astype(np.uint8)
        return table[ch]

    channels = [tune_channel(ch) for ch in cv2.split(img)]
    out = cv2.merge(channels)
    return out


def equalize_func(img):
    '''
    直方图均衡化函数，与PIL.ImageOps.equalize功能相同
    PIL的实现与cv2.equalize不同
    
    参数:
        img: 输入图像
        
    返回:
        均衡化后的图像
    '''
    n_bins = 256

    def tune_channel(ch):
        hist = cv2.calcHist([ch], [0], None, [n_bins], [0, n_bins])
        non_zero_hist = hist[hist != 0].reshape(-1)
        step = np.sum(non_zero_hist[:-1]) // (n_bins - 1)
        if step == 0:
            return ch
        n = np.empty_like(hist)
        n[0] = step // 2
        n[1:] = hist[:-1]
        table = (np.cumsum(n) // step).clip(0, 255).astype(np.uint8)
        return table[ch]

    channels = [tune_channel(ch) for ch in cv2.split(img)]
    out = cv2.merge(channels)
    return out


def rotate_func(img, degree, fill=(0, 0, 0)):
    '''
    旋转图像函数，与PIL类似，使用角度而非弧度
    
    参数:
        img: 输入图像
        degree: 旋转角度
        fill: 填充颜色
        
    返回:
        旋转后的图像
    '''
    H, W = img.shape[0], img.shape[1]
    center = W / 2, H / 2
    M = cv2.getRotationMatrix2D(center, degree, 1)
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill)
    return out


def solarize_func(img, thresh=128):
    '''
    曝光效果函数，大于阈值的像素值反转
    
    参数:
        img: 输入图像
        thresh: 阈值
        
    返回:
        应用曝光效果后的图像
    '''
    table = np.array([el if el < thresh else 255 - el for el in range(256)])
    table = table.clip(0, 255).astype(np.uint8)
    out = table[img]
    return out


def color_func(img, factor):
    '''
    颜色增强函数，与PIL.ImageEnhance.Color功能相同
    
    参数:
        img: 输入图像
        factor: 增强系数
        
    返回:
        颜色增强后的图像
    '''
    # 根据PIL定义的实现，速度较慢
    #  degenerate = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
    #  out = blend(degenerate, img, factor)
    #  M = (
    #      np.eye(3) * factor
    #      + np.float32([0.114, 0.587, 0.299]).reshape(3, 1) * (1. - factor)
    #  )[np.newaxis, np.newaxis, :]
    M = (
        np.float32([
            [0.886, -0.114, -0.114],
            [-0.587, 0.413, -0.587],
            [-0.299, -0.299, 0.701]]) * factor
        + np.float32([[0.114], [0.587], [0.299]])
    )
    out = np.matmul(img, M).clip(0, 255).astype(np.uint8)
    return out


def contrast_func(img, factor):
    """
    对比度增强函数，与PIL.ImageEnhance.Contrast功能相同
    
    参数:
        img: 输入图像
        factor: 增强系数
        
    返回:
        对比度增强后的图像
    """
    mean = np.sum(np.mean(img, axis=(0, 1)) * np.array([0.114, 0.587, 0.299]))
    table = np.array([(
        el - mean) * factor + mean
        for el in range(256)
    ]).clip(0, 255).astype(np.uint8)
    out = table[img]
    return out


def brightness_func(img, factor):
    '''
    亮度增强函数，与PIL.ImageEnhance.Brightness功能相同
    
    参数:
        img: 输入图像
        factor: 增强系数
        
    返回:
        亮度增强后的图像
    '''
    table = (np.arange(256, dtype=np.float32) *
             factor).clip(0, 255).astype(np.uint8)
    out = table[img]
    return out


def sharpness_func(img, factor):
    '''
    锐度增强函数，与PIL.ImageEnhance.Sharpness功能相同
    
    参数:
        img: 输入图像
        factor: 增强系数
        
    返回:
        锐度增强后的图像
    '''
    kernel = np.ones((3, 3), dtype=np.float32)
    kernel[1][1] = 5
    kernel /= 13
    degenerate = cv2.filter2D(img, -1, kernel)
    if factor == 0.0:
        out = degenerate
    elif factor == 1.0:
        out = img
    else:
        out = img.astype(np.float32)
        degenerate = degenerate.astype(np.float32)[1:-1, 1:-1, :]
        out[1:-1, 1:-1, :] = degenerate + factor * \
            (out[1:-1, 1:-1, :] - degenerate)
        out = out.astype(np.uint8)
    return out


def shear_x_func(img, factor, fill=(0, 0, 0)):
    '''
    水平剪切函数
    
    参数:
        img: 输入图像
        factor: 剪切系数
        fill: 填充颜色
        
    返回:
        水平剪切后的图像
    '''
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, factor, 0], [0, 1, 0]])
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill,
                         flags=cv2.INTER_LINEAR).astype(np.uint8)
    return out


def translate_x_func(img, offset, fill=(0, 0, 0)):
    '''
    水平平移函数，与PIL.Image.transform功能相同
    
    参数:
        img: 输入图像
        offset: 平移距离
        fill: 填充颜色
        
    返回:
        水平平移后的图像
    '''
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, 0, -offset], [0, 1, 0]])
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill,
                         flags=cv2.INTER_LINEAR).astype(np.uint8)
    return out


def translate_y_func(img, offset, fill=(0, 0, 0)):
    '''
    垂直平移函数，与PIL.Image.transform功能相同
    
    参数:
        img: 输入图像
        offset: 平移距离
        fill: 填充颜色
        
    返回:
        垂直平移后的图像
    '''
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, 0, 0], [0, 1, -offset]])
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill,
                         flags=cv2.INTER_LINEAR).astype(np.uint8)
    return out


def posterize_func(img, bits):
    '''
    色调分离函数，与PIL.ImageOps.posterize功能相同
    
    参数:
        img: 输入图像
        bits: 分离位数
        
    返回:
        色调分离后的图像
    '''
    out = np.bitwise_and(img, np.uint8(255 << (8 - bits)))
    return out


def shear_y_func(img, factor, fill=(0, 0, 0)):
    '''
    垂直剪切函数
    
    参数:
        img: 输入图像
        factor: 剪切系数
        fill: 填充颜色
        
    返回:
        垂直剪切后的图像
    '''
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, 0, 0], [factor, 1, 0]])
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill,
                         flags=cv2.INTER_LINEAR).astype(np.uint8)
    return out


def cutout_func(img, pad_size, replace=(0, 0, 0)):
    '''
    随机遮挡函数
    
    参数:
        img: 输入图像
        pad_size: 遮挡大小
        replace: 替换颜色
        
    返回:
        随机遮挡后的图像
    '''
    replace = np.array(replace, dtype=np.uint8)
    H, W = img.shape[0], img.shape[1]
    rh, rw = np.random.random(2)
    pad_size = pad_size // 2
    ch, cw = int(rh * H), int(rw * W)
    x1, x2 = max(ch - pad_size, 0), min(ch + pad_size, H)
    y1, y2 = max(cw - pad_size, 0), min(cw + pad_size, W)
    out = img.copy()
    out[x1:x2, y1:y2, :] = replace
    return out


# 增强级别到参数的转换函数
def enhance_level_to_args(MAX_LEVEL):
    """
    增强级别到参数转换函数
    
    参数:
        MAX_LEVEL: 最大增强级别
        
    返回:
        转换函数
    """
    def level_to_args(level):
        return ((level / MAX_LEVEL) * 1.8 + 0.1,)
    return level_to_args


def shear_level_to_args(MAX_LEVEL, replace_value):
    """
    剪切级别到参数转换函数
    
    参数:
        MAX_LEVEL: 最大剪切级别
        replace_value: 替换值
        
    返回:
        转换函数
    """
    def level_to_args(level):
        level = (level / MAX_LEVEL) * 0.3
        if np.random.random() > 0.5:
            level = -level
        return (level, replace_value)

    return level_to_args


def translate_level_to_args(translate_const, MAX_LEVEL, replace_value):
    """
    平移级别到参数转换函数
    
    参数:
        translate_const: 平移常数
        MAX_LEVEL: 最大级别
        replace_value: 替换值
        
    返回:
        转换函数
    """
    def level_to_args(level):
        level = (level / MAX_LEVEL) * float(translate_const)
        if np.random.random() > 0.5:
            level = -level
        return (level, replace_value)

    return level_to_args


def cutout_level_to_args(cutout_const, MAX_LEVEL, replace_value):
    """
    遮挡级别到参数转换函数
    
    参数:
        cutout_const: 遮挡常数
        MAX_LEVEL: 最大级别
        replace_value: 替换值
        
    返回:
        转换函数
    """
    def level_to_args(level):
        level = int((level / MAX_LEVEL) * cutout_const)
        return (level, replace_value)

    return level_to_args


def solarize_level_to_args(MAX_LEVEL):
    """
    曝光级别到参数转换函数
    
    参数:
        MAX_LEVEL: 最大级别
        
    返回:
        转换函数
    """
    def level_to_args(level):
        level = int((level / MAX_LEVEL) * 256)
        return (level, )
    return level_to_args


def none_level_to_args(level):
    """
    空级别转换函数，返回空元组
    
    参数:
        level: 级别
        
    返回:
        空元组
    """
    return ()


def posterize_level_to_args(MAX_LEVEL):
    """
    色调分离级别到参数转换函数
    
    参数:
        MAX_LEVEL: 最大级别
        
    返回:
        转换函数
    """
    def level_to_args(level):
        level = int((level / MAX_LEVEL) * 4)
        return (level, )
    return level_to_args


def rotate_level_to_args(MAX_LEVEL, replace_value):
    """
    旋转级别到参数转换函数
    
    参数:
        MAX_LEVEL: 最大级别
        replace_value: 替换值
        
    返回:
        转换函数
    """
    def level_to_args(level):
        level = (level / MAX_LEVEL) * 30
        if np.random.random() < 0.5:
            level = -level
        return (level, replace_value)

    return level_to_args

# level to args
func_dict = {
    'Identity': identity_func,
    'AutoContrast': autocontrast_func,
    'Equalize': equalize_func,
    'Rotate': rotate_func,
    'Solarize': solarize_func,
    'Color': color_func,
    'Contrast': contrast_func,
    'Brightness': brightness_func,
    'Sharpness': sharpness_func,
    'ShearX': shear_x_func,
    'TranslateX': translate_x_func,
    'TranslateY': translate_y_func,
    'Posterize': posterize_func,
    'ShearY': shear_y_func,
}

translate_const = 10
MAX_LEVEL = 10
replace_value = (128, 128, 128)
arg_dict = {
    'Identity': none_level_to_args,
    'AutoContrast': none_level_to_args,
    'Equalize': none_level_to_args,
    'Rotate': rotate_level_to_args(MAX_LEVEL, replace_value),
    'Solarize': solarize_level_to_args(MAX_LEVEL),
    'Color': enhance_level_to_args(MAX_LEVEL),
    'Contrast': enhance_level_to_args(MAX_LEVEL),
    'Brightness': enhance_level_to_args(MAX_LEVEL),
    'Sharpness': enhance_level_to_args(MAX_LEVEL),
    'ShearX': shear_level_to_args(MAX_LEVEL, replace_value),
    'TranslateX': translate_level_to_args(
        translate_const, MAX_LEVEL, replace_value
    ),
    'TranslateY': translate_level_to_args(
        translate_const, MAX_LEVEL, replace_value
    ),
    'Posterize': posterize_level_to_args(MAX_LEVEL),
    'ShearY': shear_level_to_args(MAX_LEVEL, replace_value),
}


class RandomAugment(object):
    """
    随机图像增强类
    
    在给定的增强操作列表中随机选择N个操作应用于图像
    """
    def __init__(self, N=2, M=10, isPIL=False, augs=[]):
        """
        初始化随机增强器
        
        参数:
            N: 要应用的增强操作数量
            M: 增强操作的强度
            isPIL: 输入是否为PIL图像
            augs: 可用的增强操作列表，为空则使用所有操作
        """
        self.N = N
        self.M = M
        self.isPIL = isPIL
        if augs:
            self.augs = augs
        else:
            self.augs = list(arg_dict.keys())

    def get_random_ops(self):
        """
        获取随机增强操作列表
        
        返回:
            操作列表，每个操作为(操作名, 概率, 强度)的元组
        """
        sampled_ops = np.random.choice(self.augs, self.N)
        return [(op, 0.5, self.M) for op in sampled_ops]

    def __call__(self, img):
        """
        对图像应用随机增强
        
        参数:
            img: 输入图像
            
        返回:
            增强后的图像
        """
        if self.isPIL:
            img = np.array(img)
        ops = self.get_random_ops()
        for name, prob, level in ops:
            if np.random.random() > prob:
                continue
            args = arg_dict[name](level)
            img = func_dict[name](img, *args)
        return img


def build_transform(is_train, randaug=True, input_size=224, interpolation='bicubic', std_mode='IMAGENET_INCEPTION'):
    """
    构建图像转换管道
    
    参数:
        is_train: 是否为训练模式
        randaug: 是否应用随机增强
        input_size: 输入图像大小
        interpolation: 插值方法
        std_mode: 标准化模式，可选'IMAGENET_INCEPTION'或'OPENAI_CLIP'
        
    返回:
        转换管道
    """
    if std_mode == 'IMAGENET_INCEPTION':
        mean = IMAGENET_INCEPTION_MEAN
        std = IMAGENET_INCEPTION_STD
    elif std_mode == 'OPENAI_CLIP':
        mean = OPENAI_CLIP_MEAN
        std = OPENAI_CLIP_STD
    else:
        raise NotImplementedError

    if is_train:
        crop_scale = float(os.environ.get('TRAIN_CROP_SCALE', 0.9999))
        t = [
            RandomResizedCropAndInterpolation(
                input_size, scale=(crop_scale, 1.0), interpolation='bicubic'),
            # transforms.RandomHorizontalFlip(),
        ]
        if randaug and os.environ.get('TRAIN_DO_AUG', 'False') == 'True':
            print(f'@@@@@ Do random aug during training', flush=True)
            t.append(
                RandomAugment(
                    2, 7, isPIL=True,
                    augs=[
                        'Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                        'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate',
                    ]))
        else:
            print(f'@@@@@ Skip random aug during training', flush=True)
        t += [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        t = transforms.Compose(t)
    else:
        t = transforms.Compose([
            transforms.Resize((input_size, input_size),
                               interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    return t


def img2b64(img_path):
    """
    将图像转换为base64编码字符串
    
    参数:
        img_path: 图像路径
        
    返回:
        base64编码的字符串
    """
    img = Image.open(img_path)  # 打开图像文件
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)  # 二进制数据
    base64_str = base64_str.decode("utf-8")  # 转为字符串
    return base64_str


def str2b64(str):
    """
    将字符串转换为base64编码
    
    参数:
        str: 输入字符串
        
    返回:
        base64编码的字符串
    """
    return base64.b64encode(str.encode('utf-8')).decode('utf-8')


def b642str(b64):
    """
    将base64编码解码为字符串
    
    参数:
        b64: base64编码的字符串
        
    返回:
        解码后的字符串
    """
    return base64.b64decode(b64).decode('utf-8')


def is_dist_avail_and_initialized():
    """
    检查分布式环境是否可用且已初始化
    
    返回:
        布尔值，表示分布式环境是否可用且已初始化
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """
    获取分布式训练的世界大小（进程数）
    
    返回:
        整数，表示分布式训练的进程总数，单进程训练时为1
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    获取当前进程的rank
    
    返回:
        整数，表示当前进程的rank，单进程训练时为0
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def all_gather(data):
    """
    收集所有进程中的数据（不一定是张量）
    
    参数:
        data: 任何可序列化对象
        
    返回:
        list[data]: 从各个进程收集的数据列表
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # 序列化为张量
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # 获取每个进程的张量大小
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # 接收所有进程的张量
    # 我们对张量进行填充，因为torch all_gather不支持收集不同形状的张量
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def mean(lst):
    """
    计算列表元素的平均值
    
    参数:
        lst: 数值列表
        
    返回:
        平均值
    """
    return sum(lst) / len(lst)


def stop_gradient_by_name(name: str):
    """
    通过名称停止梯度传播的应用函数
    
    参数:
        name: 模块属性名称
        
    返回:
        应用函数，用于在模块上执行梯度停止
    """
    def apply_fn(module):
        if hasattr(module, name):
            getattr(module, name).requires_grad_(False)

    return apply_fn
