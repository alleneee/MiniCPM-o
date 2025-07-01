# 训练工具函数模块
# 包含用于处理和预处理训练数据的函数

import os
import gc
import copy
import time

import torch
import warnings
import transformers

import numpy as np

from typing import Dict, Optional, Sequence
from omnilmm import conversation as conversation_lib

# 忽略标记索引
IGNORE_INDEX = -100
# 默认图像标记
DEFAULT_IMAGE_TOKEN = "<image>"
# 默认图像块标记
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
# 默认图像开始标记
DEFAULT_IM_START_TOKEN = "<im_start>"
# 默认图像结束标记
DEFAULT_IM_END_TOKEN = "<im_end>"


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """
    使用提供的分词器对字符串列表进行分词
    
    参数:
        strings (Sequence[str]): 需要分词的字符串列表
        tokenizer (transformers.PreTrainedTokenizer): HuggingFace Transformers库的分词器实例
        
    返回:
        Dict: 包含以下键的字典:
            - input_ids: 分词后的输入ID列表
            - labels: 标签ID列表（与输入ID相同）
            - input_ids_lens: 输入ID的长度列表（不包括填充部分）
            - labels_lens: 标签的长度列表（不包括填充部分）
    """
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def omni_preprocess(sources,
                      tokenizer: transformers.PreTrainedTokenizer,
                      generation=False):
    """
    预处理一批对话样本用于训练或生成
    
    参数:
        sources (list): 对话样本列表。每个样本是一个包含以下键的字典列表:
            - 'role' 或 'from': 指定说话者（'user', 'assistant', 'system', 'human', 'gpt'）
            - 'content' 或 'value': 对话内容文本
        tokenizer (transformers.PreTrainedTokenizer): HuggingFace Transformers库的分词器实例
        generation (bool, optional): 如果为True，则准备用于生成的输入（仅包含提示部分）。默认为False
        
    返回:
        dict: 包含以下键的字典:
            - input_ids: 每个样本的分词后输入ID列表
            - labels: 每个样本的标签ID列表，非响应部分的标记设置为ignore_index
    """
    system_content = '您是一名人工智能助手，为人类提供有用的、详细的和礼貌的回答。'
    ignore_index = -100

    response_template = '\n<|assistant|>\n'
    instruction_template = '\n\n'
    response_token_ids = tokenizer.encode(
        response_template, add_special_tokens=False)
    instruction_token_ids = tokenizer.encode(
        instruction_template, add_special_tokens=False)

    batch_input_ids = []
    batch_labels = []
    for i in range(len(sources)):
        new_source = []
        prev_role = 'unexpect'
        for conv_turn in sources[i]:
            role = conv_turn['from'] if 'from' in conv_turn else conv_turn['role']
            content = conv_turn['value'] if 'value' in conv_turn else conv_turn['content']

            role = 'user' if role == 'human' else role
            role = 'assistant' if role == 'gpt' else role

            assert role in ['user', 'assistant']
            assert role != prev_role, f'role={role}, prev_role={prev_role}'
            prev_role = role

            new_turn = {
                'role': role,
                'content': content
            }
            new_source.append(new_turn)
        if new_source[0]['role'] != 'system':
            new_source.insert(0, {'role': 'system', 'content': system_content})

        # TODO: 这里自动添加 '\n' 到结尾
        res_text = tokenizer.apply_chat_template(
            new_source, tokenize=False, add_generation_prompt=generation)
        if not generation:
            res_text = res_text.strip()

        conversations_tokenized = _tokenize_fn([res_text], tokenizer)
        res_input_ids = conversations_tokenized["input_ids"][0]

        # 由于标签和输入ID是指向同一个对象的
        res_labels = copy.deepcopy(conversations_tokenized["labels"][0])

        response_token_ids_idxs = []
        human_token_ids_idxs = []

        for assistant_idx in np.where(res_labels == response_token_ids[0])[0]:
            # 找到响应开始的索引
            if (response_token_ids == res_labels[assistant_idx: assistant_idx + len(
                        response_token_ids)].tolist()
                    ):
                response_token_ids_idxs.append(
                    assistant_idx + len(response_token_ids))

        if len(response_token_ids_idxs) == 0:
            warnings.warn(
                f"无法在以下实例中找到响应键 `{response_template}`: @===>{tokenizer.decode(res_input_ids)}<===@ "
                f'原始文本是 @===>{res_text}<===@'
                f'原始源是 @===>{new_source}<===@'
                f"此实例将在损失计算中被忽略。 "
                f"注意，如果这种情况经常发生，请考虑增加 `max_seq_length`。"
            )
            res_labels[:] = ignore_index

        human_token_ids = instruction_token_ids
        for human_idx in np.where(res_labels == human_token_ids[0])[0]:
            # 找到人类回答开始的索引
            if human_token_ids == res_labels[human_idx: human_idx + len(human_token_ids)].tolist():
                human_token_ids_idxs.append(human_idx)

        if len(human_token_ids_idxs) == 0:
            warnings.warn(
                f"无法在以下实例中找到指令键 `{instruction_template}`: @===>{tokenizer.decode(res_input_ids)}<===@ "
                f'原始文本是 @===>{res_text}<===@'
                f'原始源是 @===>{new_source}<===@'
                f"此实例将在损失计算中被忽略。 "
                f"注意，如果这种情况经常发生，请考虑增加 `max_seq_length`。"
            )
            res_labels[:] = ignore_index

        for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
            # 使PyTorch损失函数忽略所有非响应标记
            if idx != 0:
                res_labels[start:end] = ignore_index
            else:
                res_labels[:end] = ignore_index

        if len(response_token_ids_idxs) < len(human_token_ids_idxs):
            res_labels[human_token_ids_idxs[-1]:] = ignore_index

        batch_input_ids.append(res_input_ids)
        batch_labels.append(res_labels)

    return dict(input_ids=batch_input_ids, labels=batch_labels)
