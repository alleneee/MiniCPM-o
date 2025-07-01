# 工具函数库
# 包含日志记录、流处理和其他实用功能

import datetime
import logging
import logging.handlers
import os
import sys

import requests

from omnilmm.constants import LOGDIR

# 服务器错误消息
server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
# 内容审核不通过消息
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None


def build_logger(logger_name, logger_filename):
    """
    构建日志记录器
    
    参数:
        logger_name: 日志记录器名称
        logger_filename: 日志文件名
        
    返回:
        logger: 配置好的日志记录器实例
    """
    global handler

    # 设置日志格式
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 设置根处理器格式
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # 将标准输出和标准错误重定向到日志记录器
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # 获取日志记录器
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # 为所有日志记录器添加文件处理器
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True)
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    将流对象重定向到日志记录器的伪文件对象
    """

    def __init__(self, logger, log_level=logging.INFO):
        """
        初始化流到日志记录器的转换器
        
        参数:
            logger: 日志记录器对象
            log_level: 日志级别，默认为INFO
        """
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        """获取原始终端对象的属性"""
        return getattr(self.terminal, attr)

    def write(self, buf):
        """
        写入数据到日志
        
        参数:
            buf: 要写入的数据缓冲区
        """
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # 根据io.TextIOWrapper文档说明:
            #   在输出时，如果newline为None，任何写入的'\n'字符
            #   将转换为系统默认行分隔符
            # 默认情况下，sys.stdout.write()期望'\n'作为换行符，然后进行转换
            # 因此这仍然是跨平台兼容的
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        """刷新缓冲区中的数据到日志"""
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def disable_torch_init():
    """
    禁用冗余的PyTorch默认初始化，以加速模型创建过程
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    检查文本是否违反OpenAI内容审核API规定
    
    参数:
        text: 需要检查的文本
        
    返回:
        flagged: 布尔值，表示文本是否被标记为违规
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    """
    美化信号量对象的打印输出
    
    参数:
        semaphore: 信号量对象
        
    返回:
        str: 格式化后的信号量状态字符串
    """
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"
