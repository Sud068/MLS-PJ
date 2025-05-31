"""
日志系统
提供统一的日志记录功能
"""

import logging
import logging.handlers
import os
import sys
from typing import Optional, Dict, Any
import json
import time
import inspect


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""

    COLORS = {
        'DEBUG': '\033[94m',  # 蓝色
        'INFO': '\033[92m',  # 绿色
        'WARNING': '\033[93m',  # 黄色
        'ERROR': '\033[91m',  # 红色
        'CRITICAL': '\033[95m',  # 紫色
        'RESET': '\033[0m'  # 重置颜色
    }

    def format(self, record):
        """格式化日志记录"""
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logger(name: str = 'xai_toolkit',
                 level: int = logging.INFO,
                 log_file: Optional[str] = None,
                 console: bool = True,
                 json_format: bool = False) -> logging.Logger:
    """
    设置并返回配置好的日志记录器

    参数:
    name: 日志记录器名称
    level: 日志级别 (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file: 日志文件路径 (None表示不写入文件)
    console: 是否在控制台输出
    json_format: 是否使用JSON格式

    返回:
    配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # 清除现有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        if json_format:
            console_handler.setFormatter(logging.Formatter('%(message)s'))
        else:
            formatter = ColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        if json_format:
            file_handler.setFormatter(logging.Formatter('%(message)s'))
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class JSONFormatter(logging.Formatter):
    """JSON格式的日志格式化器"""

    def format(self, record):
        """格式化日志记录为JSON"""
        log_record = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(record.created)),
            'name': record.name,
            'level': record.levelname,
            'message': record.getMessage(),
            'pathname': record.pathname,
            'lineno': record.lineno,
            'funcName': record.funcName
        }

        # 添加额外字段
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)
        if record.stack_info:
            log_record['stack'] = self.formatStack(record.stack_info)

        return json.dumps(log_record)


def log_execution(logger: logging.Logger, level: int = logging.INFO):
    """
    记录函数执行的装饰器

    参数:
    logger: 日志记录器
    level: 日志级别
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # 获取函数信息
            func_name = func.__name__
            module = inspect.getmodule(func).__name__
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={repr(v)}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)

            # 记录开始
            logger.log(level, f"Executing {module}.{func_name}({signature})")

            try:
                # 执行函数
                result = func(*args, **kwargs)

                # 记录结果
                logger.log(level, f"Completed {module}.{func_name} => {repr(result)}")
                return result
            except Exception as e:
                # 记录异常
                logger.error(f"Error in {module}.{func_name}: {str(e)}", exc_info=True)
                raise

        return wrapper

    return decorator


def log_to_dict(log_file: str) -> list:
    """
    读取日志文件并转换为字典列表

    参数:
    log_file: 日志文件路径

    返回:
    日志记录字典列表
    """
    logs = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                log_data = json.loads(line)
                logs.append(log_data)
            except json.JSONDecodeError:
                # 处理非JSON格式的日志
                parts = line.strip().split(' - ', 3)
                if len(parts) == 4:
                    timestamp, name, level, message = parts
                    logs.append({
                        'timestamp': timestamp,
                        'name': name,
                        'level': level,
                        'message': message
                    })
    return logs