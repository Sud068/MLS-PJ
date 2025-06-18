"""
输入验证工具
提供数据验证和清理功能
"""
import os

import numpy as np
import re
from typing import Any, Union, Tuple, List, Dict, Optional


class ValidationError(ValueError):
    """自定义验证错误"""
    pass


def validate_not_none(value: Any, name: str = "value") -> Any:
    """
    验证值不为None

    参数:
    value: 要验证的值
    name: 值名称 (用于错误消息)

    返回:
    验证后的值

    抛出:
    ValidationError: 如果值为None
    """
    if value is None:
        raise ValidationError(f"{name} cannot be None")
    return value


def validate_type(value: Any, expected_type: type, name: str = "value") -> Any:
    """
    验证值的类型

    参数:
    value: 要验证的值
    expected_type: 期望的类型
    name: 值名称 (用于错误消息)

    返回:
    验证后的值

    抛出:
    ValidationError: 如果类型不匹配
    """
    if not isinstance(value, expected_type):
        raise ValidationError(
            f"{name} must be of type {expected_type.__name__}, got {type(value).__name__}"
        )
    return value


def validate_numeric(value: Union[int, float],
                     min_val: Optional[float] = None,
                     max_val: Optional[float] = None,
                     name: str = "value") -> Union[int, float]:
    """
    验证数值范围

    参数:
    value: 要验证的数值
    min_val: 最小值 (包含)
    max_val: 最大值 (包含)
    name: 值名称 (用于错误消息)

    返回:
    验证后的值

    抛出:
    ValidationError: 如果值不在范围内
    """
    if min_val is not None and value < min_val:
        raise ValidationError(f"{name} must be >= {min_val}, got {value}")
    if max_val is not None and value > max_val:
        raise ValidationError(f"{name} must be <= {max_val}, got {value}")
    return value


def validate_array(arr: np.ndarray,
                   shape: Optional[Tuple] = None,
                   dtype: Optional[type] = None,
                   name: str = "array") -> np.ndarray:
    """
    验证NumPy数组

    参数:
    arr: 要验证的数组
    shape: 期望的形状 (None表示不验证)
    dtype: 期望的数据类型 (None表示不验证)
    name: 数组名称 (用于错误消息)

    返回:
    验证后的数组

    抛出:
    ValidationError: 如果验证失败
    """
    if not isinstance(arr, np.ndarray):
        raise ValidationError(f"{name} must be a NumPy array, got {type(arr).__name__}")

    if shape is not None:
        if arr.shape != shape:
            raise ValidationError(
                f"{name} must have shape {shape}, got {arr.shape}"
            )

    if dtype is not None:
        if arr.dtype != dtype:
            raise ValidationError(
                f"{name} must have dtype {dtype.__name__}, got {arr.dtype.name}"
            )

    return arr


def validate_file_path(path: str,
                       must_exist: bool = False,
                       extension: Optional[str] = None,
                       name: str = "file path") -> str:
    """
    验证文件路径

    参数:
    path: 文件路径
    must_exist: 文件是否必须存在
    extension: 期望的文件扩展名 (None表示不验证)
    name: 路径名称 (用于错误消息)

    返回:
    验证后的路径

    抛出:
    ValidationError: 如果验证失败
    """
    if not isinstance(path, str):
        raise ValidationError(f"{name} must be a string, got {type(path).__name__}")

    if must_exist and not os.path.isfile(path):
        raise ValidationError(f"{name} does not exist: {path}")

    if extension is not None:
        if not path.lower().endswith(extension.lower()):
            raise ValidationError(
                f"{name} must have extension '{extension}', got '{os.path.splitext(path)[1]}'"
            )

    return path


def validate_choice(value: Any, choices: List[Any], name: str = "value") -> Any:
    """
    验证值在允许的选择范围内

    参数:
    value: 要验证的值
    choices: 允许的选择列表
    name: 值名称 (用于错误消息)

    返回:
    验证后的值

    抛出:
    ValidationError: 如果值不在选择范围内
    """
    if value not in choices:
        choices_str = ", ".join(str(c) for c in choices)
        raise ValidationError(
            f"{name} must be one of [{choices_str}], got {value}"
        )
    return value


def validate_dict(value: dict,
                  required_keys: Optional[List[str]] = None,
                  optional_keys: Optional[List[str]] = None,
                  name: str = "dictionary") -> dict:
    """
    验证字典结构

    参数:
    value: 要验证的字典
    required_keys: 必须存在的键列表
    optional_keys: 可选键列表 (如果提供，字典不能包含其他键)
    name: 字典名称 (用于错误消息)

    返回:
    验证后的字典

    抛出:
    ValidationError: 如果验证失败
    """
    if not isinstance(value, dict):
        raise ValidationError(f"{name} must be a dictionary, got {type(value).__name__}")

    if required_keys:
        for key in required_keys:
            if key not in value:
                raise ValidationError(f"{name} is missing required key: '{key}'")

    if optional_keys is not None:
        allowed_keys = set(required_keys or []) | set(optional_keys)
        for key in value.keys():
            if key not in allowed_keys:
                raise ValidationError(f"{name} contains unexpected key: '{key}'")

    return value


def validate_email(email: str, name: str = "email") -> str:
    """
    验证电子邮件格式

    参数:
    email: 要验证的电子邮件
    name: 值名称 (用于错误消息)

    返回:
    验证后的电子邮件

    抛出:
    ValidationError: 如果电子邮件格式无效
    """
    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    if not re.match(pattern, email):
        raise ValidationError(f"Invalid {name} format: {email}")
    return email


def validate_url(url: str, name: str = "URL") -> str:
    """
    验证URL格式

    参数:
    url: 要验证的URL
    name: 值名称 (用于错误消息)

    返回:
    验证后的URL

    抛出:
    ValidationError: 如果URL格式无效
    """
    pattern = r'^(https?|ftp)://[^\s/$.?#].[^\s]*$'
    if not re.match(pattern, url, re.IGNORECASE):
        raise ValidationError(f"Invalid {name} format: {url}")
    return url


def sanitize_string(value: str) -> str:
    """
    清理字符串，移除潜在危险字符

    参数:
    value: 要清理的字符串

    返回:
    清理后的字符串
    """
    # 移除控制字符
    sanitized = re.sub(r'[\x00-\x1F\x7F]', '', value)
    # 移除HTML标签
    sanitized = re.sub(r'<[^>]*>', '', sanitized)
    # 移除危险字符
    sanitized = re.sub(r'[;\\\'"(){}[\]<>]', '', sanitized)
    return sanitized.strip()


def sanitize_dict(data: dict) -> dict:
    """
    清理字典中的所有字符串值

    参数:
    data: 要清理的字典

    返回:
    清理后的字典
    """
    return {k: sanitize_string(v) if isinstance(v, str) else v for k, v in data.items()}