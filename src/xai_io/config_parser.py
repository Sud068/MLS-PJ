"""
配置文件解析器
支持YAML和JSON格式的配置文件
"""

import os
import yaml
import json
import logging
from typing import Any, Dict, Optional
from .data_loader import DataLoader

logger = logging.getLogger(__name__)


class ConfigParser:
    """
    配置文件解析器

    加载和解析配置文件，支持变量替换和继承
    """

    def __init__(self, config_path: str, env_vars: bool = True, **kwargs):
        """
        初始化配置解析器

        参数:
        config_path: 配置文件路径
        env_vars: 是否替换环境变量
        kwargs: 额外变量
        """
        self.config_path = config_path
        self.env_vars = env_vars
        self.extra_vars = kwargs
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        # 加载原始配置
        config = DataLoader.load(self.config_path)

        # 处理继承
        if 'base_config' in config:
            base_path = config['base_config']
            # 解析基本路径
            if not os.path.isabs(base_path):
                base_dir = os.path.dirname(self.config_path)
                base_path = os.path.join(base_dir, base_path)

            base_parser = ConfigParser(base_path, self.env_vars, **self.extra_vars)
            base_config = base_parser.config

            # 合并配置 (当前配置覆盖基本配置)
            merged_config = {**base_config, **config}
            config = merged_config

        # 处理变量替换
        config = self._replace_vars(config)

        return config

    def _replace_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """递归替换变量"""
        if isinstance(config, dict):
            return {k: self._replace_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._replace_vars(item) for item in config]
        elif isinstance(config, str):
            return self._replace_str_vars(config)
        else:
            return config

    def _replace_str_vars(self, s: str) -> str:
        """替换字符串中的变量"""
        # 替换环境变量
        if self.env_vars:
            s = os.path.expandvars(s)

        # 替换额外变量
        for key, value in self.extra_vars.items():
            placeholder = f"${{{key}}}"
            s = s.replace(placeholder, str(value))

        return s

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        获取配置值

        参数:
        key: 配置键 (支持点分隔路径)
        default: 默认值

        返回:
        配置值
        """
        keys = key.split('.')
        current = self.config

        try:
            for k in keys:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                elif isinstance(current, list) and k.isdigit():
                    current = current[int(k)]
                else:
                    return default
            return current
        except (KeyError, IndexError, TypeError):
            return default

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        获取整个配置节

        参数:
        section: 节名称 (支持点分隔路径)

        返回:
        配置字典
        """
        result = self.get(section)
        if not isinstance(result, dict):
            return {}
        return result

    def save(self, output_path: str, format: Optional[str] = None):
        """
        保存配置到文件

        参数:
        output_path: 输出文件路径
        format: 输出格式 (自动检测)
        """
        DataLoader.save(self.config, output_path, format=format)

    def update(self, updates: Dict[str, Any]):
        """
        更新配置

        参数:
        updates: 要更新的键值对
        """
        for key, value in updates.items():
            keys = key.split('.')
            current = self.config
            for i, k in enumerate(keys[:-1]):
                if k not in current or not isinstance(current[k], dict):
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """返回配置字典"""
        return self.config

    @staticmethod
    def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        验证配置是否符合模式

        参数:
        config: 配置字典
        schema: 模式字典

        返回:
        是否有效
        """
        # 简化版验证 - 实际应用中可以使用JSON Schema
        for key, expected_type in schema.items():
            if key not in config:
                logger.warning(f"缺少必需的配置项: {key}")
                return False
            if not isinstance(config[key], expected_type):
                logger.warning(f"配置项 {key} 类型错误: 期望 {expected_type}, 实际 {type(config[key])}")
                return False
        return True