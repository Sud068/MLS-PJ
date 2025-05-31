"""
XAI Toolkit Core Module
-----------------------
提供可解释AI工具箱的核心抽象接口和基础功能
"""

from .explainer import BaseExplainer, ExplanationResult
from .model_loader import ModelLoader

# 导出核心类
__all__ = [
    'BaseExplainer',
    'ExplanationResult',
    'ModelLoader'
]

# 版本信息
__version__ = "0.1.0"
__author__ = "XAI Toolkit Team"
__license__ = "Apache 2.0"