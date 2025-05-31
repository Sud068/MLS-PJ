"""
模型加载工具
支持从不同框架和文件格式加载模型
"""

import pickle
import joblib
import numpy as np
from typing import Any, Dict, Optional
import logging

# 尝试导入深度学习框架
try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    onnx = None
    ort = None

logger = logging.getLogger(__name__)


class ModelLoader:
    """统一模型加载接口"""

    @staticmethod
    def load(model_path: str,
             framework: Optional[str] = None,
             **kwargs) -> Any:
        """
        加载模型

        参数:
        model_path: 模型文件路径
        framework: 指定框架 ('sklearn', 'pytorch', 'tensorflow', 'onnx')
        kwargs: 框架特定参数

        返回:
        加载的模型对象
        """
        # 自动检测框架 (如果未指定)
        if framework is None:
            framework = ModelLoader.detect_framework(model_path)
            logger.info(f"检测到模型框架: {framework}")

        loader_map = {
            'sklearn': ModelLoader.load_sklearn,
            'pytorch': ModelLoader.load_pytorch,
            'tensorflow': ModelLoader.load_tensorflow,
            'onnx': ModelLoader.load_onnx,
            'joblib': ModelLoader.load_sklearn  # joblib 作为 sklearn 别名
        }

        if framework not in loader_map:
            raise ValueError(f"不支持的框架: {framework}")

        return loader_map[framework](model_path, **kwargs)

    @staticmethod
    def detect_framework(model_path: str) -> str:
        """根据文件扩展名检测框架"""
        if model_path.endswith('.pth') or model_path.endswith('.pt'):
            return 'pytorch'
        elif model_path.endswith('.h5') or model_path.endswith('.keras'):
            return 'tensorflow'
        elif model_path.endswith('.onnx'):
            return 'onnx'
        elif model_path.endswith('.joblib') or model_path.endswith('.pkl') or model_path.endswith('.pickle'):
            return 'sklearn'
        else:
            # 尝试根据内容推断
            try:
                with open(model_path, 'rb') as f:
                    header = f.read(100)
                    if b'ONNX' in header:
                        return 'onnx'
                    elif b'PK' in header:  # ZIP格式 (sklearn/pytorch)
                        return 'sklearn'  # 优先尝试sklearn
            except:
                pass

            raise RuntimeError(f"无法推断模型框架: {model_path}")

    @staticmethod
    def load_sklearn(model_path: str, **kwargs) -> Any:
        """加载scikit-learn风格模型 (包括joblib/pickle)"""
        if model_path.endswith('.joblib'):
            return joblib.load(model_path)
        else:
            with open(model_path, 'rb') as f:
                return pickle.load(f)

    @staticmethod
    def load_pytorch(model_path: str, **kwargs) -> Any:
        """加载PyTorch模型"""
        if torch is None:
            raise ImportError("PyTorch未安装，无法加载模型")

        device = kwargs.get('device', 'cpu')
        map_location = torch.device(device)

        # 检查是否是完整模型还是state_dict
        model = torch.load(model_path, map_location=map_location)

        if isinstance(model, torch.nn.Module):
            return model
        else:
            # 假设是state_dict，需要模型架构
            model_arch = kwargs.get('model_arch')
            if model_arch is None:
                raise ValueError("需要提供model_arch参数来加载state_dict")

            model = model_arch()
            model.load_state_dict(torch.load(model_path, map_location=map_location))
            model.to(device)
            model.eval()
            return model

    @staticmethod
    def load_tensorflow(model_path: str, **kwargs) -> Any:
        """加载TensorFlow/Keras模型"""
        if tf is None:
            raise ImportError("TensorFlow未安装，无法加载模型")

        if model_path.endswith('.h5') or model_path.endswith('.keras'):
            return tf.keras.models.load_model(model_path)
        else:
            # 尝试加载SavedModel格式
            return tf.saved_model.load(model_path)

    @staticmethod
    def load_onnx(model_path: str, **kwargs) -> Any:
        """加载ONNX模型"""
        if onnx is None or ort is None:
            raise ImportError("ONNX或ONNX Runtime未安装")

        # 验证模型
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)

        # 创建推理会话
        providers = ['CPUExecutionProvider']
        if ort.get_device() == 'GPU':
            providers.insert(0, 'CUDAExecutionProvider')

        session_options = kwargs.get('session_options', ort.SessionOptions())
        return ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=providers
        )

    @staticmethod
    def save(model: Any, model_path: str, **kwargs):
        """保存模型到文件 (统一接口)"""
        if isinstance(model, (ort.InferenceSession, onnx.ModelProto)):
            raise NotImplementedError("ONNX模型保存需使用专用方法")

        if hasattr(model, 'save') and callable(model.save):
            # Keras风格保存
            model.save(model_path)
        elif hasattr(model, 'save_model') and callable(model.save_model):
            # PyTorch Lightning风格
            model.save_model(model_path)
        else:
            # 默认使用joblib保存
            joblib.dump(model, model_path)