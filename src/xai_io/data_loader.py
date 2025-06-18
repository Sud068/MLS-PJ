"""
数据加载器
支持多种格式的数据加载和预处理
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import json
import pickle
import logging
import yaml
from typing import Any, Dict, List, Optional, Tuple, Union
import csv
import h5py
import torch
import tensorflow as tf

logger = logging.getLogger(__name__)


class DataLoader:
    """
    通用数据加载器

    支持加载多种格式的数据文件
    """

    @staticmethod
    def load(file_path: str,
             format: Optional[str] = None,
             **kwargs) -> Union[np.ndarray, pd.DataFrame, dict, list, Any]:
        """
        加载数据文件

        参数:
        file_path: 文件路径
        format: 文件格式 (自动检测)
        kwargs: 格式特定参数

        返回:
        加载的数据对象
        """
        # 自动检测格式
        if format is None:
            format = DataLoader._detect_format(file_path)

        # 调用特定加载方法
        loader = getattr(DataLoader, f"_load_{format}", None)
        if loader is None:
            raise ValueError(f"不支持的格式: {format}")

        return loader(file_path, **kwargs)

    @staticmethod
    def _detect_format(file_path: str) -> str:
        """根据文件扩展名检测格式"""
        ext = os.path.splitext(file_path)[1].lower()

        if ext in ['.csv']:
            return 'csv'
        elif ext in ['.json']:
            return 'json'
        elif ext in ['.pkl', '.pickle']:
            return 'pickle'
        elif ext in ['.npy']:
            return 'numpy'
        elif ext in ['.npz']:
            return 'npz'
        elif ext in ['.txt']:
            return 'text'
        elif ext in ['.yaml', '.yml']:
            return 'yaml'
        elif ext in ['.h5', '.hdf5']:
            return 'hdf5'
        elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']:
            return 'image'
        elif ext in ['.pt', '.pth']:
            return 'pytorch'
        elif ext in ['.tflite', '.h5']:  # .h5可能是Keras模型
            return 'tensorflow'
        else:
            raise ValueError(f"无法识别的文件格式: {file_path}")

    @staticmethod
    def _load_csv(file_path: str, **kwargs) -> pd.DataFrame:
        """加载CSV文件"""
        return pd.read_csv(file_path, **kwargs)

    @staticmethod
    def _load_json(file_path: str, **kwargs) -> Union[dict, list]:
        """加载JSON文件"""
        with open(file_path, 'r') as f:
            return json.load(f, **kwargs)

    @staticmethod
    def _load_pickle(file_path: str, **kwargs) -> Any:
        """加载Pickle文件"""
        with open(file_path, 'rb') as f:
            return pickle.load(f, **kwargs)

    @staticmethod
    def _load_numpy(file_path: str, **kwargs) -> np.ndarray:
        """加载NumPy .npy文件"""
        return np.load(file_path, **kwargs)

    @staticmethod
    def _load_npz(file_path: str, **kwargs) -> dict:
        """加载NumPy .npz文件"""
        return dict(np.load(file_path, **kwargs))

    @staticmethod
    def _load_text(file_path: str, **kwargs) -> str:
        """加载文本文件"""
        encoding = kwargs.get('encoding', 'utf-8')
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()

    @staticmethod
    def _load_yaml(file_path: str, **kwargs) -> dict:
        """加载YAML文件"""
        with open(file_path, 'r') as f:
            return yaml.safe_load(f, **kwargs)

    @staticmethod
    def _load_hdf5(file_path: str, **kwargs) -> dict:
        """加载HDF5文件"""
        data = {}
        with h5py.File(file_path, 'r') as f:
            for key in f.keys():
                data[key] = f[key][()]
        return data

    @staticmethod
    def _load_image(file_path: str, **kwargs) -> np.ndarray:
        """加载图像文件"""
        mode = kwargs.get('mode', 'RGB')
        img = Image.open(file_path)
        if mode:
            img = img.convert(mode)
        return np.array(img.convert('RGB'))

    @staticmethod
    def _load_pytorch(file_path: str, **kwargs) -> Any:
        """加载PyTorch文件"""
        device = kwargs.get('device', 'cpu')
        return torch.load(file_path, map_location=device)

    @staticmethod
    def _load_tensorflow(file_path: str, **kwargs) -> Any:
        """加载TensorFlow模型或数据"""
        if file_path.endswith('.h5'):
            # 可能是Keras模型
            return tf.keras.models.load_model(file_path, **kwargs)
        else:
            # 其他TensorFlow格式
            return tf.saved_model.load(file_path, **kwargs)

    @staticmethod
    def save(data: Any, file_path: str, **kwargs):
        """
        保存数据到文件

        参数:
        data: 要保存的数据
        file_path: 文件路径
        kwargs: 格式特定参数
        """
        # 根据扩展名确定格式
        print(file_path)
        format = DataLoader._detect_format(file_path)
        # 调用特定保存方法
        saver = getattr(DataLoader, f"_save_{format}", None)
        if saver is None:
            raise ValueError(f"不支持的保存格式: {format}")
        saver(data, file_path, **kwargs)

    @staticmethod
    def _save_csv(data: pd.DataFrame, file_path: str, **kwargs):
        """保存为CSV文件"""
        data.to_csv(file_path, index=False, **kwargs)

    @staticmethod
    def _save_json(data: Union[dict, list], file_path: str, **kwargs):
        """保存为JSON文件"""
        indent = kwargs.get('indent', 4)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent, **kwargs)

    @staticmethod
    def _save_pickle(data: Any, file_path: str, **kwargs):
        """保存为Pickle文件"""
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, **kwargs)

    @staticmethod
    def _save_numpy(data: np.ndarray, file_path: str, **kwargs):
        """保存为NumPy .npy文件"""
        np.save(file_path, data, **kwargs)

    @staticmethod
    def _save_npz(data: dict, file_path: str, **kwargs):
        """保存为NumPy .npz文件"""
        np.savez(file_path, **data, **kwargs)

    @staticmethod
    def _save_text(data: str, file_path: str, **kwargs):
        """保存为文本文件"""
        encoding = kwargs.get('encoding', 'utf-8')
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(data)

    @staticmethod
    def _save_yaml(data: dict, file_path: str, **kwargs):
        """保存为YAML文件"""
        with open(file_path, 'w') as f:
            yaml.dump(data, f, **kwargs)

    @staticmethod
    def _save_image(data: np.ndarray, file_path: str, **kwargs):
        """保存为图像文件"""
        img = Image.fromarray(data)
        img.save(file_path, **kwargs)

    @staticmethod
    def _save_pytorch(data: Any, file_path: str, **kwargs):
        """保存为PyTorch文件"""
        torch.save(data, file_path, **kwargs)

    @staticmethod
    def _save_tensorflow(model: tf.keras.Model, file_path: str, **kwargs):
        """保存TensorFlow模型"""
        if file_path.endswith('.h5'):
            model.save(file_path, **kwargs)
        else:
            tf.saved_model.save(model, file_path, **kwargs)

    @staticmethod
    def batch_load(file_paths: List[str], **kwargs) -> List[Any]:
        """
        批量加载多个文件

        参数:
        file_paths: 文件路径列表
        kwargs: 传递给加载函数的参数

        返回:
        加载的数据列表
        """
        results = []
        for file_path in file_paths:
            try:
                data = DataLoader.load(file_path, **kwargs)
                results.append(data)
            except Exception as e:
                logger.error(f"加载文件 {file_path} 失败: {str(e)}")
                results.append(None)
        return results
    

def main():
    img = DataLoader.load("/data/duyongkun/CPX/classify/MLS-PJ/test_images/cat.png")  # 自动识别图片格式
    print(type(img))           # 应该是 <class 'numpy.ndarray'>
    print(img.shape)           # (H, W, 3) 对于RGB图像


if __name__ == "__main__":
    main()