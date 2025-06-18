"""
热力图生成器
创建各种热力图可视化解释结果
"""
import base64
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import cv2
from PIL import Image
import logging
from utils.validation import validate_not_none, validate_type

logger = logging.getLogger(__name__)


class HeatmapGenerator:
    """
    热力图生成器

    创建各种热力图可视化解释结果
    """

    def __init__(self, cmap: str = 'viridis', alpha: float = 0.5):
        """
        初始化热力图生成器

        参数:
        cmap: 颜色映射 (viridis, plasma, inferno, magma, coolwarm)
        alpha: 叠加透明度 (0-1)
        """
        self.cmap = cmap
        self.alpha = alpha

    def overlay_heatmap(self,
                        image: np.ndarray,
                        heatmap: np.ndarray,
                        normalize: bool = True,
                        clip_percentile: Optional[float] = None) -> np.ndarray:
        """
        将热力图叠加到原始图像上

        参数:
        image: 原始图像 (H, W, C)
        heatmap: 热力图 (H, W)
        normalize: 是否归一化热力图
        clip_percentile: 裁剪百分位 (0-100) 增强可视化效果

        返回:
        叠加后的图像 (H, W, C)
        """
        validate_not_none(image, "image")
        validate_not_none(heatmap, "heatmap")
        validate_type(image, np.ndarray, "image")
        validate_type(heatmap, np.ndarray, "heatmap")

        # 确保图像是3通道
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.concatenate([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] > 3:
            image = image[..., :3]

        # 处理热力图
        if normalize:
            heatmap = self.normalize_heatmap(heatmap, clip_percentile)

        # 应用颜色映射
        colored_heatmap = self.apply_colormap(heatmap)

        # 叠加热力图到原始图像
        overlay = cv2.addWeighted(image, 1 - self.alpha, colored_heatmap, self.alpha, 0)

        return overlay.astype(np.uint8)

    def normalize_heatmap(self,
                          heatmap: np.ndarray,
                          clip_percentile: Optional[float] = None) -> np.ndarray:
        """
        归一化热力图

        参数:
        heatmap: 原始热力图
        clip_percentile: 裁剪百分位 (0-100)

        返回:
        归一化后的热力图 (0-1)
        """
        # 裁剪极端值
        if clip_percentile is not None and 0 < clip_percentile < 100:
            min_val = np.percentile(heatmap, clip_percentile)
            max_val = np.percentile(heatmap, 100 - clip_percentile)
            heatmap = np.clip(heatmap, min_val, max_val)

        # 归一化到0-1
        min_val, max_val = np.min(heatmap), np.max(heatmap)
        if max_val - min_val > 1e-8:
            heatmap = (heatmap - min_val) / (max_val - min_val)
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap

    def apply_colormap(self, heatmap: np.ndarray) -> np.ndarray:
        """
        应用颜色映射到热力图

        参数:
        heatmap: 归一化热力图 (0-1)

        返回:
        彩色热力图 (H, W, 3)
        """
        # 使用matplotlib颜色映射
        cmap = plt.get_cmap(self.cmap)
        colored_heatmap = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
        return colored_heatmap

    def create_correlation_heatmap(self,
                                   corr_matrix: np.ndarray,
                                   feature_names: List[str],
                                   title: str = 'Feature Correlation',
                                   figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        创建特征相关性热力图

        参数:
        corr_matrix: 相关性矩阵 (n x n)
        feature_names: 特征名称列表
        title: 图表标题
        figsize: 图表大小

        返回:
        matplotlib Figure对象
        """
        validate_not_none(corr_matrix, "corr_matrix")
        validate_not_none(feature_names, "feature_names")
        validate_type(corr_matrix, np.ndarray, "corr_matrix")
        validate_type(feature_names, list, "feature_names")

        if corr_matrix.ndim != 2 or corr_matrix.shape[0] != corr_matrix.shape[1]:
            raise ValueError("相关性矩阵必须是方阵")
        if len(feature_names) != corr_matrix.shape[0]:
            raise ValueError("特征名称数量必须与矩阵维度一致")

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=self.cmap,
                    xticklabels=feature_names, yticklabels=feature_names, ax=ax)
        ax.set_title(title)
        plt.tight_layout()
        return fig

    def attention_heatmap(self,
                          tokens: List[str],
                          attention_weights: np.ndarray,
                          title: str = 'Attention Heatmap',
                          figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        创建注意力热力图 (用于文本)

        参数:
        tokens: token列表
        attention_weights: 注意力权重矩阵 (n x n)
        title: 图表标题
        figsize: 图表大小

        返回:
        matplotlib Figure对象
        """
        validate_not_none(tokens, "tokens")
        validate_not_none(attention_weights, "attention_weights")
        validate_type(tokens, list, "tokens")
        validate_type(attention_weights, np.ndarray, "attention_weights")

        if attention_weights.ndim != 2 or attention_weights.shape[0] != attention_weights.shape[1]:
            raise ValueError("注意力权重必须是方阵")
        if len(tokens) != attention_weights.shape[0]:
            raise ValueError("token数量必须与矩阵维度一致")

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(attention_weights, annot=False, cmap=self.cmap,
                    xticklabels=tokens, yticklabels=tokens, ax=ax)

        # 设置tick标签
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens, rotation=0)

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def to_base64(self, image: np.ndarray, format: str = 'png') -> str:
        """
        将图像转换为Base64编码的字符串

        参数:
        image: 图像数组 (H, W, C)
        format: 图像格式 (png, jpg)

        返回:
        Base64编码的图像字符串
        """
        validate_not_none(image, "image")
        validate_type(image, np.ndarray, "image")

        # 将NumPy数组转换为PIL图像
        if image.ndim == 2:
            img = Image.fromarray(image.astype(np.uint8))
        else:
            img = Image.fromarray(image.astype(np.uint8))

        # 转换为Base64
        buffer = BytesIO()
        img.save(buffer, format=format)
        buffer.seek(0)

        base64_str = base64.b64encode(buffer.read()).decode('utf-8')
        return f"data:image/{format};base64,{base64_str}"

    def to_html(self, image: np.ndarray) -> str:
        """
        将图像转换为HTML img标签

        参数:
        image: 图像数组

        返回:
        包含图像的HTML字符串
        """
        base64_img = self.to_base64(image)
        return f'<img src="{base64_img}" alt="Heatmap">'