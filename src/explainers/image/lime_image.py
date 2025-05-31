"""
LIME图像解释器实现
用于局部可解释的图像解释
"""

import numpy as np
import cv2
from typing import Any, Dict, List, Optional, Union
from src.core.explainer import BaseExplainer, ExplanationResult
import logging
import lime
from lime import lime_image
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import sklearn

# 忽略警告
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class LimeImageExplainer(BaseExplainer):
    """
    LIME图像解释器实现

    提供局部可解释的模型无关解释
    """

    def __init__(self,
                 model: Any,
                 task_type: str,
                 **kwargs):
        """
        初始化LIME图像解释器

        参数:
        model: 待解释的模型
        task_type: 任务类型 ('classification'/'regression')
        kwargs:
          - top_labels: 返回前N个类别的解释
          - num_samples: 生成的样本数
          - segmentation_fn: 分割函数 ('quickshift', 'slic', 'felzenszwalb')
          - random_state: 随机种子
          - kernel_size: 分割核大小
        """
        super().__init__(model, task_type, **kwargs)

        # 验证任务类型
        if task_type not in ['classification', 'regression']:
            raise ValueError(f"LIME不支持的任务类型: {task_type}")

        # 设置参数
        self.top_labels = kwargs.get('top_labels', 5)
        self.num_samples = kwargs.get('num_samples', 1000)
        self.segmentation_fn = kwargs.get('segmentation_fn', 'quickshift')
        self.random_state = kwargs.get('random_state', 42)
        self.kernel_size = kwargs.get('kernel_size', 4)

        # 创建LIME解释器
        self.explainer = lime_image.LimeImageExplainer(
            kernel_size=self.kernel_size,
            random_state=self.random_state
        )

        logger.info(f"LIME图像解释器初始化完成: num_samples={self.num_samples}")

    def explain(self,
                input_image: Union[np.ndarray, Image.Image, str],
                target: Optional[Any] = None,
                **kwargs) -> ExplanationResult:
        """
        解释单个图像

        参数:
        input_image: 输入图像 (路径/PIL图像/NumPy数组)
        target: 目标类别 (分类任务)
        kwargs:
          - resize: 调整大小 (宽, 高)
          - hide_color: 隐藏颜色 (None表示平均像素)
          - positive_only: 只显示正贡献
          - num_features: 显示的特征数
        """
        # 预处理图像
        img = self._preprocess_image(input_image, kwargs.get('resize'))

        # 解释图像
        explanation = self._explain_image(img, target, **kwargs)

        # 创建解释结果
        result = ExplanationResult(
            raw_result=explanation,
            metadata={
                'method': 'lime_image',
                'num_samples': self.num_samples,
                'target_class': target
            }
        )

        # 添加可视化
        result.visualization = self._generate_visualization(img, explanation, **kwargs)

        return result

    def _preprocess_image(self, image, resize=None) -> np.ndarray:
        """预处理图像为NumPy数组"""
        if isinstance(image, str):
            # 从文件加载
            img = Image.open(image)
        elif isinstance(image, Image.Image):
            img = image
        else:
            img = image

        # 转换为NumPy数组
        if isinstance(img, np.ndarray):
            img_array = img
        else:
            img_array = np.array(img)

        # 确保3通道
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[-1] > 3:
            img_array = img_array[..., :3]

        # 调整大小
        if resize:
            img_array = cv2.resize(img_array, resize)

        # 归一化到0-1
        img_array = img_array.astype(np.float32) / 255.0

        return img_array

    def _explain_image(self,
                       img: np.ndarray,
                       target: Optional[int] = None,
                       **kwargs) -> Any:
        """使用LIME解释图像"""

        # 定义预测函数
        def predict_fn(images):
            # LIME输入是RGB图像数组
            if len(images.shape) == 4:
                # 批次输入
                return self.model.predict(images)
            else:
                # 单样本输入
                return self.model.predict(np.expand_dims(images, axis=0))

        # 解释参数
        hide_color = kwargs.get('hide_color', None)
        positive_only = kwargs.get('positive_only', False)
        num_features = kwargs.get('num_features', 10)

        # 生成解释
        explanation = self.explainer.explain_instance(
            img,
            predict_fn,
            top_labels=self.top_labels,
            num_samples=self.num_samples,
            hide_color=hide_color,
            segmentation_fn=self._get_segmentation_fn()
        )

        return explanation

    def _get_segmentation_fn(self):
        """获取分割函数"""
        if self.segmentation_fn == 'quickshift':
            return None  # 默认使用quickshift
        elif self.segmentation_fn == 'slic':
            return self._slic_segmentation
        elif self.segmentation_fn == 'felzenszwalb':
            return self._felzenszwalb_segmentation
        else:
            raise ValueError(f"不支持的分割方法: {self.segmentation_fn}")

    def _slic_segmentation(self, image):
        """SLIC分割"""
        from skimage.segmentation import slic
        return slic(image, n_segments=100, compactness=10, sigma=1)

    def _felzenszwalb_segmentation(self, image):
        """Felzenszwalb分割"""
        from skimage.segmentation import felzenszwalb
        return felzenszwalb(image, scale=100, sigma=0.5, min_size=50)

    def _generate_visualization(self,
                                img: np.ndarray,
                                explanation: Any,
                                **kwargs) -> Dict[str, Any]:
        """生成可视化结果"""
        # 参数
        positive_only = kwargs.get('positive_only', False)
        negative_only = kwargs.get('negative_only', False)
        num_features = kwargs.get('num_features', 10)
        hide_rest = kwargs.get('hide_rest', True)

        # 确定目标类别
        if explanation.metadata['target_class'] is not None:
            target = explanation.metadata['target_class']
        else:
            # 使用预测概率最高的类别
            target = np.argmax(explanation.predict_proba)

        # 获取解释图像
        temp, mask = explanation.get_image_and_mask(
            label=target,
            positive_only=positive_only,
            negative_only=negative_only,
            num_features=num_features,
            hide_rest=hide_rest
        )

        # 转换为RGB
        img_uint8 = np.uint8(255 * img)

        # 创建可视化字典
        visualization = {
            'original_image': img_uint8,
            'explanation_image': temp,
            'mask': mask,
            'segments': explanation.segments,
            'local_exp': explanation.local_exp.get(target, []),
            'type': 'lime_image'
        }

        return visualization

    def batch_explain(self,
                      input_batch: List[Union[np.ndarray, Image.Image, str]],
                      targets: Optional[List[Any]] = None,
                      **kwargs) -> List[ExplanationResult]:
        """
        批量解释多个图像
        """
        results = []

        # 处理目标值
        if targets is None:
            targets = [None] * len(input_batch)

        for img, target in zip(input_batch, targets):
            results.append(self.explain(img, target=target, **kwargs))

        return results