"""
Integrated Gradients解释器实现
用于可视化输入特征重要性
"""

import numpy as np
import cv2
from typing import Any, Dict, List, Optional, Union
from src.core.explainer import BaseExplainer, ExplanationResult
import logging
import torch
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class IntegratedGradientsExplainer(BaseExplainer):
    """
    Integrated Gradients解释器实现

    通过积分路径计算特征重要性
    """

    def __init__(self,
                 model: Any,
                 task_type: str,
                 **kwargs):
        """
        初始化Integrated Gradients解释器

        参数:
        model: 待解释的模型
        task_type: 任务类型 ('classification'/'regression')
        kwargs:
          - baseline: 基线输入 (None表示使用黑色图像)
          - steps: 积分步数
          - use_cuda: 是否使用GPU (PyTorch)
          - model_type: 模型框架 ('pytorch', 'tensorflow', 'keras')
        """
        super().__init__(model, task_type, **kwargs)

        # 设置参数
        self.baseline = kwargs.get('baseline', None)
        self.steps = kwargs.get('steps', 50)
        self.use_cuda = kwargs.get('use_cuda', False)
        self.model_type = kwargs.get('model_type', self._detect_model_type())

        logger.info(f"Integrated Gradients解释器初始化完成: steps={self.steps}")

    def _detect_model_type(self) -> str:
        """自动检测模型类型"""
        model_type = str(type(self.model)).lower()

        if 'torch' in model_type or 'pytorch' in model_type:
            return 'pytorch'
        elif 'tensorflow' in model_type or 'keras' in model_type:
            return 'tensorflow'
        else:
            raise ValueError("无法自动检测模型类型，请通过model_type参数指定")

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
          - alpha: 热力图叠加透明度
          - colormap: 热力图颜色映射
          - absolute: 是否取绝对值
        """
        # 预处理图像
        img = self._preprocess_image(input_image, kwargs.get('resize'))

        # 获取积分梯度
        ig = self._compute_integrated_gradients(img, target, **kwargs)

        # 创建解释结果
        result = ExplanationResult(
            raw_result=ig,
            metadata={
                'method': 'integrated_gradients',
                'steps': self.steps,
                'target_class': target
            }
        )

        # 添加可视化
        result.visualization = self._generate_visualization(img, ig, **kwargs)

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

    def _compute_integrated_gradients(self,
                                      img: np.ndarray,
                                      target: Optional[int] = None,
                                      **kwargs) -> np.ndarray:
        """计算Integrated Gradients"""
        # 应用框架特定计算
        if self.model_type == 'pytorch':
            return self._compute_pytorch(img, target, **kwargs)
        else:
            return self._compute_tensorflow(img, target, **kwargs)

    def _compute_pytorch(self,
                         img: np.ndarray,
                         target: Optional[int],
                         **kwargs) -> np.ndarray:
        """PyTorch实现Integrated Gradients"""
        # 准备输入
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        if self.use_cuda:
            img_tensor = img_tensor.cuda()
            self.model.cuda()

        # 设置基线
        if self.baseline is None:
            baseline = torch.zeros_like(img_tensor)
        else:
            baseline = self._preprocess_image(self.baseline)
            baseline = torch.from_numpy(baseline).permute(2, 0, 1).unsqueeze(0).float()
            if self.use_cuda:
                baseline = baseline.cuda()

        # 计算梯度
        integrated_grads = torch.zeros_like(img_tensor)

        # 创建路径
        alphas = torch.linspace(0, 1, self.steps)
        if self.use_cuda:
            alphas = alphas.cuda()

        # 遍历alpha值
        for alpha in alphas:
            # 插值输入
            input_step = baseline + alpha * (img_tensor - baseline)
            input_step.requires_grad = True

            # 前向传播
            output = self.model(input_step)

            # 确定目标
            if target is None:
                target = torch.argmax(output)

            # 计算目标类别的梯度
            self.model.zero_grad()
            output[0, target].backward(retain_graph=True)

            # 累加梯度
            integrated_grads += input_step.grad

        # 平均梯度
        integrated_grads /= self.steps

        # 计算最终积分梯度
        ig = (img_tensor - baseline) * integrated_grads
        ig = ig.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

        # 取绝对值
        absolute = kwargs.get('absolute', False)
        if absolute:
            ig = np.abs(ig)

        # 聚合通道
        ig = np.sum(ig, axis=-1)

        # 归一化
        ig = (ig - np.min(ig)) / (np.max(ig) - np.min(ig) + 1e-8)

        return ig

    def _compute_tensorflow(self,
                            img: np.ndarray,
                            target: Optional[int],
                            **kwargs) -> np.ndarray:
        """TensorFlow/Keras实现Integrated Gradients"""
        # 准备输入
        img_tensor = np.expand_dims(img, axis=0)

        # 设置基线
        if self.baseline is None:
            baseline = np.zeros_like(img_tensor)
        else:
            baseline = self._preprocess_image(self.baseline)
            baseline = np.expand_dims(baseline, axis=0)

        # 创建路径
        alphas = np.linspace(0, 1, self.steps)

        # 初始化梯度
        integrated_grads = np.zeros_like(img_tensor)

        # 遍历alpha值
        for alpha in alphas:
            # 插值输入
            input_step = baseline + alpha * (img_tensor - baseline)

            # 计算梯度
            with tf.GradientTape() as tape:
                tape.watch(input_step)
                output = self.model(input_step)

                # 确定目标
                if target is None:
                    target_idx = tf.argmax(output[0])
                else:
                    target_idx = target

                target_output = output[:, target_idx]

            # 计算梯度
            grads = tape.gradient(target_output, input_step)

            # 累加梯度
            integrated_grads += grads.numpy()

        # 平均梯度
        integrated_grads /= self.steps

        # 计算最终积分梯度
        ig = (img_tensor - baseline) * integrated_grads
        ig = np.squeeze(ig)

        # 取绝对值
        absolute = kwargs.get('absolute', False)
        if absolute:
            ig = np.abs(ig)

        # 聚合通道
        ig = np.sum(ig, axis=-1)

        # 归一化
        ig = (ig - np.min(ig)) / (np.max(ig) - np.min(ig) + 1e-8)

        return ig

    def _generate_visualization(self,
                                img: np.ndarray,
                                ig: np.ndarray,
                                **kwargs) -> Dict[str, Any]:
        """生成可视化结果"""
        # 参数
        alpha = kwargs.get('alpha', 0.7)
        colormap = kwargs.get('colormap', cv2.COLORMAP_VIRIDIS)

        # 调整热力图大小匹配原图
        ig_resized = cv2.resize(ig, (img.shape[1], img.shape[0]))

        # 应用颜色映射
        ig_colored = cv2.applyColorMap(np.uint8(255 * ig_resized), colormap)
        ig_colored = cv2.cvtColor(ig_colored, cv2.COLOR_BGR2RGB)

        # 叠加到原图
        img_uint8 = np.uint8(255 * img)
        superimposed_img = cv2.addWeighted(img_uint8, 1 - alpha, ig_colored, alpha, 0)

        # 创建可视化字典
        visualization = {
            'original_image': img_uint8,
            'attribution_map': ig_resized,
            'colored_attribution': ig_colored,
            'superimposed': superimposed_img,
            'type': 'integrated_gradients'
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