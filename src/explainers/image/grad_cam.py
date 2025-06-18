"""
Grad-CAM解释器实现
用于可视化CNN模型关注区域
"""

import numpy as np
import cv2
from typing import Any, Dict, List, Optional, Tuple, Union
from core.explainer import BaseExplainer, ExplanationResult
import logging
import torch
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class GradCAMExplainer(BaseExplainer):
    """
    Grad-CAM解释器实现

    生成类激活热力图，显示模型关注区域
    """

    def __init__(self,
                 model: Any,
                 task_type: str,
                 **kwargs):
        """
        初始化Grad-CAM解释器

        参数:
        model: 待解释的CNN模型
        task_type: 任务类型 ('classification'/'regression')
        kwargs:
          - target_layer: 目标层名称 (如 'layer4' 或 'conv5_block3_out')
          - use_cuda: 是否使用GPU (PyTorch)
          - model_type: 模型框架 ('pytorch', 'tensorflow', 'keras')
        """
        super().__init__(model, task_type, **kwargs)

        # 设置参数
        self.target_layer = kwargs.get('target_layer')
        self.use_cuda = kwargs.get('use_cuda', False)
        self.model_type = kwargs.get('model_type', self._detect_model_type())

        # 验证目标层
        if not self.target_layer:
            self.target_layer = self._find_default_layer()
            logger.info(f"使用默认目标层: {self.target_layer}")

        self.target_layer = 'layer4'
        # 设置梯度钩子
        self.feature_maps = None
        self.gradients = None

        if self.model_type == 'pytorch':
            self._register_hooks_pytorch()
        elif self.model_type in ['tensorflow', 'keras']:
            self._prepare_tensorflow()

        logger.info(f"Grad-CAM解释器初始化完成: target_layer={self.target_layer}")

    def _detect_model_type(self) -> str:
        """自动检测模型类型"""
        model_type = str(type(self.model)).lower()

        if 'torch' in model_type or 'pytorch' in model_type:
            return 'pytorch'
        elif 'tensorflow' in model_type or 'keras' in model_type:
            return 'tensorflow'
        else:
            raise ValueError("无法自动检测模型类型，请通过model_type参数指定")

    def _find_default_layer(self) -> str:
        """查找合适的默认层"""
        if self.model_type == 'pytorch':
            # 先转为 list
            named_modules = list(self.model.named_modules())
            # 查找最后一个卷积层
            for name, module in reversed(named_modules):
                if isinstance(module, torch.nn.Conv2d):
                    return name
            # 如果没找到卷积层，使用最后一层
            return named_modules[-1][0]
        elif self.model_type in ['tensorflow', 'keras']:
            # 查找最后一个卷积层
            for layer in reversed(self.model.layers):
                if 'conv' in layer.name.lower() or 'conv2d' in layer.name:
                    return layer.name
            # 如果没找到
            return self.model.layers[-1].name

    def _register_hooks_pytorch(self):
        """为PyTorch模型注册前向/反向钩子"""
        # 获取目标层
        target_layer = None
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target_layer = module
                break

        if target_layer is None:
            raise ValueError(f"找不到目标层: {self.target_layer}")

        # 前向钩子保存特征图
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()

        # 反向钩子保存梯度
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        # 注册钩子
        self.forward_handle = target_layer.register_forward_hook(forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(backward_hook)

    def _prepare_tensorflow(self):
        """准备TensorFlow/Keras模型"""
        # 创建模型获取特征图和梯度
        target_layer = self.model.get_layer(self.target_layer)
        if target_layer is None:
            raise ValueError(f"找不到目标层: {self.target_layer}")

        # 创建子模型获取特征图
        self.feature_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=target_layer.output
        )

        # 创建模型计算梯度
        with tf.GradientTape() as tape:
            inputs = tf.cast(self.model.input, tf.float32)
            tape.watch(inputs)
            predictions = self.model(inputs)
            target = predictions[:, 0]  # 默认使用第一个输出

        self.gradient_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[target_layer.output, predictions]
        )

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
          - apply_relu: 是否应用ReLU
        """
        # 预处理图像
        img = self._preprocess_image(input_image, kwargs.get('resize'))

        # 获取热力图
        heatmap = self._compute_heatmap(img, target, **kwargs)

        # 创建解释结果
        result = ExplanationResult(
            raw_result=heatmap,
            metadata={
                'method': 'grad_cam',
                'target_layer': self.target_layer,
                'target_class': target
            }
        )

        # 添加可视化
        result.visualization = self._generate_visualization(img, heatmap, **kwargs)

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
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        return img_array

    def _compute_heatmap(self,
                         img: np.ndarray,
                         target: Optional[int] = None,
                         **kwargs) -> np.ndarray:
        """计算Grad-CAM热力图"""
        # 应用框架特定计算
        if self.model_type == 'pytorch':
            return self._compute_pytorch(img, target, **kwargs)
        else:
            return self._compute_tensorflow(img, target, **kwargs)

    def _compute_pytorch(self,
                         img: np.ndarray,
                         target: Optional[int],
                         **kwargs) -> np.ndarray:
        """PyTorch实现Grad-CAM"""
        # 准备输入
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        if self.use_cuda:
            img_tensor = img_tensor.cuda()
            self.model.cuda()

        # 前向传播
        self.model.zero_grad()
        output = self.model(img_tensor)

        # 确定目标
        if target is None:
            target = torch.argmax(output)

        # 创建one-hot目标
        one_hot = torch.zeros_like(output)
        one_hot[0][target] = 1

        # 反向传播
        output.backward(gradient=one_hot, retain_graph=True)

        # 获取特征图和梯度
        feature_maps = self.feature_maps.cpu().numpy()[0]
        gradients = self.gradients.cpu().numpy()[0]

        # 计算权重
        weights = np.mean(gradients, axis=(1, 2))

        # 计算热力图
        heatmap = np.zeros(feature_maps.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * feature_maps[i]

        # 应用ReLU
        apply_relu = kwargs.get('apply_relu', True)
        if apply_relu:
            heatmap = np.maximum(heatmap, 0)

        # 归一化
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)

        return heatmap

    def _compute_tensorflow(self,
                            img: np.ndarray,
                            target: Optional[int],
                            **kwargs) -> np.ndarray:
        """TensorFlow/Keras实现Grad-CAM"""
        # 准备输入
        img_tensor = np.expand_dims(img, axis=0)

        # 获取特征图和预测
        with tf.GradientTape() as tape:
            inputs = tf.cast(img_tensor, tf.float32)
            tape.watch(inputs)
            feature_maps, predictions = self.gradient_model(inputs)

            # 确定目标
            if target is None:
                target = tf.argmax(predictions[0])
            output = predictions[:, target]

        # 计算梯度
        grads = tape.gradient(output, feature_maps)

        # 计算权重
        weights = tf.reduce_mean(grads, axis=(1, 2))

        # 计算热力图
        heatmap = tf.reduce_sum(feature_maps * weights, axis=-1)
        heatmap = tf.squeeze(heatmap).numpy()

        # 应用ReLU
        apply_relu = kwargs.get('apply_relu', True)
        if apply_relu:
            heatmap = np.maximum(heatmap, 0)

        # 归一化
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)

        return heatmap

    def _generate_visualization(self,
                                img: np.ndarray,
                                heatmap: np.ndarray,
                                **kwargs) -> Dict[str, Any]:
        """生成可视化结果"""
        # 参数
        alpha = kwargs.get('alpha', 0.5)
        colormap = kwargs.get('colormap', cv2.COLORMAP_JET)

        # 调整热力图大小匹配原图
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # 应用颜色映射
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # 叠加到原图
        img_uint8 = np.uint8(255 * img)
        superimposed_img = cv2.addWeighted(img_uint8, 1 - alpha, heatmap_colored, alpha, 0)

        # 创建可视化字典
        visualization = {
            'original_image': img_uint8,
            'heatmap': heatmap_resized,
            'colored_heatmap': heatmap_colored,
            'superimposed': superimposed_img,
            'type': 'grad_cam'
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

    def __del__(self):
        """清理钩子"""
        if hasattr(self, 'forward_handle'):
            self.forward_handle.remove()
        if hasattr(self, 'backward_handle'):
            self.backward_handle.remove()

import torch
import types
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def main():
    # 1. 加载预训练模型
    model = torch.load("resnet18_full_model.pth", map_location="cpu",weights_only = False)
    # model = models.resnet18(pretrained=True)
    model.eval()
    # 2. 给模型添加predict方法
    def predict(self, x):
        with torch.no_grad():
            return self(x)
    model.predict = types.MethodType(predict, model)

    # 3. 选择目标层
    target_layer = 'layer4'

    # 4. 初始化Grad-CAM解释器
    explainer = GradCAMExplainer(
        model=model,
        task_type='classification',
        target_layer=target_layer,
        use_cuda=False,
        model_type='pytorch'
    )

    # 5. 读取和预处理图片
    img_path = '/data/duyongkun/CPX/classify/MLS-PJ/test_images/cat.png'
    input_image = Image.open(img_path).convert('RGB')
    img_np = np.array(input_image)  # HWC, uint8, [0,255]
    # 6. 生成Grad-CAM解释
    result = explainer.explain(img_np, target=283)

    # 7. 显示可视化结果
    superimposed = result.visualization['superimposed']
    plt.imshow(superimposed)
    plt.title('Grad-CAM Result')
    plt.axis('off')
    plt.show()

    # 选做：保存图片
    Image.fromarray(superimposed).save('gradcam_output.jpg')
    print("Grad-CAM 结果已保存为 gradcam_output.jpg")

    

if __name__ == "__main__":
    main()