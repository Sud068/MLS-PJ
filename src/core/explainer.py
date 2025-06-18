"""
解释器抽象基类
定义所有解释器必须实现的统一接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL.Image import Image


@dataclass
class ExplanationResult:
    """
    解释结果数据容器

    属性:
    - raw_result: 解释器原始输出
    - feature_importance: 特征重要性字典 {特征名: 重要性值}
    - visualization: 可视化数据 (matplotlib图/热力图数组等)
    - metrics: 解释质量指标 {指标名: 值}
    - metadata: 元数据 (模型名、解释方法、时间戳等)
    - counterfactuals: 反事实解释列表 [{'特征': 变化, '新预测': 值}]
    """
    raw_result: Any
    feature_importance: Dict[str, float] = field(default_factory=dict)
    visualization: Any = None
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    counterfactuals: List[Dict[str, Any]] = field(default_factory=list)
    @staticmethod
    def to_serializable(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: ExplanationResult.to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ExplanationResult.to_serializable(v) for v in obj]
        else:
            return obj

    def to_dict(self) -> Dict[str, Any]:
        """将解释结果转换为字典"""
        return ExplanationResult.to_serializable(self.__dict__)
        

class BaseExplainer(ABC):
    """
    解释器抽象基类
    所有具体解释器必须继承此类并实现抽象方法
    """

    def __init__(self,
                 model: Any,
                 task_type: str,
                 feature_names: Optional[List[str]] = None,
                 **kwargs):
        """
        初始化解释器

        参数:
        model: 待解释的模型对象
        task_type: 任务类型 ('classification', 'regression')
        feature_names: 特征名称列表
        kwargs: 解释器特定参数
        """
        self.model = model
        self.task_type = task_type
        self.feature_names = feature_names
        self.params = kwargs

        # 验证模型和任务类型
        self._validate_model()

    def _validate_model(self):
        """验证模型兼容性"""
        if not callable(getattr(self.model, "predict", None)):
            raise ValueError("模型必须实现 predict 方法")

        if self.task_type not in ['classification', 'regression']:
            raise ValueError(f"不支持的任务类型: {self.task_type}")

    @abstractmethod
    def explain(self,
                input_data: Union[np.ndarray, list, dict],
                target: Optional[Any] = None,
                **kwargs) -> ExplanationResult:
        """
        解释单个输入样本

        参数:
        input_data: 输入数据 (数组/列表/字典)
        target: 解释的目标类别/值 (分类任务中可选)
        kwargs: 解释过程附加参数

        返回:
        ExplanationResult 对象
        """
        pass

    def batch_explain(self,
                      input_batch: Union[np.ndarray, list],
                      targets: Optional[List[Any]] = None,
                      **kwargs) -> List[ExplanationResult]:
        """
        批量解释多个样本 (默认实现，可被覆盖)

        参数:
        input_batch: 输入数据批次
        targets: 每个样本的目标类别/值列表
        kwargs: 解释过程附加参数

        返回:
        ExplanationResult 对象列表
        """
        results = []
        targets = targets or [None] * len(input_batch)

        for data, target in zip(input_batch, targets):
            results.append(self.explain(data, target=target, **kwargs))

        return results

    def evaluate_explanation(self,
                             explanation: ExplanationResult,
                             metrics: List[str] = ['stability'],
                             **kwargs) -> ExplanationResult:
        """
        评估解释质量 (默认实现占位符)

        参数:
        explanation: 要评估的解释结果
        metrics: 要计算的指标列表
        kwargs: 评估参数

        返回:
        更新了metrics属性的ExplanationResult
        """
        # 具体实现在子类中完成
        # 这里仅添加元数据
        explanation.metadata['evaluation_metrics'] = metrics
        return explanation

def display_explanation_result(result: ExplanationResult):
    """格式化展示 ExplanationResult 内容"""
    from pprint import pprint
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    print("\n==== Explanation Result ====\n")

    print("[元数据 metadata]")
    pprint(result.metadata)

    if result.feature_importance:
        print("\n[特征重要性 feature_importance]")
        for feat, score in result.feature_importance.items():
            print(f" - {feat}: {score:.4f}")
    else:
        print("\n[特征重要性 feature_importance] 无")

    if result.metrics:
        print("\n[解释质量指标 metrics]")
        for key, val in result.metrics.items():
            print(f" - {key}: {val}")
    else:
        print("\n[解释质量指标 metrics] 无")

    if result.counterfactuals:
        print("\n[反事实 counterfactuals]")
        for i, cf in enumerate(result.counterfactuals):
            print(f" - CF#{i+1}: {cf}")
    else:
        print("\n[反事实 counterfactuals] 无")

    print("\n[原始结果 raw_result]")
    if isinstance(result.raw_result, np.ndarray):
        print(f" - NumPy 数组，形状: {result.raw_result.shape}")
    else:
        print(f" - 类型: {type(result.raw_result)}, 内容摘要: {str(result.raw_result)[:200]}")

    print("\n[可视化 visualization]")

    def show_image_array(arr, title=None):
        """显示 NumPy 图像数组"""
        plt.figure()
        if arr.ndim == 2:  # 灰度图
            plt.imshow(arr, cmap='gray')
        else:
            plt.imshow(arr)
        if title:
            plt.title(title)
        plt.axis('off')
        plt.show()

    if isinstance(result.visualization, dict):
        for k, v in result.visualization.items():
            print(f" - {k}: ", end="")
            if isinstance(v, np.ndarray):
                print(f"ndarray, 形状 {v.shape}")
                try:
                    show_image_array(v, title=k)
                except Exception as e:
                    print(f"   >> 显示失败: {e}")
            elif isinstance(v, Image.Image):
                print(f"PIL图像, 尺寸 {v.size}")
                try:
                    v.show(title=k)
                except Exception as e:
                    print(f"   >> 显示失败: {e}")
            else:
                print(f"{type(v)}（无法可视化）")
    elif isinstance(result.visualization, (np.ndarray, Image.Image)):
        print(f"单一图像对象: {type(result.visualization)}")
        try:
            if isinstance(result.visualization, np.ndarray):
                show_image_array(result.visualization)
            else:
                result.visualization.show()
        except Exception as e:
            print(f"   >> 显示失败: {e}")
    elif result.visualization is not None:
        print(f"类型: {type(result.visualization)}")
    else:
        print(" - 无可视化数据")

    print("\n============================\n")
