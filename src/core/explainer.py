"""
解释器抽象基类
定义所有解释器必须实现的统一接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np


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

    def to_dict(self) -> Dict[str, Any]:
        """将解释结果转换为字典"""
        return {
            "feature_importance": self.feature_importance,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "counterfactuals": self.counterfactuals
        }


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
        if not callable(getattr(self.model, "predict", None):
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