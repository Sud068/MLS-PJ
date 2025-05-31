"""
LIME解释器实现
用于表格数据的局部可解释模型
"""

import numpy as np
import lime
import lime.lime_tabular
from typing import Any, Dict, List, Optional, Union
from src.core.explainer import BaseExplainer, ExplanationResult
import logging
import pandas as pd
import warnings

# 忽略LIME的警告
warnings.filterwarnings("ignore", category=UserWarning, module="lime")

logger = logging.getLogger(__name__)


class LIMEExplainer(BaseExplainer):
    """
    LIME解释器实现

    提供局部可解释的模型无关解释
    """

    def __init__(self,
                 model: Any,
                 task_type: str,
                 feature_names: Optional[List[str]] = None,
                 training_data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 categorical_features: Optional[List[int]] = None,
                 **kwargs):
        """
        初始化LIME解释器

        参数:
        model: 待解释的模型
        task_type: 任务类型 ('classification'/'regression')
        feature_names: 特征名称列表
        training_data: 训练数据集 (用于计算统计信息)
        categorical_features: 分类特征的索引列表
        kwargs:
          - mode: 'classification' 或 'regression'
          - kernel_width: LIME内核宽度
          - num_samples: 生成的样本数
          - discretize_continuous: 是否离散化连续特征
          - discretizer: 离散化方法 ('quartile', 'decile', 'entropy')
          - random_state: 随机种子
        """
        super().__init__(model, task_type, feature_names, **kwargs)

        # 验证任务类型
        if task_type not in ['classification', 'regression']:
            raise ValueError(f"LIME不支持的任务类型: {task_type}")

        # 获取训练数据统计信息
        if training_data is None:
            if hasattr(model, 'X_train_'):
                training_data = model.X_train_
            else:
                raise ValueError("需要训练数据来初始化LIME解释器")

        # 设置参数
        self.mode = 'classification' if task_type == 'classification' else 'regression'
        self.kernel_width = kwargs.get('kernel_width', 0.75)
        self.num_samples = kwargs.get('num_samples', 5000)
        self.discretize_continuous = kwargs.get('discretize_continuous', True)
        self.discretizer = kwargs.get('discretizer', 'quartile')
        self.random_state = kwargs.get('random_state', 42)
        self.categorical_features = categorical_features

        # 创建LIME解释器
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            mode=self.mode,
            feature_names=feature_names,
            categorical_features=categorical_features,
            kernel_width=self.kernel_width,
            discretize_continuous=self.discretize_continuous,
            discretizer=self.discretizer,
            random_state=self.random_state
        )

        logger.info(f"LIME解释器初始化完成: mode={self.mode}, num_samples={self.num_samples}")

    def explain(self,
                input_data: Union[np.ndarray, list, pd.DataFrame],
                target: Optional[Any] = None,
                **kwargs) -> ExplanationResult:
        """
        解释单个样本

        参数:
        input_data: 输入数据 (单样本)
        target: 目标类别 (分类任务)
        kwargs:
          - num_features: 显示的特征数
          - num_samples: 覆盖初始化的样本数
          - top_labels: 返回前N个类别的解释
        """
        # 转换输入数据
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)

        if input_data.ndim > 1:
            input_data = input_data.flatten()

        # 获取参数
        num_features = kwargs.get('num_features', len(input_data))
        num_samples = kwargs.get('num_samples', self.num_samples)
        top_labels = kwargs.get('top_labels', None)

        # 对于分类任务，确定目标类别
        if self.task_type == 'classification':
            if target is None:
                # 如果没有指定目标，使用预测概率最高的类别
                pred_proba = self.model.predict_proba(input_data.reshape(1, -1))[0]
                target = int(np.argmax(pred_proba))
        else:
            target = None

        # 生成解释
        explanation = self.explainer.explain_instance(
            data_row=input_data,
            predict_fn=self.model.predict_proba if self.task_type == 'classification' else self.model.predict,
            labels=[target] if self.task_type == 'classification' else None,
            top_labels=top_labels,
            num_features=num_features,
            num_samples=num_samples
        )

        # 创建解释结果
        result = ExplanationResult(
            raw_result=explanation,
            metadata={
                'method': 'lime',
                'num_samples': num_samples,
                'target_class': target if self.task_type == 'classification' else None
            }
        )

        # 设置特征重要性
        feature_importance = self._get_feature_importance(explanation, num_features)
        result.feature_importance = feature_importance

        # 添加可视化数据
        result.visualization = self._generate_visualization(explanation)

        return result

    def _get_feature_importance(self, explanation, num_features: int) -> Dict[str, float]:
        """从LIME解释中提取特征重要性"""
        feature_importance = {}

        # 获取解释的本地特征重要性
        if self.task_type == 'classification':
            # 分类任务：获取指定类别的特征重要性
            if explanation.top_labels:
                label = explanation.top_labels[0]
                for feature, weight in explanation.as_list(label=label):
                    feature_importance[feature] = weight
            else:
                for feature, weight in explanation.as_list():
                    feature_importance[feature] = weight
        else:
            # 回归任务
            for feature, weight in explanation.as_list():
                feature_importance[feature] = weight

        # 如果特征名称未设置，从解释中提取
        if not self.feature_names:
            self.feature_names = [f.split('=')[0].strip() for f in feature_importance.keys()]

        return feature_importance

    def _generate_visualization(self, explanation):
        """生成LIME可视化数据"""
        # 创建可视化字典
        visualization = {
            'as_list': explanation.as_list(),
            'as_map': explanation.as_map(),
            'local_pred': explanation.local_pred,
            'local_exp': explanation.local_exp,
            'type': 'lime'
        }

        # 添加HTML可视化
        try:
            visualization['html'] = explanation.as_html()
        except Exception as e:
            logger.warning(f"无法生成HTML可视化: {str(e)}")
            visualization['html'] = None

        return visualization

    def batch_explain(self,
                      input_batch: Union[np.ndarray, pd.DataFrame],
                      targets: Optional[List[Any]] = None,
                      **kwargs) -> List[ExplanationResult]:
        """
        批量解释多个样本
        """
        results = []

        # 处理目标值
        if targets is None:
            targets = [None] * len(input_batch)

        for data, target in zip(input_batch, targets):
            results.append(self.explain(data, target=target, **kwargs))

        return results