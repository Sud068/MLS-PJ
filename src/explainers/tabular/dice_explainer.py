"""
DiCE解释器实现
用于生成反事实解释
"""

import dice_ml
from dice_ml import Dice
from dice_ml.utils import helpers
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
from src.core.explainer import BaseExplainer, ExplanationResult
import logging
import warnings

# 忽略DiCE的警告
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class DiCEExplainer(BaseExplainer):
    """
    DiCE解释器实现

    生成反事实解释 (Counterfactual Explanations)
    """

    def __init__(self,
                 model: Any,
                 task_type: str,
                 feature_names: Optional[List[str]] = None,
                 training_data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 continuous_features: Optional[List[str]] = None,
                 **kwargs):
        """
        初始化DiCE解释器

        参数:
        model: 待解释的模型
        task_type: 任务类型 ('classification'/'regression')
        feature_names: 特征名称列表
        training_data: 训练数据集
        continuous_features: 连续特征的名称列表
        kwargs:
          - method: 生成方法 ('random', 'genetic', 'kdtree')
          - total_CFs: 生成的反事实数量
          - desired_class: 期望的目标类别 (分类任务)
          - desired_range: 期望的输出范围 (回归任务)
          - proximity_weight: 反事实接近原始输入的权重
          - diversity_weight: 反事实多样性的权重
        """
        super().__init__(model, task_type, feature_names, **kwargs)

        # 验证任务类型
        if task_type not in ['classification', 'regression']:
            raise ValueError(f"DiCE不支持的任务类型: {task_type}")

        # 获取训练数据
        if training_data is None:
            if hasattr(model, 'X_train_'):
                training_data = model.X_train_
            else:
                raise ValueError("需要训练数据来初始化DiCE解释器")

        # 创建数据对象
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(training_data.shape[1])]

        # 创建pandas DataFrame
        if not isinstance(training_data, pd.DataFrame):
            training_data = pd.DataFrame(training_data, columns=feature_names)

        # 设置连续特征
        if continuous_features is None:
            continuous_features = feature_names  # 默认所有特征都是连续的

        # 创建DiCE数据对象
        self.data = dice_ml.Data(
            dataframe=training_data,
            continuous_features=continuous_features,
            outcome_name='target'  # 占位符，实际不使用
        )

        # 创建DiCE模型对象
        self.dice_model = dice_ml.Model(model=model, backend='sklearn')

        # 设置参数
        self.method = kwargs.get('method', 'random')
        self.total_CFs = kwargs.get('total_CFs', 5)
        self.desired_class = kwargs.get('desired_class', 1)  # 默认期望正类
        self.desired_range = kwargs.get('desired_range', [0.7, 1.0])  # 回归任务默认范围
        self.proximity_weight = kwargs.get('proximity_weight', 0.5)
        self.diversity_weight = kwargs.get('diversity_weight', 1.0)

        # 创建DiCE解释器
        self.explainer = Dice(
            data_interface=self.data,
            model_interface=self.dice_model,
            method=self.method
        )

        logger.info(f"DiCE解释器初始化完成: method={self.method}, total_CFs={self.total_CFs}")

    def explain(self,
                input_data: Union[np.ndarray, list, pd.DataFrame, dict],
                target: Optional[Any] = None,
                **kwargs) -> ExplanationResult:
        """
        生成反事实解释

        参数:
        input_data: 输入数据 (单样本)
        target: 目标类别 (分类任务) 或目标值 (回归任务)
        kwargs:
          - total_CFs: 生成的反事实数量
          - desired_class: 期望的目标类别 (分类任务)
          - desired_range: 期望的输出范围 (回归任务)
          - features_to_vary: 允许变化的特征列表
          - permitted_range: 特征的允许范围
        """
        # 处理参数
        total_CFs = kwargs.get('total_CFs', self.total_CFs)
        desired_class = kwargs.get('desired_class', self.desired_class)
        desired_range = kwargs.get('desired_range', self.desired_range)
        features_to_vary = kwargs.get('features_to_vary', 'all')
        permitted_range = kwargs.get('permitted_range', None)

        # 处理目标值
        if target is not None:
            if self.task_type == 'classification':
                desired_class = target
            else:
                desired_range = [target * 0.9, target * 1.1]  # 创建围绕目标的小范围

        # 准备输入数据
        if isinstance(input_data, np.ndarray):
            if input_data.ndim == 1:
                input_data = input_data.reshape(1, -1)
            input_df = pd.DataFrame(input_data, columns=self.feature_names)
        elif isinstance(input_data, list):
            input_df = pd.DataFrame([input_data], columns=self.feature_names)
        elif isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data

        # 生成反事实
        dice_exp = self.explainer.generate_counterfactuals(
            input_df,
            total_CFs=total_CFs,
            desired_class=desired_class if self.task_type == 'classification' else None,
            desired_range=desired_range if self.task_type == 'regression' else None,
            features_to_vary=features_to_vary,
            permitted_range=permitted_range,
            proximity_weight=self.proximity_weight,
            diversity_weight=self.diversity_weight
        )

        # 创建解释结果
        result = ExplanationResult(
            raw_result=dice_exp,
            metadata={
                'method': self.method,
                'total_CFs': total_CFs,
                'desired_class': desired_class if self.task_type == 'classification' else None,
                'desired_range': desired_range if self.task_type == 'regression' else None
            }
        )

        # 设置反事实结果
        result.counterfactuals = self._get_counterfactuals(dice_exp)

        return result

    def _get_counterfactuals(self, dice_exp) -> List[Dict[str, Any]]:
        """从DiCE结果中提取反事实信息"""
        counterfactuals = []

        # 获取反事实数据
        cf_df = dice_exp.cf_examples_list[0].final_cfs_df

        # 获取原始输入和预测
        original_input = dice_exp.test_instance_df.iloc[0].to_dict()
        original_pred = self.model.predict(
            dice_exp.test_instance_df.values.reshape(1, -1))[0]

        # 对于分类任务，获取概率
        if self.task_type == 'classification':
            original_prob = self.model.predict_proba(
                dice_exp.test_instance_df.values.reshape(1, -1))[0]

        # 处理每个反事实
        for idx, row in cf_df.iterrows():
            cf_dict = row.to_dict()

            # 获取反事实的预测
            cf_pred = self.model.predict(
                row.values.reshape(1, -1))[0]

            # 对于分类任务，获取概率
            if self.task_type == 'classification':
                cf_prob = self.model.predict_proba(
                    row.values.reshape(1, -1))[0]

            # 计算变化
            changes = {}
            for feature in self.feature_names:
                orig_val = original_input[feature]
                cf_val = cf_dict[feature]

                if orig_val != cf_val:
                    changes[feature] = {
                        'original': orig_val,
                        'counterfactual': cf_val,
                        'change': cf_val - orig_val if isinstance(cf_val, (int, float)) else 'categorical'
                    }

            # 添加到结果
            counterfactuals.append({
                'features': cf_dict,
                'prediction': cf_pred,
                'probability': cf_prob if self.task_type == 'classification' else None,
                'changes': changes,
                'distance': self._calculate_distance(original_input, cf_dict)
            })

        return counterfactuals

    def _calculate_distance(self, original: dict, counterfactual: dict) -> float:
        """计算原始输入和反事实之间的距离"""
        distance = 0.0

        for feature in self.feature_names:
            orig_val = original[feature]
            cf_val = counterfactual[feature]

            # 只计算数值特征的距离
            if isinstance(orig_val, (int, float)) and isinstance(cf_val, (int, float)):
                # 标准化距离 (避免量纲影响)
                if hasattr(self, 'feature_ranges_'):
                    feat_range = self.feature_ranges_.get(feature, 1.0)
                    if feat_range == 0:
                        feat_range = 1.0
                    distance += abs(orig_val - cf_val) / feat_range
                else:
                    distance += abs(orig_val - cf_val)

        return distance

    def batch_explain(self,
                      input_batch: Union[np.ndarray, pd.DataFrame],
                      targets: Optional[List[Any]] = None,
                      **kwargs) -> List[ExplanationResult]:
        """
        批量生成反事实解释
        """
        results = []

        # 处理目标值
        if targets is None:
            targets = [None] * len(input_batch)

        for data, target in zip(input_batch, targets):
            results.append(self.explain(data, target=target, **kwargs))

        return results

    def set_feature_ranges(self, feature_ranges: Dict[str, float]):
        """设置特征范围用于距离计算"""
        self.feature_ranges_ = feature_ranges