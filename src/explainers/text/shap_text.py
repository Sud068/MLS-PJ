"""
SHAP文本解释器实现
用于解释文本模型的SHAP值
"""

import numpy as np
import shap
from typing import Any, Dict, List, Optional, Union
from src.core.explainer import BaseExplainer, ExplanationResult
import logging
import warnings
import re
import string

# 忽略SHAP的警告
warnings.filterwarnings("ignore", category=UserWarning, module="shap")

logger = logging.getLogger(__name__)


class SHAPTextExplainer(BaseExplainer):
    """
    SHAP文本解释器实现

    使用SHAP值解释文本模型
    """

    def __init__(self,
                 model: Any,
                 task_type: str,
                 **kwargs):
        """
        初始化SHAP文本解释器

        参数:
        model: 待解释的模型
        task_type: 任务类型 ('classification'/'regression')
        kwargs:
          - tokenizer: 自定义分词器函数
          - masker: SHAP的掩码器 (默认使用文本掩码)
          - nsamples: 生成的样本数
          - class_names: 类别名称列表
        """
        super().__init__(model, task_type, **kwargs)

        # 设置参数
        self.tokenizer = kwargs.get('tokenizer', self._default_tokenizer)
        self.masker = kwargs.get('masker', None)
        self.nsamples = kwargs.get('nsamples', 1000)
        self.class_names = kwargs.get('class_names', None)

        # 创建SHAP解释器
        self.explainer = self._create_explainer()

        logger.info(f"SHAP文本解释器初始化完成: nsamples={self.nsamples}")

    def _default_tokenizer(self, text: str) -> List[str]:
        """默认分词器"""
        # 简单分词：转换为小写，移除标点，按空格分割
        text = text.lower()
        text = re.sub(f'[{string.punctuation}]', ' ', text)
        return text.split()

    def _create_explainer(self):
        """创建SHAP文本解释器"""
        # 如果未提供掩码器，创建默认文本掩码器
        if self.masker is None:
            self.masker = shap.maskers.Text(tokenizer=self.tokenizer)

        # 创建解释器
        if self.task_type == 'classification':
            return shap.Explainer(self.model, self.masker, output_names=self.class_names)
        else:
            return shap.Explainer(self.model, self.masker)

    def explain(self,
                input_text: str,
                target: Optional[Any] = None,
                **kwargs) -> ExplanationResult:
        """
        解释文本样本

        参数:
        input_text: 输入文本
        target: 目标类别 (分类任务)
        kwargs:
          - nsamples: 覆盖初始化的样本数
          - fixed_context: 固定上下文设置
          - batch_size: 批处理大小
        """
        # 获取参数
        nsamples = kwargs.get('nsamples', self.nsamples)
        fixed_context = kwargs.get('fixed_context', 1)
        batch_size = kwargs.get('batch_size', 50)

        # 计算SHAP值
        shap_values = self.explainer(
            [input_text],
            nsamples=nsamples,
            fixed_context=fixed_context,
            batch_size=batch_size
        )

        # 处理多输出
        if len(shap_values.shape) == 3 and shap_values.shape[0] == 1:
            # 单样本多类别
            if target is not None:
                # 选择特定类别
                values = shap_values[0, :, target].values
            else:
                # 默认选择最大绝对值影响的类别
                max_impact_idx = np.argmax(np.max(np.abs(shap_values.values[0]), axis=1))
                values = shap_values[0, :, max_impact_idx].values
                target = max_impact_idx
        else:
            values = shap_values.values[0]

        # 获取token
        tokens = self.tokenizer(input_text)

        # 创建解释结果
        result = ExplanationResult(
            raw_result=shap_values,
            metadata={
                'method': 'shap_text',
                'nsamples': nsamples,
                'target_class': target,
                'tokens': tokens
            }
        )

        # 设置特征重要性
        result.feature_importance = self._get_feature_importance(tokens, values)

        # 添加可视化
        result.visualization = self._generate_visualization(input_text, shap_values, target)

        return result

    def _get_feature_importance(self, tokens: List[str], values: np.ndarray) -> Dict[str, float]:
        """创建特征重要性字典"""
        # 确保长度匹配
        min_len = min(len(tokens), len(values))
        tokens = tokens[:min_len]
        values = values[:min_len]

        return {token: float(value) for token, value in zip(tokens, values)}

    def _generate_visualization(self,
                                text: str,
                                shap_values: Any,
                                target: Optional[int]) -> Dict[str, Any]:
        """生成可视化结果"""
        # 获取SHAP可视化
        if self.class_names and target is not None:
            class_name = self.class_names[target]
        else:
            class_name = f"Class {target}" if target is not None else "Output"

        # 创建HTML可视化
        html_plot = shap.plots.text(shap_values, display=False)

        # 创建摘要图
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.bar(shap_values, show=False)
        plt.tight_layout()

        # 创建可视化字典
        visualization = {
            'html_plot': html_plot,
            'bar_plot': fig,
            'shap_values': shap_values.values,
            'base_value': shap_values.base_values,
            'type': 'shap_text'
        }

        return visualization

    def batch_explain(self,
                      input_batch: List[str],
                      targets: Optional[List[Any]] = None,
                      **kwargs) -> List[ExplanationResult]:
        """
        批量解释多个文本样本
        """
        results = []

        # 处理目标值
        if targets is None:
            targets = [None] * len(input_batch)

        for text, target in zip(input_batch, targets):
            results.append(self.explain(text, target=target, **kwargs))

        return results