"""
LIME文本解释器实现
用于局部可解释的文本解释
"""

import numpy as np
import re
import string
from typing import Any, Dict, List, Optional, Union
from core.explainer import BaseExplainer, ExplanationResult
import logging
import lime
from lime.lime_text import LimeTextExplainer
import nltk
from nltk.corpus import stopwords
import warnings
import matplotlib.pyplot as plt
# 下载停用词资源
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# 忽略警告
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class LimeTextExplainerWrapper(BaseExplainer):
    """
    LIME文本解释器实现

    提供局部可解释的模型无关文本解释
    """

    def __init__(self,
                 model: Any,
                 task_type: str,
                 **kwargs):
        """
        初始化LIME文本解释器

        参数:
        model: 待解释的模型
        task_type: 任务类型 ('classification'/'regression')
        kwargs:
          - class_names: 类别名称列表 (分类任务)
          - num_features: 显示的特征数
          - num_samples: 生成的样本数
          - random_state: 随机种子
          - bow: 是否使用词袋表示 (默认True)
          - mask_string: 掩盖字符串 (如 'UNK')
          - tokenizer: 自定义分词器函数
        """
        super().__init__(model, task_type, **kwargs)

        # 验证任务类型
        if task_type not in ['classification', 'regression']:
            raise ValueError(f"LIME不支持的任务类型: {task_type}")

        # 设置参数
        self.class_names = kwargs.get('class_names', None)
        self.num_features = kwargs.get('num_features', 10)
        self.num_samples = kwargs.get('num_samples', 5000)
        self.random_state = kwargs.get('random_state', 42)
        self.bow = kwargs.get('bow', True)
        self.mask_string = kwargs.get('mask_string', '[MASK]')
        self.tokenizer = kwargs.get('tokenizer', self._default_tokenizer)

        # 创建LIME解释器
        self.explainer = LimeTextExplainer(
            class_names=self.class_names,
            bow=self.bow,
            mask_string=self.mask_string,
            random_state=self.random_state
        )

        logger.info(f"LIME文本解释器初始化完成: num_features={self.num_features}, num_samples={self.num_samples}")

    def _default_tokenizer(self, text: str) -> List[str]:
        """默认分词器"""
        # 简单分词：转换为小写，移除标点，按空格分割
        text = text.lower()
        text = re.sub(f'[{string.punctuation}]', ' ', text)
        return text.split()

    def explain(self,
                input_text: str,
                target: Optional[Any] = None,
                **kwargs) -> ExplanationResult:
        """
        解释单个文本样本

        参数:
        input_text: 输入文本
        target: 目标类别 (分类任务)
        kwargs:
          - num_features: 覆盖初始化的特征数
          - top_labels: 返回前N个类别的解释
          - hide_rest: 是否隐藏无关特征
        """
        # 获取参数
        num_features = kwargs.get('num_features', self.num_features)
        top_labels = kwargs.get('top_labels', 1)
        hide_rest = kwargs.get('hide_rest', False)

        # 定义预测函数
        def predict_fn(texts):
            return self.model.predict_proba(texts)

        # 生成解释
        explanation = self.explainer.explain_instance(
            text_instance=input_text,
            classifier_fn=predict_fn,
            labels=(target,) if target is not None else None,
            top_labels=top_labels,
            num_features=num_features,
            num_samples=self.num_samples
        )

        # 创建解释结果
        result = ExplanationResult(
            raw_result=explanation,
            metadata={
                'method': 'lime_text',
                'num_samples': self.num_samples,
                'target_class': target,
                'num_features': num_features
            }
        )

        # 设置特征重要性
        result.feature_importance = self._get_feature_importance(explanation, num_features)

        # 添加可视化
        result.visualization = self._generate_visualization(input_text, explanation, hide_rest)

        return result

    def _get_feature_importance(self, explanation, num_features: int) -> Dict[str, float]:
        """从LIME解释中提取特征重要性"""
        feature_importance = {}

        # 获取解释的本地特征重要性
        if explanation.top_labels:
            label = explanation.top_labels[0]
            for feature, weight in explanation.as_list(label=label):
                feature_importance[feature] = weight
        else:
            for feature, weight in explanation.as_list():
                feature_importance[feature] = weight

        return feature_importance

    def _generate_visualization(self,
                                text: str,
                                explanation: Any,
                                hide_rest: bool) -> Dict[str, Any]:
        """生成可视化结果"""
        # 获取高亮文本
        if explanation.top_labels:
            label = explanation.top_labels[0]
            exp_list = explanation.as_list(label=label)
        else:
            exp_list = explanation.as_list()

        # 创建高亮HTML
        highlighted_text = self._highlight_text(text, exp_list, hide_rest)

        # 创建可视化字典
        visualization = {
            'highlighted_text': highlighted_text,
            'explanation_list': exp_list,
            'local_pred': explanation.local_pred,
            'type': 'lime_text'
        }

        return visualization

    def _highlight_text(self, text: str, exp_list: list, hide_rest: bool) -> str:
        """生成高亮文本HTML"""
        # 获取重要特征及其权重
        features = {word: weight for word, weight in exp_list}

        # 分词
        tokens = self.tokenizer(text)

        # 创建高亮HTML
        html_output = '<div style="font-family: monospace; font-size: 14px; line-height: 1.5;">'

        for token in tokens:
            # 清理token用于匹配
            clean_token = token.strip(string.punctuation).lower()

            # 检查是否在重要特征中
            if clean_token in features:
                weight = features[clean_token]
                # 根据权重设置颜色 (红色表示负贡献，绿色表示正贡献)
                if weight < 0:
                    color = f"rgba(255, 0, 0, {min(0.9, abs(weight) * 2)})"
                else:
                    color = f"rgba(0, 255, 0, {min(0.9, abs(weight) * 2)})"

                # 添加高亮
                html_output += f'<span style="background-color: {color};">{token}</span> '
            else:
                if not hide_rest:
                    html_output += f'{token} '
                else:
                    # 隐藏不重要的词
                    if token in string.punctuation:
                        html_output += token
                    else:
                        html_output += '... '

        html_output += '</div>'
        return html_output

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
    
def main():
    # 1. 构造一个假模型（具有predict_proba方法）
    class DummyTextModel:
        def predict_proba(self, texts):
            # 假设二分类，返回 [p0, p1]，简单用长度决定概率
            results = []
            for text in texts:
                p1 = min(1.0, max(0.0, len(text) / 40))
                results.append([1 - p1, p1])
            return np.array(results)
        def predict(self, texts):
            # 返回类别
            proba = self.predict_proba(texts)
            return np.argmax(proba, axis=1)

    # 2. 初始化LIME解释器
    model = DummyTextModel()
    class_names = ['neg', 'pos']
    explainer = LimeTextExplainerWrapper(
        model=model,
        task_type='classification',
        class_names=class_names,
        num_features=6
    )

    # 3. 输入文本
    text = "LIME is a great tool for understanding model decisions in NLP tasks!"
    result = explainer.explain(text, target=1, num_features=6)

    print("原文：", text)
    print("特征贡献：")
    for feat, val in result.feature_importance.items():
        print(f"{feat}: {val:.4f}")

    print("\n高亮HTML：\n", result.visualization['highlighted_text'])

    # 如需浏览器查看高亮：
    with open('lime_highlight.html', 'w', encoding='utf-8') as f:
        f.write(result.visualization['highlighted_text'])
    print("\n已保存高亮HTML到 lime_highlight.html，可用浏览器打开。")

if __name__ == '__main__':
    main()