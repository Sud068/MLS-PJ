"""
解释敏感度评估
衡量解释对输入变化的敏感程度
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class SensitivityEvaluator:
    """
    解释敏感度评估器

    评估解释对输入变化的敏感程度
    """

    @staticmethod
    def input_sensitivity(explainer, data: np.ndarray,
                          explanations: List[Dict[str, float]],
                          feature_index: int,
                          perturbation_range: Tuple[float, float] = (-1.0, 1.0),
                          num_steps: int = 10, **kwargs) -> float:
        """
        计算输入特征敏感度

        参数:
        explainer: 解释器
        data: 输入数据
        explanations: 原始解释结果列表
        feature_index: 要扰动的特征索引
        perturbation_range: 扰动范围
        num_steps: 扰动步数

        返回:
        敏感度分数 (0-1之间，越高表示越敏感)
        """
        # 选择样本子集进行评估
        sample_indices = np.random.choice(len(data), min(10, len(data)), replace=False)

        sensitivity_scores = []

        for idx in sample_indices:
            sample = data[idx].copy()
            original_exp = explanations[idx]

            # 获取原始特征重要性
            original_importance = original_exp.get(f'feature_{feature_index}', 0.0)

            # 生成扰动值
            perturbations = np.linspace(perturbation_range[0], perturbation_range[1], num_steps)
            importance_changes = []

            for p in perturbations:
                # 扰动特征
                perturbed_sample = sample.copy()
                perturbed_sample[feature_index] += p

                # 计算扰动后的解释
                perturbed_exp = explainer.explain(perturbed_sample)
                perturbed_importance = perturbed_exp.feature_importance.get(f'feature_{feature_index}', 0.0)

                # 记录重要性变化
                importance_changes.append(perturbed_importance - original_importance)

            # 计算变化率
            changes = np.abs(importance_changes)
            sensitivity = np.mean(changes) / (np.max(changes) + 1e-8)
            sensitivity_scores.append(sensitivity)

        return np.mean(sensitivity_scores)

    @staticmethod
    def max_sensitivity(explainer, data: np.ndarray,
                        explanations: List[Dict[str, float]],
                        num_perturbations: int = 5,
                        noise_scale: float = 0.1, **kwargs) -> float:
        """
        计算最大敏感度指标

        参数:
        explainer: 解释器
        data: 输入数据
        explanations: 原始解释结果列表
        num_perturbations: 每个样本的扰动次数
        noise_scale: 噪声比例

        返回:
        最大敏感度分数 (0-1之间)
        """
        max_sensitivities = []

        for i, sample in enumerate(data):
            original_exp = explanations[i]

            # 生成扰动样本
            perturbed_data = SensitivityEvaluator._perturb_sample(
                sample, num_perturbations, noise_scale
            )

            max_change = 0
            for perturbed_sample in perturbed_data:
                # 计算扰动后的解释
                perturbed_exp = explainer.explain(perturbed_sample)

                # 计算解释变化
                change = SensitivityEvaluator._explanation_change(
                    original_exp, perturbed_exp.feature_importance
                )
                max_change = max(max_change, change)

            max_sensitivities.append(max_change)

        return np.mean(max_sensitivities)

    @staticmethod
    def explanation_robustness(explainer, data: np.ndarray,
                               explanations: List[Dict[str, float]],
                               num_perturbations: int = 10,
                               noise_scale: float = 0.05, **kwargs) -> float:
        """
        计算解释鲁棒性指标 (敏感度的倒数)

        参数:
        explainer: 解释器
        data: 输入数据
        explanations: 原始解释结果列表
        num_perturbations: 每个样本的扰动次数
        noise_scale: 噪声比例

        返回:
        鲁棒性分数 (0-1之间，越高越好)
        """
        # 计算最大敏感度
        max_sens = SensitivityEvaluator.max_sensitivity(
            explainer, data, explanations, num_perturbations, noise_scale
        )

        # 鲁棒性为敏感度的倒数
        robustness = 1.0 - max_sens
        return max(0, robustness)

    @staticmethod
    def evaluate_all(explainer, data: np.ndarray,
                     explanations: List[Dict[str, float]],
                     **kwargs) -> Dict[str, float]:
        """
        计算所有敏感度指标

        返回:
        包含所有敏感度指标的字典
        """
        # 随机选择一个特征进行评估
        if data.shape[1] > 0:
            feature_idx = np.random.randint(0, data.shape[1])
        else:
            feature_idx = 0

        results = {
            'max_sensitivity': SensitivityEvaluator.max_sensitivity(
                explainer, data, explanations, **kwargs
            ),
            'input_feature_sensitivity': SensitivityEvaluator.input_sensitivity(
                explainer, data, explanations, feature_idx, **kwargs
            ),
            'explanation_robustness': SensitivityEvaluator.explanation_robustness(
                explainer, data, explanations, **kwargs
            )
        }

        # 计算综合敏感度分数
        results['composite_sensitivity'] = np.mean([
            results['max_sensitivity'],
            results['input_feature_sensitivity'],
            1 - results['explanation_robustness']  # 鲁棒性是敏感度的倒数
        ])

        return results

    @staticmethod
    def _perturb_sample(sample: np.ndarray, num_perturbations: int,
                        noise_scale: float) -> List[np.ndarray]:
        """生成扰动样本"""
        perturbed_samples = []
        for _ in range(num_perturbations):
            noise = np.random.normal(0, noise_scale * np.std(sample), sample.shape)
            perturbed = sample + noise
            perturbed_samples.append(perturbed)
        return perturbed_samples

    @staticmethod
    def _explanation_change(exp1: Dict[str, float], exp2: Dict[str, float]) -> float:
        """计算两个解释之间的变化量"""
        # 确保特征顺序一致
        features = sorted(exp1.keys())
        vec1 = np.array([exp1[f] for f in features])
        vec2 = np.array([exp2[f] for f in features])

        # 计算欧氏距离
        distance = np.linalg.norm(vec1 - vec2)

        # 归一化距离
        max_possible = np.linalg.norm(np.ones_like(vec1) * np.max(np.abs(vec1)))
        if max_possible > 0:
            normalized_distance = distance / max_possible
        else:
            normalized_distance = 0

        return min(1.0, normalized_distance)
    

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
def main():
    # 1. 加载数据
    X, y = fetch_california_housing(return_X_y=True)
    X = X[:50]  # 选前50条做测试
    y = y[:50]

    # 2. 拆分训练/测试
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # 3. 训练模型
    model = RandomForestRegressor(random_state=0)
    model.fit(X_train, y_train)

    # 4. 定义解释器
    class DummyExplainer:
        def __init__(self, model):
            self.model = model
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        def explain(self, x):
            coef = np.linspace(1, 2, X.shape[1])
            vals = x * coef
            # 返回一个对象，有 .feature_importance 属性（字典）
            return type('Exp', (), {'feature_importance': {f: v for f, v in zip(self.feature_names, vals)}})()

    # 5. 生成原始解释
    explainer = DummyExplainer(model)
    original_explanations = [explainer.explain(x).feature_importance for x in X_test]

    # 6. 敏感度评估
    # -- 这里直接用你上面给的 SensitivityEvaluator
    scores = SensitivityEvaluator.evaluate_all(explainer, X_test, original_explanations)

    # 7. 打印结果
    print("解释敏感度评估指标：")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()