"""
解释稳定性评估
衡量解释对于输入微小变化的鲁棒性
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class StabilityEvaluator:
    """
    解释稳定性评估器

    评估解释对输入微小变化的鲁棒性
    """

    @staticmethod
    def local_stability(explainer, data: np.ndarray,
                        explanations: List[Dict[str, float]],
                        num_perturbations: int = 10,
                        noise_scale: float = 0.05, **kwargs) -> float:
        """
        计算局部稳定性指标

        参数:
        explainer: 解释器
        data: 输入数据
        explanations: 原始解释结果列表
        num_perturbations: 每个样本的扰动次数
        noise_scale: 噪声比例

        返回:
        平均稳定性分数 (0-1之间，越高越好)
        """
        stability_scores = []

        for i, sample in enumerate(data):
            original_exp = explanations[i]

            # 生成扰动样本
            perturbed_data = StabilityEvaluator._perturb_sample(
                sample, num_perturbations, noise_scale
            )

            # 计算扰动样本的解释
            perturbed_exps = []
            for perturbed_sample in perturbed_data:
                exp = explainer.explain(perturbed_sample)
                perturbed_exps.append(exp.feature_importance)

            # 计算稳定性分数
            exp_similarities = []
            for perturbed_exp in perturbed_exps:
                # 计算解释相似度
                similarity = StabilityEvaluator._explanation_similarity(
                    original_exp, perturbed_exp
                )
                exp_similarities.append(similarity)

            # 平均相似度作为当前样本的稳定性
            stability_scores.append(np.mean(exp_similarities))

        return np.mean(stability_scores)

    # @staticmethod
    # def explanation_consistency(explainer, data: np.ndarray,
    #                             explanations: List[Dict[str, float]],
    #                             num_subsets: int = 5,
    #                             subset_ratio: float = 0.8, **kwargs) -> float:
    #     """
    #     计算解释一致性指标

    #     参数:
    #     explainer: 解释器
    #     data: 输入数据
    #     explanations: 解释结果列表
    #     num_subsets: 子集数量
    #     subset_ratio: 子集比例

    #     返回:
    #     平均一致性分数 (0-1之间，越高越好)
    #     """
    #     consistency_scores = []

    #     for i, sample in enumerate(data):
    #         original_exp = explanations[i]

    #         # 生成数据子集
    #         subsets = StabilityEvaluator._create_subsets(
    #             data, i, num_subsets, subset_ratio
    #         )

    #         # 计算每个子集的解释
    #         subset_exps = []
    #         for subset in subsets:
    #             exp = explainer.explain(subset)
    #             subset_exps.append(exp.feature_importance)

    #         # 计算一致性分数
    #         exp_similarities = []
    #         for subset_exp in subset_exps:
    #             similarity = StabilityEvaluator._explanation_similarity(
    #                 original_exp, subset_exp
    #             )
    #             exp_similarities.append(similarity)

    #         consistency_scores.append(np.mean(exp_similarities))

    #     return np.mean(consistency_scores)
    @staticmethod
    def explanation_consistency(explainer, data: np.ndarray,
                                explanations: List[Dict[str, float]],
                                num_subsets: int = 5,
                                subset_ratio: float = 0.8, **kwargs) -> float:
        consistency_scores = []

        for i, sample in enumerate(data):
            original_exp = explanations[i]

            # 生成数据子集
            subsets = StabilityEvaluator._create_subsets(
                data, i, num_subsets, subset_ratio
            )

            # 用不同子集设定下，对同一个样本解释
            subset_exps = []
            for subset in subsets:
                exp = explainer.explain(sample)
                subset_exps.append(exp.feature_importance)

            # 计算一致性分数
            exp_similarities = []
            for subset_exp in subset_exps:
                similarity = StabilityEvaluator._explanation_similarity(
                    original_exp, subset_exp
                )
                exp_similarities.append(similarity)

            consistency_scores.append(np.mean(exp_similarities))

        return np.mean(consistency_scores)


    @staticmethod
    def model_confidence_stability(model, explainer, data: np.ndarray,
                                   explanations: List[Dict[str, float]],
                                   confidence_threshold: float = 0.7, **kwargs) -> float:
        """
        计算模型置信度稳定性指标

        参数:
        model: 原始模型
        explainer: 解释器
        data: 输入数据
        explanations: 解释结果列表
        confidence_threshold: 置信度阈值

        返回:
        稳定性分数 (0-1之间，越高越好)
        """
        # 获取模型预测置信度
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(data)
            confidences = np.max(probas, axis=1)
        else:
            # 对于回归模型，使用预测值的归一化
            preds = model.predict(data)
            confidences = 1 - (np.abs(preds - np.mean(preds)) / np.max(np.abs(preds - np.mean(preds))))

        # 高置信度样本索引
        high_conf_indices = np.where(confidences >= confidence_threshold)[0]

        if len(high_conf_indices) == 0:
            logger.warning("没有高置信度样本可用于稳定性评估")
            return 0.0

        # 计算高置信度样本的解释相似度
        high_conf_exps = [explanations[i] for i in high_conf_indices]

        similarities = []
        for i in range(len(high_conf_exps)):
            for j in range(i + 1, len(high_conf_exps)):
                similarity = StabilityEvaluator._explanation_similarity(
                    high_conf_exps[i], high_conf_exps[j]
                )
                similarities.append(similarity)

        return np.mean(similarities) if similarities else 0.0

    @staticmethod
    def evaluate_all(explainer, data: np.ndarray,
                     explanations: List[Dict[str, float]],
                     model=None, **kwargs) -> Dict[str, float]:
        """
        计算所有稳定性指标

        返回:
        包含所有稳定性指标的字典
        """
        results = {
            'local_stability': StabilityEvaluator.local_stability(
                explainer, data, explanations, **kwargs
            )
        }

        if model is not None:
            results['model_confidence_stability'] = StabilityEvaluator.model_confidence_stability(
                model, explainer, data, explanations, **kwargs
            )

        results['explanation_consistency'] = StabilityEvaluator.explanation_consistency(
            explainer, data, explanations, **kwargs
        )

        # 计算综合稳定性分数
        results['composite_stability'] = np.mean(list(results.values()))

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
    def _create_subsets(data: np.ndarray, exclude_index: int,
                        num_subsets: int, subset_ratio: float) -> List[np.ndarray]:
        """创建数据子集"""
        subsets = []
        n = len(data)
        subset_size = int(n * subset_ratio)

        for _ in range(num_subsets):
            # 随机选择索引 (排除当前样本)
            indices = np.random.choice(np.delete(np.arange(n), exclude_index),
                                       size=subset_size, replace=False)
            subset = data[indices]
            subsets.append(subset)

        return subsets

    @staticmethod
    def _explanation_similarity(exp1: Dict[str, float], exp2: Dict[str, float]) -> float:
        """计算两个解释之间的相似度"""
        # 确保特征顺序一致
        features = sorted(exp1.keys())
        vec1 = np.array([exp1[f] for f in features])
        vec2 = np.array([exp2[f] for f in features])

        # 归一化向量
        vec1 = (vec1 - np.min(vec1)) / (np.max(vec1) - np.min(vec1) + 1e-8)
        vec2 = (vec2 - np.min(vec2)) / (np.max(vec2) - np.min(vec2) + 1e-8)

        # 计算余弦相似度
        cos_sim = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]

        # 计算Jensen-Shannon散度
        js_div = jensenshannon(vec1, vec2)
        js_sim = 1 - js_div

        # 平均相似度
        return (cos_sim + js_sim) / 2
    

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
def main():
    # 1. 加载数据（替换成加州房价数据集）
    X, y = fetch_california_housing(return_X_y=True)
    X = X[:50]  # 只用一小部分数据便于测试
    y = y[:50]

    # 2. 拆分训练/测试
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # 3. 训练模型
    model = RandomForestRegressor(random_state=0)
    model.fit(X_train, y_train)

    # 4. 定义一个简单的解释器
    class DummyExplainer:
        def __init__(self, model):
            self.model = model
            self.feature_names = [f"f{i}" for i in range(X.shape[1])]
        def explain(self, x):
            coef = np.linspace(1, 2, X.shape[1])
            vals = x * coef
            return type('Exp', (), {'feature_importance': {f: v for f, v in zip(self.feature_names, vals)}})()

    # 5. 生成原始解释
    explainer = DummyExplainer(model)
    original_explanations = [explainer.explain(x).feature_importance for x in X_test]



    scores = StabilityEvaluator.evaluate_all(explainer, X_test, original_explanations, model=model)

    # 7. 打印结果
    print("解释稳定性评估指标：")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")

        
if __name__ == "__main__":
    main()