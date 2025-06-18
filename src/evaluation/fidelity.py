"""
解释保真度评估
衡量解释是否忠实于原始模型的行为
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class FidelityEvaluator:
    """
    解释保真度评估器

    评估解释是否忠实反映原始模型的行为
    """

    @staticmethod
    def fidelity_plus(model, explainer, data: np.ndarray,
                      explanations: List[Dict[str, float]],
                      top_k: int = 5, **kwargs) -> float:
        """
        计算Fidelity+指标

        参数:
        model: 原始模型
        explainer: 解释器
        data: 输入数据
        explanations: 解释结果列表 (每个样本的特征重要性字典)
        top_k: 使用最重要的前k个特征

        返回:
        保真度分数 (0-1之间，越高越好)
        """
        # 原始模型预测
        original_preds = model.predict(data)

        # 使用解释创建简化模型
        simplified_preds = []

        for i, sample in enumerate(data):
            # 获取当前样本的解释
            exp = explanations[i]

            # 获取最重要的top_k特征索引
            sorted_features = sorted(exp.items(), key=lambda x: abs(x[1]), reverse=True)
            top_indices = [int(f.split('_')[-1]) for f, _ in sorted_features[:top_k]]

            # 创建简化输入 (只保留重要特征，其他设为零)
            simplified_sample = np.zeros_like(sample)
            simplified_sample[top_indices] = sample[top_indices]

            # 使用原始模型预测简化输入
            pred = model.predict(simplified_sample.reshape(1, -1))[0]
            simplified_preds.append(pred)

        # 计算相关系数作为保真度
        fidelity = np.corrcoef(original_preds, simplified_preds)[0, 1]
        return max(0, fidelity)  # 确保非负

    @staticmethod
    def fidelity_minus(model, explainer, data: np.ndarray,
                       explanations: List[Dict[str, float]],
                       top_k: int = 5, **kwargs) -> float:
        """
        计算Fidelity-指标

        参数:
        model: 原始模型
        explainer: 解释器
        data: 输入数据
        explanations: 解释结果列表 (每个样本的特征重要性字典)
        top_k: 使用最不重要的前k个特征

        返回:
        保真度分数 (0-1之间，越高越好)
        """
        # 原始模型预测
        original_preds = model.predict(data)

        # 使用解释创建简化模型
        simplified_preds = []

        for i, sample in enumerate(data):
            # 获取当前样本的解释
            exp = explanations[i]

            # 获取最不重要的top_k特征索引
            sorted_features = sorted(exp.items(), key=lambda x: abs(x[1]))
            bottom_indices = [int(f.split('_')[-1]) for f, _ in sorted_features[:top_k]]

            # 创建简化输入 (只保留不重要特征，其他设为零)
            simplified_sample = np.zeros_like(sample)
            simplified_sample[bottom_indices] = sample[bottom_indices]

            # 使用原始模型预测简化输入
            pred = model.predict(simplified_sample.reshape(1, -1))[0]
            simplified_preds.append(pred)

        # 计算与原始预测的差异
        differences = np.abs(np.array(original_preds) - np.array(simplified_preds))
        fidelity = 1.0 - np.mean(differences) / np.max(differences)
        return max(0, min(1, fidelity))

    @staticmethod
    def prediction_similarity(explainer, data: np.ndarray,
                              explanations: List[Dict[str, float]],
                              surrogate_model, **kwargs) -> float:
        """
        通过代理模型计算预测相似度

        参数:
        explainer: 解释器
        data: 输入数据
        explanations: 解释结果列表
        surrogate_model: 代理模型 (如线性模型)

        返回:
        预测相似度分数 (R²分数)
        """
        # 原始模型预测
        original_preds = explainer.model.predict(data)

        # 使用解释训练代理模型
        X_surrogate = []
        for exp in explanations:
            # 将特征重要性转换为向量
            features = sorted(exp.items(), key=lambda x: x[0])
            feature_vector = [val for _, val in features]
            X_surrogate.append(feature_vector)

        X_surrogate = np.array(X_surrogate)

        # 训练代理模型
        surrogate_model.fit(X_surrogate, original_preds)

        # 代理模型预测
        surrogate_preds = surrogate_model.predict(X_surrogate)

        # 计算R²分数
        r2 = r2_score(original_preds, surrogate_preds)
        return max(0, r2)

    @staticmethod
    def evaluate_all(model, explainer, data: np.ndarray,
                     explanations: List[Dict[str, float]],
                     surrogate_model, **kwargs) -> Dict[str, float]:
        """
        计算所有保真度指标

        返回:
        包含所有保真度指标的字典
        """
        top_k = kwargs.get('top_k', 5)

        results = {
            'fidelity_plus': FidelityEvaluator.fidelity_plus(
                model, explainer, data, explanations, top_k
            ),
            'fidelity_minus': FidelityEvaluator.fidelity_minus(
                model, explainer, data, explanations, top_k
            ),
            'prediction_similarity': FidelityEvaluator.prediction_similarity(
                explainer, data, explanations, surrogate_model
            )
        }

        # 计算综合保真度分数
        results['composite_fidelity'] = np.mean(list(results.values()))

        return results
    
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
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
            return type('Exp', (), {'feature_importance': {f: v for f, v in zip(self.feature_names, vals)}})()

    # 5. 生成原始解释
    explainer = DummyExplainer(model)
    original_explanations = [explainer.explain(x).feature_importance for x in X_test]

    # 6. 定义代理模型（用于 prediction_similarity）
    surrogate_model = LinearRegression()

    # 7. 保真度评估
    scores = FidelityEvaluator.evaluate_all(model, explainer, X_test, original_explanations, surrogate_model=surrogate_model)

    # 8. 打印结果
    print("解释保真度评估指标：")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()