"""
SHAP解释器实现
用于表格数据的SHAP值计算
"""

import numpy as np
import shap
from typing import Any, Dict, List, Optional, Union
from core.explainer import BaseExplainer, ExplanationResult
import logging
import warnings
import pandas as pd

# 忽略SHAP的警告
warnings.filterwarnings("ignore", category=UserWarning, module="shap")

logger = logging.getLogger(__name__)


class SHAPExplainer(BaseExplainer):
    """
    SHAP解释器实现

    支持:
    - KernelSHAP (模型无关)
    - TreeSHAP (树模型专用)
    - DeepSHAP (深度学习模型)
    """

    def __init__(self,
                 model: Any,
                 task_type: str,
                 feature_names: Optional[List[str]] = None,
                 background_data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 **kwargs):
        """
        初始化SHAP解释器

        参数:
        model: 待解释的模型
        task_type: 任务类型 ('classification'/'regression')
        feature_names: 特征名称列表
        background_data: 背景数据集 (用于KernelSHAP)
        kwargs:
          - method: 指定SHAP方法 ('auto', 'kernel', 'tree', 'deep', 'linear')
          - nsamples: KernelSHAP的样本数
          - link: 链接函数 ('identity'/'logit')
        """
        super().__init__(model, task_type, feature_names, **kwargs)

        # 设置默认参数
        self.method = kwargs.get('method', 'auto')
        self.nsamples = kwargs.get('nsamples', 1000)
        self.link = kwargs.get('link', 'identity')

        # 创建背景数据
        self.background_data = self._prepare_background(background_data)

        # 初始化解释器
        self.explainer = self._create_explainer()

        # 日志记录
        logger.info(f"SHAP解释器初始化完成: method={self.method}, nsamples={self.nsamples}")

    def _prepare_background(self, background_data):
        """准备背景数据集"""
        if background_data is not None:
            return background_data

        # 如果没有提供背景数据，尝试使用训练数据
        if hasattr(self.model, 'X_train_'):
            return self.model.X_train_

        # 对于树模型，使用特征重要性计算
        if hasattr(self.model, 'feature_importances_'):
            return None

        # 对于深度学习模型，使用零值或均值
        return None

    def _create_explainer(self):
        """根据模型类型创建合适的SHAP解释器"""
        model_type = str(type(self.model)).lower()

        # 自动检测最佳方法
        if self.method == 'auto':
            if 'tree' in model_type or 'forest' in model_type or 'xgboost' in model_type:
                self.method = 'tree'
            elif 'linear' in model_type or 'logistic' in model_type:
                self.method = 'linear'
            elif 'torch' in model_type or 'tensorflow' in model_type:
                self.method = 'deep'
            else:
                self.method = 'kernel'

        # 创建特定类型的解释器
        if self.method == 'tree':
            return shap.TreeExplainer(
                self.model,
                data=self.background_data,
                feature_perturbation='tree_path_dependent'
            )
        elif self.method == 'linear':
            return shap.LinearExplainer(self.model, self.background_data)
        elif self.method == 'kernel':
            return shap.KernelExplainer(
                self.model.predict,
                self.background_data,
                link=self.link
            )
        elif self.method == 'deep':
            return shap.DeepExplainer(self.model, self.background_data)
        else:
            raise ValueError(f"不支持的SHAP方法: {self.method}")

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
          - nsamples: 覆盖初始化的样本数
          - check_additivity: 验证SHAP值相加等于预测值
        """
        # 转换输入数据
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)

        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)

        # 获取SHAP值
        nsamples = kwargs.get('nsamples', self.nsamples)
        check_additivity = kwargs.get('check_additivity', True)
        is_kernel = 'kernel' in str(type(self.explainer)).lower()
        # 对于分类模型，计算指定类别的SHAP值
        if self.task_type == 'classification' and target is not None:
            if is_kernel:
                shap_values = self.explainer.shap_values(
                    input_data,
                    nsamples=nsamples,
                    check_additivity=check_additivity
                )
            else:
                shap_values = self.explainer.shap_values(
                    input_data,
                    check_additivity=check_additivity
                )

            # 多分类模型返回列表
            if isinstance(shap_values, list):
                if target < len(shap_values):
                    shap_vals = shap_values[target]
                else:
                    logger.warning(f"目标类别{target}超出范围，使用第一个类别")
                    shap_vals = shap_values[0]
            else:
                shap_vals = shap_values
        else:
            if is_kernel:
                shap_vals = self.explainer.shap_values(
                    input_data,
                    nsamples=nsamples,
                    check_additivity=check_additivity
                )
            else:
                shap_vals = self.explainer.shap_values(
                    input_data,
                    check_additivity=check_additivity
                )

        # 确保是二维数组
        if shap_vals.ndim == 1:
            shap_vals = shap_vals.reshape(1, -1)

        # 创建解释结果
        result = ExplanationResult(
            raw_result=shap_vals,
            metadata={
                'method': self.method,
                'nsamples': nsamples,
                'explainer_type': str(type(self.explainer))
            }
        )
        # print(shap_vals.shape)
        if shap_vals.ndim == 3 and shap_vals.shape[0] == 1:
            # (1, n_features, n_outputs) => (n_features, n_outputs)
            shap_vals = shap_vals[0]

        # 处理特征重要性
        if shap_vals.ndim == 2 and shap_vals.shape[0] == len(self.feature_names):
            # shap_vals: (n_features, n_outputs)          
            if target is not None and shap_vals.shape[1] > 1:
                feature_importance = self._get_feature_importance(shap_vals[:, target])
            else:
                feature_importance = self._get_feature_importance(shap_vals.mean(axis=1))
        elif shap_vals.ndim == 2 and shap_vals.shape[0] == 1:
            feature_importance = self._get_feature_importance(shap_vals[0])
        else:
            feature_importance = self._get_feature_importance(shap_vals.squeeze())
        result.feature_importance = feature_importance

        # 添加可视化数据
        result.visualization = self._generate_visualization(input_data, shap_vals, target)

        return result

    def _get_feature_importance(self, shap_values: np.ndarray) -> Dict[str, float]:
        """从SHAP值创建特征重要性字典"""
        if self.feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(len(shap_values))]
        return {name: float(np.squeeze(val)) for name, val in zip(self.feature_names, shap_values)}

    # def _generate_visualization(self, input_data: np.ndarray, shap_values: np.ndarray):
    #     """生成SHAP可视化数据"""
    #     # 创建SHAP对象
    #     if self.method == 'kernel':
    #         shap_obj = shap.Explanation(
    #             values=shap_values,
    #             base_values=self.explainer.expected_value,
    #             data=input_data,
    #             feature_names=self.feature_names
    #         )
    #     else:
    #         shap_obj = shap.Explanation(
    #             values=shap_values,
    #             base_values=self.explainer.expected_value,
    #             data=input_data,
    #             feature_names=self.feature_names
    #         )

    #     return {
    #         'force_plot': shap.force_plot(
    #             self.explainer.expected_value,
    #             shap_values,
    #             input_data,
    #             feature_names=self.feature_names,
    #             matplotlib=False
    #         ),
    #         'summary_plot': shap_obj,
    #         'type': 'shap'
    #     }
    def _generate_visualization(self, input_data: np.ndarray, shap_values: np.ndarray, target: Optional[int] = None):
        """生成SHAP可视化数据"""
        # 取目标类别shap值和base_value
        if shap_values.ndim == 2 and shap_values.shape[1] > 1 and target is not None:
            shap_for_target = shap_values[:, target]
            base_value = self.explainer.expected_value[target] if isinstance(self.explainer.expected_value, (list, np.ndarray)) else self.explainer.expected_value
        else:
            shap_for_target = shap_values.squeeze()
            base_value = self.explainer.expected_value if not isinstance(self.explainer.expected_value, (list, np.ndarray)) else self.explainer.expected_value[0]

        shap_obj = shap.Explanation(
            values=shap_for_target,
            base_values=base_value,
            data=input_data,
            feature_names=self.feature_names
        )

        return {
            'force_plot': shap.force_plot(
                base_value,
                shap_for_target,
                input_data,
                feature_names=self.feature_names,
                matplotlib=False
            ),
            'summary_plot': shap_obj,
            'type': 'shap'
        }
    def batch_explain(self,
                      input_batch: Union[np.ndarray, pd.DataFrame],
                      targets: Optional[List[Any]] = None,
                      **kwargs) -> List[ExplanationResult]:
        """
        批量解释优化版本 (避免重复初始化)
        """
        # 转换输入数据
        if not isinstance(input_batch, np.ndarray):
            input_batch = np.array(input_batch)

        # 处理目标值
        if targets is None:
            targets = [None] * len(input_batch)

        # 一次性计算所有SHAP值
        nsamples = kwargs.get('nsamples', self.nsamples)
        check_additivity = kwargs.get('check_additivity', True)

        # 对于分类任务的特殊处理
        if self.task_type == 'classification':
            all_shap_values = self.explainer.shap_values(
                input_batch,
                nsamples=nsamples,
                check_additivity=check_additivity
            )

            # 多分类模型返回列表
            if isinstance(all_shap_values, list):
                # 为每个样本选择正确的目标类别
                results = []
                for i, (data, target) in enumerate(zip(input_batch, targets)):
                    if target is None:
                        target = 0

                    if target < len(all_shap_values):
                        shap_vals = all_shap_values[target][i]
                    else:
                        shap_vals = all_shap_values[0][i]

                    results.append(self._create_explanation(data, shap_vals))
                return results
            else:
                shap_vals = all_shap_values
        else:
            shap_vals = self.explainer.shap_values(
                input_batch,
                nsamples=nsamples,
                check_additivity=check_additivity
            )

        # 创建解释结果列表
        results = []
        for i, data in enumerate(input_batch):
            results.append(self._create_explanation(data, shap_vals[i]))

        return results

    def _create_explanation(self, input_data: np.ndarray, shap_vals: np.ndarray) -> ExplanationResult:
        """从原始数据创建解释结果"""
        # 创建解释结果
        result = ExplanationResult(
            raw_result=shap_vals,
            metadata={
                'method': self.method,
                'nsamples': self.nsamples,
                'explainer_type': str(type(self.explainer))
            }
        )

        # 设置特征重要性
        feature_importance = self._get_feature_importance(shap_vals)
        result.feature_importance = feature_importance

        # 添加可视化数据
        result.visualization = self._generate_visualization(
            input_data.reshape(1, -1),
            shap_vals.reshape(1, -1)
        )

        return result
    
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
    # 1. 加载和准备数据
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    # 只用0/1两类
    X = X[y < 2]
    y = y[y < 2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # 2. 训练模型
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 3. 初始化SHAP解释器
    feature_names = list(X.columns)
    # background_data可以用训练集的前100条
    background_data = X_train.values[:100]

    explainer = SHAPExplainer(
        model=model,
        task_type='classification',
        feature_names=feature_names,
        background_data=background_data,
        method='auto',
        nsamples=100
    )

    # 4. 选一条测试样本生成SHAP解释
    x0 = X_test.iloc[0].values
    print("原始样本：")
    print(X_test.iloc[0])
    print("原始类别:", model.predict([x0])[0])
    print("原始概率:", model.predict_proba([x0])[0])

    result = explainer.explain(x0, target=1)

    # 5. 打印SHAP解释结果
    print("\nSHAP特征贡献：")
    for feat, val in result.feature_importance.items():
        print(f"{feat}: {val:.4f}")

    print("\n可视化（force_plot对象）：", type(result.visualization['force_plot']))
    print("SHAP summary_plot对象:", type(result.visualization['summary_plot']))

    # 如果你想保存force_plot为html
    import shap
    shap.save_html("shap_force_plot.html", result.visualization['force_plot'])

if __name__ == "__main__":
    main()