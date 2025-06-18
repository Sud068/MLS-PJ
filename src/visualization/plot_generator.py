"""
静态图表生成器
创建各种静态图表可视化解释结果
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from io import BytesIO
import base64
from utils.validation import validate_not_none, validate_type

logger = logging.getLogger(__name__)


class PlotGenerator:
    """
    静态图表生成器

    创建各种图表可视化解释结果
    """

    def __init__(self, style: str = 'whitegrid', palette: str = 'viridis'):
        """
        初始化图表生成器

        参数:
        style: seaborn样式 (whitegrid, darkgrid, white, dark, ticks)
        palette: seaborn调色板 (viridis, magma, plasma, inferno, cividis)
        """
        self.style = style
        self.palette = palette
        self.set_style()

    def set_style(self):
        """设置图表样式"""
        sns.set_style(self.style)
        sns.set_palette(self.palette)
        plt.rcParams['font.family'] = 'DejaVu Sans'  # 支持中文
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

    def feature_importance(self,
                           importance: Dict[str, float],
                           title: str = 'Feature Importance',
                           top_n: Optional[int] = None,
                           figsize: tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        创建特征重要性条形图

        参数:
        importance: 特征重要性字典 {特征名: 重要性值}
        title: 图表标题
        top_n: 只显示前n个最重要的特征
        figsize: 图表大小

        返回:
        matplotlib Figure对象
        """
        validate_not_none(importance, "importance")
        validate_type(importance, dict, "importance")

        # 排序特征
        sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)

        # 选择前n个特征
        if top_n is not None and top_n > 0:
            sorted_features = sorted_features[:top_n]

        features, values = zip(*sorted_features)
        values = np.array(values)

        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        colors = ['#3498db' if v >= 0 else '#e74c3c' for v in values]
        y_pos = np.arange(len(features))

        # 绘制条形图
        ax.barh(y_pos, values, align='center', color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # 最重要的特征在顶部
        ax.set_xlabel('Importance Score')
        ax.set_title(title)

        # 添加数值标签
        for i, v in enumerate(values):
            ax.text(v if v >= 0 else v - 0.02,
                    i,
                    f"{v:.4f}",
                    color='white' if abs(v) > 0.1 else 'black',
                    va='center',
                    ha='right' if v < 0 else 'left')

        plt.tight_layout()
        return fig

    def prediction_distribution(self,
                                predictions: np.ndarray,
                                title: str = 'Prediction Distribution',
                                bins: int = 30,
                                figsize: tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        创建预测分布直方图

        参数:
        predictions: 预测值数组
        title: 图表标题
        bins: 直方图箱数
        figsize: 图表大小

        返回:
        matplotlib Figure对象
        """
        validate_not_none(predictions, "predictions")
        validate_type(predictions, np.ndarray, "predictions")

        fig, ax = plt.subplots(figsize=figsize)
        sns.histplot(predictions, bins=bins, kde=True, ax=ax)
        ax.set_xlabel('Prediction Value')
        ax.set_ylabel('Count')
        ax.set_title(title)
        plt.tight_layout()
        return fig

    def confusion_matrix(self,
                         cm: np.ndarray,
                         class_names: List[str],
                         title: str = 'Confusion Matrix',
                         figsize: tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        创建混淆矩阵热力图

        参数:
        cm: 混淆矩阵 (n x n)
        class_names: 类别名称列表
        title: 图表标题
        figsize: 图表大小

        返回:
        matplotlib Figure对象
        """
        validate_not_none(cm, "cm")
        validate_type(cm, np.ndarray, "cm")
        validate_type(class_names, list, "class_names")

        if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
            raise ValueError("混淆矩阵必须是方阵")
        if len(class_names) != cm.shape[0]:
            raise ValueError("类别名称数量必须与混淆矩阵维度一致")

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(title)
        plt.tight_layout()
        return fig

    def roc_curve(self,
                  fpr: np.ndarray,
                  tpr: np.ndarray,
                  roc_auc: float,
                  title: str = 'ROC Curve',
                  figsize: tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        创建ROC曲线图

        参数:
        fpr: 假正率数组
        tpr: 真正率数组
        roc_auc: AUC分数
        title: 图表标题
        figsize: 图表大小

        返回:
        matplotlib Figure对象
        """
        validate_not_none(fpr, "fpr")
        validate_not_none(tpr, "tpr")
        validate_type(fpr, np.ndarray, "fpr")
        validate_type(tpr, np.ndarray, "tpr")

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        plt.tight_layout()
        return fig

    def precision_recall_curve(self,
                               precision: np.ndarray,
                               recall: np.ndarray,
                               average_precision: float,
                               title: str = 'Precision-Recall Curve',
                               figsize: tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        创建精确率-召回率曲线图

        参数:
        precision: 精确率数组
        recall: 召回率数组
        average_precision: 平均精确率分数
        title: 图表标题
        figsize: 图表大小

        返回:
        matplotlib Figure对象
        """
        validate_not_none(precision, "precision")
        validate_not_none(recall, "recall")
        validate_type(precision, np.ndarray, "precision")
        validate_type(recall, np.ndarray, "recall")

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(recall, precision, color='blue', lw=2,
                label=f'Precision-Recall (AP = {average_precision:.2f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend(loc="lower left")
        plt.tight_layout()
        return fig

    def scatter_plot(self,
                     x: np.ndarray,
                     y: np.ndarray,
                     hue: Optional[np.ndarray] = None,
                     title: str = 'Scatter Plot',
                     x_label: str = 'X',
                     y_label: str = 'Y',
                     figsize: tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        创建散点图

        参数:
        x: x值数组
        y: y值数组
        hue: 用于着色的类别数组 (可选)
        title: 图表标题
        x_label: x轴标签
        y_label: y轴标签
        figsize: 图表大小

        返回:
        matplotlib Figure对象
        """
        validate_not_none(x, "x")
        validate_not_none(y, "y")
        validate_type(x, np.ndarray, "x")
        validate_type(y, np.ndarray, "y")

        fig, ax = plt.subplots(figsize=figsize)

        if hue is not None:
            sns.scatterplot(x=x, y=y, hue=hue, palette='viridis', ax=ax)
        else:
            sns.scatterplot(x=x, y=y, ax=ax)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        plt.tight_layout()
        return fig

    def to_base64(self, fig: plt.Figure, format: str = 'png') -> str:
        """
        将图表转换为Base64编码的字符串

        参数:
        fig: matplotlib图表对象
        format: 图像格式 (png, jpg, svg)

        返回:
        Base64编码的图像字符串
        """
        validate_not_none(fig, "fig")

        buffer = BytesIO()
        fig.savefig(buffer, format=format, bbox_inches='tight')
        plt.close(fig)  # 关闭图表释放内存
        buffer.seek(0)

        base64_str = base64.b64encode(buffer.read()).decode('utf-8')
        return f"data:image/{format};base64,{base64_str}"

    def to_html(self, fig: plt.Figure) -> str:
        """
        将图表转换为HTML img标签

        参数:
        fig: matplotlib图表对象

        返回:
        包含图表的HTML字符串
        """
        base64_img = self.to_base64(fig)
        return f'<img src="{base64_img}" alt="Chart">'