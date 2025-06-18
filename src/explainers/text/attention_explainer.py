"""
注意力解释器实现
用于可视化基于注意力的模型的注意力权重
"""

import numpy as np
import torch
from typing import Any, Dict, List, Optional, Union
from core.explainer import BaseExplainer, ExplanationResult
import logging
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
import re
import matplotlib.pyplot as plt
# 忽略警告
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class AttentionExplainer(BaseExplainer):
    """
    注意力解释器实现

    专门用于基于注意力的模型（如Transformer）
    """

    def __init__(self,
                 model: Any,
                 task_type: str,
                 **kwargs):
        """
        初始化注意力解释器

        参数:
        model: 基于注意力的模型 (如BERT, GPT)
        task_type: 任务类型 ('classification'/'regression'/'generation')
        kwargs:
          - layer_index: 要可视化的层索引 (None表示所有层)
          - head_index: 要可视化的注意力头索引 (None表示所有头)
          - tokenizer: 模型的分词器
          - special_tokens: 特殊token列表 (如['[CLS]', '[SEP]'])
        """
        super().__init__(model, task_type, **kwargs)

        # 验证模型类型
        if not self._is_attention_model():
            raise ValueError("模型似乎不是基于注意力的模型")

        # 设置参数
        self.layer_index = kwargs.get('layer_index', None)
        self.head_index = kwargs.get('head_index', None)
        self.tokenizer = kwargs.get('tokenizer', None)
        self.special_tokens = kwargs.get('special_tokens', ['[CLS]', '[SEP]', '[PAD]'])

        logger.info(f"注意力解释器初始化完成: layer_index={self.layer_index}, head_index={self.head_index}")

    def _is_attention_model(self) -> bool:
        """检查模型是否包含注意力机制"""
        # 检查PyTorch模型
        if isinstance(self.model, torch.nn.Module):
            for module in self.model.modules():
                if isinstance(module, torch.nn.MultiheadAttention):
                    return True
        return False

    def explain(self,
                input_text: str,
                target: Optional[Any] = None,
                **kwargs) -> ExplanationResult:
        """
        解释文本样本的注意力权重

        参数:
        input_text: 输入文本
        target: 目标类别 (分类任务) 或目标token (生成任务)
        kwargs:
          - layer_index: 覆盖初始化的层索引
          - head_index: 覆盖初始化的注意力头索引
          - include_special: 是否包含特殊token
          - aggregation: 聚合方法 ('mean', 'max', 'min', 'sum')
        """
        # 获取参数
        layer_index = kwargs.get('layer_index', self.layer_index)
        head_index = kwargs.get('head_index', self.head_index)
        include_special = kwargs.get('include_special', False)
        aggregation = kwargs.get('aggregation', 'mean')

        # 分词
        tokens = self._tokenize(input_text)

        # 获取注意力权重
        attn_weights = self._get_attention_weights(input_text, layer_index, head_index)

        # 处理注意力权重
        aggregated_weights = self._aggregate_attention(attn_weights, aggregation)

        # 创建解释结果
        result = ExplanationResult(
            raw_result=attn_weights,
            metadata={
                'method': 'attention',
                'tokens': tokens,
                'layer_index': layer_index,
                'head_index': head_index,
                'aggregation': aggregation
            }
        )

        # 设置特征重要性
        result.feature_importance = self._get_feature_importance(tokens, aggregated_weights, include_special)

        # 添加可视化
        result.visualization = self._generate_visualization(tokens, aggregated_weights, include_special)

        return result

    def _tokenize(self, text: str) -> List[str]:
        """分词文本"""
        if self.tokenizer:
            tokens = self.tokenizer.tokenize(text)
            # 移除特殊token
            tokens = [token for token in tokens if token not in self.special_tokens]
            return tokens

        # 简单分词
        tokens = re.findall(r'\w+|[^\w\s]', text)
        tokens = [token for token in tokens if token.strip() != '']
        return tokens

    def _get_attention_weights(self,
                               text: str,
                               layer_index: Optional[int],
                               head_index: Optional[int]) -> np.ndarray:
        """获取模型的注意力权重"""
        # 准备输入
        if self.tokenizer:
            inputs = self.tokenizer(text, return_tensors="pt")
        else:
            # 简单编码
            tokens = self._tokenize(text)
            inputs = {"input_ids": torch.tensor([[0] * len(tokens)])}

        # 前向传播获取注意力权重
        if isinstance(self.model, torch.nn.Module):
            # PyTorch模型
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
            attentions = outputs.attentions
        else:
            # 其他框架模型 (占位符)
            # 在实际应用中需要根据具体框架实现
            num_layers = 12
            num_heads = 12
            seq_len = len(inputs["input_ids"][0])
            attentions = [np.random.rand(num_heads, seq_len, seq_len) for _ in range(num_layers)]

        # 转换为NumPy数组
        attn_weights = [attn.detach().numpy() for attn in attentions] if isinstance(attentions, list) else attentions

        # 选择特定层
        if layer_index is not None:
            attn_weights = [attn_weights[layer_index]]

        # 选择特定头
        if head_index is not None:
            for i in range(len(attn_weights)):
                attn_weights[i] = attn_weights[i][head_index:head_index + 1]

        return attn_weights

    def _aggregate_attention(self,
                             attn_weights: List[np.ndarray],
                             method: str) -> np.ndarray:
        """聚合注意力权重"""
        if not attn_weights:
            return np.array([])

        # 合并所有层
        all_heads = np.concatenate(attn_weights, axis=0)

        # 聚合方法
        if method == 'mean':
            aggregated = np.mean(all_heads, axis=0)
        elif method == 'max':
            aggregated = np.max(all_heads, axis=0)
        elif method == 'min':
            aggregated = np.min(all_heads, axis=0)
        elif method == 'sum':
            aggregated = np.sum(all_heads, axis=0)
        else:
            raise ValueError(f"不支持的聚合方法: {method}")

        # 平均接收到的注意力
        aggregated = np.mean(aggregated, axis=0)

        return aggregated

    def _get_feature_importance(self,
                                tokens: List[str],
                                weights: np.ndarray,
                                include_special: bool) -> Dict[str, float]:
        """创建特征重要性字典"""
        # 过滤特殊token
        if not include_special:
            tokens = [token for token in tokens if token not in self.special_tokens]
            weights = weights[:len(tokens)]

        # 创建字典
        return {token: float(weight) for token, weight in zip(tokens, weights)}

    def _generate_visualization(self,
                                tokens: List[str],
                                weights: np.ndarray,
                                include_special: bool) -> Dict[str, Any]:
        """生成可视化结果"""
        # 过滤特殊token
        if not include_special:
            tokens = [token for token in tokens if token not in self.special_tokens]
            weights = weights[:len(tokens)]

        # 创建热力图
        fig, ax = plt.subplots(figsize=(12, 3))
        im = ax.imshow([weights], cmap='viridis', aspect='auto')

        # 设置坐标轴
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticks([])

        # 添加颜色条
        plt.colorbar(im, ax=ax)

        # 创建HTML高亮文本
        html_text = self._create_highlighted_html(tokens, weights)

        # 创建可视化字典
        visualization = {
            'heatmap_figure': fig,
            'tokens': tokens,
            'weights': weights.tolist(),
            'highlighted_text': html_text,
            'type': 'attention'
        }

        return visualization

    def _create_highlighted_html(self, tokens: List[str], weights: np.ndarray) -> str:
        """创建带颜色高亮的HTML文本"""
        # 归一化权重
        min_w, max_w = np.min(weights), np.max(weights)
        norm_weights = (weights - min_w) / (max_w - min_w) if max_w > min_w else weights

        # 创建颜色映射 (蓝色到红色)
        cmap = LinearSegmentedColormap.from_list("attn_cmap", ["#1f77b4", "#ff7f0e"])

        html_output = '<div style="font-family: monospace; font-size: 14px; line-height: 1.5;">'

        for token, weight in zip(tokens, norm_weights):
            # 获取颜色
            r, g, b, _ = cmap(weight)
            color = f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"

            # 添加高亮
            html_output += f'<span style="background-color: {color};">{token}</span> '

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
    # 构造一个假输入和假模型
    class DummyAttention(torch.nn.Module):
        def __init__(self, num_layers=2, num_heads=2, seq_len=6):
            super().__init__()
            self.attn = torch.nn.MultiheadAttention(embed_dim=16, num_heads=2, batch_first=True)
            self.num_layers = num_layers
            self.num_heads = num_heads
            self.seq_len = seq_len
        def forward(self, input_ids=None, output_attentions=True, **kwargs):
            class Output:
                def __init__(self, num_layers, num_heads, seq_len):
                    self.attentions = [torch.rand(num_heads, seq_len, seq_len) for _ in range(num_layers)]
            return Output(self.num_layers, self.num_heads, self.seq_len)
        def predict(self, *args, **kwargs):
            return np.array([0])
    # 假分词器
    class DummyTokenizer:
        def tokenize(self, text):
            return text.split()
        def __call__(self, text, return_tensors=None):
            # 简单编码
            tokens = text.split()
            return {"input_ids": torch.tensor([[i for i in range(len(tokens))]])}

    # 初始化假模型和分词器
    model = DummyAttention(num_layers=2, num_heads=2, seq_len=6)
    tokenizer = DummyTokenizer()

    explainer = AttentionExplainer(
        model=model,
        task_type='classification',
        tokenizer=tokenizer,
        layer_index=0,  # 可指定层
        head_index=0    # 可指定头
    )

    # 输入文本
    text = "The quick brown fox jumps over"
    result = explainer.explain(text, aggregation='mean', include_special=False)

    print("Tokens:", result.metadata['tokens'])
    print("特征重要性:", result.feature_importance)
    print("HTML高亮：\n", result.visualization['highlighted_text'])

    # 可视化热力图
    fig = result.visualization['heatmap_figure']
    fig.suptitle("Attention Heatmap")
    
    plt.savefig("attention_explainer.png")
    print("热力图保存至attention_explainer.png")

if __name__ == "__main__":
    main()