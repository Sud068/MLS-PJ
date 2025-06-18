"""
PDF报告生成器
创建包含可视化结果的PDF报告
"""

import numpy as np
from PIL.Image import Image
from fpdf import FPDF
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from typing import Union
import logging
import base64
from io import BytesIO
from utils.validation import validate_not_none, validate_type

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    PDF报告生成器

    创建包含可视化结果的PDF报告
    """

    def __init__(self,
                 title: str = "XAI Report",
                 author: str = "XAI Toolkit",
                 font: str = "Arial",
                 font_size: int = 12):
        """
        初始化报告生成器

        参数:
        title: 报告标题
        author: 报告作者
        font: 默认字体
        font_size: 默认字体大小
        """
        self.title = title
        self.author = author
        self.font = font
        self.font_size = font_size
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.pdf.set_title(title)
        self.pdf.set_author(author)

    def add_title_page(self, subtitle: Optional[str] = None):
        """
        添加标题页

        参数:
        subtitle: 报告副标题
        """
        self.pdf.add_page()
        self.pdf.set_font(self.font, 'B', 24)
        self.pdf.cell(0, 40, self.title, 0, 1, 'C')

        if subtitle:
            self.pdf.set_font(self.font, 'I', 18)
            self.pdf.cell(0, 20, subtitle, 0, 1, 'C')

        self.pdf.set_font(self.font, '', 14)
        self.pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
        self.pdf.cell(0, 10, f"Author: {self.author}", 0, 1, 'C')

        # 添加空行
        self.pdf.ln(20)

    def add_section_title(self, title: str, level: int = 1):
        """
        添加章节标题

        参数:
        title: 章节标题
        level: 标题级别 (1-3)
        """
        if level == 1:
            font_size = 18
            style = 'B'
        elif level == 2:
            font_size = 16
            style = 'B'
        else:
            font_size = 14
            style = 'B'

        self.pdf.set_font(self.font, style, font_size)
        self.pdf.cell(0, 10, title, 0, 1)
        self.pdf.ln(5)

    def add_text(self, text: str):
        """
        添加文本段落

        参数:
        text: 要添加的文本
        """
        self.pdf.set_font(self.font, '', self.font_size)
        self.pdf.multi_cell(0, 8, text)
        self.pdf.ln(5)

    def add_image(self,
                  image_data: Union[plt.Figure, np.ndarray, str],
                  caption: Optional[str] = None,
                  width: int = 180,
                  height: Optional[int] = None):
        """
        添加图像到报告

        参数:
        image_data: 图像数据 (matplotlib图表、NumPy数组或文件路径)
        caption: 图像标题
        width: 图像宽度 (mm)
        height: 图像高度 (mm, 自动计算比例)
        """
        validate_not_none(image_data, "image_data")

        # 获取图像文件路径或数据
        if isinstance(image_data, plt.Figure):
            img_path = self._save_figure(image_data)
        elif isinstance(image_data, np.ndarray):
            img_path = self._save_numpy_image(image_data)
        elif isinstance(image_data, str):
            img_path = image_data
        else:
            raise ValueError("不支持的图像数据类型")

        # 添加图像
        if height is None:
            # 保持宽高比
            self.pdf.image(img_path, x=10, y=None, w=width)
        else:
            self.pdf.image(img_path, x=10, y=None, w=width, h=height)

        # 添加标题
        if caption:
            self.pdf.set_font(self.font, 'I', self.font_size - 2)
            self.pdf.cell(0, 5, caption, 0, 1, 'C')

        self.pdf.ln(10)

    def add_table(self,
                  data: List[Dict[str, Any]],
                  headers: Optional[List[str]] = None,
                  col_widths: Optional[List[int]] = None,
                  title: Optional[str] = None):
        """
        添加表格到报告

        参数:
        data: 表格数据 (字典列表)
        headers: 表头列表 (如果为None，使用字典键)
        col_widths: 列宽列表 (mm)
        title: 表格标题
        """
        validate_not_none(data, "data")
        validate_type(data, list, "data")

        if title:
            self.add_section_title(title, level=3)

        if not data:
            return

        # 确定表头
        if headers is None:
            headers = list(data[0].keys())

        # 确定列宽
        if col_widths is None:
            col_widths = [40] * len(headers)  # 默认40mm

        # 设置表格样式
        self.pdf.set_font(self.font, 'B', self.font_size - 2)

        # 添加表头
        for i, header in enumerate(headers):
            self.pdf.cell(col_widths[i], 10, str(header), border=1)
        self.pdf.ln()

        # 添加数据行
        self.pdf.set_font(self.font, '', self.font_size - 2)
        for row in data:
            for i, key in enumerate(headers):
                value = str(row.get(key, ''))
                self.pdf.cell(col_widths[i], 10, value, border=1)
            self.pdf.ln()

        self.pdf.ln(10)

    def add_explanation(self,
                        explanation: Dict[str, Any],
                        section_title: str = "Explanation Details"):
        """
        添加解释结果到报告

        参数:
        explanation: 解释结果字典
        section_title: 章节标题
        """
        self.add_section_title(section_title, level=2)

        # 添加元数据
        if 'metadata' in explanation:
            self.add_text("Metadata:")
            for key, value in explanation['metadata'].items():
                self.add_text(f"- {key}: {value}")
            self.pdf.ln(5)

        # 添加特征重要性
        if 'feature_importance' in explanation:
            self.add_section_title("Feature Importance", level=3)
            self.add_table(
                [{'Feature': k, 'Importance': v} for k, v in explanation['feature_importance'].items()],
                headers=['Feature', 'Importance'],
                col_widths=[100, 80]
            )

        # 添加可视化
        if 'visualization' in explanation:
            self.add_section_title("Visualization", level=3)
            vis = explanation['visualization']

            if 'image' in vis:
                self.add_image(vis['image'], caption="Explanation Visualization")
            elif 'html' in vis:
                # HTML可视化无法直接添加到PDF，转换为图像
                # 实际应用中需要将HTML渲染为图像
                self.add_text("HTML visualization is available in the interactive report.")
            elif 'heatmap' in vis:
                self.add_image(vis['heatmap'], caption="Heatmap Visualization")

    def save(self, output_path: str):
        """
        保存PDF报告

        参数:
        output_path: 输出文件路径
        """
        self.pdf.output(output_path)
        logger.info(f"PDF report saved to: {output_path}")

    def _save_figure(self, fig: plt.Figure) -> str:
        """保存matplotlib图表为临时文件"""
        temp_path = "temp_plot.png"
        fig.savefig(temp_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return temp_path

    def _save_numpy_image(self, image: np.ndarray) -> str:
        """保存NumPy图像为临时文件"""
        temp_path = "temp_image.png"
        img = Image.fromarray(image.astype(np.uint8))
        img.save(temp_path)
        return temp_path

    def generate_report(self,
                        content: List[Dict[str, Any]],
                        output_path: str):
        """
        生成完整报告

        参数:
        content: 报告内容列表 [{'type': 'title', 'text': '...'}, ...]
        output_path: 输出文件路径
        """
        self.add_title_page()

        for item in content:
            item_type = item.get('type')

            if item_type == 'title':
                self.add_section_title(item['text'], level=item.get('level', 1))
            elif item_type == 'text':
                self.add_text(item['text'])
            elif item_type == 'image':
                self.add_image(
                    image_data=item['data'],
                    caption=item.get('caption'),
                    width=item.get('width', 180),
                    height=item.get('height')
                )
            elif item_type == 'table':
                self.add_table(
                    data=item['data'],
                    headers=item.get('headers'),
                    col_widths=item.get('col_widths'),
                    title=item.get('title')
                )
            elif item_type == 'explanation':
                self.add_explanation(
                    explanation=item['data'],
                    section_title=item.get('title', 'Explanation')
                )

        self.save(output_path)

    @staticmethod
    def fig_to_base64(fig: plt.Figure, format: str = 'png') -> str:
        """
        将图表转换为Base64编码的字符串

        参数:
        fig: matplotlib图表对象
        format: 图像格式 (png, jpg)

        返回:
        Base64编码的图像字符串
        """
        buffer = BytesIO()
        fig.savefig(buffer, format=format, bbox_inches='tight')
        plt.close(fig)
        buffer.seek(0)

        base64_str = base64.b64encode(buffer.read()).decode('utf-8')
        return f"data:image/{format};base64,{base64_str}"

    @staticmethod
    def image_to_base64(image: np.ndarray, format: str = 'png') -> str:
        """
        将图像数组转换为Base64编码的字符串

        参数:
        image: 图像数组
        format: 图像格式 (png, jpg)

        返回:
        Base64编码的图像字符串
        """
        img = Image.fromarray(image.astype(np.uint8))
        buffer = BytesIO()
        img.save(buffer, format=format)
        buffer.seek(0)

        base64_str = base64.b64encode(buffer.read()).decode('utf-8')
        return f"data:image/{format};base64,{base64_str}"