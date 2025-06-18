"""
结果写入器
支持多种格式的结果输出
"""

import os
import json
import csv
import pickle

import numpy as np
import pandas as pd
from PIL import Image
import logging
from typing import Any, Dict, List, Optional, Union
from xai_io.data_loader import DataLoader

logger = logging.getLogger(__name__)

import os
import numpy as np
from PIL import Image

import os
from PIL import Image
import numpy as np

import os
from PIL import Image
import numpy as np

def save_visualization_images(visualization: dict, output_dir: str, keys_to_save=None, prefix=""):
    """
    将visualization中指定key的ndarray保存为图片
    :param visualization: dict
    :param output_dir: 保存目录
    :param keys_to_save: 只保存这些key，为None表示全部ndarray都保存
    :param prefix: 文件名前缀
    :return: {key: 路径}
    """
    os.makedirs(output_dir, exist_ok=True)
    img_paths = {}
    for k, v in visualization.items():
        if (keys_to_save is None or k in keys_to_save) and isinstance(v, np.ndarray):
            arr = v
            # 如果不是uint8，自动归一化并转
            if arr.dtype != np.uint8:
                arr_min, arr_max = arr.min(), arr.max()
                if arr_max > arr_min:
                    arr = (arr - arr_min) / (arr_max - arr_min) * 255
                else:
                    arr = np.zeros_like(arr)
                arr = arr.astype(np.uint8)
            fname = f"{prefix}{k}.png"
            path = os.path.join(output_dir, fname)
            Image.fromarray(arr).save(path)
            img_paths[k] = path
    return img_paths

class ResultWriter:
    """
    结果写入器

    将解释结果保存到不同格式的文件
    """
    import os
    import numpy as np
    from PIL import Image
    
    @staticmethod
    def write(result: Any, output_path: str, format: Optional[str] = None, **kwargs):
        """
        写入结果到文件

        参数:
        result: 要保存的结果
        output_path: 输出文件路径
        format: 输出格式 (自动检测)
        kwargs: 格式特定参数
        """
        # 创建输出目录

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # 自动检测格式
        if format is None:
            format = ResultWriter._detect_format(output_path)
        
        # 调用特定写入方法
        writer = getattr(ResultWriter, f"_write_{format}", None)
        if writer is None:
            # 使用DataLoader的保存方法作为后备
            DataLoader.save(result, output_path, **kwargs)
        else:
            writer(result, output_path, **kwargs)

    @staticmethod
    def _detect_format(file_path: str) -> str:
        """根据文件扩展名检测格式"""
        ext = os.path.splitext(file_path)[1].lower()

        if ext in ['.json']:
            return 'json'
        elif ext in ['.csv']:
            return 'csv'
        elif ext in ['.html']:
            return 'html'
        elif ext in ['.txt']:
            return 'text'
        elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']:
            return 'image'
        elif ext in ['.pdf']:
            return 'pdf'
        elif ext in ['.pkl', '.pickle']:
            return 'pickle'
        else:
            return 'auto'  # 使用自动检测

    @staticmethod
    def _write_json(result: Union[dict, list], output_path: str, **kwargs):
        """写入JSON文件，自动处理不可序列化对象"""
        indent = kwargs.get('indent', 4)

        # 设置默认序列化方式为 str（若用户未手动指定）
        if 'default' not in kwargs:
            kwargs['default'] = str

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=indent, ensure_ascii=False, **kwargs)

    @staticmethod
    def _write_csv(result: Union[list, pd.DataFrame, np.ndarray], output_path: str, **kwargs):
        """写入CSV文件"""
        if isinstance(result, pd.DataFrame):
            result.to_csv(output_path, index=False, **kwargs)
        elif isinstance(result, np.ndarray):
            pd.DataFrame(result).to_csv(output_path, index=False, **kwargs)
        elif isinstance(result, list) and all(isinstance(item, dict) for item in result):
            # 字典列表
            keys = result[0].keys()
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(result)
        elif isinstance(result, list):
            # 简单列表
            with open(output_path, 'w') as f:
                for item in result:
                    f.write(f"{item}\n")
        else:
            raise ValueError(f"不支持的结果类型: {type(result)}")

    @staticmethod
    def _write_html(result: Union[str, dict], output_path: str, **kwargs):
        """写入HTML文件"""
        if isinstance(result, dict) and 'html' in result:
            content = result['html']
        elif isinstance(result, str):
            content = result
        else:
            raise ValueError("结果必须包含HTML内容或为字符串")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

    @staticmethod
    def _write_text(result: Union[str, list], output_path: str, **kwargs):
        """写入文本文件"""
        if isinstance(result, list):
            content = "\n".join(str(item) for item in result)
        else:
            content = str(result)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

    @staticmethod
    def _write_image(result: Union[np.ndarray, dict], output_path: str, **kwargs):
        """写入图像文件"""
        # if isinstance(result, dict) and 'image' in result:
        #     img_data = result['image']
        # elif isinstance(result, np.ndarray):
        #     img_data = result
        # else:
        #     raise ValueError("结果必须包含图像数据或为NumPy数组")

        # img = Image.fromarray(img_data)
        # img.save(output_path, **kwargs)
        # 列出所有 key
        print(list(result.visualization.keys()))
        # 比如: ['heatmap', 'superimposed', 'original_image', ...]
        # 假设你只想保存 heatmap 和 superimposed
        img_paths = save_visualization_images(result.visualization, "output_imgs", keys_to_save=['colored_heatmap','heatmap', 'superimposed'])

        # {'heatmap': 'output_imgs/heatmap.png', 'superimposed': 'output_imgs/superimposed.png'}

    @staticmethod
    def _write_pdf(result: Any, output_path: str, **kwargs):
        """写入PDF文件 (需要报告生成功能)"""
        # 实际实现需要额外的PDF生成库
        # 这里作为占位符
        with open(output_path, 'wb') as f:
            f.write(b"PDF content would be here")
        logger.warning("PDF写入功能需要额外实现")

    @staticmethod
    def _write_pickle(result: Any, output_path: str, **kwargs):
        """写入Pickle文件"""
        with open(output_path, 'wb') as f:
            pickle.dump(result, f, **kwargs)

    @staticmethod
    def write_explanation(explanation: dict, output_path: str, **kwargs):
        """
        写入解释结果

        参数:
        explanation: 解释结果字典
        output_path: 输出文件路径
        kwargs: 格式特定参数
        """
        # 根据结果类型选择最佳格式
        format = ResultWriter._detect_format(output_path)

        # 如果格式是自动的，根据内容决定
        if format == 'auto':
            if 'html' in explanation:
                format = 'html'
            elif 'image' in explanation:
                format = 'image'
            else:
                format = 'json'

        # 写入结果
        ResultWriter.write(explanation, output_path, format=format, **kwargs)

    @staticmethod
    def batch_write(results: List[Any], output_paths: List[str], **kwargs):
        """
        批量写入多个结果

        参数:
        results: 结果列表
        output_paths: 输出文件路径列表
        kwargs: 传递给写入函数的参数
        """
        if len(results) != len(output_paths):
            raise ValueError("结果列表和输出路径列表长度必须相同")

        for result, path in zip(results, output_paths):
            try:
                ResultWriter.write(result, path, **kwargs)
            except Exception as e:
                logger.error(f"写入文件 {path} 失败: {str(e)}")