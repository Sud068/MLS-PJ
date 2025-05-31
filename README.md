# XAI Toolkit CLI - 可解释人工智能命令行工具包

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/your-username/xai-toolkit-cli/actions/workflows/tests.yml/badge.svg)](https://github.com/your-username/xai-toolkit-cli/actions)
[![Documentation Status](https://readthedocs.org/projects/xai-toolkit-cli/badge/?version=latest)](https://xai-toolkit-cli.readthedocs.io/)

XAI Toolkit CLI 是一个开箱即用的命令行工具，用于解释和分析机器学习模型的行为。它集成了多种先进的解释算法，支持表格数据、图像和文本模型，并提供专业的解释评估和可视化功能。

## 主要特性

- **多模态支持**：表格数据、图像分类、文本分析
- **多种解释方法**：SHAP、LIME、Grad-CAM、反事实解释等
- **专业评估**：保真度、稳定性、敏感性量化分析
- **丰富可视化**：热力图、特征重要性图、PDF报告
- **配置驱动**：YAML配置文件简化复杂工作流
- **可扩展架构**：轻松集成新的解释算法和模型类型

## 安装指南

### 前置要求
- Python 3.8+
- pip 20.0+

### 推荐安装方法 (使用虚拟环境)
```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/MacOS
.venv\Scripts\activate    # Windows

# 安装XAI Toolkit CLI
pip install git+https://github.com/your-username/xai-toolkit-cli.git

# 或者从本地安装
git clone https://github.com/your-username/xai-toolkit-cli.git
cd xai-toolkit-cli
pip install -e .