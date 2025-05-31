# XAI Toolkit CLI 命令参考手册

## 概述
XAI Toolkit CLI 是一个用于解释机器学习模型行为的命令行工具。它支持多种解释算法、评估指标和可视化输出。

## 全局选项
| 选项 | 缩写 | 描述 | 默认值 |
|------|------|------|--------|
| `--config` | `-c` | 指定配置文件路径 | `configs/default.yaml` |
| `--log-level` | `-l` | 设置日志级别 (debug, info, warning, error) | `info` |
| `--output-dir` | `-o` | 指定输出目录 | `results/` |
| `--help` | `-h` | 显示帮助信息 | - |

## 主要命令

### `explain`
生成模型预测的解释

**选项:**
- `--method`：指定解释方法（如 shap, lime, grad_cam）
- `--sample-idx`：指定要解释的样本索引
- `--batch-size`：批量解释的样本数

**示例:**
```bash
# 使用默认配置生成解释
xai-cli explain

# 使用SHAP方法解释特定样本
xai-cli explain --method shap --sample-idx 42

# 使用自定义配置文件
xai-cli explain -c configs/financial.yaml