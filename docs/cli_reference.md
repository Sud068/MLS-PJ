以下是根据你的 `CommandParser` 代码整理出的 `cli_reference.md` 文件内容：

---

# `xai-cli` 命令行参考文档

```
命令解析器  
解析命令行参数和选项  
```

---

## 基本信息

* **程序名**: `xai-cli`
* **描述**: eXplainable AI Toolkit 命令行工具
* **版本**: `1.0.0`

---

## 全局选项（Global Options）

| 选项                | 说明                                                    |
| ----------------- | ----------------------------------------------------- |
| `--config`        | 指定配置文件路径                                              |
| `--log-level`     | 日志等级，选项：`debug`, `info`, `warning`, `error`，默认：`info` |
| `--log-file`      | 指定日志文件路径                                              |
| `--output-dir`    | 指定输出目录                                                |
| `-h`, `--help`    | 显示帮助信息并退出                                             |
| `-v`, `--version` | 显示当前版本信息并退出                                           |

---

## 子命令（Commands）

### 1. `explain` - 生成模型解释

**参数：**

| 参数                  | 说明                                                                                                                                                                           |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model_path`        | 模型文件路径（必填）                                                                                                                                                                   |
| `data_path`         | 输入数据路径（必填）                                                                                                                                                                   |
| `-m`, `--method`    | 指定解释方法（必填），可选值：<br> `shap_text`, `shap_tabular`, `lime_image`, `lime_text`, `lime_tabular`, `grad_cam_image`, `integrated_gradients_image`, `dice_tabular`, `attention_text` |
| `-t`, `--task-type` | 任务类型：`classification`（默认）或 `regression`                                                                                                                                      |
| `--target`          | 目标分类（分类任务中可选）                                                                                                                                                                |
| `--batch`           | 启用批处理模式                                                                                                                                                                      |
| `--feature-names`   | 特征名称列表                                                                                                                                                                       |
| `-o`, `--output`    | 解释结果输出路径                                                                                                                                                                     |

---

### 2. `evaluate` - 评估解释质量

**参数：**

| 参数                  | 说明                                                       |
| ------------------- | -------------------------------------------------------- |
| `explanation_path`  | 解释文件路径（必填）                                               |
| `model_path`        | 模型文件路径（必填）                                               |
| `data_path`         | 输入数据路径（必填）                                               |
| `-m`, `--metrics`   | 要计算的评估指标，默认：`fidelity`, `stability`，可选值还包括 `sensitivity` |
| `-t`, `--task-type` | 任务类型：`classification`（默认）或 `regression`                  |
| `-o`, `--output`    | 输出评估结果路径                                                 |

---

### 3. `visualize` - 生成解释可视化结果

**参数：**

| 参数                 | 说明                                                       |
| ------------------ | -------------------------------------------------------- |
| `explanation_path` | 解释文件路径（必填）                                               |
| `-t`, `--type`     | 可视化类型（必填），选项包括：`feature_importance`, `heatmap`, `report` |
| `--title`          | 可视化图表标题                                                  |
| `-o`, `--output`   | 输出图像路径                                                   |



如需查看更多帮助信息，可使用：

```bash
xai-cli --help
xai-cli <command> --help
```

---

如需我以 markdown 文件形式导出，也可以告诉我。
