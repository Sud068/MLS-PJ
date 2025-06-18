"""
CLI入口点
提供命令行接口的主程序
"""

import os

from torchvision.transforms import transforms

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from core.explainer import display_explanation_result

import sys
import os
import logging
import types
import torch
import numpy as np
# 第三方/标准库：放最上面
# 自定义模块：按功能层次组织顺序整理

# CLI核心模块
from cli.command_parser import CommandParser
from cli.help_generator import HelpGenerator

# 配置与日志
from xai_io.config_parser import ConfigParser
from utils.logging import setup_logger
from utils.file_utils import ensure_dir_exists

# 核心模型与解释器
from core.model_loader import ModelLoader
from core.explainer import BaseExplainer
from explainers.factory import get_explainer
from explainers.image.grad_cam import GradCAMExplainer

# 数据与IO
from xai_io.data_loader import DataLoader
from xai_io.result_writer import ResultWriter

# 评估组件

from evaluation.fidelity import FidelityEvaluator
from evaluation.stability import StabilityEvaluator
from evaluation.sensitivity import SensitivityEvaluator

# 可视化组件
from visualization.heatmap_generator import HeatmapGenerator
from visualization.plot_generator import PlotGenerator
from visualization.report_generator import ReportGenerator



def main():
    """主入口函数"""
    try:
        # 创建命令解析器
        parser = CommandParser()
        args = parser.parse_args(sys.argv[1:])

        # 设置日志
        log_level = getattr(logging, args.log_level.upper(), logging.INFO)
        logger = setup_logger(level=log_level, log_file=args.log_file)

        # 加载配置文件
        config = {}
        if args.config:
            config_parser = ConfigParser(args.config)
            config = config_parser.to_dict()
            logger.info(f"Loaded configuration from {args.config}")

        # 确保输出目录存在
        if args.output_dir:
            ensure_dir_exists(args.output_dir)

        # 执行命令
        if args.command == 'explain':
            _handle_explain_command(args, config, logger)
        elif args.command == 'evaluate':
            _handle_evaluate_command(args, config, logger)
        elif args.command == 'visualize':
            _handle_visualize_command(args, config, logger)
        else:
            HelpGenerator.print_main_help()

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def _handle_explain_command(args, config, logger):
    """处理解释命令"""
    # 加载模型
    model = ModelLoader.load(args.model_path, **config.get('model', {}))
    logger.info(f"Loaded model from {args.model_path}")

    if(args.method == 'lime_image'):
        def predict(self, images):
            # images: numpy array, shape (N, H, W, 3), [0,1]
            transform = transforms.Compose([
                transforms.ToTensor(),  # (3, H, W), [0,1]
            ])
            batch = []
            for img in images:
                img_t = transform(img.astype(np.float32)).unsqueeze(0)  # (1,3,H,W)
                batch.append(img_t)
            batch = torch.cat(batch, dim=0)
            with torch.no_grad():
                logits = model(batch)
                probs = torch.nn.functional.softmax(logits, dim=1)
            return probs.cpu().numpy()  # (N, 1000)

        model.predict = types.MethodType(predict, model)
    else:
        def predict(self, x):
            with torch.no_grad():
                return self(x)

        model.predict = types.MethodType(predict, model)
    model.predict = types.MethodType(predict, model)

    logger.info("为PyTorch模型自动添加了predict方法。")

    # 加载数据
    data = DataLoader.load(args.data_path, **config.get('data', {}))
    logger.info(f"Loaded data from {args.data_path}")

    # 创建解释器
    explainer = get_explainer(
        model=model,
        method=args.method,
        task_type=args.task_type,
        feature_names=args.feature_names,
        **config.get('explainer', {})
    )
    logger.info(f"Created {args.method} explainer")

    # 生成解释
    if args.batch:
        results = explainer.batch_explain(data, targets=args.target)
    else:
        results = explainer.explain(data, target=args.target)
    if isinstance(results, list):
        for i, res in enumerate(results):
            print(f"\n=== 第{i + 1}个解释结果 ===")
            display_explanation_result(res)
    else:
        display_explanation_result(results)

    # 保存结果
    output_path = args.output or f"./explanation_{args.method}.json"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ResultWriter.write(results, output_path)
    output_path = "output/output_json"
    logger.info(f"Saved explanation to {output_path}")


def _handle_evaluate_command(args, config, logger):
    """处理评估命令"""
    # 加载解释结果
    explanation = DataLoader.load(args.explanation_path)
    logger.info(f"Loaded explanation from {args.explanation_path}")

    # 加载数据
    data = DataLoader.load(args.data_path, **config.get('data', {}))
    logger.info(f"Loaded data from {args.data_path}")

    # 加载模型
    model = ModelLoader.load(args.model_path, **config.get('model', {}))
    logger.info(f"Loaded model from {args.model_path}")

    # 加载解释器
    explainer = get_explainer(
        model=model,
        method=explanation.get('method', 'shap'),
        task_type=args.task_type,
        **config.get('explainer', {})
    )

    # 执行评估
    eval_results = {}
    if 'fidelity' in args.metrics:
        fidelity = FidelityEvaluator.evaluate_all(
            model, explainer, data, explanation
        )
        eval_results['fidelity'] = fidelity

    if 'stability' in args.metrics:
        stability = StabilityEvaluator.evaluate_all(
            explainer, data, explanation
        )
        eval_results['stability'] = stability

    if 'sensitivity' in args.metrics:
        sensitivity = SensitivityEvaluator.evaluate_all(
            explainer, data, explanation
        )
        eval_results['sensitivity'] = sensitivity

    # 保存结果
    output_path = args.output or f"evaluation_{args.metrics}.json"
    ResultWriter.write(eval_results, output_path)
    logger.info(f"Saved evaluation results to {output_path}")


def _handle_visualize_command(args, config, logger):
    """处理可视化命令"""
    # 加载解释结果
    explanation = DataLoader.load(args.explanation_path)
    logger.info(f"Loaded explanation from {args.explanation_path}")

    # 创建可视化
    if args.type == 'feature_importance':
        generator = PlotGenerator()
        fig = generator.feature_importance(
            explanation.get('feature_importance', {}),
            title=args.title
        )
        result = {'figure': fig}
    elif args.type == 'heatmap':
        generator = HeatmapGenerator()
        if 'image' in explanation and 'heatmap' in explanation:
            overlay = generator.overlay_heatmap(
                explanation['image'], explanation['heatmap']
            )
            result = {'image': overlay}
        else:
            fig = generator.attention_heatmap(
                explanation.get('tokens', []),
                explanation.get('attention_weights', [])
            )
            result = {'figure': fig}
    elif args.type == 'report':
        generator = ReportGenerator(
            title=args.title or "XAI Explanation Report"
        )
        generator.add_title_page()
        generator.add_explanation(explanation)
        result = generator
    else:
        raise ValueError(f"Unsupported visualization type: {args.type}")

    # 保存结果
    output_path = args.output or f"visualization_{args.type}.pdf"
    if isinstance(result, ReportGenerator):
        result.save(output_path)
    else:
        ResultWriter.write(result, output_path)
    logger.info(f"Saved visualization to {output_path}")




if __name__ == "__main__":
    main()
