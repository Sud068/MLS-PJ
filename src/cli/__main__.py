"""
CLI入口点
提供命令行接口的主程序
"""

import sys
import logging
from .command_parser import CommandParser
from .help_generator import HelpGenerator
from src.io.config_parser import ConfigParser
from src.utils.logging import setup_logger
from src.utils.file_utils import ensure_dir_exists
from src.core import ModelLoader, BaseExplainer


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
        elif args.command == 'compare':
            _handle_compare_command(args, config, logger)
        elif args.command == 'serve':
            _handle_serve_command(args, config, logger)
        else:
            HelpGenerator.print_main_help()

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


def _handle_explain_command(args, config, logger):
    """处理解释命令"""
    from src.explainers import get_explainer
    from src.io.data_loader import DataLoader
    from src.io.result_writer import ResultWriter

    # 加载模型
    model = ModelLoader.load(args.model_path, **config.get('model', {}))
    logger.info(f"Loaded model from {args.model_path}")

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

    # 保存结果
    output_path = args.output or f"explanation_{args.method}.json"
    ResultWriter.write(results, output_path)
    logger.info(f"Saved explanation to {output_path}")


def _handle_evaluate_command(args, config, logger):
    """处理评估命令"""
    from src.evaluation import FidelityEvaluator, StabilityEvaluator, SensitivityEvaluator
    from src.io.data_loader import DataLoader
    from src.io.result_writer import ResultWriter

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
    from src.explainers import get_explainer
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
    from src.visualization import PlotGenerator, HeatmapGenerator, ReportGenerator
    from src.io.data_loader import DataLoader
    from src.io.result_writer import ResultWriter

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


def _handle_compare_command(args, config, logger):
    """处理比较命令"""
    from src.io.data_loader import DataLoader
    from src.io.result_writer import ResultWriter
    from src.core.explainer import ExplanationComparator

    # 加载解释结果
    explanations = []
    for path in args.inputs:
        exp = DataLoader.load(path)
        explanations.append(exp)
        logger.info(f"Loaded explanation from {path}")

    # 创建比较器
    comparator = ExplanationComparator()
    for i, exp in enumerate(explanations):
        comparator.add_method(f"method_{i + 1}", exp)

    # 执行比较
    comparison = comparator.compare(
        metrics=args.metrics,
        visualization=args.visualize
    )

    # 保存结果
    output_path = args.output or "comparison_report.pdf"
    ResultWriter.write(comparison, output_path)
    logger.info(f"Saved comparison report to {output_path}")


def _handle_serve_command(args, config, logger):
    """处理服务命令"""
    from src.api.server import start_server

    logger.info(f"Starting XAI API server on port {args.port}")
    start_server(
        host=args.host,
        port=args.port,
        config=config
    )


if __name__ == "__main__":
    main()