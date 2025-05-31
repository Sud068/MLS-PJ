"""
命令解析器
解析命令行参数和选项
"""

import argparse
from typing import List, Any


class CommandParser:
    """命令行参数解析器"""

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """创建主解析器"""
        parser = argparse.ArgumentParser(
            prog='xai-cli',
            description='eXplainable AI Toolkit Command Line Interface',
            add_help=False
        )

        # 全局选项
        global_group = parser.add_argument_group('Global Options')
        global_group.add_argument(
            '--config',
            help='Path to configuration file'
        )
        global_group.add_argument(
            '--log-level',
            choices=['debug', 'info', 'warning', 'error'],
            default='info',
            help='Logging level'
        )
        global_group.add_argument(
            '--log-file',
            help='Path to log file'
        )
        global_group.add_argument(
            '--output-dir',
            help='Output directory for results'
        )
        global_group.add_argument(
            '-h', '--help',
            action='store_true',
            help='Show this help message and exit'
        )

        # 子命令
        subparsers = parser.add_subparsers(
            title='Commands',
            dest='command',
            metavar='<command>'
        )

        # explain命令
        explain_parser = subparsers.add_parser(
            'explain',
            help='Generate model explanations'
        )
        self._add_explain_args(explain_parser)

        # evaluate命令
        evaluate_parser = subparsers.add_parser(
            'evaluate',
            help='Evaluate explanation quality'
        )
        self._add_evaluate_args(evaluate_parser)

        # visualize命令
        visualize_parser = subparsers.add_parser(
            'visualize',
            help='Generate visualizations from explanations'
        )
        self._add_visualize_args(visualize_parser)

        # compare命令
        compare_parser = subparsers.add_parser(
            'compare',
            help='Compare multiple explanations'
        )
        self._add_compare_args(compare_parser)

        # serve命令
        serve_parser = subparsers.add_parser(
            'serve',
            help='Start a local API server'
        )
        self._add_serve_args(serve_parser)

        return parser

    def _add_explain_args(self, parser: argparse.ArgumentParser):
        """添加explain命令参数"""
        parser.add_argument(
            'model_path',
            help='Path to the model file'
        )
        parser.add_argument(
            'data_path',
            help='Path to the input data'
        )
        parser.add_argument(
            '-m', '--method',
            required=True,
            choices=['shap', 'lime', 'grad_cam', 'integrated_gradients', 'dice'],
            help='Explanation method'
        )
        parser.add_argument(
            '-t', '--task-type',
            choices=['classification', 'regression'],
            default='classification',
            help='Type of machine learning task'
        )
        parser.add_argument(
            '--target',
            type=int,
            help='Target class for explanation (classification)'
        )
        parser.add_argument(
            '--batch',
            action='store_true',
            help='Process data in batch mode'
        )
        parser.add_argument(
            '--feature-names',
            nargs='+',
            help='List of feature names'
        )
        parser.add_argument(
            '-o', '--output',
            help='Output file path for explanation'
        )

    def _add_evaluate_args(self, parser: argparse.ArgumentParser):
        """添加evaluate命令参数"""
        parser.add_argument(
            'explanation_path',
            help='Path to the explanation file'
        )
        parser.add_argument(
            'model_path',
            help='Path to the model file'
        )
        parser.add_argument(
            'data_path',
            help='Path to the input data'
        )
        parser.add_argument(
            '-m', '--metrics',
            nargs='+',
            choices=['fidelity', 'stability', 'sensitivity'],
            default=['fidelity', 'stability'],
            help='Evaluation metrics to compute'
        )
        parser.add_argument(
            '-t', '--task-type',
            choices=['classification', 'regression'],
            default='classification',
            help='Type of machine learning task'
        )
        parser.add_argument(
            '-o', '--output',
            help='Output file path for evaluation results'
        )

    def _add_visualize_args(self, parser: argparse.ArgumentParser):
        """添加visualize命令参数"""
        parser.add_argument(
            'explanation_path',
            help='Path to the explanation file'
        )
        parser.add_argument(
            '-t', '--type',
            required=True,
            choices=['feature_importance', 'heatmap', 'report'],
            help='Type of visualization'
        )
        parser.add_argument(
            '--title',
            help='Title for the visualization'
        )
        parser.add_argument(
            '-o', '--output',
            help='Output file path for visualization'
        )

    def _add_compare_args(self, parser: argparse.ArgumentParser):
        """添加compare命令参数"""
        parser.add_argument(
            'inputs',
            nargs='+',
            help='Paths to explanation files to compare'
        )
        parser.add_argument(
            '-m', '--metrics',
            nargs='+',
            choices=['fidelity', 'stability', 'sensitivity'],
            default=['fidelity'],
            help='Metrics to compare'
        )
        parser.add_argument(
            '-v', '--visualize',
            action='store_true',
            help='Generate visual comparison'
        )
        parser.add_argument(
            '-o', '--output',
            help='Output file path for comparison report'
        )

    def _add_serve_args(self, parser: argparse.ArgumentParser):
        """添加serve命令参数"""
        parser.add_argument(
            '-H', '--host',
            default='127.0.0.1',
            help='Host to bind the server to'
        )
        parser.add_argument(
            '-p', '--port',
            type=int,
            default=8080,
            help='Port to run the server on'
        )

    def parse_args(self, args: List[str]) -> Any:
        """解析命令行参数"""
        parsed = self.parser.parse_args(args)

        # 处理帮助命令
        if parsed.command is None or parsed.help:
            from .help_generator import HelpGenerator
            if parsed.command:
                HelpGenerator.print_command_help(parsed.command)
            else:
                HelpGenerator.print_main_help()
            sys.exit(0)

        return parsed