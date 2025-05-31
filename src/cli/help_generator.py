"""
帮助系统生成器
生成命令行帮助信息
"""

import textwrap


class HelpGenerator:
    """命令行帮助信息生成器"""

    @staticmethod
    def print_main_help():
        """打印主帮助信息"""
        help_text = """
        eXplainable AI Toolkit (XAI-Toolkit)
        ===================================

        A comprehensive toolkit for explainable artificial intelligence.

        Usage:
          xai-cli [global-options] <command> [command-options]

        Global Options:
          --config <file>      Path to configuration file
          --log-level <level>  Set logging level (debug, info, warning, error)
          --log-file <file>    Path to log file
          --output-dir <dir>   Output directory for results
          -h, --help           Show this help message

        Commands:
          explain     Generate model explanations
          evaluate    Evaluate explanation quality
          visualize   Generate visualizations from explanations
          compare     Compare multiple explanations
          serve       Start a local API server

        Use 'xai-cli <command> --help' for command-specific help.
        """
        print(textwrap.dedent(help_text).strip())

    @staticmethod
    def print_command_help(command: str):
        """打印命令特定帮助信息"""
        help_texts = {
            'explain': """
            Generate Model Explanations
            ==========================

            Usage:
              xai-cli explain [options] <model_path> <data_path>

            Options:
              -m, --method <method>    Explanation method (shap, lime, grad_cam, 
                                       integrated_gradients, dice)
              -t, --task-type <type>   Task type (classification, regression) 
                                       [default: classification]
              --target <class>         Target class for explanation (classification)
              --batch                  Process data in batch mode
              --feature-names <names>  List of feature names
              -o, --output <file>      Output file path for explanation

            Examples:
              # Explain a single sample
              xai-cli explain model.pkl data.csv --method shap

              # Explain a batch of samples
              xai-cli explain model.pkl data.csv --method lime --batch
            """,

            'evaluate': """
            Evaluate Explanation Quality
            ===========================

            Usage:
              xai-cli evaluate [options] <explanation_path> <model_path> <data_path>

            Options:
              -m, --metrics <metrics>  Evaluation metrics (fidelity, stability, sensitivity)
                                       [default: fidelity,stability]
              -t, --task-type <type>   Task type (classification, regression) 
                                       [default: classification]
              -o, --output <file>      Output file path for evaluation results

            Examples:
              # Evaluate explanation fidelity and stability
              xai-cli evaluate explanation.json model.pkl data.csv
            """,

            'visualize': """
            Generate Visualizations
            =======================

            Usage:
              xai-cli visualize [options] <explanation_path>

            Options:
              -t, --type <type>    Visualization type (feature_importance, heatmap, report)
              --title <title>      Title for the visualization
              -o, --output <file>  Output file path for visualization

            Examples:
              # Generate a feature importance plot
              xai-cli visualize explanation.json --type feature_importance

              # Generate a comprehensive PDF report
              xai-cli visualize explanation.json --type report -o report.pdf
            """,

            'compare': """
            Compare Multiple Explanations
            ============================

            Usage:
              xai-cli compare [options] <explanation_path>...

            Options:
              -m, --metrics <metrics>  Metrics to compare (fidelity, stability, sensitivity)
                                       [default: fidelity]
              -v, --visualize          Generate visual comparison
              -o, --output <file>      Output file path for comparison report

            Examples:
              # Compare two explanations
              xai-cli compare expl1.json expl2.json

              # Compare with visualization
              xai-cli compare expl1.json expl2.json --visualize
            """,

            'serve': """
            Start Local API Server
            ======================

            Usage:
              xai-cli serve [options]

            Options:
              -H, --host <host>  Host to bind the server to [default: 127.0.0.1]
              -p, --port <port>  Port to run the server on [default: 8080]

            Examples:
              # Start server on default port
              xai-cli serve

              # Start server on custom port
              xai-cli serve --port 8000
            """
        }

        if command in help_texts:
            print(textwrap.dedent(help_texts[command]).strip())
        else:
            print(f"Unknown command: {command}")
            HelpGenerator.print_main_help()

    @staticmethod
    def generate_markdown_docs() -> str:
        """生成Markdown格式的文档"""
        docs = """
        # XAI Toolkit Command Line Interface

        ## Global Options

        | Option | Description |
        |--------|-------------|
        | `--config <file>` | Path to configuration file |
        | `--log-level <level>` | Set logging level (debug, info, warning, error) |
        | `--log-file <file>` | Path to log file |
        | `--output-dir <dir>` | Output directory for results |
        | `-h, --help` | Show help message |

        ## Commands

        ### `explain`
        Generate model explanations

        **Usage:**
        ```
        xai-cli explain [options] <model_path> <data_path>
        ```

        **Options:**
        | Option | Description |
        |--------|-------------|
        | `-m, --method <method>` | Explanation method (shap, lime, grad_cam, integrated_gradients, dice) |
        | `-t, --task-type <type>` | Task type (classification, regression) [default: classification] |
        | `--target <class>` | Target class for explanation (classification) |
        | `--batch` | Process data in batch mode |
        | `--feature-names <names>` | List of feature names |
        | `-o, --output <file>` | Output file path for explanation |

        ### `evaluate`
        Evaluate explanation quality

        **Usage:**
        ```
        xai-cli evaluate [options] <explanation_path> <model_path> <data_path>
        ```

        **Options:**
        | Option | Description |
        |--------|-------------|
        | `-m, --metrics <metrics>` | Evaluation metrics (fidelity, stability, sensitivity) [default: fidelity,stability] |
        | `-t, --task-type <type>` | Task type (classification, regression) [default: classification] |
        | `-o, --output <file>` | Output file path for evaluation results |

        ### `visualize`
        Generate visualizations from explanations

        **Usage:**
        ```
        xai-cli visualize [options] <explanation_path>
        ```

        **Options:**
        | Option | Description |
        |--------|-------------|
        | `-t, --type <type>` | Visualization type (feature_importance, heatmap, report) |
        | `--title <title>` | Title for the visualization |
        | `-o, --output <file>` | Output file path for visualization |

        ### `compare`
        Compare multiple explanations

        **Usage:**
        ```
        xai-cli compare [options] <explanation_path>...
        ```

        **Options:**
        | Option | Description |
        |--------|-------------|
        | `-m, --metrics <metrics>` | Metrics to compare (fidelity, stability, sensitivity) [default: fidelity] |
        | `-v, --visualize` | Generate visual comparison |
        | `-o, --output <file>` | Output file path for comparison report |

        ### `serve`
        Start a local API server

        **Usage:**
        ```
        xai-cli serve [options]
        ```

        **Options:**
        | Option | Description |
        |--------|-------------|
        | `-H, --host <host>` | Host to bind the server to [default: 127.0.0.1] |
        | `-p, --port <port>` | Port to run the server on [default: 8080] |
        """
        return textwrap.dedent(docs).strip()