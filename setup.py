from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="xai-toolkit-cli",
    version="0.8.0",
    description="可解释人工智能(XAI)命令行工具包",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/xai-toolkit-cli",
    author="Your Name",
    author_email="your.email@example.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="xai, explainable ai, machine learning, interpretability",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8, <4",
    install_requires=[
        "numpy>=1.22.0",
        "scipy>=1.8.0",
        "pandas>=1.4.0",
        "scikit-learn>=1.0.0",
        "shap>=0.41.0",
        "lime>=0.2.0.1",
        "dice-ml>=0.8",
        "captum>=0.5.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.2",
        "Pillow>=9.0.0",
        "plotly>=5.8.0",
        "PyYAML>=6.0",
        "click>=8.0.0",
        "rich>=12.0.0",
        "tqdm>=4.62.0",
        "Jinja2>=3.0.0",
        "weasyprint>=54.0"
    ],
    extras_require={
        "full": ["torch>=1.12.0", "tensorflow>=2.8.0", "keras>=2.8.0"],
        "dev": ["pytest>=7.0.0", "coverage>=6.0.0", "sphinx>=4.0.0"],
        "test": ["pytest>=7.0.0"],
    },
    entry_points={
        "console_scripts": [
            "xai-cli=src.cli.__main__:main",
        ],
    },
    package_data={
        "src": [
            "templates/*.html",
            "templates/*.css",
            "configs/*.yaml"
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/your-username/xai-toolkit-cli/issues",
        "Source": "https://github.com/your-username/xai-toolkit-cli",
        "Documentation": "https://your-username.github.io/xai-toolkit-cli/",
    },
)