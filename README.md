# MLBoardKit

<div align="center">
  <img src="/Users/sohan/Documents/GitHub/scripts/logo.svg" width="120" alt="MLBoardKit Logo" />
</div>

A Python library that provides utilities for streamlined data processing, model training, and analysis tasks in machine learning workflows.

**mlboardkit** offers easy CLI commands and Python interfaces for dataset quality checks, format conversion, metric computation, plot generation, and model training — with support for popular frameworks and minimal setup.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/sohv/mlboardkit?style=social)](https://github.com/sohv/mlboardkit/stargazers)
[![PyPI version](https://img.shields.io/pypi/v/mlboardkit.svg)](https://pypi.org/project/mlboardkit/)



## Install

```bash
# from source (editable)
pip install -e .

# from PyPI (published)
pip install mlboardkit
```

## Quick start

```python
# After installing mlboardkit, import via the mlboardkit namespace
from mlboardkit.data_utils.dataset_processor import main as dataset_processor_main
from mlboardkit.analysis_tools.metrics_utils import classification_report

report = classification_report([1,0,1], [1,0,0])
```

CLI via python -m:
```bash
python -m mlboardkit.data_utils.dataset_processor quality-check dataset.csv --report report.json
python -m mlboardkit.data_utils.data_converter convert input.json output.csv --format csv
python -m mlboardkit.analysis_tools.plot_metrics training_log.json --plot-type training --output curves.png
python -m mlboardkit.model_utils.train_model --model-name bert-base-uncased --train-file train.jsonl --epochs 3
```

Python requirement: 3.9+

Full usage and CLI examples are in `usage.md`. Here is a [demo notebook](https://colab.research.google.com/drive/1Z7ltGDY89NFUT3Vyzl71nWls2y0DsjLb?usp=sharing) that demonstrates the usage of this library in a ML project.

