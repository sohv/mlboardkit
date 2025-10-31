# ML-Scripts

Minimal setup and installation instructions. Detailed usage has moved to `usage.md`.

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
from mlboardkit.analysis_tools.metrics_utils import compute_classification_metrics

compute_classification_metrics([1,0,1],[1,0,0])
```

CLI via python -m:
```bash
python -m mlboardkit.data_utils.dataset_processor quality-check dataset.csv --report report.json
python -m mlboardkit.data_utils.data_converter convert input.json output.csv --format csv
python -m mlboardkit.analysis_tools.plot_metrics training_log.json --plot-type training --output curves.png
python -m mlboardkit.model_utils.train_model --model-name bert-base-uncased --train-file train.jsonl --epochs 3
```

Python requirement: 3.9+

Full usage and CLI examples are in `usage.md`.

