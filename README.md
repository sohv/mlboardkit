# ML-Scripts

Minimal setup and installation instructions. Detailed usage has moved to `usage.md`.

## Install

```bash
# from source (editable)
pip install -e .

# or after publishing to PyPI
pip install ml-scripts
```

## Quick start

```python
from data_utils.dataset_processor import main as dataset_processor_main
from analysis_tools.metrics_utils import compute_classification_metrics
```

Python requirement: 3.9+

Full usage and CLI examples are in `usage.md`.

