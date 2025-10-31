"""
mlboardkit namespace: import subpackages as mlboardkit.<subpackage>

This module aliases existing top-level packages (e.g., data_utils) under the
mlboardkit namespace so users can write `import mlboardkit.data_utils ...`.
"""

from importlib import import_module
import sys

_SUBPACKAGES = [
    "analysis_tools",
    "automation",
    "data_analysis",
    "data_utils",
    "file_conversion",
    "file_management",
    "llm_experiments",
    "misc",
    "model_utils",
    "safety_utils",
    "text_processing",
    "text_sql_tools",
    "visualization",
]

for _name in _SUBPACKAGES:
    try:
        _mod = import_module(_name)
        setattr(sys.modules[__name__], _name, _mod)
        sys.modules[f"{__name__}.{_name}"] = _mod
    except Exception:
        # Subpackage may be missing in some distributions; skip aliasing.
        pass


