import importlib
import pkgutil


def test_namespace_root_import():
    mod = importlib.import_module("mlboardkit")
    assert mod is not None


def test_subpackages_aliases_exist():
    mlbk = importlib.import_module("mlboardkit")
    expected = {
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
    }
    for name in expected:
        # Accessible as attribute and importable as module
        assert hasattr(mlbk, name), f"mlboardkit.{name} missing"
        importlib.import_module(f"mlboardkit.{name}")


def test_modules_discoverable_in_each_subpackage():
    # Ensure at least one module is present in each subpackage
    for sub in [
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
    ]:
        pkg = importlib.import_module(f"mlboardkit.{sub}")
        modules = [m.name for m in pkgutil.iter_modules(pkg.__path__)]
        assert len(modules) >= 1, f"no modules found in mlboardkit.{sub}"
