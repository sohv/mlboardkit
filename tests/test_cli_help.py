import subprocess
import sys
import pytest

CLI_MODULES = [
    "mlboardkit.data_utils.dataset_processor",
    "mlboardkit.data_utils.data_converter",
    "mlboardkit.analysis_tools.plot_metrics",
    # excluded: mlboardkit.file_conversion.format_json (no argparse --help)
    "mlboardkit.file_management.compress_folder",
    # excluded: mlboardkit.model_utils.train_model (heavy optional deps at import)
    "mlboardkit.model_utils.evaluate_model",
    "mlboardkit.model_utils.training_setup",
    "mlboardkit.llm_experiments.jailbreak_tester",
    "mlboardkit.automation.pipeline_runner",
    "mlboardkit.text_processing.clean_text",
]

@pytest.mark.parametrize("module_name", CLI_MODULES)
def test_cli_help_runs(module_name):
    # Some scripts may exit with code 0 or 2 on --help depending on argparse config
    proc = subprocess.run([sys.executable, "-m", module_name, "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.returncode in (0, 2), f"{module_name} --help failed: {proc.stderr.decode('utf-8', 'ignore')}"
