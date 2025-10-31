## Subtools included in the ml-scripts package

The package installs the following subpackages and tools:

- analysis_tools: compare_runs, experiment_tracker, feature_importance, metrics_utils, plot_metrics
- automation: pipeline_runner, scheduler_job, task_scheduler, workflow_engine
- data_analysis: data_profiler
- data_utils: csv_to_json, data_converter, data_label_inspector, data_sampler, dataset_processor, deduplicate, remove_duplicates, split_data
- file_conversion: format_json, json_to_csv, jsonl_splitter, merge_json_files, txt_to_jsonl
- file_management: compress_folder, project_organizer, smart_organizer
- llm_experiments: adversarial_prompting, alignment_metrics, jailbreak_tester, prompt_eval, response_consistency
- misc: cli_parser, config_manager, db_utilities, env_report, git_ml_workflow, logger_utils, timer
- model_utils: embedding_utils, evaluate_model, hp_search, model_card_generator, model_compression, model_convert, model_validator, reproducibility, seed_everything, train_model, training_setup
- safety_utils: red_teaming_tools, toxicity_filter
- text_processing: augment_text, clean_text, text_analyzer, text_utilities
- text_sql_tools: nl_to_sql, sql_to_nl
- visualization: chart_generator, viz_utils

## How to use

Install:
```bash
pip install -e .
# or after publish
pip install ml-scripts
```

Use from Python:
```python
from data_utils.dataset_processor import main as dataset_cli
from analysis_tools.metrics_utils import compute_classification_metrics

compute_classification_metrics([1,0,1],[1,0,0])
```

Use as CLI modules:
```bash
python -m data_utils.dataset_processor quality-check dataset.csv --report report.json
python -m data_utils.data_converter convert input.json output.csv --format csv
python -m analysis_tools.plot_metrics training_log.json --plot-type training --output curves.png
python -m file_conversion.format_json data.json
python -m file_management.compress_folder my_project --format zip
python -m model_utils.train_model --model-name bert-base-uncased --train-file train.jsonl --epochs 3
python -m model_utils.evaluate_model predictions.json --task classification
python -m model_utils.training_setup check --gpu-only
python -m llm_experiments.jailbreak_tester --categories violence illegal_activities --simulate
python -m automation.pipeline_runner --config pipeline.json --output results.json
python -m text_processing.clean_text input.txt --stats
python -m text_sql_tools.nl_to_sql --query "top 10 users by spend"
python -m visualization.chart_generator --input data.json --chart bar --output chart.png
```

Discover flags:
```bash
python -m data_utils.dataset_processor --help
```

## Available Scripts

### 1. Configuration Management (`config_manager.py`)
Manage ML training configurations with YAML/JSON support, validation, and templating.

**Features:**
- Pre-built templates for common ML tasks (finetuning, classification, generation)
- Environment variable overrides
- Configuration validation and merging
- Nested parameter support

**Usage Examples:**
```bash
# Create a configuration from template
python3 config_manager.py create --template finetuning --output config.yaml

# Validate existing configuration
python3 config_manager.py validate config.yaml

# Merge base and override configurations
python3 config_manager.py merge base.yaml override.yaml --output final.yaml

# Set specific configuration values
python3 config_manager.py set config.yaml training.learning_rate 0.001

# Get configuration values
python3 config_manager.py get config.yaml model.name

# List available templates
python3 config_manager.py list-templates
```

**Available Templates:**
- `finetuning`: Complete setup for model fine-tuning
- `classification`: Text/image classification tasks
- `generation`: Text generation models

### 2. Advanced CLI Parser (`cli_parser.py`)
Sophisticated command-line argument parsing with dataclass integration and nested configuration support.

**Usage Examples:**
```bash
# Use with configuration file
python3 cli_parser.py --config config.yaml --model.name bert-base-uncased

# Override specific parameters
python3 cli_parser.py train --learning-rate 0.001 --batch-size 32 --num-epochs 5

# Print final configuration
python3 cli_parser.py --config config.yaml --print-config

# Save merged configuration
python3 cli_parser.py --config base.yaml --save-config final.yaml --learning-rate 0.001

# Use config overrides
python3 cli_parser.py --config config.yaml --config-overrides training.optimizer=adam model.max_length=1024
```

### 3. Data Conversion Utilities (`data_converter.py`)
Convert between data formats with validation, cleaning, and quality checks.

**Usage Examples:**
```bash
# Convert between formats
python3 data_converter.py convert input.json output.csv --format csv
python3 data_converter.py convert data.csv data.parquet

# Validate data against schema
python3 data_converter.py validate data.csv --schema schema.json --quality-check

# Clean data with multiple options
python3 data_converter.py clean messy_data.csv clean_data.csv \
    --remove-duplicates \
    --handle-missing fill \
    --standardize-text text_column description \
    --remove-outliers price rating

# Split dataset for ML training
python3 data_converter.py split dataset.csv \
    --train 0.7 --val 0.15 --test 0.15 \
    --stratify label_column \
    --output-dir ./splits

# Get dataset information
python3 data_converter.py info dataset.csv --detailed
```

**Schema Example (`schema.json`):**
```json
{
  "required_columns": ["text", "label"],
  "column_types": {
    "text": "string",
    "label": "integer",
    "score": "float"
  },
  "constraints": {
    "score": {
      "min_value": 0,
      "max_value": 10
    },
    "label": {
      "allowed_values": [0, 1, 2]
    }
  }
}
```

### 4. Training Environment Setup (`training_setup.py`)
Comprehensive environment validation and setup for ML training.

**Usage Examples:**
```bash
# Check system capabilities
python3 training_setup.py check --verbose
python3 training_setup.py check --gpu-only --framework pytorch

# Setup training environment
python3 training_setup.py setup --framework pytorch \
    --optimize-env \
    --create-template \
    --log-dir ./logs

# Benchmark system performance
python3 training_setup.py benchmark --quick
python3 training_setup.py benchmark --cpu --gpu --memory --duration 10

# Install framework dependencies
python3 training_setup.py install --framework pytorch --gpu
python3 training_setup.py install --requirements requirements.txt
```

### 5. Dataset Processing (`dataset_processor.py`)
Advanced dataset manipulation for ML workflows.

**Usage Examples:**
```bash
# Sample datasets
python3 dataset_processor.py sample large_dataset.csv \
    --size 0.1 \
    --method stratified \
    --stratify-column label

# Balance imbalanced datasets
python3 dataset_processor.py balance dataset.csv label_column \
    --method oversample \
    --output balanced_dataset.csv

# Analyze dataset quality
python3 dataset_processor.py quality-check dataset.csv \
    --report quality_report.json

# Augment text data
python3 dataset_processor.py augment text_dataset.csv text_column \
    --techniques synonym insertion swap \
    --multiplier 3

# Remove duplicates
python3 dataset_processor.py deduplicate dataset.csv \
    --columns text_column \
    --keep first
```

**Quality Report Example:**
```json
{
  "quality_score": {
    "overall": 85.2,
    "duplicate_score": 95.0,
    "missing_score": 82.1,
    "consistency_score": 78.5
  },
  "dataset_info": {
    "shape": [10000, 5],
    "memory_usage_mb": 2.4
  },
  "duplicates": {
    "duplicate_rows": 45,
    "duplicate_percentage": 0.45
  }
}
```

### 6. JSON Formatter (`format_json.py`)
Clean and format JSON files with control character handling.

**Usage Examples:**
```bash
# Format JSON file
python3 format_json.py data.json

# Format multiple files
for file in *.json; do
    python3 format_json.py "$file"
done
```

### 7. Complete ML Project Setup
```bash
# 1. Check system capabilities
python3 training_setup.py check --verbose

# 2. Create project configuration
python3 config_manager.py create --template finetuning --output project_config.yaml

# 3. Prepare your dataset
python3 data_converter.py convert raw_data.json clean_data.csv
python3 dataset_processor.py quality-check clean_data.csv --report quality.json
python3 data_converter.py split clean_data.csv --train 0.8 --val 0.1 --test 0.1

# 4. Setup training environment
python3 training_setup.py setup --framework pytorch --optimize-env --create-template

# 5. Run training with advanced CLI
python3 cli_parser.py train --config project_config.yaml --output-dir ./outputs
```

### 8. Data Science Workflow
```bash
# Explore and clean data
python3 data_converter.py info dataset.csv --detailed
python3 dataset_processor.py quality-check dataset.csv --report report.json
python3 data_converter.py clean dataset.csv cleaned.csv --remove-duplicates --handle-missing fill

# Sample for prototyping
python3 dataset_processor.py sample cleaned.csv --size 1000 --method stratified --stratify-column target

# Balance if needed
python3 dataset_processor.py balance cleaned.csv target --method smote
```

### 9. NLP Project Pipeline
```bash
# Prepare text data
python3 data_converter.py convert text_data.jsonl text_data.csv
python3 dataset_processor.py augment text_data.csv text_column --techniques synonym insertion --multiplier 2

# Setup NLP environment
python3 training_setup.py install --framework transformers
python3 config_manager.py create --template classification --output nlp_config.yaml

# Configure for NLP
python3 config_manager.py set nlp_config.yaml model.name distilbert-base-uncased
python3 config_manager.py set nlp_config.yaml training.learning_rate 2e-5
```

## Utilities — How to run

The repository includes a few lightweight utility scripts useful during experiments. Below are prerequisites and simple run examples for each utility.

Prerequisites:

- Python 3.8+ recommended
- Install basic libs:

```bash
python3 -m pip install --user numpy pandas matplotlib requests
# Optional: torch, setuptools
python3 -m pip install --user torch setuptools
```

Usage examples (no expected output shown):

- Metrics utilities

```bash
python3 metrics_utils.py classification --y_true '[1,0,1,1]' --y_pred '[1,0,0,1]'
python3 metrics_utils.py regression --y_true '[1.2,2.3]' --y_pred '[1.0,2.5]'
```

- Visualization (use from a Python script)

```python
from viz_utils import plot_confusion_matrix
cm = [[50, 5], [4, 41]]
plot_confusion_matrix(cm, labels=['neg','pos'], outfile='cm.png')
```

- Reproducibility helpers

```python
from reproducibility import set_seed, capture_experiment_metadata
set_seed(42)
capture_experiment_metadata('metadata.json', extras={'notes': 'test run'})
```

- Logger setup

```python
from logger_utils import setup_logger
logger = setup_logger('exp', 'logs/exp.log')
logger.info('Starting experiment')
```

- Environment report

```bash
python3 env_report.py --output env_report.json
```

## New AI/ML Project Scripts — How to run

Additional utility scripts for AI/ML workflows:

**Data Utils:**

- Text augmentation:
```bash
python3 augment_text.py --input texts.json --output augmented.json --techniques synonym insertion --multiplier 3
```

- File format conversion:
```bash
python3 csv_to_json.py data.csv data.json --format records
python3 txt_to_jsonl.py text_file.txt output.jsonl --metadata
```

- JSONL processing:
```bash
python3 jsonl_splitter.py large_file.jsonl chunks/ --chunk-size 1000
python3 merge_json_files.py "data/*.json" merged.jsonl --format jsonl
```

**Model Utils:**

- Model card generation:
```bash
python3 model_card_generator.py --config model_config.json --metrics results.json --output MODEL_CARD.md
```

**LLM Experiments:**

- Prompt evaluation:
```bash
python3 prompt_eval.py --prompts prompts.json --responses responses.json --output evaluation.md
```

**Analysis Tools:**

- Compare experiment runs:
```bash
python3 compare_runs.py --pattern "results/*.json" --output comparison.md --metric accuracy
```

**Automation:**

- Pipeline runner:
```bash
python3 pipeline_runner.py --config pipeline.json --output results.json
# Create sample config:
python3 pipeline_runner.py --create-sample
```

### 27. Comprehensive Job Scheduler (`scheduler_job.py`)
Advanced job scheduler with cron-like scheduling, dependency management, and resource monitoring.

**Usage Examples:**
```bash
# Run scheduler with configuration file
python3 scheduler_job.py --config scheduler_config.json

# Create sample configuration
python3 scheduler_job.py --create-sample-config

# Check scheduler status
python3 scheduler_job.py --status

# Simple mode (backward compatibility)
python3 scheduler_job.py --cmd "python train_model.py" --interval 3600 --iterations 0

# Run single command
python3 scheduler_job.py --cmd "rsync -av /data/ /backup/" --iterations 1
```

**Sample Configuration:**
```json
{
  "jobs": [
    {
      "name": "data_backup",
      "command": "rsync -av /data/ /backup/",
      "schedule": "0 2 * * *",
      "max_retries": 2,
      "timeout": 7200
    },
    {
      "name": "model_training", 
      "command": "python train_model.py --config config.yaml",
      "schedule": "interval:3600",
      "depends_on": ["data_backup"],
      "resource_limits": {"max_memory_mb": 8192}
    }
  ]
}
```

### 28. Advanced Reproducibility Manager (`seed_everything.py`)
Comprehensive reproducibility management beyond basic seeding for ML experiments.

**Features:**
- Framework-specific seeding (PyTorch, TensorFlow, NumPy, scikit-learn)
- Environment configuration and reporting
- Reproducibility verification and testing
- Deterministic data splitting
- Complete environment snapshots

**Usage Examples:**
```bash
# Set global seeds and save reproducibility info
python3 seed_everything.py --seed 42 --save-info --output-dir ./reproducibility

# Enable deterministic mode (slower but fully reproducible)
python3 seed_everything.py --seed 42 --deterministic

# Generate reproducible data splits
python3 seed_everything.py --seed 42 --split-data 10000 --train-ratio 0.8 --val-ratio 0.1

# Load configuration and set seeds
python3 seed_everything.py --seed 42 --config repro_config.json
```

**Features:**
- Auto-generates setup scripts for environment reproduction
- Creates comprehensive environment reports
- Saves package requirements snapshots
- Provides reproducibility verification tools

### 29. High-Performance Timer & Profiler (`timer.py`)
Advanced timing and profiling utilities with context managers, decorators, and benchmarking.

**Usage Examples:**
```bash
# Run demonstration of timing features
python3 timer.py --demo

# Benchmark functions with profiling
python3 timer.py --benchmark benchmark_funcs.py --iterations 1000 --profile

# Save results to file
python3 timer.py --demo --output timing_results.json
```

**Code Integration:**
```python
from timer import Timer, TimingManager, time_it, benchmark_it

# Context manager usage
with Timer("data_loading") as timer:
    data = load_large_dataset()
print(f"Loading took {timer.duration} seconds")

# Decorator usage
@time_it(name="training_step")
def train_step(model, batch):
    return model(batch)

@benchmark_it(iterations=100)
def optimized_function():
    return expensive_computation()

# Advanced timing management
manager = TimingManager(auto_save=True)
with manager.time_block("preprocessing"):
    preprocess_data()
print(manager.get_summary_report())
```

### 30. Comprehensive Text Cleaner (`clean_text.py`)
Advanced text cleaning and preprocessing with configurable pipelines for NLP tasks.

**Features:**
- HTML/XML tag removal with BeautifulSoup integration
- Encoding issue fixes with ftfy library
- Unicode normalization and accent removal
- URL, email, phone number extraction
- Language filtering with langdetect
- Custom regex patterns and replacements

**Usage Examples:**
```bash
# Basic text cleaning
python3 clean_text.py input.txt --output cleaned.txt

# Advanced cleaning with all options
python3 clean_text.py input.txt --lowercase --remove-accents --language-filter en --min-length 10

# JSON file processing
python3 clean_text.py data.json --input-format json --output-format json --remove-duplicates

# Custom patterns and replacements
python3 clean_text.py input.txt --custom-patterns "\d{4}-\d{4}" --replacements '{"old_term": "new_term"}'

# Show cleaning statistics
python3 clean_text.py input.txt --stats

# Save/load configuration
python3 clean_text.py --save-config cleaning_config.json
python3 clean_text.py input.txt --config cleaning_config.json
```

**Configuration Options:**
- HTML/URL/email removal
- Unicode normalization and encoding fixes  
- Whitespace and punctuation normalization
- Length filtering and deduplication
- Language detection and filtering
- Custom regex patterns and replacements

## **Data Utils**

### 31. Deduplication (`deduplicate.py`)
Remove duplicate entries with various similarity metrics:
```bash
# Exact deduplication
python3 deduplicate.py data.txt --method exact --output clean_data.txt

# Jaccard similarity deduplication  
python3 deduplicate.py data.jsonl --method jaccard --threshold 0.8

# Normalize and deduplicate
python3 deduplicate.py data.json --method normalize --stats
```

### 32. Data Splitting (`split_data.py`)
Split datasets with various strategies:
```bash
# Random split
python3 split_data.py data.json --train-ratio 0.8 --val-ratio 0.1

# Stratified split
python3 split_data.py data.jsonl --method stratified --label-key category

# Temporal split
python3 split_data.py data.json --method temporal --time-key timestamp

# Group split (no data leakage)
python3 split_data.py data.json --method group --group-key user_id
```

## **Model Utils**

### 33. Model Training (`train_model.py`)
Generic PyTorch/Transformers training:
```bash
# Quick training
python3 train_model.py --model-name bert-base-uncased --train-file train.jsonl --epochs 3

# With configuration
python3 train_model.py --config train_config.json

# Create sample config
python3 train_model.py --create-config
```

### 34. Model Evaluation (`evaluate_model.py`)
Standardized evaluation metrics:
```bash
# Classification evaluation
python3 evaluate_model.py predictions.json --task classification

# Generation evaluation (BLEU, ROUGE)
python3 evaluate_model.py predictions.json --task generation --target-key reference

# Regression evaluation
python3 evaluate_model.py predictions.json --task regression --format json
```

### 35. Model Compression (`model_compression.py`)
Pruning, quantization, and distillation:
```bash
# Magnitude pruning
python3 model_compression.py bert-base-uncased --compression pruning --output-dir ./compressed --pruning-amount 0.3

# Dynamic quantization
python3 model_compression.py bert-base-uncased --compression quantization --output-dir ./quantized

# Create student model for distillation
python3 model_compression.py bert-base-uncased --compression distillation --student-reduction 0.5
```

## **LLM Experiments**

### 36. Jailbreak Testing (`jailbreak_tester.py`)
Test LLM robustness against adversarial prompts:
```bash
# Test with default categories
python3 jailbreak_tester.py --categories violence illegal_activities --simulate

# Test with custom prompts
python3 jailbreak_tester.py --custom-prompts my_prompts.txt --output results.json

# Generate report from existing results
python3 jailbreak_tester.py --report-only --output results.json
```

## **Analysis Tools**

### 37. Metrics Plotting (`plot_metrics.py`)
Plot losses, confusion matrices, embeddings:
```bash
# Training curves
python3 plot_metrics.py training_log.json --plot-type training --output curves.png

# Confusion matrix
python3 plot_metrics.py eval_results.json --plot-type confusion --true-key labels --pred-key predictions

# Embedding visualization
python3 plot_metrics.py embeddings.json --plot-type embeddings --reduction-method tsne

# Distribution plot
python3 plot_metrics.py scores.json --plot-type distribution --value-key accuracy_scores
```

## **File Conversion & Management**

### 38. Folder Compression (`compress_folder.py`)
Zip folders for sharing or backup:
```bash
# Create ZIP archive
python3 compress_folder.py my_project --format zip --exclude .git __pycache__

# Create TAR.GZ archive
python3 compress_folder.py my_data --format tar.gz --output backup.tar.gz
```

## **Text Processing**

### 39. Duplicate Removal (`remove_duplicates.py`)
Remove duplicate lines or JSON entries:
```bash
# Remove duplicate lines
python3 remove_duplicates.py data.txt --case-sensitive

# Remove duplicate JSON entries by key
python3 remove_duplicates.py data.jsonl --key-field id --stats

# Auto-detect format and remove duplicates
python3 remove_duplicates.py data.json --format auto --output clean.json
```

## Running scripts (single / all) — safe examples

This repository contains many small utility scripts. Below are recommended, zsh-friendly ways to run a single script and to safely run many scripts in a folder. These examples assume you're in the repository root (`/Users/sohan/Desktop/scripts`).

1) Use a virtual environment (recommended)

```bash
# create + activate a venv (zsh)
python3 -m venv .venv
source .venv/bin/activate
```

2) Run a single script (preferred for testing)

```bash
# Example: run the dataset processor
python3 dataset_processor.py --help
python3 dataset_processor.py sample data/my.csv --size 0.1
```

3) Dry-run: list what would be executed (no-op)

```bash
# Print the python commands for each .py file in a folder without executing
find . -maxdepth 1 -name "*.py" -print0 | xargs -0 -n1 -I{} echo python3 {}
```

4) Run all scripts in a folder (CAUTION: side-effects possible)

The following pattern runs each top-level `*.py` file using `python3` and prompts for confirmation before running to avoid accidental destructive operations. Use only when you know what each script does.

```bash
for f in ./*.py; do
  echo "---"
  echo "About to run: $f"
  read -q "REPLY?Run $f? (y/N): " && echo && [[ $REPLY =~ ^[Yy]$ ]] || { echo; echo "Skipping $f"; continue; }
  python3 "$f"
done
```

5) Run all scripts non-interactively (use with care!)

```bash
for f in ./*.py; do
  echo "Running $f" >> run_all.log
  python3 "$f" >> run_all.log 2>&1 || echo "FAILED: $f" >> run_all.log
done
```

Notes:
- Prefer running individual scripts during development.
- Add `--dry-run`, `--yes`, or `--output` flags to scripts where available to reduce risk.
- If a script requires specific dependencies, install them in the virtualenv before running.


### Custom Configuration Templates
Add your own templates to `config_manager.py`:

```python
CUSTOM_TEMPLATE = {
    "model": {
        "architecture": "custom_model",
        "layers": 12
    },
    "training": {
        "custom_param": "value"
    }
}

# add to ConfigManager.TEMPLATES
ConfigManager.TEMPLATES["custom"] = CUSTOM_TEMPLATE
```

### Custom CLI Arguments
Extend the CLI parser with custom dataclasses:

```python
@dataclass
class CustomConfig:
    custom_param: str = "default"
    advanced_setting: bool = False

parser = AdvancedArgumentParser(CustomConfig)
```

### Batch Processing Scripts
```bash
for dataset in data/*.csv; do
    echo "Processing $dataset"
    python3 dataset_processor.py quality-check "$dataset" --report "${dataset%.csv}_report.json"
    python3 data_converter.py clean "$dataset" "clean_$(basename $dataset)" --remove-duplicates
done
```