# Usage Guide

This document consolidates detailed usage examples and script references from the repository.

- For installation and quick setup, see README.md

## Configuration Management (`config_manager.py`)
Features: templates, env overrides, validation/merging, nested params.

```bash
python3 config_manager.py create --template finetuning --output config.yaml
python3 config_manager.py validate config.yaml
python3 config_manager.py merge base.yaml override.yaml --output final.yaml
python3 config_manager.py set config.yaml training.learning_rate 0.001
python3 config_manager.py get config.yaml model.name
python3 config_manager.py list-templates
```

## Advanced CLI Parser (`cli_parser.py`)
```bash
python3 cli_parser.py --config config.yaml --model.name bert-base-uncased
python3 cli_parser.py train --learning-rate 0.001 --batch-size 32 --num-epochs 5
python3 cli_parser.py --config base.yaml --save-config final.yaml --learning-rate 0.001
python3 cli_parser.py --config config.yaml --config-overrides training.optimizer=adam model.max_length=1024
```

## Data Conversion Utilities (`data_converter.py`)
```bash
python3 data_converter.py convert input.json output.csv --format csv
python3 data_converter.py convert data.csv data.parquet
python3 data_converter.py validate data.csv --schema schema.json --quality-check
python3 data_converter.py clean messy_data.csv clean_data.csv --remove-duplicates --handle-missing fill --standardize-text text_column description --remove-outliers price rating
python3 data_converter.py split dataset.csv --train 0.7 --val 0.15 --test 0.15 --stratify label_column --output-dir ./splits
python3 data_converter.py info dataset.csv --detailed
```

## Training Environment (`training_setup.py`)
```bash
python3 training_setup.py check --verbose
python3 training_setup.py check --gpu-only --framework pytorch
python3 training_setup.py setup --framework pytorch --optimize-env --create-template --log-dir ./logs
python3 training_setup.py benchmark --cpu --gpu --memory --duration 10
python3 training_setup.py install --framework pytorch --gpu
python3 training_setup.py install --requirements requirements.txt
```

## Dataset Processing (`dataset_processor.py`)
```bash
python3 dataset_processor.py sample large_dataset.csv --size 0.1 --method stratified --stratify-column label
python3 dataset_processor.py balance dataset.csv label_column --method oversample --output balanced_dataset.csv
python3 dataset_processor.py quality-check dataset.csv --report quality_report.json
python3 dataset_processor.py augment text_dataset.csv text_column --techniques synonym insertion swap --multiplier 3
python3 dataset_processor.py deduplicate dataset.csv --columns text_column --keep first
```

## JSON Formatter (`format_json.py`)
```bash
python3 format_json.py data.json
for file in *.json; do python3 format_json.py "$file"; done
```

## Complete ML Project Setup
```bash
python3 training_setup.py check --verbose
python3 config_manager.py create --template finetuning --output project_config.yaml
python3 data_converter.py convert raw_data.json clean_data.csv
python3 dataset_processor.py quality-check clean_data.csv --report quality.json
python3 data_converter.py split clean_data.csv --train 0.8 --val 0.1 --test 0.1
python3 training_setup.py setup --framework pytorch --optimize-env --create-template
python3 cli_parser.py train --config project_config.yaml --output-dir ./outputs
```

## Data Science Workflow
```bash
python3 data_converter.py info dataset.csv --detailed
python3 dataset_processor.py quality-check dataset.csv --report report.json
python3 data_converter.py clean dataset.csv cleaned.csv --remove-duplicates --handle-missing fill
python3 dataset_processor.py sample cleaned.csv --size 1000 --method stratified --stratify-column target
python3 dataset_processor.py balance cleaned.csv target --method smote
```

## NLP Project Pipeline
```bash
python3 data_converter.py convert text_data.jsonl text_data.csv
python3 dataset_processor.py augment text_data.csv text_column --techniques synonym insertion --multiplier 2
python3 training_setup.py install --framework transformers
python3 config_manager.py create --template classification --output nlp_config.yaml
python3 config_manager.py set nlp_config.yaml model.name distilbert-base-uncased
python3 config_manager.py set nlp_config.yaml training.learning_rate 2e-5
```

## Utilities — How to run
```bash
python3 -m pip install --user numpy pandas matplotlib requests
# Optional: torch, setuptools
python3 -m pip install --user torch setuptools
```
Examples:
```bash
python3 metrics_utils.py classification --y_true '[1,0,1,1]' --y_pred '[1,0,0,1]'
python3 metrics_utils.py regression --y_true '[1.2,2.3]' --y_pred '[1.0,2.5]'
```
```python
from viz_utils import plot_confusion_matrix
cm = [[50, 5], [4, 41]]
plot_confusion_matrix(cm, labels=['neg','pos'], outfile='cm.png')
```
```python
from reproducibility import set_seed, capture_experiment_metadata
set_seed(42)
capture_experiment_metadata('metadata.json', extras={'notes': 'test run'})
```
```bash
python3 env_report.py --output env_report.json
```

## More Scripts
```bash
python3 augment_text.py --input texts.json --output augmented.json --techniques synonym insertion --multiplier 3
python3 csv_to_json.py data.csv data.json --format records
python3 txt_to_jsonl.py text_file.txt output.jsonl --metadata
python3 jsonl_splitter.py large_file.jsonl chunks/ --chunk-size 1000
python3 merge_json_files.py "data/*.json" merged.jsonl --format jsonl
python3 model_card_generator.py --config model_config.json --metrics results.json --output MODEL_CARD.md
python3 prompt_eval.py --prompts prompts.json --responses responses.json --output evaluation.md
python3 compare_runs.py --pattern "results/*.json" --output comparison.md --metric accuracy
python3 pipeline_runner.py --config pipeline.json --output results.json
python3 pipeline_runner.py --create-sample
```

## Scheduler (`scheduler_job.py`)
```bash
python3 scheduler_job.py --config scheduler_config.json
python3 scheduler_job.py --create-sample-config
python3 scheduler_job.py --status
python3 scheduler_job.py --cmd "python train_model.py" --interval 3600 --iterations 0
python3 scheduler_job.py --cmd "rsync -av /data/ /backup/" --iterations 1
```

## Reproducibility (`seed_everything.py`)
```bash
python3 seed_everything.py --seed 42 --save-info --output-dir ./reproducibility
python3 seed_everything.py --seed 42 --deterministic
python3 seed_everything.py --seed 42 --split-data 10000 --train-ratio 0.8 --val-ratio 0.1
python3 seed_everything.py --seed 42 --config repro_config.json
```

## Timer & Profiler (`timer.py`)
```bash
python3 timer.py --demo
python3 timer.py --benchmark benchmark_funcs.py --iterations 1000 --profile
python3 timer.py --demo --output timing_results.json
```
```python
from timer import Timer, TimingManager, time_it, benchmark_it
with Timer("data_loading") as timer:
    data = load_large_dataset()
print(f"Loading took {timer.duration} seconds")
@time_it(name="training_step")
def train_step(model, batch):
    return model(batch)
@benchmark_it(iterations=100)
def optimized_function():
    return expensive_computation()
manager = TimingManager(auto_save=True)
with manager.time_block("preprocessing"):
    preprocess_data()
print(manager.getSummaryReport())
```

## Text Cleaner (`clean_text.py`)
```bash
python3 clean_text.py input.txt --output cleaned.txt
python3 clean_text.py input.txt --lowercase --remove-accents --language-filter en --min-length 10
python3 clean_text.py data.json --input-format json --output-format json --remove-duplicates
python3 clean_text.py input.txt --custom-patterns "\d{4}-\d{4}" --replacements '{"old_term":"new_term"}'
python3 clean_text.py input.txt --stats
python3 clean_text.py --save-config cleaning_config.json
python3 clean_text.py input.txt --config cleaning_config.json
```

## Data/Model Utilities (additional)
```bash
python3 deduplicate.py data.txt --method exact --output clean_data.txt
python3 deduplicate.py data.jsonl --method jaccard --threshold 0.8
python3 deduplicate.py data.json --method normalize --stats
python3 split_data.py data.json --train-ratio 0.8 --val-ratio 0.1
python3 split_data.py data.jsonl --method stratified --label-key category
python3 split_data.py data.json --method temporal --time-key timestamp
python3 split_data.py data.json --method group --group-key user_id
python3 train_model.py --model-name bert-base-uncased --train-file train.jsonl --epochs 3
python3 evaluate_model.py predictions.json --task classification
python3 evaluate_model.py predictions.json --task generation --target-key reference
python3 evaluate_model.py predictions.json --task regression --format json
python3 model_compression.py bert-base-uncased --compression pruning --output-dir ./compressed --pruning-amount 0.3
python3 model_compression.py bert-base-uncased --compression quantization --output-dir ./quantized
python3 model_compression.py bert-base-uncased --compression distillation --student-reduction 0.5
python3 plot_metrics.py training_log.json --plot-type training --output curves.png
python3 plot_metrics.py eval_results.json --plot-type confusion --true-key labels --pred-key predictions
python3 plot_metrics.py embeddings.json --plot-type embeddings --reduction-method tsne
python3 plot_metrics.py scores.json --plot-type distribution --value-key accuracy_scores
python3 compress_folder.py my_project --format zip --exclude .git __pycache__
python3 compress_folder.py my_data --format tar.gz --output backup.tar.gz
```

## Running scripts — safe examples
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 dataset_processor.py --help
python3 dataset_processor.py sample data/my.csv --size 0.1
find . -maxdepth 1 -name "*.py" -print0 | xargs -0 -n1 -I{} echo python3 {}
for f in ./*.py; do echo "---"; echo "About to run: $f"; read -q "REPLY?Run $f? (y/N): " && echo && [[ $REPLY =~ ^[Yy]$ ]] || { echo; echo "Skipping $f"; continue; }; python3 "$f"; done
for f in ./*.py; do echo "Running $f" >> run_all.log; python3 "$f" >> run_all.log 2>&1 || echo "FAILED: $f" >> run_all.log; done
```

## Troubleshooting & Performance
```bash
pip install imbalanced-learn nltk pyarrow
python3 training_setup.py check --gpu-only
python3 dataset_processor.py sample large_dataset.csv --size 0.1
python3 config_manager.py validate config.yaml
```
Tips: use `--quick`, set `OMP_NUM_THREADS`, prefer Parquet, consider reservoir sampling.
