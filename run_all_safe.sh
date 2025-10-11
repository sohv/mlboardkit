#!/usr/bin/env zsh
# run_all_safe.sh — safely run top-level Python scripts with optional non-interactive mode
# Usage:
#   ./run_all_safe.sh         # prompts before each script
#   ./run_all_safe.sh --yes   # run all without prompting (use with care)

set -euo pipefail

YES=false
if [[ "${1-}" == "--yes" || "${1-}" == "-y" ]]; then
  YES=true
fi

LOGFILE="run_all.log"
: > "$LOGFILE"

echo "Run-all started: $(date)" | tee -a "$LOGFILE"

for f in ./*.py; do
  # skip this helper and README or non-files
  [[ "$f" == "./run_all_safe.sh" ]] && continue
  [[ ! -f "$f" ]] && continue

  echo "---" | tee -a "$LOGFILE"
  echo "Script: $f" | tee -a "$LOGFILE"

  if [[ "$YES" == "true" ]]; then
    echo "Running (non-interactive): $f" | tee -a "$LOGFILE"
    python3 "$f" >> "$LOGFILE" 2>&1 || echo "FAILED: $f" | tee -a "$LOGFILE"
  else
    echo "About to run: $f"
    read -q "REPLY?Run $f? (y/N): "
    echo
    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
      echo "Running: $f" | tee -a "$LOGFILE"
      python3 "$f" >> "$LOGFILE" 2>&1 || echo "FAILED: $f" | tee -a "$LOGFILE"
    else
      echo "Skipping: $f" | tee -a "$LOGFILE"
    fi
  fi

done

echo "Run-all finished: $(date)" | tee -a "$LOGFILE"
