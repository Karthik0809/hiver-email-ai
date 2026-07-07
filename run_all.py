"""
Single entry point that runs the whole pipeline end-to-end:
  1. (re)build the dataset
  2. generate suggested replies for the held-out incoming emails
  3. score them with the evaluation rubric
  4. validate that the rubric discriminates good/mediocre/bad replies
  5. check judge self-consistency

Uses whatever LLM_PROVIDER is set in the environment (default: ollama).
See README.md for setup instructions.

Run: python3 run_all.py
"""

import os
import subprocess
import sys

# Use the same interpreter that launched this script (sys.executable) rather
# than a hardcoded "python3"/"python" name, so the pipeline runs the same way
# on Windows (where the launcher is usually "python") and macOS/Linux.
PY = sys.executable or "python3"

STEPS = [
    ("Building dataset", [PY, "dataset/build_dataset.py"]),
    ("Generating suggested replies", [PY, "generator/generate.py"]),
    ("Scoring replies (evaluation)", [PY, "evaluation/evaluate.py"]),
    ("Validating the metric (good/mediocre/bad)", [PY, "evaluation/validate_metric.py"]),
    ("Checking judge self-consistency", [PY, "evaluation/consistency_check.py"]),
]


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    provider = os.environ.get("LLM_PROVIDER", "ollama")
    print(f"Running full pipeline with LLM_PROVIDER={provider}\n{'=' * 60}")

    for label, cmd in STEPS:
        print(f"\n>>> {label}\n{'-' * 60}")
        result = subprocess.run(cmd, cwd=root)
        if result.returncode != 0:
            print(f"\nStep failed: {label} (exit code {result.returncode}). Stopping.")
            sys.exit(result.returncode)

    print(f"\n{'=' * 60}\nDone. See evaluation/scores.json, evaluation/validation_report.json, "
          f"and evaluation/consistency_report.json for results.")


if __name__ == "__main__":
    main()
