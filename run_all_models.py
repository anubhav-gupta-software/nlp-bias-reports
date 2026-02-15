#!/usr/bin/env python3
"""
Run baseline_run.py for each (model, dimension) pair.
One dimension per run; outputs saved as {model_slug}_{dimension}_results.csv in outputs/.
Usage: python run_all_models.py
"""

import subprocess
import sys
from pathlib import Path

# Model name (ollama) -> slug for filename (e.g. qwen2.5:3b -> qwen2_5_3b)
def model_to_slug(model: str) -> str:
    return model.replace(":", "_").replace(".", "_")

MODELS = [
    "qwen2.5:3b",
    "llama3.2:3b",
    "mistral:7b",
]
DIMENSIONS = [
    "age",
    "disability",
    "veteran",
    "parental",
    "mental_health",
    "language",
]
OUTPUT_DIR = "outputs"
INPUT_CSV = "power_scenarios.csv"


def main():
    base = Path(__file__).parent
    baseline = base / "baseline_run.py"
    if not baseline.exists():
        print(f"ERROR: {baseline} not found.")
        sys.exit(1)

    configs = []
    for model in MODELS:
        slug = model_to_slug(model)
        for dimension in DIMENSIONS:
            output = f"{OUTPUT_DIR}/{slug}_{dimension}_results.csv"
            configs.append({"model": model, "dimension": dimension, "output": output})

    for i, cfg in enumerate(configs, 1):
        model = cfg["model"]
        dimension = cfg["dimension"]
        output = cfg["output"]
        print("\n" + "=" * 60)
        print(f"[{i}/{len(configs)}] {model} / {dimension} -> {output}")
        print("=" * 60)
        cmd = [
            sys.executable,
            str(baseline),
            "--model", model,
            "--dimension", dimension,
            "--output", output,
            "--input", INPUT_CSV,
        ]
        r = subprocess.run(cmd, cwd=base)
        if r.returncode != 0:
            print(f"ERROR: {model} / {dimension} failed with exit code {r.returncode}")
            sys.exit(r.returncode)
        print(f"Done: {output}\n")

    print(f"All {len(configs)} runs finished successfully.")


if __name__ == "__main__":
    main()
