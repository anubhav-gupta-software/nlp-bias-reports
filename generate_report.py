#!/usr/bin/env python3
"""
Analyze all outputs/*_results.csv and generate a JSON report.
Compares demographics (dimensions), models, and behavior (compliant / loophole / non-compliant).
Usage: python generate_report.py
Output: outputs/report.json
"""

import json
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path(__file__).parent / "outputs"
REPORT_PATH = OUTPUT_DIR / "report.json"
DIMENSIONS = ["age", "disability", "veteran", "parental", "mental_health", "language"]


def slug_to_model(slug: str) -> str:
    """Convert filename slug to model name (qwen2_5_3b -> qwen2.5:3b, mistral_7b -> mistral:7b)."""
    parts = slug.split("_")
    if len(parts) == 3:
        return f"{parts[0]}.{parts[1]}:{parts[2]}"
    if len(parts) == 2:
        return f"{parts[0]}:{parts[1]}"
    return slug


def parse_filename(name: str):
    """From e.g. qwen2_5_3b_mental_health_results.csv return (model_slug, dimension)."""
    stem = name.replace("_results.csv", "")
    for dim in DIMENSIONS:
        if stem.endswith("_" + dim):
            model_slug = stem[: -len(dim) - 1]
            return model_slug, dim
    return None, None


def identity_column(dimension: str) -> str:
    return dimension  # age -> age, mental_health -> mental_health


def pct(x: int, total: int) -> float:
    return round(100.0 * x / total, 2) if total else 0.0


def main():
    files = sorted(OUTPUT_DIR.glob("*_results.csv"))
    if not files:
        print(f"No *_results.csv found in {OUTPUT_DIR}")
        return

    report = {
        "summary": {
            "total_files": len(files),
            "dimensions": DIMENSIONS,
            "models": [],
        },
        "by_dimension": {},
        "by_model": {},
    }

    for dim in DIMENSIONS:
        report["by_dimension"][dim] = {"models": {}}

    for f in files:
        model_slug, dimension = parse_filename(f.name)
        if not dimension:
            continue
        model_name = slug_to_model(model_slug)
        if model_name not in report["summary"]["models"]:
            report["summary"]["models"].append(model_name)
        report["summary"]["models"].sort()

        df = pd.read_csv(f)
        df_valid = df.dropna(subset=["task"])
        n_total = len(df)
        n_valid = len(df_valid)
        id_col = identity_column(dimension)

        # Overall task distribution for this file
        task_counts = df_valid["task"].value_counts()
        overall = {
            "total_responses": n_total,
            "valid_responses": n_valid,
            "compliant": int(task_counts.get("compliant", 0)),
            "loophole": int(task_counts.get("loophole", 0)),
            "non_compliant": int(task_counts.get("non-compliant", 0)),
        }
        overall["compliant_pct"] = pct(overall["compliant"], n_valid)
        overall["loophole_pct"] = pct(overall["loophole"], n_valid)
        overall["non_compliant_pct"] = pct(overall["non_compliant"], n_valid)

        # By identity (e.g. young/old for age)
        identities = {}
        for val in df_valid[id_col].dropna().unique():
            sub = df_valid[df_valid[id_col] == val]
            n = len(sub)
            c = (sub["task"] == "compliant").sum()
            l = (sub["task"] == "loophole").sum()
            nc = (sub["task"] == "non-compliant").sum()
            identities[str(val)] = {
                "n": int(n),
                "compliant": int(c),
                "loophole": int(l),
                "non_compliant": int(nc),
                "compliant_pct": pct(c, n),
                "loophole_pct": pct(l, n),
                "non_compliant_pct": pct(nc, n),
            }
        # Gap (compliant % difference between the two identities)
        vals = list(identities.keys())
        if len(vals) == 2:
            p1 = identities[vals[0]]["compliant_pct"]
            p2 = identities[vals[1]]["compliant_pct"]
            overall["compliant_gap_pp"] = round(abs(p1 - p2), 2)
        else:
            overall["compliant_gap_pp"] = None

        report["by_dimension"][dimension]["models"][model_name] = {
            "overall": overall,
            "by_identity": identities,
        }

        # By model (aggregate per model across dimensions)
        if model_name not in report["by_model"]:
            report["by_model"][model_name] = {"dimensions": {}}
        report["by_model"][model_name]["dimensions"][dimension] = {
            "overall": overall,
            "by_identity": identities,
        }

    # Validity summary across all files
    all_valid = 0
    all_total = 0
    for f in files:
        df = pd.read_csv(f)
        all_total += len(df)
        all_valid += df["task"].notna().sum()
    report["summary"]["total_responses"] = all_total
    report["summary"]["valid_responses"] = int(all_valid)
    report["summary"]["valid_pct"] = round(100.0 * all_valid / all_total, 2) if all_total else 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as out:
        json.dump(report, out, indent=2)

    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
