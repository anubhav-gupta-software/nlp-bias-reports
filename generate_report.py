#!/usr/bin/env python3
"""
Analyze all outputs/*_results.csv and generate a JSON report + HTML dashboard.
Compares demographics (dimensions), models, and behavior (compliant / loophole / non-compliant).
Usage: python generate_report.py
Output: outputs/report.json, index.html (in project root; open in browser to view charts)
"""

import json
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path(__file__).parent / "outputs"
REPORT_JSON = OUTPUT_DIR / "report.json"
REPORT_HTML = Path(__file__).parent / "index.html"
DIMENSIONS = ["age", "disability", "veteran", "parental", "mental_health", "language"]
DIMENSION_LABELS = {
    "age": "Age",
    "disability": "Disability",
    "veteran": "Veteran status",
    "parental": "Parental status",
    "mental_health": "Mental health",
    "language": "Language",
}


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
    with open(REPORT_JSON, "w") as out:
        json.dump(report, out, indent=2)
    print(f"JSON report: {REPORT_JSON}")

    # Generate HTML dashboard with charts
    html = build_html(report)
    with open(REPORT_HTML, "w") as out:
        out.write(html)
    print(f"HTML report: {REPORT_HTML} (open in browser)")


def build_html(report: dict) -> str:
    """Build a single HTML file with embedded JSON and Chart.js graphs."""
    summary = report["summary"]
    by_dim = report["by_dimension"]
    models = summary["models"]
    # Escape JSON for embedding in HTML
    data_js = json.dumps(report).replace("</", "<\\/")

    by_dim = report["by_dimension"]
    charts_markup = "\n".join(
        f'<div class="chart-container"><h3>{DIMENSION_LABELS.get(dim, dim)}</h3><canvas id="chart-{dim}"></canvas></div>'
        for dim in DIMENSIONS if dim in by_dim
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Power & Compliance Report</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: system-ui, sans-serif; margin: 0; padding: 24px; background: #f8fafc; }}
    h1 {{ color: #0f172a; margin-bottom: 8px; }}
    .summary {{ background: #fff; padding: 16px 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,.08); margin-bottom: 24px; }}
    .summary p {{ margin: 4px 0; color: #475569; }}
    .chart-container {{ background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,.08); margin-bottom: 24px; max-width: 700px; }}
    .chart-container h3 {{ margin: 0 0 16px 0; color: #334155; font-size: 1.1rem; }}
    .gap-table {{ font-size: 14px; margin-top: 12px; }}
    .gap-table th, .gap-table td {{ padding: 4px 12px 4px 0; text-align: left; }}
    .gap-table th {{ color: #64748b; font-weight: 500; }}
  </style>
</head>
<body>
  <h1>Power & Compliance Report</h1>
  <div class="summary">
    <p><strong>Total responses:</strong> {summary["total_responses"]:,} &nbsp;|&nbsp; <strong>Valid:</strong> {summary["valid_responses"]:,} ({summary["valid_pct"]}%)</p>
    <p><strong>Models:</strong> {", ".join(summary["models"])}</p>
    <p><strong>Dimensions:</strong> {", ".join(DIMENSION_LABELS.get(d, d) for d in summary["dimensions"])}</p>
  </div>

  {charts_markup}

  <div class="chart-container">
    <h3>Compliant gap by dimension (percentage points)</h3>
    <canvas id="chart-gaps"></canvas>
  </div>

  <script>
    window.REPORT_DATA = {data_js};

    const DIMENSION_LABELS = {json.dumps(DIMENSION_LABELS)};
    const DIMENSIONS = {json.dumps(DIMENSIONS)};
    const MODELS = {json.dumps(models)};
    const colors = ['#3b82f6', '#a855f7', '#f59e0b'];

    function initCharts() {{
      const byDim = window.REPORT_DATA.by_dimension;
      DIMENSIONS.forEach((dim, idx) => {{
        if (!byDim[dim]) return;
        const modelsData = byDim[dim].models;
        const identities = [];
        for (const m of MODELS) {{
          if (!modelsData[m]) continue;
          Object.keys(modelsData[m].by_identity).forEach(id => {{
            if (!identities.includes(id)) identities.push(id);
          }});
        }}
        identities.sort();
        if (identities.length === 0) return;

        const datasets = MODELS.map((model, i) => {{
          if (!modelsData[model]) return null;
          const byId = modelsData[model].by_identity;
          return {{
            label: model,
            data: identities.map(id => byId[id] ? byId[id].compliant_pct : 0),
            backgroundColor: colors[i % colors.length],
          }};
        }}).filter(Boolean);

        new Chart(document.getElementById('chart-' + dim), {{
          type: 'bar',
          data: {{ labels: identities, datasets }},
          options: {{
            responsive: true,
            plugins: {{ legend: {{ position: 'top' }} }},
            scales: {{
              y: {{ beginAtZero: true, max: 100, title: {{ display: true, text: 'Compliant %' }} }}
            }}
          }}
        }});
      }});

      // Gap chart: one bar per (dimension, model)
      const gapLabels = [];
      const gapData = [];
      const gapColors = [];
      DIMENSIONS.forEach((dim, di) => {{
        if (!byDim[dim]) return;
        const modelsData = byDim[dim].models;
        MODELS.forEach((model, mi) => {{
          if (!modelsData[model] || modelsData[model].overall.compliant_gap_pp == null) return;
          gapLabels.push(DIMENSION_LABELS[dim] + ' — ' + model);
          gapData.push(modelsData[model].overall.compliant_gap_pp);
          gapColors.push(colors[mi % colors.length]);
        }});
      }});
      new Chart(document.getElementById('chart-gaps'), {{
        type: 'bar',
        data: {{ labels: gapLabels, datasets: [{{ label: 'Compliant gap (pp)', data: gapData, backgroundColor: gapColors }}] }},
        options: {{
          indexAxis: 'y',
          responsive: true,
          scales: {{ x: {{ beginAtZero: true, title: {{ display: true, text: 'Compliant % gap' }} }} }}
        }}
      }});
    }}

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initCharts);
    else initCharts();
  </script>
</body>
</html>"""


if __name__ == "__main__":
    main()
