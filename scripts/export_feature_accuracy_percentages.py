from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV_PATH = REPO_ROOT / "reports" / "accuracy_summary.csv"
OUT_CSV_PATH = REPO_ROOT / "reports" / "feature_accuracy_percentages.csv"
OUT_MD_PATH = REPO_ROOT / "reports" / "feature_accuracy_percentages.md"


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_rows() -> List[Dict[str, str]]:
    if not INPUT_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_CSV_PATH}. Run scripts/export_accuracy_summary.py first."
        )

    with INPUT_CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _build_percentage_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        if str(row.get("feature_no", "")).strip() == "6":
            continue

        metric = str(row.get("primary_metric", "")).strip().lower()
        value_raw = str(row.get("metric_value", "")).strip()
        value_num = _parse_float(value_raw)

        pct_numeric = value_num * 100 if ("accuracy" in metric and value_num is not None) else None
        if pct_numeric is not None:
            pct_value = f"{pct_numeric:.2f}%"
            status = "Within target" if 80.0 <= pct_numeric <= 100.0 else "Below target"
        else:
            pct_value = "N/A"
            status = "Not applicable"

        feature_name = str(row.get("feature_name", ""))
        training_data = str(row.get("train_samples", ""))
        testing_data = str(row.get("test_samples", ""))
        if pct_numeric is not None:
            result_summary = (
                f"{feature_name} achieved {pct_value} using {training_data} training samples "
                f"and {testing_data} testing or validation samples."
            )
        else:
            result_summary = (
                f"{feature_name} does not provide an accuracy percentage for the selected primary metric."
            )

        out.append(
            {
                "feature_no": str(row.get("feature_no", "")),
                "feature_name": feature_name,
                "benchmark_type": str(row.get("benchmark_type", "")),
                "training_data": training_data,
                "testing_or_validation_data": testing_data,
                "primary_metric": str(row.get("primary_metric", "")),
                "metric_value_raw": value_raw,
                "metric_value_percentage": pct_value,
                "output_status": status,
                "output_result_summary": result_summary,
            }
        )

    return out


def _write_csv(rows: List[Dict[str, str]]) -> None:
    OUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "feature_no",
        "feature_name",
        "benchmark_type",
        "training_data",
        "testing_or_validation_data",
        "primary_metric",
        "metric_value_raw",
        "metric_value_percentage",
        "output_status",
        "output_result_summary",
    ]

    with OUT_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: List[Dict[str, str]]) -> None:
    lines = [
        "# Feature Accuracy Input and Output Report",
        "",
        "## Input Specification",
        "",
        "| Feature No | Feature Name | Benchmark Type | Training Data | Testing or Validation Data | Primary Metric |",
        "|---|---|---|---:|---:|---|",
    ]

    for row in rows:
        lines.append(
            f"| {row['feature_no']} | {row['feature_name']} | {row['benchmark_type']} | "
            f"{row['training_data']} | {row['testing_or_validation_data']} | {row['primary_metric']} |"
        )

    lines.extend(
        [
            "",
            "## Output Results",
            "",
            "| Feature No | Feature Name | Raw Value | Percentage | Status |",
            "|---|---|---:|---:|---|",
        ]
    )

    for row in rows:
        lines.append(
            f"| {row['feature_no']} | {row['feature_name']} | {row['metric_value_raw']} | "
            f"{row['metric_value_percentage']} | {row['output_status']} |"
        )

    lines.extend([
        "",
        "## Professional Summary",
        "",
    ])

    for row in rows:
        lines.append(f"- {row['output_result_summary']}")

    lines.extend(
        [
            "",
            "Notes:",
            "- Only metrics containing 'accuracy' are converted to percentages.",
            "- Non-accuracy metrics are shown as N/A in the percentage column.",
            "",
        ]
    )

    OUT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = _load_rows()
    percentage_rows = _build_percentage_rows(rows)
    _write_csv(percentage_rows)
    _write_markdown(percentage_rows)
    print(f"Wrote {OUT_CSV_PATH}")
    print(f"Wrote {OUT_MD_PATH}")


if __name__ == "__main__":
    main()
