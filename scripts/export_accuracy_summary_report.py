from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON_PATH = REPO_ROOT / "reports" / "ml_benchmark_report.json"
OUT_MD_PATH = REPO_ROOT / "reports" / "accuracy_summary.md"
OUT_CSV_PATH = REPO_ROOT / "reports" / "accuracy_summary.csv"


def _load_report() -> Dict[str, Any]:
    if not REPORT_JSON_PATH.exists():
        raise FileNotFoundError(f"Missing report file: {REPORT_JSON_PATH}")
    return json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))


def _fmt_num(value: Any, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return "N/A"


def _build_feature_rows(report: Dict[str, Any]) -> List[Dict[str, str]]:
    external = report.get("external_benchmarks", {})

    f1 = external.get("document_classifier", {})
    f2 = external.get("eligibility_models", {})
    f3 = external.get("recommendation_ranker", {})
    f4 = external.get("phase2_anomaly_detector", {})
    f5 = external.get("phase4_scheduler", {})
    f6 = external.get("phase3_forecasting", {})

    # Feature 2 combines 3 submodels; use macro-average accuracy for single-row reporting.
    f2_models = f2.get("models", {})
    f2_acc_values = []
    for sub in ("tier_classifier", "match_classifier", "alignment_classifier"):
        acc = f2_models.get(sub, {}).get("metrics", {}).get("accuracy")
        if isinstance(acc, (int, float)):
            f2_acc_values.append(float(acc))
    f2_acc = sum(f2_acc_values) / len(f2_acc_values) if f2_acc_values else None

    # Feature 6 is forecasting, so report MAPE across currencies instead of classification accuracy.
    f6_currencies = f6.get("currencies", {})
    f6_mape_values = []
    f6_train = 0
    f6_test = 0
    for cur in f6_currencies.values():
        mape = cur.get("metrics", {}).get("mape_pct")
        if isinstance(mape, (int, float)):
            f6_mape_values.append(float(mape))
        if isinstance(cur.get("train_samples"), int):
            f6_train += cur.get("train_samples", 0)
        if isinstance(cur.get("test_samples"), int):
            f6_test += cur.get("test_samples", 0)
    f6_mape = sum(f6_mape_values) / len(f6_mape_values) if f6_mape_values else None

    rows = [
        {
            "feature_no": "1",
            "feature_name": "Document Classifier",
            "benchmark_type": str(f1.get("benchmark_type", "N/A")),
            "train_samples": str(f1.get("train_samples", "N/A")),
            "test_samples": str(f1.get("test_samples", "N/A")),
            "primary_metric": "accuracy",
            "metric_value": _fmt_num(f1.get("metrics", {}).get("accuracy")),
        },
        {
            "feature_no": "2",
            "feature_name": "Eligibility Models (tier/match/alignment)",
            "benchmark_type": str(f2.get("benchmark_type", "N/A")),
            "train_samples": str(f2.get("train_rows", "N/A")),
            "test_samples": str(f2.get("test_rows", "N/A")),
            "primary_metric": "avg_accuracy_3_submodels",
            "metric_value": _fmt_num(f2_acc),
        },
        {
            "feature_no": "3",
            "feature_name": "Recommendation Ranker",
            "benchmark_type": str(f3.get("benchmark_type", "N/A")),
            "train_samples": str(f3.get("train_samples", "N/A")),
            "test_samples": str(f3.get("test_samples", "N/A")),
            "primary_metric": "accuracy",
            "metric_value": _fmt_num(f3.get("metrics", {}).get("accuracy")),
        },
        {
            "feature_no": "4",
            "feature_name": "Phase2 Anomaly Detector",
            "benchmark_type": str(f4.get("benchmark_type", "N/A")),
            "train_samples": str(f4.get("train_samples", "N/A")),
            "test_samples": str(f4.get("test_samples", "N/A")),
            "primary_metric": "accuracy",
            "metric_value": _fmt_num(f4.get("metrics", {}).get("accuracy")),
        },
        {
            "feature_no": "5",
            "feature_name": "Phase4 Scheduler",
            "benchmark_type": str(f5.get("benchmark_type", "N/A")),
            "train_samples": "N/A",
            "test_samples": "N/A",
            "primary_metric": "accuracy",
            "metric_value": _fmt_num(f5.get("metrics", {}).get("accuracy")),
        },
        {
            "feature_no": "6",
            "feature_name": "Phase3 Forecasting",
            "benchmark_type": str(f6.get("benchmark_type", "N/A")),
            "train_samples": str(f6_train if f6_train else "N/A"),
            "test_samples": str(f6_test if f6_test else "N/A"),
            "primary_metric": "avg_mape_pct_lower_is_better",
            "metric_value": _fmt_num(f6_mape),
        },
    ]

    return rows


def _write_csv(rows: List[Dict[str, str]]) -> None:
    OUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "feature_no",
        "feature_name",
        "benchmark_type",
        "train_samples",
        "test_samples",
        "primary_metric",
        "metric_value",
    ]
    with OUT_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: List[Dict[str, str]], generated_at: str) -> None:
    lines = [
        "# Accuracy Summary (Features 1-6)",
        "",
        f"Source report generated_at: {generated_at}",
        "",
        "| Feature No | Feature Name | Benchmark Type | Train Samples | Test Samples | Primary Metric | Value |",
        "|---|---|---|---:|---:|---|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {row['feature_no']} | {row['feature_name']} | {row['benchmark_type']} | {row['train_samples']} | {row['test_samples']} | {row['primary_metric']} | {row['metric_value']} |"
        )

    lines.extend(
        [
            "",
            "Notes:",
            "- Feature 2 value is the arithmetic mean of the 3 eligibility submodel accuracies.",
            "- Feature 6 is a forecasting task, so MAPE is reported (lower is better) instead of accuracy.",
            "",
        ]
    )

    OUT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    report = _load_report()
    rows = _build_feature_rows(report)
    _write_csv(rows)
    _write_markdown(rows, str(report.get("generated_at", "N/A")))
    print(f"Wrote {OUT_CSV_PATH}")
    print(f"Wrote {OUT_MD_PATH}")


if __name__ == "__main__":
    main()