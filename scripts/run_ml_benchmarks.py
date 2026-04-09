from __future__ import annotations

import ast
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multiagent.core.agents.eligibility_verification_agent import EligibilityVerificationAgent
from multiagent.core.agents.recommendation_agent import RecommendationAgent
from multiagent.core.database.phase2_web_scraper import Phase2AnomalyDetector
from multiagent.core.database.phase3_api_integration import HISTORICAL_RATES_LKR, Phase3MLEngine, _months_since_2020
from multiagent.core.database.phase4_scheduler import Phase4MLEngine


REPORT_DIR = REPO_ROOT / "reports"
JSON_REPORT_PATH = REPORT_DIR / "ml_benchmark_report.json"
MD_REPORT_PATH = REPORT_DIR / "ml_benchmark_report.md"
API_SERVER_PATH = REPO_ROOT / "multiagent" / "api_server.py"


def _round(value: float) -> float:
    return round(float(value), 4)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _counter_dict(values: Iterable[Any]) -> Dict[str, int]:
    return {str(key): int(count) for key, count in Counter(values).items()}


def _parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


def _load_document_training_data() -> Dict[str, List[str]]:
    module = ast.parse(API_SERVER_PATH.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TRAINING_DATA":
                    return ast.literal_eval(node.value)
    raise RuntimeError(f"TRAINING_DATA not found in {API_SERVER_PATH}")


def _augment_text(text: str) -> List[str]:
    rng = np.random.default_rng(abs(hash(text)) % (2 ** 32))
    words = text.split()
    variants = [text.lower()]
    variants.append(" ".join(w.upper() if rng.random() > 0.55 else w for w in words))
    if len(words) > 3:
        idx = int(rng.integers(0, len(words) - 1))
        swapped = words[:]
        swapped[idx], swapped[idx + 1] = swapped[idx + 1], swapped[idx]
        variants.append(" ".join(swapped))
    else:
        variants.append(text)
    if len(words) > 4:
        drop_count = min(2, max(1, len(words) // 5))
        drop_idx = set(int(i) for i in rng.choice(len(words), size=drop_count, replace=False))
        variants.append(" ".join(w for idx, w in enumerate(words) if idx not in drop_idx))
    else:
        variants.append(text.lower())
    return variants


def _build_document_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=8000,
            sublinear_tf=True,
            min_df=1,
            token_pattern=r"[a-zA-Z0-9]{2,}",
        )),
        ("nb", MultinomialNB(alpha=0.3)),
    ])


def _build_document_corpus(training_data: Dict[str, List[str]], augment: bool = True) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    labels: List[str] = []
    for label, samples in training_data.items():
        for sample in samples:
            texts.append(sample)
            labels.append(label)
            if augment:
                for variant in _augment_text(sample):
                    texts.append(variant)
                    labels.append(label)
    return texts, labels


def _evaluate_document_classifier_internal(training_data: Dict[str, List[str]]) -> Dict[str, Any]:
    texts, labels = _build_document_corpus(training_data, augment=True)
    counts = Counter(labels)
    min_class = min(counts.values()) if counts else 0
    folds = min(5, min_class) if min_class >= 2 else 0
    if folds < 2:
        return {
            "available": False,
            "reason": "Need at least 2 samples per class for stratified cross-validation.",
            "samples": len(labels),
            "classes": _counter_dict(labels),
        }
    results = cross_validate(
        _build_document_pipeline(),
        texts,
        labels,
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=42),
        scoring={
            "accuracy": "accuracy",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
            "f1_macro": "f1_macro",
        },
        n_jobs=1,
    )
    return {
        "available": True,
        "samples": len(labels),
        "classes": _counter_dict(labels),
        "cv_folds": folds,
        "metrics": {
            name.replace("test_", ""): {
                "mean": _round(values.mean()),
                "std": _round(values.std()),
            }
            for name, values in results.items()
            if name.startswith("test_")
        },
    }


def _classification_metrics(y_true: Sequence[Any], y_pred: Sequence[Any], average: str) -> Dict[str, float]:
    return {
        "accuracy": _round(accuracy_score(y_true, y_pred)),
        "precision": _round(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": _round(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": _round(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


def _binary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float],
    include_brier: bool = True,
) -> Dict[str, float]:
    metrics = {
        "accuracy": _round(accuracy_score(y_true, y_pred)),
        "precision": _round(precision_score(y_true, y_pred, zero_division=0)),
        "recall": _round(recall_score(y_true, y_pred, zero_division=0)),
        "f1": _round(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _round(roc_auc_score(y_true, y_score)),
        "average_precision": _round(average_precision_score(y_true, y_score)),
    }
    if include_brier:
        metrics["brier_score"] = _round(brier_score_loss(y_true, y_score))
    return metrics


def _train_test_split_by_ratio(items: Sequence[Any], test_ratio: float = 0.2) -> Tuple[List[Any], List[Any]]:
    test_count = max(1, int(math.ceil(len(items) * test_ratio)))
    train_count = max(len(items) - test_count, 1)
    return list(items[:train_count]), list(items[train_count:])


def _benchmark_document_classifier() -> Dict[str, Any]:
    training_data = _load_document_training_data()
    train_texts: List[str] = []
    train_labels: List[str] = []
    test_texts: List[str] = []
    test_labels: List[str] = []

    for label, samples in training_data.items():
        ordered = list(samples)
        split_index = max(1, int(math.ceil(len(ordered) * 0.2)))
        holdout = ordered[-split_index:]
        train_seed = ordered[:-split_index] or ordered
        for sample in train_seed:
            train_texts.append(sample)
            train_labels.append(label)
            for augmented in _augment_text(sample):
                train_texts.append(augmented)
                train_labels.append(label)
        test_texts.extend(holdout)
        test_labels.extend([label] * len(holdout))

    pipeline = _build_document_pipeline()
    pipeline.fit(train_texts, train_labels)
    predictions = pipeline.predict(test_texts)
    return {
        "benchmark_type": "seed_holdout_proxy_external",
        "train_samples": len(train_labels),
        "test_samples": len(test_labels),
        "test_classes": _counter_dict(test_labels),
        "metrics": _classification_metrics(test_labels, predictions, average="macro"),
    }


def _benchmark_eligibility_models() -> Dict[str, Any]:
    agent = EligibilityVerificationAgent()
    rows = sorted(
        agent._load_historical_outcomes(agent._historical_outcomes_path),
        key=lambda row: _parse_timestamp(str(row.get("timestamp", ""))),
    )
    train_rows, test_rows = _train_test_split_by_ratio(rows, test_ratio=0.2)
    train_sets = agent._prepare_training_datasets(train_rows)
    test_sets = agent._prepare_training_datasets(test_rows)

    results: Dict[str, Any] = {
        "benchmark_type": "temporal_holdout",
        "source_file": str(Path(agent._historical_outcomes_path).relative_to(REPO_ROOT)),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "models": {},
    }

    model_specs = {
        "tier_classifier": (
            RandomForestClassifier(n_estimators=80, random_state=42),
            train_sets["tier"]["X"],
            train_sets["tier"]["y"],
            test_sets["tier"]["X"],
            test_sets["tier"]["y"],
            "macro",
        ),
        "match_classifier": (
            RandomForestClassifier(n_estimators=80, random_state=42),
            train_sets["match"]["X"],
            train_sets["match"]["y"],
            test_sets["match"]["X"],
            test_sets["match"]["y"],
            "macro",
        ),
    }
    for name, (estimator, X_train, y_train, X_test, y_test, average) in model_specs.items():
        if len(X_train) < 20 or len(X_test) == 0 or len(set(y_train)) < 2 or len(set(y_test)) < 2:
            results["models"][name] = {
                "available": False,
                "reason": "Temporal holdout does not contain enough class diversity.",
                "train_samples": len(y_train),
                "test_samples": len(y_test),
            }
            continue
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        results["models"][name] = {
            "available": True,
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "test_classes": _counter_dict(y_test),
            "metrics": _classification_metrics(y_test, y_pred, average=average),
        }

    align_train_X = train_sets["alignment"]["X"]
    align_train_y = train_sets["alignment"]["y"]
    align_test_X = test_sets["alignment"]["X"]
    align_test_y = test_sets["alignment"]["y"]
    if len(align_train_X) >= 20 and len(set(align_train_y)) >= 2 and len(align_test_X) > 0 and len(set(align_test_y)) >= 2:
        align_model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("svc", LinearSVC(C=1.0, max_iter=3000, dual=True)),
        ])
        align_model.fit(align_train_X, align_train_y)
        align_pred = align_model.predict(align_test_X)
        results["models"]["alignment_classifier"] = {
            "available": True,
            "train_samples": len(align_train_y),
            "test_samples": len(align_test_y),
            "test_classes": _counter_dict(align_test_y),
            "metrics": _classification_metrics(align_test_y, align_pred, average="binary"),
        }
    else:
        results["models"]["alignment_classifier"] = {
            "available": False,
            "reason": "Temporal holdout does not contain enough alignment labels.",
            "train_samples": len(align_train_y),
            "test_samples": len(align_test_y),
        }

    return results


def _benchmark_recommendation_model() -> Dict[str, Any]:
    agent = RecommendationAgent(disable_direct_visa_sources=True)
    rows = sorted(
        agent._load_outcomes(agent.historical_outcomes_path),
        key=lambda row: _parse_timestamp(str(row.get("timestamp", ""))),
    )
    train_rows, test_rows = _train_test_split_by_ratio(rows, test_ratio=0.2)
    X_train, y_train = agent._build_training_dataset(train_rows)
    X_test, y_test = agent._build_training_dataset(test_rows)
    if len(X_train) < 30 or len(X_test) == 0 or len(set(y_train)) < 2 or len(set(y_test)) < 2:
        return {
            "benchmark_type": "temporal_holdout",
            "available": False,
            "reason": "Temporal holdout does not contain enough labeled outcomes.",
            "train_samples": len(y_train),
            "test_samples": len(y_test),
        }

    model = RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "benchmark_type": "temporal_holdout",
        "available": True,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "test_classes": _counter_dict(y_test),
        "metrics": _binary_metrics(y_test, predictions, probabilities),
    }


def _benchmark_phase2_anomaly_detector() -> Dict[str, Any]:
    detector = Phase2AnomalyDetector()
    train_normal = detector._build_baseline_samples(np.random.default_rng(101), samples_per_uni=25)
    test_normal = detector._build_baseline_samples(np.random.default_rng(202), samples_per_uni=8)
    anomalies = detector._build_anomaly_samples(np.random.default_rng(303), samples_per_uni=8)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_normal)
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(train_scaled)

    X_test = np.vstack([test_normal, anomalies])
    y_true = np.concatenate([
        np.zeros(len(test_normal), dtype=int),
        np.ones(len(anomalies), dtype=int),
    ])
    X_test_scaled = scaler.transform(X_test)
    anomaly_scores = -model.decision_function(X_test_scaled)
    y_pred = (model.predict(X_test_scaled) == -1).astype(int)
    metrics = _binary_metrics(y_true, y_pred, anomaly_scores, include_brier=False)
    metrics["false_positive_rate"] = _round(float(((y_pred == 1) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)))
    return {
        "benchmark_type": "synthetic_stress_suite",
        "available": True,
        "train_samples": int(len(train_normal)),
        "test_samples": int(len(X_test)),
        "metrics": metrics,
    }


def _build_phase3_model() -> Pipeline:
    return Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=True)),
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=1.0)),
    ])


def _benchmark_phase3_forecasting() -> Dict[str, Any]:
    _ = Phase3MLEngine()
    currencies: Dict[str, Any] = {}
    for currency, history in HISTORICAL_RATES_LKR.items():
        train_history, test_history = _train_test_split_by_ratio(history, test_ratio=0.2)
        if len(train_history) < 8 or len(test_history) == 0:
            currencies[currency] = {
                "available": False,
                "reason": "Not enough temporal samples for holdout evaluation.",
            }
            continue
        X_train = np.array([[_months_since_2020(y, m)] for y, m, _ in train_history])
        y_train = np.array([rate for _, _, rate in train_history])
        X_test = np.array([[_months_since_2020(y, m)] for y, m, _ in test_history])
        y_test = np.array([rate for _, _, rate in test_history])
        model = _build_phase3_model()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mape = float(np.mean(np.abs((y_test - predictions) / np.clip(y_test, 1e-9, None))) * 100)
        currencies[currency] = {
            "available": True,
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "metrics": {
                "mae_lkr": _round(mean_absolute_error(y_test, predictions)),
                "rmse_lkr": _round(np.sqrt(mean_squared_error(y_test, predictions))),
                "mape_pct": _round(mape),
            },
        }
    return {
        "benchmark_type": "temporal_holdout",
        "currencies": currencies,
    }


def _benchmark_phase4_scheduler() -> Dict[str, Any]:
    engine = Phase4MLEngine()
    train_result = engine.train(verbose=False)
    if "accuracy" not in train_result:
        return {
            "benchmark_type": "synthetic_train_test_split",
            "available": False,
            "reason": train_result.get("error", "Phase4 scheduler training did not return accuracy."),
        }
    return {
        "benchmark_type": "synthetic_train_test_split",
        "available": True,
        "metrics": {
            "accuracy": _round(train_result["accuracy"]),
        },
        "trained_at": train_result.get("trained_at"),
    }


def _production_metric_recommendations() -> Dict[str, List[str]]:
    return {
        "document_classifier": [
            "accuracy",
            "macro_f1",
            "macro_recall",
            "per_class_precision_recall",
        ],
        "eligibility_models": [
            "tier_macro_f1",
            "match_macro_f1",
            "alignment_f1",
            "confusion_matrix_per_submodel",
        ],
        "recommendation_ranker": [
            "roc_auc",
            "average_precision",
            "brier_score",
            "calibration_curve",
        ],
        "phase2_anomaly_detector": [
            "anomaly_precision",
            "anomaly_recall",
            "false_positive_rate",
            "alert_review_rate",
        ],
        "phase3_forecasting": [
            "walk_forward_mae",
            "walk_forward_rmse",
            "mape",
            "forecast_bias",
        ],
        "phase4_scheduler": [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "skip_rate",
        ],
        "rag_retrieval": [
            "recall_at_k",
            "mrr",
            "ndcg_at_k",
            "grounded_answer_rate",
        ],
    }


def _build_report() -> Dict[str, Any]:
    document_training_data = _load_document_training_data()
    report = {
        "generated_at": _iso_now(),
        "internal_metrics": {
            "document_classifier": _evaluate_document_classifier_internal(document_training_data),
            "eligibility_models": EligibilityVerificationAgent().get_model_metrics(),
            "recommendation_ranker": RecommendationAgent(disable_direct_visa_sources=True).get_model_metrics(),
            "phase2_anomaly_detector": Phase2AnomalyDetector().fit_baseline(verbose=False).get("evaluation", {}),
            "phase3_forecasting": Phase3MLEngine().train(verbose=False).get("metrics", {}),
            "phase4_scheduler": _benchmark_phase4_scheduler(),
        },
        "external_benchmarks": {
            "document_classifier": _benchmark_document_classifier(),
            "eligibility_models": _benchmark_eligibility_models(),
            "recommendation_ranker": _benchmark_recommendation_model(),
            "phase2_anomaly_detector": _benchmark_phase2_anomaly_detector(),
            "phase3_forecasting": _benchmark_phase3_forecasting(),
            "phase4_scheduler": _benchmark_phase4_scheduler(),
            "rag_retrieval": {
                "available": False,
                "reason": "No labeled retrieval benchmark exists in the repository. Use a query-document relevance set before reporting retrieval quality externally.",
            },
        },
        "production_metric_recommendations": _production_metric_recommendations(),
    }
    return report


def _to_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# ML Benchmark Report",
        "",
        f"Generated at: {report['generated_at']}",
        "",
        "## Internal Metrics",
        "",
    ]
    for name, payload in report["internal_metrics"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(payload, indent=2))
        lines.append("```")
        lines.append("")

    lines.append("## External Benchmarks")
    lines.append("")
    for name, payload in report["external_benchmarks"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(payload, indent=2))
        lines.append("```")
        lines.append("")

    lines.append("## Production Metric Recommendations")
    lines.append("")
    for name, metrics in report["production_metric_recommendations"].items():
        lines.append(f"### {name}")
        lines.append("")
        for metric in metrics:
            lines.append(f"- {metric}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = _build_report()
    JSON_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    MD_REPORT_PATH.write_text(_to_markdown(report), encoding="utf-8")
    print(f"Wrote {JSON_REPORT_PATH}")
    print(f"Wrote {MD_REPORT_PATH}")


if __name__ == "__main__":
    main()