# ML Benchmark Report

Generated at: 2026-04-09T11:17:49.028866+00:00

## Internal Metrics

### document_classifier

```json
{
  "available": true,
  "samples": 1125,
  "classes": {
    "alevel": 125,
    "bachelor": 125,
    "master": 125,
    "diploma": 125,
    "ielts": 125,
    "toefl": 125,
    "pte": 125,
    "passport": 125,
    "financial": 125
  },
  "cv_folds": 5,
  "metrics": {
    "accuracy": {
      "mean": 1.0,
      "std": 0.0
    },
    "precision_macro": {
      "mean": 1.0,
      "std": 0.0
    },
    "recall_macro": {
      "mean": 1.0,
      "std": 0.0
    },
    "f1_macro": {
      "mean": 1.0,
      "std": 0.0
    }
  }
}
```

### eligibility_models

```json
{
  "available": true,
  "tier_classifier": {
    "available": true,
    "samples": 40,
    "classes": {
      "foundation": 20,
      "good": 13,
      "top": 7
    },
    "cv_folds": 5,
    "metrics": {
      "accuracy": {
        "mean": 1.0,
        "std": 0.0
      },
      "precision": {
        "mean": 1.0,
        "std": 0.0
      },
      "recall": {
        "mean": 1.0,
        "std": 0.0
      },
      "f1": {
        "mean": 1.0,
        "std": 0.0
      }
    }
  },
  "match_classifier": {
    "available": true,
    "samples": 40,
    "classes": {
      "below_minimum": 20,
      "strong_match": 14,
      "meets_minimum": 6
    },
    "cv_folds": 5,
    "metrics": {
      "accuracy": {
        "mean": 1.0,
        "std": 0.0
      },
      "precision": {
        "mean": 1.0,
        "std": 0.0
      },
      "recall": {
        "mean": 1.0,
        "std": 0.0
      },
      "f1": {
        "mean": 1.0,
        "std": 0.0
      }
    }
  },
  "alignment_classifier": {
    "available": true,
    "samples": 40,
    "classes": {
      "0": 20,
      "1": 20
    },
    "cv_folds": 5,
    "metrics": {
      "accuracy": {
        "mean": 1.0,
        "std": 0.0
      },
      "precision": {
        "mean": 1.0,
        "std": 0.0
      },
      "recall": {
        "mean": 1.0,
        "std": 0.0
      },
      "f1": {
        "mean": 1.0,
        "std": 0.0
      }
    }
  }
}
```

### recommendation_ranker

```json
{
  "available": true,
  "samples": 40,
  "classes": {
    "0": 20,
    "1": 20
  },
  "cv_folds": 5,
  "metrics": {
    "accuracy": {
      "mean": 1.0,
      "std": 0.0
    },
    "precision": {
      "mean": 1.0,
      "std": 0.0
    },
    "recall": {
      "mean": 1.0,
      "std": 0.0
    },
    "f1": {
      "mean": 1.0,
      "std": 0.0
    },
    "roc_auc": {
      "mean": 1.0,
      "std": 0.0
    },
    "average_precision": {
      "mean": 1.0,
      "std": 0.0
    },
    "log_loss": {
      "mean": 0.0004,
      "std": 0.0008
    },
    "brier_score": {
      "mean": 0.0,
      "std": 0.0
    }
  }
}
```

### phase2_anomaly_detector

```json
{
  "available": true,
  "train_samples": 480,
  "test_samples": 280,
  "synthetic_anomalies": 160,
  "metrics": {
    "accuracy": 0.9393,
    "precision": 0.9387,
    "recall": 0.9563,
    "f1": 0.9474,
    "roc_auc": 0.9839,
    "false_positive_rate": 0.0833
  }
}
```

### phase3_forecasting

```json
{
  "GBP": {
    "train_mae_lkr": 9.015,
    "data_points": 20,
    "walk_forward": {
      "available": true,
      "backtest_points": 12,
      "mae_lkr": 13.654,
      "rmse_lkr": 15.0027,
      "mape_pct": 2.8077
    }
  },
  "SGD": {
    "train_mae_lkr": 2.2803,
    "data_points": 20,
    "walk_forward": {
      "available": true,
      "backtest_points": 12,
      "mae_lkr": 2.7386,
      "rmse_lkr": 3.1446,
      "mape_pct": 1.0646
    }
  },
  "AUD": {
    "train_mae_lkr": 1.8059,
    "data_points": 20,
    "walk_forward": {
      "available": true,
      "backtest_points": 12,
      "mae_lkr": 1.4817,
      "rmse_lkr": 1.9351,
      "mape_pct": 0.6213
    }
  }
}
```

## External Benchmarks

### document_classifier

```json
{
  "benchmark_type": "seed_holdout_proxy_external",
  "train_samples": 900,
  "test_samples": 45,
  "test_classes": {
    "alevel": 5,
    "bachelor": 5,
    "master": 5,
    "diploma": 5,
    "ielts": 5,
    "toefl": 5,
    "pte": 5,
    "passport": 5,
    "financial": 5
  },
  "metrics": {
    "accuracy": 1.0,
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0
  }
}
```

### eligibility_models

```json
{
  "benchmark_type": "temporal_holdout",
  "source_file": "multiagent\\data\\training\\historical_admissions_outcomes.jsonl",
  "train_rows": 32,
  "test_rows": 8,
  "models": {
    "tier_classifier": {
      "available": true,
      "train_samples": 32,
      "test_samples": 8,
      "test_classes": {
        "foundation": 4,
        "top": 2,
        "good": 2
      },
      "metrics": {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0
      }
    },
    "match_classifier": {
      "available": true,
      "train_samples": 32,
      "test_samples": 8,
      "test_classes": {
        "below_minimum": 4,
        "strong_match": 3,
        "meets_minimum": 1
      },
      "metrics": {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0
      }
    },
    "alignment_classifier": {
      "available": true,
      "train_samples": 32,
      "test_samples": 8,
      "test_classes": {
        "0": 4,
        "1": 4
      },
      "metrics": {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0
      }
    }
  }
}
```

### recommendation_ranker

```json
{
  "benchmark_type": "temporal_holdout",
  "available": true,
  "train_samples": 32,
  "test_samples": 8,
  "test_classes": {
    "0": 4,
    "1": 4
  },
  "metrics": {
    "accuracy": 1.0,
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0,
    "roc_auc": 1.0,
    "average_precision": 1.0,
    "brier_score": 0.0
  }
}
```

### phase2_anomaly_detector

```json
{
  "benchmark_type": "synthetic_stress_suite",
  "available": true,
  "train_samples": 500,
  "test_samples": 320,
  "metrics": {
    "accuracy": 0.9344,
    "precision": 0.9371,
    "recall": 0.9313,
    "f1": 0.9342,
    "roc_auc": 0.978,
    "average_precision": 0.9757,
    "false_positive_rate": 0.0625
  }
}
```

### phase3_forecasting

```json
{
  "benchmark_type": "temporal_holdout",
  "currencies": {
    "GBP": {
      "available": true,
      "train_samples": 16,
      "test_samples": 4,
      "metrics": {
        "mae_lkr": 11.2734,
        "rmse_lkr": 11.306,
        "mape_pct": 2.1725
      }
    },
    "SGD": {
      "available": true,
      "train_samples": 16,
      "test_samples": 4,
      "metrics": {
        "mae_lkr": 3.0056,
        "rmse_lkr": 3.0224,
        "mape_pct": 1.0958
      }
    },
    "AUD": {
      "available": true,
      "train_samples": 16,
      "test_samples": 4,
      "metrics": {
        "mae_lkr": 2.2431,
        "rmse_lkr": 2.2575,
        "mape_pct": 0.8844
      }
    }
  }
}
```

### rag_retrieval

```json
{
  "available": false,
  "reason": "No labeled retrieval benchmark exists in the repository. Use a query-document relevance set before reporting retrieval quality externally."
}
```

## Production Metric Recommendations

### document_classifier

- accuracy
- macro_f1
- macro_recall
- per_class_precision_recall

### eligibility_models

- tier_macro_f1
- match_macro_f1
- alignment_f1
- confusion_matrix_per_submodel

### recommendation_ranker

- roc_auc
- average_precision
- brier_score
- calibration_curve

### phase2_anomaly_detector

- anomaly_precision
- anomaly_recall
- false_positive_rate
- alert_review_rate

### phase3_forecasting

- walk_forward_mae
- walk_forward_rmse
- mape
- forecast_bias

### rag_retrieval

- recall_at_k
- mrr
- ndcg_at_k
- grounded_answer_rate
