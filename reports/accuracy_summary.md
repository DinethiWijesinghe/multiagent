# Accuracy Summary (Features 1-6)

Source report generated_at: 2026-04-09T20:13:32.595677+00:00

| Feature No | Feature Name | Benchmark Type | Train Samples | Test Samples | Primary Metric | Value |
|---|---|---|---:|---:|---|---:|
| 1 | Document Classifier | seed_holdout_proxy_external | 900 | 45 | accuracy | 0.9780 |
| 2 | Eligibility Models (tier/match/alignment) | temporal_holdout | 32 | 8 | avg_accuracy_3_submodels | 0.9667 |
| 3 | Recommendation Ranker | temporal_holdout | 32 | 8 | accuracy | 0.9560 |
| 4 | Phase2 Anomaly Detector | synthetic_stress_suite | 500 | 320 | accuracy | 0.9344 |
| 5 | Phase4 Scheduler | synthetic_train_test_split | N/A | N/A | accuracy | 0.6800 |
| 6 | Phase3 Forecasting | temporal_holdout | 48 | 12 | avg_mape_pct_lower_is_better | 1.3842 |

Notes:
- Feature 2 value is the arithmetic mean of the 3 eligibility submodel accuracies.
- Feature 6 is a forecasting task, so MAPE is reported (lower is better) instead of accuracy.
