# Accuracy Results for Supervisor Presentation

## Study Scope
This summary presents evaluation results for Features 1 to 6 of the multi-agent system.

## Evaluation Snapshot
- Source file: reports/accuracy_summary.csv
- Generated from: reports/ml_benchmark_report.json
- Note: Feature 6 is forecasting, so MAPE is used (lower is better).

## Main Results
| Feature No | Module | Benchmark Type | Train | Test | Reported Metric | Value |
|---|---|---|---:|---:|---|---:|
| 1 | Document Classifier | seed_holdout_proxy_external | 900 | 45 | Accuracy | 0.9780 |
| 2 | Eligibility (tier/match/alignment) | temporal_holdout | 32 | 8 | Avg Accuracy (3 submodels) | 0.9667 |
| 3 | Recommendation Ranker | temporal_holdout | 32 | 8 | Accuracy | 0.9560 |
| 4 | Phase2 Anomaly Detector | synthetic_stress_suite | 500 | 320 | Accuracy | 0.9344 |
| 5 | Phase4 Scheduler | synthetic_train_test_split | N/A | N/A | Accuracy | 0.6800 |
| 6 | Phase3 Forecasting | temporal_holdout | 48 | 12 | Avg MAPE (%) | 1.3842 |

## Key Messages for Viva/Defense
1. Core decision models (Features 1 to 4) show strong performance with accuracy above 0.93.
2. Eligibility and recommendation models are stable under temporal holdout evaluation.
3. Forecasting quality is strong with low average MAPE (1.3842%).
4. Scheduler module is currently the weakest area and can be improved with richer historical run logs.

## Suggested One-Minute Explanation
The system was evaluated across six implemented ML features using holdout and validation-style benchmarks. Document classification achieved 97.80% accuracy, eligibility averaged 96.67% across three submodels, recommendation ranking reached 95.60%, and anomaly detection achieved 93.44%. For forecasting, accuracy is measured with MAPE, which is low at 1.3842%, indicating good prediction quality. The scheduler currently reports 68.00% accuracy and is identified as the main optimization target.
