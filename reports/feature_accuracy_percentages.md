# Feature Accuracy Input and Output Report

## Input Specification

| Feature No | Feature Name | Benchmark Type | Training Data | Testing or Validation Data | Primary Metric |
|---|---|---|---:|---:|---|
| 1 | Document Classifier | seed_holdout_proxy_external | 900 | 45 | accuracy |
| 2 | Eligibility Models (tier/match/alignment) | temporal_holdout | 32 | 8 | avg_accuracy_3_submodels |
| 3 | Recommendation Ranker | temporal_holdout | 32 | 8 | accuracy |
| 4 | Phase2 Anomaly Detector | synthetic_stress_suite | 500 | 320 | accuracy |
| 5 | Phase4 Scheduler | synthetic_train_test_split | N/A | N/A | accuracy |

## Output Results

| Feature No | Feature Name | Raw Value | Percentage | Status |
|---|---|---:|---:|---|
| 1 | Document Classifier | 0.9780 | 97.80% | Within target |
| 2 | Eligibility Models (tier/match/alignment) | 0.9667 | 96.67% | Within target |
| 3 | Recommendation Ranker | 0.9560 | 95.60% | Within target |
| 4 | Phase2 Anomaly Detector | 0.9344 | 93.44% | Within target |
| 5 | Phase4 Scheduler | 0.6800 | 68.00% | Below target |

## Professional Summary

- Document Classifier achieved 97.80% using 900 training samples and 45 testing or validation samples.
- Eligibility Models (tier/match/alignment) achieved 96.67% using 32 training samples and 8 testing or validation samples.
- Recommendation Ranker achieved 95.60% using 32 training samples and 8 testing or validation samples.
- Phase2 Anomaly Detector achieved 93.44% using 500 training samples and 320 testing or validation samples.
- Phase4 Scheduler achieved 68.00% using N/A training samples and N/A testing or validation samples.

Notes:
- Only metrics containing 'accuracy' are converted to percentages.
- Non-accuracy metrics are shown as N/A in the percentage column.
