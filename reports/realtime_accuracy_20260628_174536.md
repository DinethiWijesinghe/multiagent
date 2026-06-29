# Real-Time Accuracy Check

Generated at: 2026-06-28T17:45:36.363230+00:00

Base URL: http://127.0.0.1:8000
Cases Run: 5
Cases Passed: 3
Accuracy: 60.0%

## Case Results

| Case ID | PASS | Intent | Factors | Actions | Meaning |
|---|---|---|---|---|---|
| EF-01 | YES | OK | OK | OK | OK |
| EF-06 | NO | NO | NO | OK | NO |
| EF-07 | YES | OK | OK | OK | OK |
| EF-08 | NO | NO | NO | OK | NO |
| EF-12 | YES | OK | OK | OK | OK |

## Failed Cases

- EF-06: intent=general, factors=['financial_constraints', 'psychological_emotional', 'reliable_information', 'educational_background']
- EF-08: intent=general, factors=['time_deadlines', 'reliable_information', 'financial_constraints', 'educational_background']
