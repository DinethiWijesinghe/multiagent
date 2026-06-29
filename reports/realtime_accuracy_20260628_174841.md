# Real-Time Accuracy Check

Generated at: 2026-06-28T17:48:41.455107+00:00

Base URL: http://127.0.0.1:8000
Cases Run: 12
Cases Passed: 5
Accuracy: 41.67%

## Case Results

| Case ID | PASS | Intent | Factors | Actions | Meaning |
|---|---|---|---|---|---|
| EF-01 | YES | OK | OK | OK | OK |
| EF-02 | YES | OK | OK | OK | OK |
| EF-03 | YES | OK | OK | OK | OK |
| EF-04 | NO | OK | OK | OK | NO |
| EF-05 | NO | NO | OK | OK | NO |
| EF-06 | NO | NO | NO | OK | NO |
| EF-07 | YES | OK | OK | OK | OK |
| EF-08 | NO | NO | NO | OK | NO |
| EF-09 | NO | OK | NO | OK | NO |
| EF-10 | NO | OK | OK | OK | NO |
| EF-11 | NO | OK | NO | OK | NO |
| EF-12 | YES | OK | OK | OK | OK |

## Failed Cases

- EF-04: intent=eligibility, factors=['language_proficiency', 'educational_background', 'financial_constraints']
- EF-05: intent=general, factors=['geographic_socioeconomic', 'global_external', 'reliable_information', 'financial_constraints', 'educational_background']
- EF-06: intent=general, factors=['financial_constraints', 'psychological_emotional', 'reliable_information', 'educational_background']
- EF-08: intent=general, factors=['time_deadlines', 'reliable_information', 'financial_constraints', 'educational_background']
- EF-09: intent=visa, factors=['educational_background', 'geographic_socioeconomic', 'global_external', 'visa_immigration', 'financial_constraints']
- EF-10: intent=general, factors=['reliable_information', 'trust_transparency', 'financial_constraints', 'educational_background']
- EF-11: intent=financial, factors=['financial_constraints', 'educational_background']
