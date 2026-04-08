# Real-Time External Factor Test Cases

## Purpose
This document provides manual and semi-automated test cases for real-time chatbot behavior when students ask practical study-abroad questions affected by external factors.

## Scope
- Validate that student questions trigger the correct external factors.
- Validate that system responses are relevant, empathetic when needed, and actionable.
- Validate that next-step actions are practical for live application support.

## Test Data Baseline
Use a context similar to the following for stable results:

- Profile includes country, program interest, and budget.
- Document data includes at least one academic document.
- Universities list includes at least 3 universities, with at least 1 upcoming deadline.
- Recommendation, eligibility, and financial agents are available in test environment.

## Pass Rule
A test is PASS when all expected checks are satisfied:

1. Intent is correct or acceptable alternative intent is observed.
2. Expected external factor IDs are present in agent data.
3. Response contains the expected meaning (not exact string match).
4. Actions include at least one expected actionable item.

## Test Cases

| Case ID | Student Question | Expected Intent | Expected External Factors | Expected System Answer (Meaning Check) | Expected Actions (At Least One) |
|---|---|---|---|---|---|
| EF-01 | My family budget is very low. Can I still apply to UK universities? | financial | financial_constraints | Explains affordability clearly, mentions feasible or borderline options, and suggests scholarships or lower-cost alternatives. | Set your annual budget in profile; Check scholarships and fee waivers; Shortlist budget-feasible universities |
| EF-02 | Why did you recommend University A over University B? I want transparent reasons. | recommendation | trust_transparency, reliable_information | Gives ranking reasons using eligibility/cost/deadline evidence and avoids vague claims. | Review ranking reasons; Check eligibility and cost evidence; Compare recommendation with backups |
| EF-03 | I have A/L results with mixed grades. Do I need a foundation pathway? | eligibility | educational_background, language_proficiency | Explains direct eligibility vs pathway route and highlights missing improvements. | Run eligibility check; Check program fit and pathway options |
| EF-04 | My IELTS is 5.5. Can I apply now or should I retake? | eligibility | language_proficiency | Compares current score with likely thresholds and gives retake guidance if needed. | Check language-score requirements; Plan retake if needed |
| EF-05 | I live in a rural area with weak internet. Can I complete this process online? | document | geographic_socioeconomic | Confirms remote-friendly process and suggests low-travel steps with saved progress. | Use remote document upload; Prioritize low-travel application steps; Keep all progress saved in your account |
| EF-06 | I am very anxious and overwhelmed. I feel I will miss everything. | emotional | psychological_emotional, time_deadlines | Responds with empathy, reduces pressure, and gives small next steps with timeline focus. | Break the process into small steps; Focus on one next action; Prioritize earliest deadlines |
| EF-07 | What visa documents do I need for Canada student permit? | visa | visa_immigration, global_external | Provides visa checklist guidance, timing advice, and indicates policy uncertainty handling. | Review visa requirements early; Prepare visa document checklist; Prioritize lower-risk destinations |
| EF-08 | Deadlines are this month. Which application should I submit first? | recommendation | time_deadlines, trust_transparency | Prioritizes by earliest deadline and explains sequencing logic. | Prioritize earliest deadlines; Sequence applications by intake; Prepare required documents first |
| EF-09 | I heard there may be policy changes and travel restrictions. Should I keep backups? | visa | global_external, trust_transparency | Advises risk-aware planning, backup destinations, and flexible options like hybrid/online. | Check country-risk updates; Keep backup destinations ready; Consider hybrid or online fallback options |
| EF-10 | Consultants gave me different advice. Which source should I trust? | general or recommendation | reliable_information, trust_transparency | Promotes verified sources, official links, and evidence-based comparison. | Check official university links; Compare evidence before deciding |
| EF-11 | Exchange rate keeps changing. How do I plan my tuition payments safely? | financial | financial_constraints, global_external | Acknowledges currency volatility and suggests budget buffers plus scholarship strategy. | Review tuition and living costs; Check scholarships and fee waivers |
| EF-12 | I want fast recommendations but also need visa-safe and affordable options. | recommendation | financial_constraints, visa_immigration, time_deadlines, trust_transparency | Produces balanced shortlist logic across cost, visa risk, and urgency, with clear reasoning. | Shortlist budget-feasible universities; Prioritize lower-risk destinations; Prioritize earliest deadlines |

## Student Question and System Answer Samples

Use these as realistic chatbot response examples for manual testing.

### Q&A 1 - Financial Constraints
Student Question: My family budget is very low. Can I still apply to UK universities?

System Answer:
Yes, you can still apply, but we should target budget-feasible universities first. Based on your budget profile, I will prioritize lower-cost options and identify scholarship and fee-waiver opportunities. We can also keep 1-2 backup countries with lower total living costs.

### Q&A 2 - Trust and Transparency
Student Question: Why did you recommend University A over University B?

System Answer:
University A was ranked higher because your eligibility match is stronger, the total cost is closer to your budget, and the deadline timing is safer for your current progress. University B is still a backup option, but it has higher financial pressure and tighter risk on submission timing.

### Q&A 3 - Educational Background and Pathway
Student Question: I have mixed A/L grades. Do I need a foundation pathway?

System Answer:
You may still have direct-entry options, but a foundation pathway can improve your chances for competitive programs. I will check your subject-level fit and show both direct and pathway routes so you can compare risk and timeline.

### Q&A 4 - Language Proficiency
Student Question: My IELTS is 5.5. Can I apply now or should I retake?

System Answer:
You can apply to programs that accept your current score, but many universities require a higher band for direct entry. I recommend shortlisting programs with 5.5 acceptance and planning a retake to unlock more options and scholarship competitiveness.

### Q&A 5 - Geographic and Access Challenges
Student Question: I live in a rural area with weak internet. Can I complete this online?

System Answer:
Yes. You can complete most steps online with saved progress, including profile updates and document uploads. To reduce connectivity issues, we can prioritize low-data tasks first and keep a checklist so you can continue from where you stopped.

### Q&A 6 - Psychological and Emotional Support
Student Question: I am overwhelmed and anxious. I feel like I will miss deadlines.

System Answer:
You are not alone, and we can handle this step by step. Let us focus on one immediate task now: finalize your top 3 universities and submit the earliest deadline first. After that, we will plan documents and visa tasks in small, manageable checkpoints.

### Q&A 7 - Visa and Immigration
Student Question: What visa documents do I need for a Canada student permit?

System Answer:
Start with your offer letter, valid passport, proof of funds, academic records, and required immigration forms. I recommend preparing your financial and document checklist early because policy updates can affect timelines. I can also suggest lower-risk backup destinations in parallel.

### Q&A 8 - Time and Deadline Pressure
Student Question: Deadlines are this month. Which application should I submit first?

System Answer:
Submit in this order: first the university with the earliest deadline and complete documents, then the next nearest deadline, then your backup option. This sequencing reduces rejection risk from late submission and keeps your options open.

### Q&A 9 - Global External Risk
Student Question: I heard about travel restrictions and policy changes. Should I keep backups?

System Answer:
Yes, keep at least two backup destinations. Global policy and travel conditions can change quickly, so a risk-aware plan should include one stable primary option and one flexible backup, including hybrid or online alternatives.

### Q&A 10 - Conflicting Consultant Advice
Student Question: Different consultants told me different things. Which source should I trust?

System Answer:
Trust decisions that are backed by official university requirements, transparent eligibility checks, and clear cost evidence. I can show you a side-by-side comparison so each recommendation is traceable and not based on hidden bias.

### Q&A 11 - Exchange Rate Volatility
Student Question: Exchange rates keep changing. How should I plan tuition safely?

System Answer:
Use a safety buffer in your budget and prioritize universities that remain feasible under a weaker exchange-rate scenario. Then combine this with scholarship-focused applications so your financial risk is controlled.

### Q&A 12 - Multi-Factor Real-Time Decision
Student Question: I need fast recommendations, but they must be affordable and visa-safe.

System Answer:
I will generate a shortlist using three filters together: budget feasibility, visa-risk profile, and nearest deadlines. You will get a primary list for immediate action and backup options with transparent reasons for each ranking.

## Real-Time Regression Set (Quick Run)
Run these 5 cases in every release candidate:

1. EF-01 (budget pressure)
2. EF-06 (emotional distress)
3. EF-07 (visa checklist)
4. EF-08 (urgent deadlines)
5. EF-12 (multi-factor balancing)

## Suggested Evaluation Notes Template
Use this note format per test case:

- Case ID:
- Timestamp:
- User message used:
- Observed intent:
- Observed external factors:
- Response summary:
- Actions returned:
- PASS or FAIL:
- If FAIL, issue details:

## Optional Automation Mapping
For automation, map each case to assertions on:

- response.intent
- response.agent_data.external_factors[].id
- response.actions[] contains expected action text
- response.response contains required meaning keywords
