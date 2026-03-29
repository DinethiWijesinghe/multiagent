# MultiAgent International Study Advisor - Complete Application Overview
## For MSc Students | Comprehensive Architecture & Design Guide

---

## 1. PROJECT MISSION & PROBLEM STATEMENT

### What Problem Does It Solve?
International students from Sri Lanka face multiple barriers when applying to universities:
- **Financial uncertainties**: Exchange rate fluctuations, hidden costs, affordability mismatches
- **Eligibility confusion**: Sri Lankan qualifications (A/Ls, GPA) don't map clearly to international requirements
- **Information fragmentation**: No centralized source; conflicting advice from consultants
- **Document processing**: Manual credential extraction from transcripts/certificates is error-prone
- **Risk management**: No intelligent system to rank universities by feasibility

### Solution: Multi-Agent AI System
A **conversational AI platform** that combines:
- RAG (Retrieval-Augmented Generation) for knowledge-rich responses
- Multiple specialized AI agents that collaborate
- Real-time financial analysis
- Academic eligibility verification
- Intelligent university recommendations

---

## 1.1 RESEARCH POSITIONING FOR MSc THESIS

### Proposed Research Title
**Resource-Aware Multi-Agent Decision Support for International Student Applications**

### Main Contribution
This project can be positioned as a single decision-support system with two operating modes:
- **FULL mode** for richer retrieval, broader indexing, and higher-quality outputs on stronger hardware.
- **LITE mode** for low-performance, low-storage student laptops using reduced OCR image sizes, smaller retrieval settings, curated data only, and free/open-source components.

The core research contribution is not only the multi-agent architecture itself, but the demonstration that the same system can balance quality, speed, and resource usage under realistic student hardware constraints.

### Evaluation Metrics
The system can be evaluated using practical and research-relevant metrics:
- **Latency**: Response time for OCR, eligibility checking, recommendation generation, and chat responses.
- **RAM usage**: Peak and average memory consumption in FULL and LITE modes.
- **Storage footprint**: Disk usage of models, vector indexes, cached data, and processed artifacts.
- **Recommendation relevance**: Quality of ranked university suggestions against known suitable options or expert judgement.
- **Eligibility accuracy**: Agreement between system decisions and manually verified eligibility outcomes.
- **User satisfaction**: Feedback scores from student testers on clarity, usefulness, trust, and ease of use.

### Practical Value
The practical value of the thesis is that it shows how AI-assisted study-abroad guidance can be delivered using free and open-source technologies on low-cost hardware. This makes the work more reproducible, affordable for student developers, and relevant to institutions or communities with limited computing resources.

### Selected Implementation Stack (No Ollama)
For this project iteration, the selected implementation stack is:
- OCR: Tesseract
- Document classifier: TF-IDF + LinearSVC
- Eligibility model: HistGradientBoostingClassifier
- Financial analysis: Rule-based + regression/classifier
- Recommendation: Weighted scoring + small ML reranker
- Embeddings: all-MiniLM-L6-v2
- Vector database: Chroma
- Generator mode: Keyless grounded responses by default (no external model key required)

This configuration is intentionally resource-aware and avoids Ollama dependencies so it can run more reliably on low-performance student hardware.

---

## 2. CORE ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE                           │
│         React Frontend (Vite SPA) - app.jsx                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Server                            │
│              api_server.py (port 8000)                      │
│  • REST endpoints for all agent interactions               │
│  • Document upload & OCR processing                        │
│  • ML model inference                                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   RAG System │  │   Chatbot    │  │  Eligibility │
│ (Retrievals) │  │    Agent     │  │ Verification │
└──────────────┘  └──────────────┘  └──────────────┘
        ↓                  ↓                  ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Financial   │  │   Document   │  │  Recommend- │
│  Feasibility │  │  Processing  │  │    ation    │
│    Agent     │  │    Agent     │  │    Agent    │
└──────────────┘  └──────────────┘  └──────────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ↓
        ┌──────────────────────────────────┐
        │    Unified Data Manager (5 Phases)
        │         (database/*)
        └──────────────────────────────────┘
```

---

## 3. THE 5 CORE AGENTS (MULTI-AGENT SYSTEM)

### Agent 1: **Chatbot Agent** (Conversational Hub)
**Location**: `multiagent/core/agents/chatbot_agent.py`

**Purpose**: 
- Acts as the main conversational interface
- Routes user queries to appropriate specialized agents
- Provides emotional support and guidance
- Answers "why" and "how" questions about the process

**Key Responsibilities**:
- Parse natural language queries
- Determine which agent(s) to invoke
- Aggregate responses from multiple agents
- Provide personalized, context-aware guidance

**Example Interaction**:
```
User: "Can I study in UK with my A/L results and €5000 budget?"
Chatbot Agent:
  1. Calls Eligibility Agent → checks A/L equivalence
  2. Calls Financial Agent → checks £ purchase power vs €5000
  3. Aggregates: "Yes, Foundation programs accept A/Ls. 
                  Your budget fits ~40 universities. 
                  See recommendations..."
```

---

### Agent 2: **Eligibility Verification Agent**
**Location**: `multiagent/core/agents/eligibility_verification_agent.py`

**Purpose**: 
- Maps Sri Lankan qualifications → International requirements
- Checks language proficiency (IELTS, TOEFL)
- Identifies pathways (direct admission vs. Foundation year)
- Validates document credentials

**Key Components**:
- **Eligibility Calculator** (processor)
  - TF-IDF + Naive Bayes classifier
  - Trained on real admission requirements
  - Outputs: Eligibility score (0-100%), pathway recommendations

**Qualification Mapping Logic**:
```
Sri Lankan Qualification  →  International Equivalent
────────────────────────────────────────────────────
A/L (3 subjects, avg 75%)  →  UK Foundation Year (or direct entry)
GPA 3.8/4.0               →  High school certificate equivalent
Diploma (2 years)         →  Associate degree (US) / HND (UK)
```

**Language Proficiency Checks**:
- IELTS ≥ 6.0 for most programs
- TOEFL ≥ 75 for US universities
- Exemptions for native speakers or pre-university programs

---

### Agent 3: **Financial Feasibility Agent**
**Location**: `multiagent/core/agents/financial_feasibility_agent.py`

**Purpose**: 
- Calculate total cost of attendance (tuition + living costs)
- Adjust for **exchange rate fluctuations** (critical for Sri Lankan students)
- Match student budget against university cost profiles
- Suggest scholarships, grants, work-study alternatives

**Financial Calculation Pipeline**:

```
Step 1: Extract Student Budget
  Input: "I have €5000 + can borrow $20,000"
  Output: Converted to common currency (USD): €5000 = $5,400 + $20,000 = $25,400

Step 2: Exchange Rate Modeling
  Input: EUR/USD historical rates (last 2 years)
  Output: Polynomial Ridge Regression predicts future rates
  Purpose: Warn students of currency risks

Step 3: University Cost Database Lookup
  Input: University name + Program
  Output: Annual tuition + estimated living cost
  Example: UC Berkeley Engineering = $46,000 tuition + $22,000 living

Step 4: Feasibility Scoring
  Formula: Student Budget / Total Cost = Feasibility Score
  ≥ 0.8 = Financially feasible
  0.5-0.8 = Marginal (needs scholarship)
  < 0.5 = Not feasible without loan

Step 5: Recommendations
  → Similar universities in affordable countries
  → Scholarship search results
  → Work-study opportunities
```

**Critical Feature: Exchange Rate Prediction**
- Uses **Polynomial Ridge Regression**
- Trains on historical EUR/USD, GBP/USD, INR/USD data
- Predicts 3-6 month ahead trends
- Alerts: "PLN exchange rate may worsen in Q2 2026 → apply to Poland NOW"

---

### Agent 4: **Document Processing (OCR) Agent**
**Location**: `multiagent/core/agents/document_processing_agent.py` + `multiagent/core/processors/ocr_processor.py`

**Purpose**: 
- Extract grades, transcripts, certificates from uploaded documents (PDF/images)
- Automate credential verification
- Populate student profile automatically

**OCR Engine Strategy (Windows-Compatible)**:
```
Priority 1: Tesseract-OCR (fast, low RAM)
  → C:\Program Files\Tesseract-OCR\tesseract.exe
  
Priority 2: EasyOCR (if Tesseract not found)
  → Pure Python, ~100 MB download
  
Priority 3: Error message with install guide
```

**Field Extraction Logic**:
```
Input Document: A/L Certificate (PDF)
     ↓
OCR reads text
     ↓
TF-IDF + Naive Bayes Classifier identifies fields:
  - Student ID
  - Exam Year
  - Subject 1 (grade): "A" → 90%
  - Subject 2 (grade): "B" → 80%
  - Subject 3 (grade): "A*" → 95%
     ↓
Extracted Profile: {
  "qualification": "GCE A/L",
  "year": 2023,
  "subjects": ["Mathematics", "Physics", "Chemistry"],
  "grades": [90, 80, 95],
  "gpa": 3.8
}
```

**ML Classifier Details**:
- **Algorithm**: TF-IDF Vectorizer + Multinomial Naive Bayes
- **Purpose**: Distinguish between document types (A/L vs. Diploma vs. GPA transcript)
- **Confidence Threshold**: 40% (tunable)
- **Cross-validation**: Prevents overfitting on limited training data

---

### Agent 5: **Recommendation Agent**
**Location**: `multiagent/core/agents/recommendation_agent.py`

**Purpose**: 
- Rank and prioritize universities based on **multi-criteria analysis**
- Consider: eligibility, financial feasibility, deadlines, risk
- Provide personalized top-N recommendations

**Ranking Algorithm** (Weighted Multi-Criteria Decision Analysis):

```
For each University in database:
  
  Score = (
    w1 × Eligibility_Score +      [40% weight]
    w2 × Financial_Feasibility +  [35% weight]
    w3 × Deadline_Risk +          [15% weight]
    w4 × Scholarship_Probability  [10% weight]
  )
  
where:
  Eligibility_Score = 0-100% (from Eligibility Agent)
  Financial_Feasibility = Student_Budget / Total_Cost
  Deadline_Risk = Days_Remaining / Avg_Processing_Time
  Scholarship_Probability = RandomForest prediction (see ML models)
```

**Random Forest for Scholarship Prediction**:
- **Training Data**: Historical scholarship awards + student profiles
- **Features**: GPA, IELTS score, program competitiveness, country of origin
- **Output**: P(scholarship) = probability student gets aid
- **Use Case**: Weight recommendations by scholarship likely

**Recommendation Output Example**:
```
Your Top 5 Universities:

1. University of Helsinki (Finland)
   Score: 92/100
   ✓ Eligible: Yes (A/L → BSc pathway)
   ✓ Affordable: Yes (€9,000/year + free tuition for EU-like programs)
   ✓ Deadline: 8 weeks away
   ✓ Scholarship: 45% probability

2. UC Dublin (Ireland)
   Score: 88/100
   ✓ Eligible: Yes (Foundation needed)
   ⚠ Affordable: Marginal (needs €2,000 scholarship)
   ✓ Deadline: 12 weeks away
   ✓ Scholarship: 60% probability
```

---

## 4. DATA LAYER: UNIFIED 5-PHASE DATABASE MANAGER

**Location**: `multiagent/core/database/manager.py` + `phase*.py` files

### Why 5 Phases?
Each phase solves a specific data problem:

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: Curated Database (Static, High-Quality)          │
│  File: phase1_curated_db.py                                │
│  ────────────────────────────────────────────────────────  │
│  • Hand-verified universities (100-500 institutions)       │
│  • Accurate tuition, entry requirements                    │
│  • Updated 2x yearly by administrators                     │
│  • High confidence: 95%+                                   │
│                                                              │
│  Data Format: /data/databases/universities_database.json   │
│  {                                                           │
│    "university_id": 1,                                      │
│    "name": "University of Cambridge",                       │
│    "country": "UK",                                         │
│    "tuition_gbp": 35000,                                    │
│    "entry_requirements": {                                  │
│      "a_level_gpa": 3.8,                                    │
│      "ielts_score": 7.0,                                    │
│      "programs": ["Engineering", "Medicine", ...]          │
│    }                                                         │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: Web Scraper (Dynamic, Broad Coverage)            │
│  File: phase2_web_scraper.py                               │
│  ────────────────────────────────────────────────────────  │
│  • Scrapes university websites for latest info             │
│  • Discovers new universities (can reach 5000+)            │
│  • Extracts: tuition, deadlines, scholarship details       │
│  • Runs daily/weekly to catch updates                      │
│  • Confidence: 60-80% (websites often outdated)            │
│                                                              │
│  Challenge: Inconsistent data formats → uses NLP cleanup   │
│  Example: "$40,000" vs "40,000 USD" vs "€35,000"          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  PHASE 3: API Integration (Real-Time, Official)            │
│  File: phase3_api_integration.py                           │
│  ────────────────────────────────────────────────────────  │
│  • Connects to official university APIs                    │
│  • Example: UCAS (UK), Common App (US)                     │
│  • Real-time deadline extensions, policy changes          │
│  • Confidence: 90%+ (official sources)                     │
│  • Updates: Instant                                        │
│                                                              │
│  APIs Integrated:                                           │
│  • UCAS Track (UK deadline tracking)                       │
│  • Common App API (US/International)                       │
│  • Nuffic (Dutch higher ed directory)                      │
│  • IDP Connect (Australian universities)                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  PHASE 4: Scheduler (Predictive, Event-Driven)             │
│  File: phase4_scheduler.py                                 │
│  ────────────────────────────────────────────────────────  │
│  • Predicts when to refresh data for each university       │
│  • GradientBoosting model: When is deadline likely to      │
│    change? When does tuition typically update?             │
│  • Optimizes API calls (limited quota)                     │
│  • Prioritizes universities by student interest            │
│                                                              │
│  Example Decision:                                          │
│  "Cambridge hasn't updated deadline in 3 years            │
│   → Check only 2x/year. Boston University updates monthly  │
│   → Check weekly."                                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  PHASE 5: Override Manager + ML Anomaly Detection          │
│  File: phase5_override_manager_ml.py                       │
│  ────────────────────────────────────────────────────────  │
│  Purpose: Handle conflicting/suspicious data               │
│                                                              │
│  Example Problem:                                           │
│  Phase 2 scraper says: "University of Tokyo tuition $100"  │
│  Phase 1 curated says: "$27,000"                           │
│  Phase 3 API says: "$27,500"                               │
│  → Which is correct?                                       │
│                                                              │
│  Solution: IsolationForest Anomaly Detector                │
│  • Detects outliers in tuition across all phases           │
│  • Flags suspicious updates (10x change overnight)         │
│  • Weights: If 2/3 sources agree → use majority           │
│  • Confidence scoring: 0.3-0.9                             │
│                                                              │
│  User Overrides:                                            │
│  /data/overrides/active_overrides.json                     │
│  {                                                           │
│    "university_id": 5,                                      │
│    "field": "tuition_gbp",                                 │
│    "value": 29500,                                         │
│    "reason": "New scholarship program reduces cost",       │
│    "override_date": "2026-03-20"                           │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow Example

```
Student Opens App
  ↓
RAGSystem loads indexed universities from ChromaDB
  ↓
Chatbot queries: "Universities under €10,000"
  ↓
Unified Manager fetches from 5 Phases:
  Phase 1: Curated DB (100 universities, 95% confidence)
  Phase 2: Recent scrapes (250 universities, 70% confidence)
  Phase 3: API calls (50 live updates from official sources, 95% conf)
  Phase 4: Predictor suggests which to refresh
  Phase 5: Anomaly detector flags $100 tuition as suspicious
  ↓
Merged Result (deduplicated, highest-confidence sources prioritized):
  325 universities under €10,000
  ↓
Recommendation Agent ranks by student profile
```

---

## 5. RETRIEVAL-AUGMENTED GENERATION (RAG) SYSTEM

**Location**: `multiagent/core/rag_system.py`

### What is RAG?
Instead of just using Gemini 2.5 Flash (the AI model), we **augment** it with real data:

```
Traditional LLM (Gemini alone):
  User: "What's the IELTS requirement for Cambridge?"
  Gemini: "Cambridge typically requires IELTS 7.0..." 
  → Problem: Hallucination risk, outdated info

RAG System (Gemini + Knowledge Base):
  User: "What's the IELTS requirement for Cambridge?"
    ↓
  ChromaDB Retriever searches indexed documents
    ↓
  Finds 3 most similar chunks:
    - "Cambridge IELTS requirement: 7.0 (Engineering)"
    - "Cambridge IELTS requirement: 6.5 (Business)"  
    - "Cambridge application deadline: March 15 2026"
    ↓
  Augmented Prompt to Gemini:
    "Based on these verified sources: [chunks]
     Answer: What's IELTS requirement for Cambridge?"
    ↓
  Gemini: "Cambridge requires IELTS 7.0 for Engineering,
           6.5 for Business, verified as of 2026."
  → Trustworthy, grounded in real data
```

### RAG Components

```
1. ChromaDB Vector Store
   ────────────────────
   • Stores embeddings of all documents
   • Location: Persistent disk storage
   • Content indexed:
     - Phase 1 curated university data
     - Phase 2 scraped pages
     - Phase 3 API responses
     - User-uploaded documents (transcripts, etc.)
     - FAQs and guidance docs

2. Sentence Transformers
   ──────────────────────
   • Converts text → semantic embeddings (384-dim vector)
   • Model: all-MiniLM-L6-v2 (compact, fast)
   • Same embeddings for both documents and queries
   • Enables similarity search: "What's the math requirement?"
     finds "Mathematics GPA ≥ 3.5 required"

3. Semantic Retriever
   ───────────────────
   • Takes user query → embedding
   • Searches ChromaDB for k=5 most similar chunks
   • Returns: Relevant university data, deadlines, requirements
   • Similarity scoring: 0-1 (higher = more relevant)

4. Gemini 2.5 Flash Generator
   ────────────────────────────
   • Takes user query + retrieved chunks
   • Generates natural language response
   • Context window: 1M tokens (can handle full university data)
   • Cost-efficient: 0.075₹/1M input tokens
```

### Example RAG Flow

```
User Query: "I have 65% in A/Ls. Where can I study?"
  ↓
Step 1: Convert to embedding
  embedding = SentenceTransformer.encode(query)
  ↓
Step 2: Search ChromaDB
  results = chromadb.query(embedding, top_k=5)
  Results:
    - "Foundation Year accepts 50-70% A/L students"
    - "UK universities requirement matrix: A/L 70%+ → BSc Direct"
    - "Boston University accepts 60% GPA equivalent to A/L 65%"
    - "Singapore universities: A/L 75%+ recommended"
    - "Malaysia: No A/L equivalent needed, STPM or Diploma"
  ↓
Step 3: Augment prompt to Gemini
  prompt = f"""
    Based on these verified sources:
    {retrieved_chunks}
    
    User's query: "I have 65% in A/Ls. Where can I study?"
    
    Provide 3-5 specific options with:
    1. University name
    2. Program pathway (Foundation / Direct / Diploma pathway)
    3. Next steps
  """
  ↓
Step 4: Gemini responds
  "With your 65% A/L results, consider:
   1. **UK Foundation Programs** - Accept 65%+ A/Ls
      Universities: University of London, Into University, etc.
   
   2. **Malaysian Universities** - No A/L requirement
      Universities: UCSI, Sunway, Monash Malaysia
   
   3. **US Community Colleges** - 2+2 pathway
      Transfer after 2 years to top universities
   
   Recommendation: Apply to Foundation now (deadline March 2026)"
  ↓
User gets grounded, specific, actionable advice
```

---

## 6. MACHINE LEARNING MODELS (ML STACK)

### Overview: 5 ML Models for Different Purposes

```
┌──────────────────────────────────────────────────────────────┐
│  MODEL 1: Document Classification (TF-IDF + Naive Bayes)    │
│  ────────────────────────────────────────────────────────   │
│  Purpose: Identify document type from OCR text               │
│  Location: api_server.py (OCR processor)                     │
│  Input: Raw text from Tesseract/EasyOCR                      │
│  Output: Class (A/L Cert, GPA Transcript, Diploma, etc.)    │
│  Accuracy: ~94% (depending on document quality)              │
│                                                               │
│  Algorithm:                                                  │
│  1. TF-IDF Vectorizer: Convert text → sparse numeric vector │
│  2. Multinomial Naive Bayes: P(class | features)             │
│  3. Confidence threshold: 40%                                │
│                                                               │
│  Example:                                                    │
│  Input: Scanned A/L certificate text                         │
│         "General Certificate of Education Advanced Level...  │
│          Subject: Mathematics Grade: A..."                   │
│  ↓                                                            │
│  TF-IDF extracts key terms: {"A/L", "Mathematics", "Grade"}  │
│  ↓                                                            │
│  Naive Bayes: P(A/L Document | features) = 0.96             │
│  ↓                                                            │
│  Output: "A/L Certificate" (96% confidence)                  │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  MODEL 2: Admission Probability (Random Forest)              │
│  ────────────────────────────────────────────────────────   │
│  Purpose: Predict if student will be admitted               │
│  Location: recommendation_agent.py                           │
│  Training Data: Historical admissions (5000+ records)        │
│                                                               │
│  Features:                                                   │
│  • Student GPA (0-4.0)                                       │
│  • IELTS/TOEFL score                                         │
│  • Program competitiveness (engineering vs art)             │
│  • Student country of origin (nationality bias)             │
│  • Test scores (SAT/ACT equivalent)                         │
│                                                               │
│  Output: P(admission) = 0-1 probability                      │
│  Example: "With your profile, 78% chance UC Berkeley admits" │
│                                                               │
│  Use Case:                                                   │
│  Filter recommendations: Only show universities where       │
│  P(admission) > 60% ("realistic" targets)                    │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  MODEL 3: Exchange Rate Prediction (Polynomial Ridge)        │
│  ────────────────────────────────────────────────────────   │
│  Purpose: Forecast future exchange rates                     │
│  Location: financial_feasibility_agent.py                    │
│  Training Data: 2 years daily EUR/USD, GBP/USD, etc.         │
│                                                               │
│  Algorithm: Polynomial Regression with Ridge regularization  │
│  • Polynomial degree: 3 (captures non-linear trends)         │
│  • Ridge alpha: 0.1 (prevents overfitting)                   │
│  • Forecast horizon: 3-6 months ahead                        │
│                                                               │
│  Example:                                                    │
│  Input: EUR/USD exchange rates (last 730 days)              │
│  Training: Fit polynomial curve to historical data           │
│  Forecast: "EUR/USD likely to be 1.08 in June 2026"         │
│  Alert: "Will worsen by 2%; consider strong USD universities"│
│                                                               │
│  Output: Risk assessment                                     │
│  - "₹ to USD weakening" → Warn Indian students               │
│  - "€ to GBP strengthening" → Recommend Germany/Ireland      │
│  - "LKR extremely volatile" → Encourage scholarships only    │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  MODEL 4: Refresh Scheduler (Gradient Boosting)              │
│  ────────────────────────────────────────────────────────   │
│  Purpose: Predict when university data needs updating       │
│  Location: database/phase4_scheduler.py                      │
│  Training Data: Historical university data change patterns    │
│                                                               │
│  Features:                                                   │
│  • University age (founded when?)                            │
│  • Past update frequency                                     │
│  • Season (deadlines change per academic cycle)              │
│  • Regional patterns (UK vs US update patterns differ)       │
│  • Number of students interested (priority weighting)        │
│                                                               │
│  Algorithm: Gradient Boosting classifier                     │
│  Output: "Refresh this university? [Yes/No] with confidence" │
│                                                               │
│  Example Schedule:                                           │
│  - Top 50 universities by interest: Check weekly             │
│  - Tier 2 universities: Check monthly                        │
│  - Niche universities: Check quarterly                       │
│  - Static universities (no recent changes): Check annually   │
│                                                               │
│  Benefit: Reuses API quotas efficiently                      │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  MODEL 5: Anomaly Detection (Isolation Forest)               │
│  ────────────────────────────────────────────────────────   │
│  Purpose: Flag suspicious/conflicting data from 5 phases     │
│  Location: database/phase5_override_manager_ml.py            │
│  Training Data: Historical university tuition trends         │
│                                                               │
│  Algorithm: Isolation Forest                                 │
│  • Unsupervised anomaly detection                            │
│  • Isolates outliers in multivariate space                   │
│  • Features: tuition, living cost, market shifts, etc.       │
│                                                               │
│  Detects:                                                    │
│  • Tuition change >30% overnight → Flag as anomaly            │
│  • Conflicting data between phases → Raise uncertainty       │
│  • Impossible values → Scraper error detected               │
│                                                               │
│  Example:                                                    │
│  Phase 1 (curated): Tuition £30,000                         │
│  Phase 2 (scraper): Tuition £300 (10x drop)                 │
│  Phase 3 (API): Tuition £29,500                             │
│  ↓                                                            │
│  Isolation Forest: Detect £300 as anomaly                    │
│  ↓                                                            │
│  Decision: Use £29,500 from Phase 3 (verified API)           │
│  Confidence: 0.85 (high — 2/3 sources agree)                │
└──────────────────────────────────────────────────────────────┘
```

---

## 7. TECHNOLOGY STACK

### Backend
```
Framework:       FastAPI (async REST API)
Language:        Python 3.11+
Dependencies:
  - langchain, langchain-google-genai  → LLM orchestration
  - chromadb                            → Vector database
  - sentence-transformers               → Embeddings
  - google-generativeai                 → Gemini 2.5 Flash API
  - scikit-learn                        → ML models (TF-IDF, NB, RF, etc.)
  - pytesseract / easyocr              → Document OCR
  - faiss-cpu                           → Fast similarity search
  - unstructured, pypdf                → Document parsing
```

### Frontend
```
Framework:       React 18+ with Vite (SPA)
Language:        JSX / JavaScript ES6+
Build Tool:      Vite (fast dev server, optimized prod builds)
Location:        multiagent/app.jsx, multiagent/main.jsx
```

### Data & ML
```
Vector Store:    ChromaDB (persistent embeddings)
Embeddings:      Sentence Transformers (all-MiniLM-L6-v2)
ML Library:      scikit-learn (algorithms)
Database:        JSON files (universities_database.json, overrides)
Storage Dirs:
  - data/databases/       → University data
  - data/ml_models/       → Trained models (.pkl files)
  - data/overrides/       → User corrections
  - data/backups/         → Backup data snapshots
```

### APIs & External Services
```
Google Gemini:       LLM for response generation
UCAS:              UK university tracking  
Common App:        US/International applications
Nuffic:            Dutch universities directory
IDP Connect:       Australian universities
```

---

## 8. HOW EVERYTHING WORKS TOGETHER: END-TO-END FLOW

### Scenario: Student Profile Creation & Recommendation

```
STEP 1: Student Uploads Documents
  ├─ A/L Certificate (PDF)
  ├─ GPA Transcript (Image)
  └─ IELTS Result (PDF)
       ↓
STEP 2: Document Processing Agent (OCR)
  ├─ Tesseract/EasyOCR extracts text from images
  ├─ TF-IDF + Naive Bayes classifier identifies document types
  ├─ Field extractor pulls: grades, scores, dates
  └─ Output: Student Profile (GPA 3.7, IELTS 6.5, A/L subjects)
       ↓
STEP 3: Eligibility Verification Agent
  ├─ Maps A/L subjects → International equivalent (Stage 1/2/3 systems)
  ├─ Checks IELTS score (6.5 → OK for most, not Elite universities)
  ├─ Eligibility calculator (TF-IDF + NB classifier) determines pathways
  └─ Output: "Foundation + Direct entry options available"
       ↓
STEP 4: Student Inputs Financial Details
  ├─ Input: "I have €8,000 saved + can borrow $15,000"
  ├─ Conversion: €8,000 = $8,640 (using current rate)
  └─ Total Available: $23,640
       ↓
STEP 5: Financial Feasibility Agent
  ├─ Converts to reference currency (USD)
  ├─ Polynomial Ridge predicts exchange rates (warn if unfavorable)
  ├─ Lookup university tuition from 5-phase database
  │  Example: Cambridge ~$46,000 → Financial score: 23,640/46,000 = 51% (marginal)
  ├─ Flags scholarship-dependent universities
  └─ Output: "Affordable universities in Ireland, Malaysia, Poland"
       ↓
STEP 6: RAG System Augments Knowledge
  ├─ ChromaDB retrieves relevant university data
  ├─ Query: "Best universities for €8,000 budget + Foundation pathway"
  ├─ Returns: 5 most similar chunks from indexed data
  └─ Augmented context sent to Gemini
       ↓
STEP 7: Recommendation Agent Ranks Universities
  ├─ Weighted scoring formula:
  │  Score = 0.4×Eligibility + 0.35×Financial + 0.15×Deadline + 0.1×Scholarship
  │
  ├─ For each university:
  │   - Eligibility Score: 85% (Foundation available)
  │   - Financial Score: (8,640 / tuition)
  │   - Random Forest predicts: P(scholarship) = 65%
  │   - Deadline Risk: Days remaining / typical processing time
  │
  ├─ Sort by total score (highest = best fit)
  └─ Filter by P(admission) > 60% (realistic targets only)
       ↓
STEP 8: Chatbot Agent Delivers Personalized Recommendations
  ├─ Conversational response:
  │  "Based on your profile (3.7 GPA, IELTS 6.5, €8k budget):
  │
  │   🎓 TOP PICKS (Highly Feasible):
  │   1. University of Helsinki (Finland)
  │      - Foundation: Yes | Cost: €9,000/yr | Scholarship: 45%
  │      - Action: Apply now (deadline April 15)
  │
  │   2. Trinity College Dublin (Ireland)
  │      - Foundation: Yes | Cost: €15,000/yr | Scholarship: 60%
  │      - Action: Apply soon (deadline May 1)
  │
  │   ⚠️  REACH UNIVERSITIES (May need scholarship):
  │   3. UC Dublin (Needs scholarship for full funding)
  │
  │   💡 FINANCIAL TIPS:
  │   - Polish zloty expected to strengthen (March 2026 forecast)
  │   - Consider Poland universities: cheaper, strong backing
  │   - Target scholarship deadline (March 31) for best funding"
  │
  └─ Student bookmarks recommendations
       ↓
STEP 9: Continuous Monitoring (Tasks in Background)
  ├─ Phase 4 Scheduler: Checks for deadline changes weekly
  ├─ Phase 5 Anomaly Detector: Flags suspicious tuition updates
  ├─ Phase 2 Web Scraper: Refreshes university websites
  ├─ Phase 3 API Integration: Pulls real-time data (UCAS, Common App)
  └─ Chatbot alerts: "Trinity deadline extended to May 15!"
```

---

## 9. KEY DESIGN PATTERNS & CONCEPTS

### 1. **Multi-Agent Collaboration**
- Each agent is independent but communicates through the Chatbot Agent
- No hard-coding of agent flows — orchestration is done by the Chatbot
- Agents can be updated/replaced without affecting others

### 2. **Data Quality Pyramid**
```
Level 1 (Highest Trust): Curated DB (admin-verified)
Level 2: API calls (official sources)
Level 3: Recent scraped data (automated but potentially stale)
Level 4: User overrides (user corrections)
Level 5 (Anomaly detection): Flag conflicts → human review
```

### 3. **RAG for Reliability**
- Never trust LLM alone → always ground responses in real data
- ChromaDB acts as the "source of truth"
- Gemini generates natural language, but facts come from indexed data

### 4. **ML for Decision Support**
- Models predict probabilities, not certainties
- Always present uncertainty: "45% probability" not "will get scholarship"
- Thresholds are tunable (confidence, feasibility, etc.)

### 5. **Financial Forecasting Under Uncertainty**
- Exchange rates are volatile → Polynomial Ridge captures trends
- Alerts warn students of currency risks
- Recommendations weighted by financial feasibility, not just eligibility

---

## 10. FILE STRUCTURE & QUICK REFERENCE

```
multiagent/
├── api_server.py                    ← FastAPI backend (start here)
├── app.jsx                          ← React frontend UI
├── main.jsx                         ← Vite entry point
├── index.html                       ← HTML template
├── vite.config.js                   ← Vite configuration
├── package.json                     ← Frontend dependencies
├── requirement.txt                  ← Python dependencies (backend)
│
├── core/
│   ├── rag_system.py               ← ChromaDB + Sentence Transformers
│   │
│   ├── agents/
│   │   ├── chatbot_agent.py        ← Orchestrator agent
│   │   ├── eligibility_verification_agent.py
│   │   ├── financial_feasibility_agent.py
│   │   ├── document_processing_agent.py
│   │   └── recommendation_agent.py
│   │
│   ├── processors/
│   │   ├── ocr_processor.py        ← Tesseract/EasyOCR wrapper
│   │   └── eligibility_calculator.py
│   │
│   └── database/
│       ├── manager.py              ← Unified data manager (5 phases)
│       ├── phase1_curated_db.py
│       ├── phase2_web_scraper.py
│       ├── phase3_api_integration.py
│       ├── phase4_scheduler.py
│       └── phase5_override_manager_ml.py
│
└── data/
    ├── databases/
    │   └── universities_database.json   ← Master university data
    ├── ml_models/                       ← Trained .pkl files
    ├── overrides/                       ← User corrections
    ├── backups/                         ← Snapshots
    ├── scraped/                         ← Phase 2 web scraper output
    └── training/                        ← ML training data
```

---

## 11. TO RUN THE APPLICATION

### Backend
```bash
cd d:\Multiagent
.\.venv\Scripts\Activate.ps1
uvicorn multiagent.api_server:app --host 0.0.0.0 --port 8000
```

### Frontend (New Terminal)
```bash
cd d:\Multiagent\multiagent
npm install
npm run dev
```

### Access UI
```
http://localhost:5173 (dev)
or http://localhost:8000 (prod)
```

---

## 12. KEY INSIGHTS FOR MSC WORK

### Research Opportunities
1. **Improve Eligibility Mapping**: Engineer better A/L → international qualification equivalences
2. **Exchange Rate Forecasting**: Compare polynomial regression vs LSTM/transformer models
3. **Fairness in Recommendations**: Audit for bias in Random Forest admission predictions
4. **Scalability**: Migrate Phase 2 scraper from BeautifulSoup to distributed architecture
5. **Zero-Shot Learning**: Use Gemini to classify documents without retraining ML models

### Thesis Topics
- "Multi-Agent Systems for EdTech: Design Patterns and Scalability"
- "RAG-Enhanced LLMs for Domain-Specific Advice: A Case Study in Higher Education"
- "Exchange Rate Prediction for International Student Financial Planning"
- "Anomaly Detection in Heterogeneous Data Sources: University Datasets"

### Performance Monitoring
- Document OCR accuracy (target: >94%)
- RAG retrieval relevance (similarity scores)
- Recommendation ranking correlation with actual admissions
- API response latency (should be <2s for recommendations)

---

**This is your comprehensive guide to understanding the MultiAgent International Study Advisor system!**
