"""
UniAssist — core/phases/ package
=================================
One file per phase. Each file contains both the data layer and its ML engine.

    phase1.py  →  CuratedDatabaseManager  +  Phase1MLEngine  (Random Forest)
    phase2.py  →  WebScraper              +  Phase2AnomalyDetector (Isolation Forest)
    phase3.py  →  APIIntegrator           +  Phase3MLEngine  (Polynomial Regression)
    phase4.py  →  UpdateScheduler         +  Phase4MLEngine  (Gradient Boosting)
    phase5.py  →  OverrideManager         +  3 ML modules    (IF + TF-IDF/NB + Z-score)

Quick imports:
    from core.phases.phase1 import Phase1
    from core.phases.phase2 import Phase2
    from core.phases.phase3 import Phase3
    from core.phases.phase4 import Phase4
    from core.phases.phase5 import Phase5

Each Phase class is the single entry point:
    p1 = Phase1()    # DB auto-loads, ML auto-trains on first run
    p2 = Phase2()    # Scraper + anomaly detector ready
    p3 = Phase3()    # API integrator + exchange rate forecaster ready
    p4 = Phase4(manager=mgr)  # Scheduler + smart skip/run ML ready
    p5 = Phase5()    # Override manager + all 3 ML modules ready
"""

from .phase1 import Phase1, CuratedDatabaseManager, Phase1MLEngine
from .phase2 import Phase2, WebScraper, Phase2AnomalyDetector, score_scraped_record
from .phase3 import Phase3, APIIntegrator, Phase3MLEngine
from .phase4 import Phase4, UpdateScheduler, Phase4MLEngine
from .phase5 import Phase5, AnomalyDetector, OverrideClassifier, SuggestionEngine

__all__ = [
    "Phase1", "Phase2", "Phase3", "Phase4", "Phase5",
    "CuratedDatabaseManager", "Phase1MLEngine",
    "WebScraper", "Phase2AnomalyDetector", "score_scraped_record",
    "APIIntegrator", "Phase3MLEngine",
    "UpdateScheduler", "Phase4MLEngine",
    "AnomalyDetector", "OverrideClassifier", "SuggestionEngine",
]