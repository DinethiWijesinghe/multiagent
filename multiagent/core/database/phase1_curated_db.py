"""
Phase 1: Curated University Database + ML Eligibility Engine
=============================================================
Location: multiagent/core/phases/phase1.py

Combines:
  - CuratedDatabaseManager  → load/save/query universities_database.json
  - Phase1MLEngine           → Random Forest acceptance probability + ranking

Why ML here:
  Without ML: pass/fail eligibility check only (GPA >= min? yes/no)
  With ML:    acceptance probability per university, ranked best-fit list,
              student tier classification — far more useful for students

Usage:
    from core.phases.phase1 import Phase1

    p1 = Phase1()
    unis = p1.db.get_universities_by_country("UK")

    # ML: rank all universities for a student
    ranked = p1.ml.rank_universities(student, unis)

    # ML: classify overall student strength
    tier = p1.ml.classify_student_tier(student)

    # ML: single acceptance probability
    result = p1.ml.predict_acceptance(student, university)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("[Phase1] scikit-learn not found — run: pip install scikit-learn")

logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────

_ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DB_PATH   = os.path.join(_ROOT, "data", "databases", "universities_database.json")
_MODEL_DIR = os.path.join(_ROOT, "data", "ml_models")

# ─── Feature names ────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "student_gpa", "student_ielts", "student_toefl",
    "uni_min_gpa", "uni_min_ielts", "uni_min_toefl",
    "gpa_gap", "ielts_gap", "toefl_gap",
    "qs_rank_normalised", "budget_ratio",
    "stream_match", "has_scholarship",
]

# ─── Synthetic training data ──────────────────────────────────────────────────

def _generate_training_data(n: int = 2000):
    np.random.seed(42)
    # (min_gpa, min_ielts, min_toefl, qs_rank, tuition_usd)
    unis = [
        (3.9,7.5,110,2,45000),(3.9,7.5,110,3,45000),(3.7,6.5,96,6,50000),
        (3.7,6.5,92,9,40000),(3.5,6.5,92,27,36000),(3.3,6.5,90,32,30000),
        (3.5,7.0,100,40,38000),(3.7,7.0,107,45,38000),(3.3,6.5,88,55,28000),
        (3.3,6.5,88,67,28000),(3.7,6.0,85,8,22000),(3.5,6.0,85,15,20000),
        (3.3,6.5,90,500,28000),(3.3,6.0,85,200,20000),(3.0,6.0,79,600,12000),
        (3.3,6.5,79,14,40000),(3.3,6.5,85,18,28000),(3.3,6.5,80,30,25000),
        (3.3,6.5,85,19,26000),(3.0,6.5,79,37,24000),(3.0,6.5,79,43,24000),
        (3.0,6.5,82,72,22000),(3.0,6.5,79,89,21000),(3.0,6.5,79,140,22000),
        (3.0,6.5,83,195,20000),(2.8,6.0,79,266,19000),(2.8,6.0,79,307,17000),
        (2.8,6.0,79,351,17000),(3.0,6.0,80,185,19000),(2.8,6.0,79,197,18000),
    ]
    budgets = [15000,25000,35000,50000,70000]
    X, y = [], []
    for _ in range(n):
        sg  = np.random.triangular(1.5,3.2,4.0)
        use_ielts = np.random.random() > 0.2
        si  = np.random.triangular(5.0,6.5,9.0) if use_ielts else 0.0
        st  = 0 if use_ielts else np.random.randint(60,120)
        sm  = int(np.random.random() > 0.25)
        bud = np.random.choice(budgets)
        u   = unis[np.random.randint(0,len(unis))]
        mg,mi,mt,qs,tui = u
        gg  = sg - mg
        ig  = (si-mi) if use_ielts else 0.0
        tg  = (st-mt) if not use_ielts else 0.0
        qn  = min(qs,500)/500.0
        br  = bud/max(tui,1)
        hs  = int(br<0.8 and sg>=3.5)
        X.append([sg,si,st,mg,mi,mt,gg,ig,tg,qn,br,sm,hs])
        eng_ok = (use_ielts and si>=mi) or (not use_ielts and st>=mt)
        gpa_ok = sg >= mg - 0.05
        if not eng_ok or not sm:
            label = 0
        elif gpa_ok and eng_ok:
            prob = np.clip(0.55 + gg*0.3 + br*0.1, 0.05, 0.95)
            label = int(np.random.random() < prob)
        else:
            label = 0
        y.append(label)
    return np.array(X), np.array(y)


# ══════════════════════════════════════════════════════════════════════════════
#  CURATED DATABASE MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class CuratedDatabaseManager:
    """
    Loads and queries universities_database.json.
    30 real universities: UK(10), Singapore(5), Australia(15).
    """

    def __init__(self, db_path: str = _DB_PATH):
        self.db_path  = db_path
        self.database: Dict[str, List[Dict]] = {}
        self._load()

    def _load(self):
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                self.database = json.load(f)
            total = sum(len(v) for v in self.database.values())
            print(f"✅ Phase1 DB: Loaded {total} universities from {self.db_path}")
        except FileNotFoundError:
            print(f"⚠️  Phase1 DB: {self.db_path} not found — starting empty")
            self.database = {}
        except json.JSONDecodeError as e:
            logger.error("Phase1 DB corrupt: %s", e)
            self.database = {}

    def save_database(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.database, f, indent=2, ensure_ascii=False)
        print(f"✅ Phase1 DB: Saved to {self.db_path}")

    def get_universities_by_country(self, country: str) -> List[Dict]:
        return self.database.get(country, [])

    def get_university_by_id(self, uni_id: str) -> Optional[Dict]:
        for unis in self.database.values():
            for uni in unis:
                if uni.get("id") == uni_id:
                    return uni
        return None

    def get_all_universities(self) -> List[Dict]:
        result = []
        for unis in self.database.values():
            result.extend(unis)
        return result

    def get_statistics(self) -> Dict:
        stats = {"total": 0, "by_country": {}}
        for country, unis in self.database.items():
            stats["by_country"][country] = len(unis)
            stats["total"] += len(unis)
        return stats


# ══════════════════════════════════════════════════════════════════════════════
#  ML ENGINE — Random Forest
# ══════════════════════════════════════════════════════════════════════════════

class Phase1MLEngine:
    """
    Random Forest classifier for acceptance probability prediction.

    What it adds over rule-based checks:
      - Probability score instead of pass/fail (87% vs just "eligible")
      - Ranks all 30 universities by best fit for student
      - Classifies student tier (Elite/Strong/Competitive/Borderline)
      - Considers all features together (GPA + IELTS + budget + stream + ranking)
    """

    def __init__(self):
        self.model   = None
        self.scaler  = StandardScaler() if ML_AVAILABLE else None
        self.trained = False
        self._try_load()

    def _try_load(self):
        mp = os.path.join(_MODEL_DIR, "phase1_rf_model.pkl")
        sp = os.path.join(_MODEL_DIR, "phase1_scaler.pkl")
        if ML_AVAILABLE and os.path.exists(mp) and os.path.exists(sp):
            try:
                with open(mp,"rb") as f: self.model  = pickle.load(f)
                with open(sp,"rb") as f: self.scaler = pickle.load(f)
                self.trained = True
                print("[Phase1-ML] Loaded Random Forest model from disk")
            except Exception: pass

    def train(self, verbose: bool = True) -> dict:
        if not ML_AVAILABLE:
            return {"error": "scikit-learn not available"}
        X, y = _generate_training_data(2000)
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
        Xtr_s = self.scaler.fit_transform(Xtr)
        Xte_s = self.scaler.transform(Xte)
        self.model = RandomForestClassifier(
            n_estimators=150, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        self.model.fit(Xtr_s, ytr)
        self.trained = True
        acc = accuracy_score(yte, self.model.predict(Xte_s))
        os.makedirs(_MODEL_DIR, exist_ok=True)
        with open(os.path.join(_MODEL_DIR,"phase1_rf_model.pkl"),"wb") as f: pickle.dump(self.model,f)
        with open(os.path.join(_MODEL_DIR,"phase1_scaler.pkl"),"wb") as f: pickle.dump(self.scaler,f)
        if verbose: print(f"[Phase1-ML] Trained | Accuracy: {acc:.1%}")
        return {"accuracy": round(acc,4), "train_samples": len(Xtr), "trained_at": datetime.now().isoformat()}

    def _features(self, student: dict, university: dict) -> np.ndarray:
        ac  = university.get("acceptance_criteria", {})
        mg  = ac.get("min_grade_point", 3.0)
        mi  = ac.get("ielts_min", 6.5)
        mt  = ac.get("toefl_min", 80)
        qs  = (university.get("rankings") or {}).get("qs_world") or 600
        tui = (university.get("tuition") or {})
        tui = tui.get("undergraduate_intl_gbp") or tui.get("undergraduate_intl_sgd") or tui.get("undergraduate_intl_aud") or 30000
        sg  = float(student.get("gpa", 0))
        si  = float(student.get("ielts", 0))
        st  = float(student.get("toefl", 0))
        bud = float(student.get("budget_usd", 30000))
        use = si > 0
        compat = {
            "Physical Science": ["Engineering","Computer Science","Science","Medicine","IT"],
            "Bio Science":      ["Medicine","Science","Engineering"],
            "Commerce":         ["Business","Law","IT","Arts"],
            "Arts":             ["Arts","Law","Social Science","Education","Business"],
            "Technology":       ["Engineering","IT","Computer Science"],
        }
        sm = int(student.get("program","") in compat.get(student.get("stream",""),[]))
        gg = sg - mg
        ig = (si-mi) if use else 0.0
        tg = (st-mt) if not use else 0.0
        qn = min(qs,500)/500.0
        br = bud/max(tui,1)
        hs = int(br<0.8 and sg>=3.5)
        return np.array([[sg,si,st,mg,mi,mt,gg,ig,tg,qn,br,sm,hs]])

    def predict_acceptance(self, student: dict, university: dict) -> dict:
        """Predict acceptance probability for one student-university pair."""
        if not ML_AVAILABLE or not self.trained:
            return self._fallback(student, university)
        feat  = self.scaler.transform(self._features(student, university))
        proba = self.model.predict_proba(feat)[0]
        p     = float(proba[1])
        if p >= 0.75:   tier, rec = "Strong",     "Excellent match — high acceptance probability"
        elif p >= 0.55: tier, rec = "Competitive","Good match — apply with strong personal statement"
        elif p >= 0.35: tier, rec = "Borderline", "Apply as a reach school"
        else:           tier, rec = "Weak",        "Low probability — consider strengthening profile"
        dist = abs(p-0.5)
        conf = "High" if dist>0.25 else ("Medium" if dist>0.1 else "Low")
        return {"probability": round(p,3), "percentage": f"{p:.0%}",
                "tier": tier, "confidence": conf, "recommendation": rec, "ml_used": True}

    def _fallback(self, student: dict, university: dict) -> dict:
        ac = university.get("acceptance_criteria",{})
        gpa_ok = student.get("gpa",0) >= ac.get("min_grade_point",3.0)
        eng_ok = student.get("ielts",0) >= ac.get("ielts_min",6.5) or \
                 student.get("toefl",0) >= ac.get("toefl_min",80)
        if gpa_ok and eng_ok:   p, tier = 0.65, "Competitive"
        elif gpa_ok or eng_ok:  p, tier = 0.35, "Borderline"
        else:                   p, tier = 0.10, "Weak"
        return {"probability": p, "percentage": f"{p:.0%}", "tier": tier,
                "confidence": "Low", "recommendation": "Rule-based estimate", "ml_used": False}

    def rank_universities(self, student: dict, universities: list) -> list:
        """Score and rank all universities for a student, best-fit first."""
        ranked = []
        for uni in universities:
            score   = self.predict_acceptance(student, uni)
            enriched = dict(uni)
            enriched["ml_score"] = score
            ranked.append(enriched)
        ranked.sort(key=lambda x: x["ml_score"]["probability"], reverse=True)
        return ranked

    def classify_student_tier(self, student: dict) -> dict:
        """Classify overall student strength across all metrics."""
        gpa   = float(student.get("gpa",0))
        ielts = float(student.get("ielts",0))
        toefl = float(student.get("toefl",0))
        eng   = max(ielts/9.0, toefl/120.0)*100
        if gpa>=3.8:   gt="Exceptional"
        elif gpa>=3.5: gt="Strong"
        elif gpa>=3.2: gt="Good"
        elif gpa>=2.8: gt="Average"
        else:          gt="Below Average"
        if eng>=85:   et="Exceptional"
        elif eng>=75: et="Strong"
        elif eng>=65: et="Good"
        elif eng>=55: et="Average"
        else:         et="Below Requirement"
        score = int(min(gpa/4.0*60,60) + min(eng*0.4,40))
        if score>=80:   overall="Elite"
        elif score>=65: overall="Strong"
        elif score>=50: overall="Competitive"
        elif score>=35: overall="Borderline"
        else:           overall="Needs Improvement"
        strengths, weaknesses = [], []
        if gpa>=3.5:   strengths.append(f"Strong GPA ({gpa:.2f}/4.0)")
        else:          weaknesses.append(f"GPA below target ({gpa:.2f}/4.0)")
        if ielts>=6.5: strengths.append(f"IELTS meets requirements ({ielts})")
        elif ielts>0:  weaknesses.append(f"IELTS below 6.5 ({ielts})")
        return {
            "overall_tier": overall, "gpa_tier": gt, "english_tier": et,
            "profile_score": score, "strengths": strengths, "weaknesses": weaknesses,
        }

    def get_feature_importance(self) -> dict:
        if not self.trained or not self.model: return {}
        return dict(zip(FEATURE_NAMES,[round(float(v),4) for v in self.model.feature_importances_]))


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINED PHASE 1 ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

class Phase1:
    """
    Single import for all Phase 1 functionality.

    Usage:
        from core.phases.phase1 import Phase1

        p1 = Phase1()
        unis   = p1.db.get_universities_by_country("UK")
        ranked = p1.ml.rank_universities(student, unis)
        tier   = p1.ml.classify_student_tier(student)
    """

    def __init__(self, db_path: str = _DB_PATH):
        self.db = CuratedDatabaseManager(db_path=db_path)
        self.ml = Phase1MLEngine()
        if not self.ml.trained:
            self.ml.train(verbose=False)
        print("✅ Phase1 ready (DB + ML)")

    def analyse_student(self, student: dict, country: str = None) -> dict:
        """
        Full Phase 1 analysis for one student.
        Returns tier + ranked universities (optionally filtered by country).
        """
        if country:
            unis = self.db.get_universities_by_country(country)
        else:
            unis = self.db.get_all_universities()

        tier   = self.ml.classify_student_tier(student)
        ranked = self.ml.rank_universities(student, unis)
        strong      = [u for u in ranked if u["ml_score"]["tier"] == "Strong"]
        competitive = [u for u in ranked if u["ml_score"]["tier"] == "Competitive"]
        borderline  = [u for u in ranked if u["ml_score"]["tier"] == "Borderline"]

        return {
            "student_tier":  tier,
            "total_analysed": len(ranked),
            "strong":        len(strong),
            "competitive":   len(competitive),
            "borderline":    len(borderline),
            "top_5":         ranked[:5],
            "all_ranked":    ranked,
        }


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1 — Curated DB + ML Self Test")
    print("=" * 60)

    p1 = Phase1()
    print(f"\nDB stats: {p1.db.get_statistics()}")

    student = {"gpa":3.67,"ielts":7.0,"toefl":0,"stream":"Physical Science",
               "program":"Engineering","budget_usd":35000}

    tier = p1.ml.classify_student_tier(student)
    print(f"\nStudent tier: {tier['overall_tier']} ({tier['profile_score']}/100)")
    print(f"Strengths:    {tier['strengths']}")
    print(f"Weaknesses:   {tier['weaknesses']}")

    unis = [
        {"id":"uk_cambridge","name":"University of Cambridge","rankings":{"qs_world":2},
         "acceptance_criteria":{"min_grade_point":3.9,"ielts_min":7.5,"toefl_min":110},
         "tuition":{"undergraduate_intl_gbp":28000}},
        {"id":"uk_manchester","name":"University of Manchester","rankings":{"qs_world":32},
         "acceptance_criteria":{"min_grade_point":3.3,"ielts_min":6.5,"toefl_min":90},
         "tuition":{"undergraduate_intl_gbp":19000}},
        {"id":"au_melbourne","name":"University of Melbourne","rankings":{"qs_world":14},
         "acceptance_criteria":{"min_grade_point":3.3,"ielts_min":6.5,"toefl_min":79},
         "tuition":{"undergraduate_intl_aud":50000}},
    ]

    ranked = p1.ml.rank_universities(student, unis)
    print("\nRanked universities:")
    for i,u in enumerate(ranked,1):
        s = u["ml_score"]
        print(f"  #{i} {u['name']:<38} {s['percentage']:>5}  [{s['tier']}]")

    print("\n✓ Phase 1 test complete")