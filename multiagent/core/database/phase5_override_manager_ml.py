"""
Phase 5: Override Manager + ML Intelligence
============================================
Location: multiagent/core/phases/phase5.py

Combines:
  - OverrideManager   → load/save/apply JSON overrides with audit trail
  - AnomalyDetector   → IsolationForest flags suspicious values (ML-1)
  - OverrideClassifier→ TF-IDF + Naive Bayes NLP classifier (ML-2)
  - SuggestionEngine  → Z-score auto-suggests stale fields (ML-3)

Usage:
    from core.phases.phase5 import Phase5

    p5 = Phase5()
    p5.load_overrides()

    # Add override — all 3 ML modules run automatically
    p5.add_override("uk_manchester", "acceptance_criteria.ielts_min",
                    6.5, updated_by="Admin", reason="IELTS raised for 2026")

    # Apply overrides to a university dict
    enriched = p5.apply_overrides(university_dict)

    # Auto-scan DB for stale/suspicious values
    suggestions = p5.suggest_overrides(universities)

    # Full ML activity report
    report = p5.get_ml_report()
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("[Phase5] scikit-learn not found — run: pip install scikit-learn")

logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────

_ROOT           = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OVERRIDES_PATH = os.path.join(_ROOT, "data", "overrides", "active_overrides.json")
_HISTORY_PATH   = os.path.join(_ROOT, "data", "overrides", "override_history.json")
_ML_LOG_PATH    = os.path.join(_ROOT, "data", "overrides", "ml_override_log.json")


# ══════════════════════════════════════════════════════════════════════════════
#  ML-1 — ANOMALY DETECTOR (IsolationForest)
# ══════════════════════════════════════════════════════════════════════════════

class AnomalyDetector:
    """
    Detects suspicious override values using IsolationForest.
    Trained on normal value ranges from the 30-university database.
    """

    FIELD_TRAINING_DATA: Dict[str, List[float]] = {
        "ielts_min":              [6.0,6.0,6.5,6.5,6.5,7.0,7.0,7.5,7.5,6.5],
        "ielts_minimum":          [6.0,6.0,6.5,6.5,6.5,7.0,7.0,7.5,7.5,6.5],
        "ielts_writing":          [6.0,6.0,6.5,6.5,6.5,7.0,7.0],
        "toefl_min":              [79,79,85,85,88,90,92,96,100,107,110],
        "toefl_minimum":          [79,79,85,85,88,90,92,96,100,107,110],
        "min_grade_point":        [2.5,2.8,3.0,3.0,3.3,3.3,3.5,3.7,3.7,3.9],
        "grade_point":            [2.5,2.8,3.0,3.0,3.3,3.3,3.5,3.7,3.7,3.9],
        "qs_world":               [2,3,6,8,9,14,15,18,19,27,32,43,55,67,89,140,185,197],
        "the_world":              [1,5,8,19,22,27,30,33,36,59,67,71,79,81,91,95,132,151],
        "undergraduate_intl_gbp": [18000,18000,19000,20000,21000,22000,22000,28000,28000,32000],
        "undergraduate_intl_aud": [22000,24000,25000,26000,27000,28000,30000,30000,32000,34000,36000,50000],
        "undergraduate_intl_sgd": [15000,26000,26000,28000,35000],
        "living_cost":            [1000,1200,1400,1500,1750,2000,2200],
    }

    HARD_BOUNDS = {
        "ielts": (4.0, 9.0), "toefl": (40, 120), "grade_point": (1.0, 4.0),
        "gpa": (1.0, 4.0), "undergraduate_intl_gbp": (8000, 80000),
        "undergraduate_intl_aud": (10000, 100000), "undergraduate_intl_sgd": (5000, 90000),
    }

    MAX_CHANGE_PCT = {
        "grade_point": 20, "gpa": 20, "ielts": 15, "toefl": 20,
        "qs_world": 60, "the_world": 60, "tuition": 40, "gbp": 40, "aud": 40, "sgd": 40,
    }

    def __init__(self):
        self._models: Dict[str, Any] = {}
        if ML_AVAILABLE:
            for field_key, values in self.FIELD_TRAINING_DATA.items():
                X = np.array(values).reshape(-1, 1)
                m = IsolationForest(n_estimators=50, contamination=0.1, random_state=42)
                m.fit(X)
                self._models[field_key] = m

    def check(self, field_path: str, new_value: Any, old_value: Any = None) -> Dict:
        """Check if new_value is anomalous for the given field."""
        result = {"is_anomaly": False, "anomaly_score": 0.0, "change_pct": None, "warning": None}
        if not isinstance(new_value, (int, float)):
            return result

        fp = field_path.lower()

        # Hard bounds
        for bound_key, (lo, hi) in self.HARD_BOUNDS.items():
            if bound_key in fp:
                if not (lo <= new_value <= hi):
                    result["is_anomaly"] = True
                    result["warning"] = f"⚠ {new_value} is outside valid range {lo}–{hi} for '{field_path}'"
                    return result

        # Change % check
        if old_value is not None and isinstance(old_value, (int, float)) and old_value != 0:
            chg = abs((new_value - old_value) / old_value) * 100
            result["change_pct"] = round(chg, 1)
            max_chg = 50
            for k, v in self.MAX_CHANGE_PCT.items():
                if k in fp: max_chg = v; break
            if chg > max_chg:
                result["is_anomaly"] = True
                result["warning"] = f"⚠ Large change: {old_value} → {new_value} ({chg:.1f}% — limit {max_chg}%)"
                return result

        # IsolationForest check
        field_key = next((k for k in self.FIELD_TRAINING_DATA if k in fp), None)
        if field_key and field_key in self._models:
            X   = np.array([[float(new_value)]])
            pred  = self._models[field_key].predict(X)[0]
            score = float(self._models[field_key].score_samples(X)[0])
            result["anomaly_score"] = round(score, 4)
            if pred == -1:
                result["is_anomaly"] = True
                lo = min(self.FIELD_TRAINING_DATA[field_key])
                hi = max(self.FIELD_TRAINING_DATA[field_key])
                result["warning"] = f"⚠ {new_value} looks unusual for '{field_path}' (typical: {lo}–{hi})"

        return result


# ══════════════════════════════════════════════════════════════════════════════
#  ML-2 — NLP CLASSIFIER (TF-IDF + Naive Bayes)
# ══════════════════════════════════════════════════════════════════════════════

class OverrideClassifier:
    """Classifies override reason text into 5 semantic categories."""

    TRAINING_EXAMPLES = [
        # criteria_update
        ("IELTS minimum raised to 6.5 for 2026 intake",                    "criteria_update"),
        ("TOEFL minimum updated to 96 verified on Imperial website",        "criteria_update"),
        ("Min GPA corrected back to 3.5 erroneous entry rolled back",       "criteria_update"),
        ("IELTS writing band minimum raised to 6.5 for CS programs",        "criteria_update"),
        ("Minimum grade point average updated based on admissions office",  "criteria_update"),
        ("English language requirement increased for engineering faculty",  "criteria_update"),
        ("GPA requirement lowered due to new pathway program",              "criteria_update"),
        ("TOEFL waived for students from English medium schools",           "criteria_update"),
        # tuition_update
        ("2026 tuition updated AUD 50000 per year",                         "tuition_update"),
        ("Tuition fee increased due to university policy change",           "tuition_update"),
        ("International student fee revised for 2026 academic year",       "tuition_update"),
        ("Sydney Research Scholarship added AUD 25000 min GPA 3.5",        "tuition_update"),
        ("Living cost updated to USD 1750 per month Numbeo data",          "tuition_update"),
        ("Fee structure changed undergraduate international students",     "tuition_update"),
        ("New scholarship opportunity available for merit students",       "tuition_update"),
        # ranking_update
        ("QS 2026 ranking updated to 43 was 47 in Phase 1 DB",             "ranking_update"),
        ("THE World ranking improved to position 22 from 27",              "ranking_update"),
        ("QS subject ranking for engineering updated to 38",               "ranking_update"),
        ("University dropped in rankings verified from official QS site",  "ranking_update"),
        ("Rankings data refreshed from latest QS World University Rankings","ranking_update"),
        # deadline_change
        ("Application deadline moved to Feb 15 for Aug 2026 intake",       "deadline_change"),
        ("Closing date changed to March 31 for international applicants",  "deadline_change"),
        ("Rolling admissions introduced no fixed deadline for 2026",       "deadline_change"),
        ("Deadline extended by two weeks due to high demand",              "deadline_change"),
        ("Early decision deadline added for scholarship consideration",    "deadline_change"),
        # new_program
        ("New MSc Artificial Intelligence program added for 2026 IELTS 6.5","new_program"),
        ("Bachelor of Data Science launched from 2026 academic year",      "new_program"),
        ("MBA program discontinued program removed from offerings",        "new_program"),
        ("New joint degree program with MIT added for computer science",   "new_program"),
        ("Postgraduate diploma in machine learning introduced 2026",       "new_program"),
    ]

    CATEGORIES = ["criteria_update","tuition_update","ranking_update","deadline_change","new_program"]

    LABELS = {
        "criteria_update": " Criteria Update",
        "tuition_update":  " Tuition/Cost Update",
        "ranking_update":  "Ranking Update",
        "deadline_change": "Deadline Change",
        "new_program":     " New Program",
    }

    def __init__(self):
        self._vec, self._model, self._trained = None, None, False
        if ML_AVAILABLE:
            texts  = [e[0] for e in self.TRAINING_EXAMPLES]
            labels = [e[1] for e in self.TRAINING_EXAMPLES]
            self._vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, lowercase=True)
            X = self._vec.fit_transform(texts)
            self._model = MultinomialNB(alpha=1.0)
            self._model.fit(X, labels)
            self._trained = True

    def classify(self, text: str) -> Dict:
        if not self._trained or not text:
            return {"category": "unknown", "label": "❓ Unknown", "confidence": 0.0, "all_probabilities": {}}
        try:
            X     = self._vec.transform([text])
            probs = self._model.predict_proba(X)[0]
            cats  = self._model.classes_
            best  = int(np.argmax(probs))
            cat   = cats[best]
            return {
                "category": cat, "label": self.LABELS.get(cat, cat),
                "confidence": round(float(probs[best]), 3),
                "all_probabilities": {c: round(float(p),3) for c,p in zip(cats,probs)},
            }
        except Exception as exc:
            logger.warning("Classify failed: %s", exc)
            return {"category": "unknown", "label": "❓ Unknown", "confidence": 0.0, "all_probabilities": {}}


# ══════════════════════════════════════════════════════════════════════════════
#  ML-3 — SUGGESTION ENGINE (Z-score)
# ══════════════════════════════════════════════════════════════════════════════

class SuggestionEngine:
    """
    Scans university database and suggests overrides for stale/suspicious values
    using Z-score outlier detection + hard bounds.
    """

    HARD_BOUNDS = {
        "ielts_min": (4.0,9.0), "ielts_minimum": (4.0,9.0), "ielts_writing": (4.0,9.0),
        "toefl_min": (40,120),  "toefl_minimum": (40,120),
        "min_grade_point": (1.0,4.0), "qs_world": (1,2000), "the_world": (1,2000),
        "undergraduate_intl_gbp": (8000,80000), "undergraduate_intl_aud": (10000,100000),
        "undergraduate_intl_sgd": (5000,90000), "living_cost_usd_monthly": (300,5000),
    }

    Z_THRESHOLD = 2.5

    def suggest(self, universities: List[Dict]) -> List[Dict]:
        field_values: Dict[str, List] = {}
        for uni in universities:
            self._collect(uni.get("id","?"), uni, field_values)
        suggestions = []
        for field, entries in field_values.items():
            vals = [v for _, v in entries if isinstance(v,(int,float))]
            if len(vals) < 3: continue
            mean, std = float(np.mean(vals)), float(np.std(vals))
            if std == 0: continue
            for uid, val in entries:
                if not isinstance(val,(int,float)): continue
                z         = abs((val - mean) / std)
                bounds    = self.HARD_BOUNDS.get(field)
                out_hard  = bounds and not (bounds[0] <= val <= bounds[1])
                if z > self.Z_THRESHOLD or out_hard:
                    reason = (f"Value {val} is outside bounds {bounds[0]}–{bounds[1]}"
                              if out_hard else
                              f"Value {val} is {z:.1f}σ from mean ({mean:.2f}±{std:.2f})")
                    suggestions.append({
                        "university_id": uid, "field": field, "current_value": val,
                        "suggested_action": "review", "reason": reason,
                        "z_score": round(z,2), "confidence": min(round(z/4,2),1.0),
                        "out_of_hard_bounds": bool(out_hard),
                    })
        suggestions.sort(key=lambda x: x["confidence"], reverse=True)
        return suggestions

    def _collect(self, uid: str, obj: Dict, out: Dict):
        for k, v in obj.items():
            if isinstance(v, dict): self._collect(uid, v, out)
            elif isinstance(v,(int,float)) and v > 0:
                out.setdefault(k.lower(), []).append((uid, v))


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 5 — COMBINED ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

class Phase5:
    """
    Single import for all Phase 5 functionality.

    Usage:
        from core.phases.phase5 import Phase5
        p5 = Phase5()
        p5.load_overrides()
        p5.add_override("uk_manchester", "acceptance_criteria.ielts_min",
                        6.5, updated_by="Admin", reason="IELTS raised 2026")
        enriched  = p5.apply_overrides(university_dict)
        suggested = p5.suggest_overrides(universities)
        report    = p5.get_ml_report()
    """

    def __init__(self,
                 overrides_path: str = _OVERRIDES_PATH,
                 history_path:   str = _HISTORY_PATH,
                 ml_log_path:    str = _ML_LOG_PATH,
                 db_manager=None):

        self.overrides_path = overrides_path
        self.history_path   = history_path
        self.ml_log_path    = ml_log_path
        self.db_manager     = db_manager

        self._overrides: Dict = {}
        self._history:   List = []
        self._ml_log:    List = self._load_ml_log()

        # ML modules
        print("🤖 Phase5: Loading ML modules...")
        self.anomaly    = AnomalyDetector()
        self.classifier = OverrideClassifier()
        self.suggester  = SuggestionEngine()
        print(f"✅ Phase5 ready — ML-1 ({len(self.anomaly._models)} field models) | "
              f"ML-2 (TF-IDF+NB, {len(OverrideClassifier.TRAINING_EXAMPLES)} examples) | "
              f"ML-3 (Z-score, {len(SuggestionEngine.HARD_BOUNDS)} bounds)")

    # ── Core override management ──────────────────────────────────────────────

    def load_overrides(self) -> Dict:
        """Load active_overrides.json from disk."""
        try:
            with open(self.overrides_path, "r", encoding="utf-8") as f:
                self._overrides = json.load(f)
            print(f"✅ Phase5: Loaded overrides for {len(self._overrides)} universities")
        except FileNotFoundError:
            self._overrides = {}
            print("ℹ️  Phase5: No active_overrides.json — starting fresh")
        except json.JSONDecodeError as e:
            logger.error("Phase5: overrides corrupt: %s", e)
            self._overrides = {}
        self._history = self._load_history()
        return self._overrides

    def apply_overrides(self, university: Dict) -> Dict:
        """Apply all active overrides to a university dict."""
        uid = university.get("id","")
        if uid not in self._overrides:
            return university
        result = dict(university)
        for field_path, entry in self._overrides[uid].items():
            value = entry.get("value") if isinstance(entry, dict) else entry
            self._set_nested(result, field_path, value)
        return result

    def add_override(self, uni_id: str, field_path: str, new_value: Any,
                     updated_by: str = "Admin", reason: str = "",
                     old_value: Any = None) -> Dict:
        """
        Add or update an override.
        ML-1 (anomaly), ML-2 (classify) run automatically.
        """
        # Get old value if not provided
        if old_value is None:
            existing = (self._overrides.get(uni_id) or {}).get(field_path)
            if isinstance(existing, dict): old_value = existing.get("value")
            elif existing is not None:     old_value = existing

        # ML-1 anomaly check
        anomaly = self.anomaly.check(field_path, new_value, old_value)
        if anomaly["is_anomaly"]:
            print(f"  🚨 ML-1 ANOMALY: {uni_id}.{field_path} = {new_value}")
            print(f"     {anomaly['warning']}")
        else:
            print(f"  ✅ ML-1: {new_value} looks normal for '{field_path}'")

        # ML-2 classify reason
        clf = self.classifier.classify(reason)
        print(f"  🏷️  ML-2: '{clf['label']}' ({clf['confidence']:.0%} confidence)")

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "value": new_value, "updated_by": updated_by,
            "updated_at": now, "reason": reason,
            "type": "update" if old_value is not None else "new",
            "ml_category": clf["category"], "ml_category_label": clf["label"],
            "ml_confidence": clf["confidence"],
            "ml_is_anomaly": anomaly["is_anomaly"],
            "ml_anomaly_score": anomaly["anomaly_score"],
            "ml_anomaly_warning": anomaly.get("warning"),
        }

        self._overrides.setdefault(uni_id, {})[field_path] = entry
        self._history.append({
            "action": entry["type"], "university_id": uni_id,
            "field": field_path, "old_value": old_value, "new_value": new_value,
            "updated_by": updated_by, "timestamp": now, "reason": reason,
            "ml_category": clf["category"], "ml_is_anomaly": anomaly["is_anomaly"],
        })
        self._ml_log.append({
            "event": "add_override", "university_id": uni_id, "field": field_path,
            "new_value": new_value, "timestamp": now,
            "anomaly": anomaly, "classification": clf,
        })
        self._save_overrides()
        self._save_history()
        self._save_ml_log()

        return {
            "status": entry["type"] + "d",
            "university_id": uni_id, "field": field_path, "new_value": new_value,
            "ml_category": clf["label"], "ml_is_anomaly": anomaly["is_anomaly"],
            "ml_warning": anomaly.get("warning"),
        }

    def remove_override(self, uni_id: str, field_path: str, updated_by: str = "Admin") -> bool:
        """Remove an override."""
        if uni_id in self._overrides and field_path in self._overrides[uni_id]:
            del self._overrides[uni_id][field_path]
            if not self._overrides[uni_id]: del self._overrides[uni_id]
            self._history.append({
                "action": "remove", "university_id": uni_id, "field": field_path,
                "old_value": None, "new_value": None, "updated_by": updated_by,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "reason": "Override removed", "ml_category": None, "ml_is_anomaly": False,
            })
            self._save_overrides(); self._save_history()
            print(f"🗑️  Phase5: Removed {uni_id}.{field_path}")
            return True
        return False

    def get_overrides(self, uni_id: str = None) -> Dict:
        return self._overrides.get(uni_id, {}) if uni_id else self._overrides

    def get_history(self, uni_id: str = None, limit: int = 50) -> List:
        h = self._history[-limit:]
        return [e for e in h if e.get("university_id") == uni_id] if uni_id else h

    # ── ML-3 Auto-suggestions ──────────────────────────────────────────────────

    def suggest_overrides(self, universities: List[Dict] = None) -> List[Dict]:
        """ML-3: Scan DB and suggest overrides for suspicious values."""
        if universities is None and self.db_manager:
            universities = []
            for c in ["UK","Singapore","Australia"]:
                universities.extend(self.db_manager.get_universities_by_country(c))
        if not universities:
            print("⚠️  ML-3: No university data for scan")
            return []
        print(f"\n  🔍 ML-3: Scanning {len(universities)} universities...")
        suggestions = self.suggester.suggest(universities)
        print(f"  📋 ML-3: {len(suggestions)} suggestion(s) found")
        for s in suggestions[:5]:
            print(f"     • {s['university_id']}.{s['field']} = {s['current_value']}")
            print(f"       {s['reason']} | Confidence: {s['confidence']:.0%}")
        self._ml_log.append({
            "event": "suggest_overrides",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "uni_count": len(universities), "suggestions": suggestions,
        })
        self._save_ml_log()
        return suggestions

    # ── ML Report ──────────────────────────────────────────────────────────────

    def get_ml_report(self) -> Dict:
        """Full summary of all ML activity across all overrides."""
        total, anomalies, categories = 0, [], {}
        for uid, fields in self._overrides.items():
            for field, entry in fields.items():
                if not isinstance(entry, dict): continue
                total += 1
                if entry.get("ml_is_anomaly"):
                    anomalies.append({
                        "university_id": uid, "field": field,
                        "value": entry.get("value"), "warning": entry.get("ml_anomaly_warning"),
                    })
                cat = entry.get("ml_category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
        return {
            "total_overrides": total, "anomaly_count": len(anomalies),
            "anomalous_overrides": anomalies,
            "category_distribution": categories,
            "recent_ml_events": self._ml_log[-10:],
            "ml_modules": {
                "ml1_anomaly":     {"algorithm": "IsolationForest", "fields": len(self.anomaly._models)},
                "ml2_classifier":  {"algorithm": "TF-IDF + Naive Bayes", "examples": len(OverrideClassifier.TRAINING_EXAMPLES), "categories": 5},
                "ml3_suggestion":  {"algorithm": "Z-score outlier detection", "z_threshold": SuggestionEngine.Z_THRESHOLD},
            },
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _set_nested(self, obj: Dict, path: str, value: Any):
        parts = path.split(".")
        for p in parts[:-1]:
            obj = obj.setdefault(p, {})
        obj[parts[-1]] = value

    def _to_serial(self, obj):
        if isinstance(obj, dict):  return {k: self._to_serial(v) for k,v in obj.items()}
        if isinstance(obj, list):  return [self._to_serial(v) for v in obj]
        if isinstance(obj, (np.bool_,)):    return bool(obj)
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        return obj

    def _save_overrides(self):
        os.makedirs(os.path.dirname(self.overrides_path), exist_ok=True)
        with open(self.overrides_path, "w", encoding="utf-8") as f:
            json.dump(self._to_serial(self._overrides), f, indent=2)

    def _save_history(self):
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(self._to_serial(self._history[-500:]), f, indent=2)

    def _load_history(self) -> List:
        try:
            with open(self.history_path, "r", encoding="utf-8") as f:
                d = json.load(f); return d if isinstance(d,list) else []
        except Exception: return []

    def _load_ml_log(self) -> List:
        try:
            with open(self.ml_log_path, "r", encoding="utf-8") as f:
                d = json.load(f); return d if isinstance(d,list) else []
        except Exception: return []

    def _save_ml_log(self):
        os.makedirs(os.path.dirname(self.ml_log_path), exist_ok=True)
        with open(self.ml_log_path, "w", encoding="utf-8") as f:
            json.dump(self._to_serial(self._ml_log[-1000:]), f, indent=2)


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    print("=" * 60)
    print("Phase 5 — Override Manager + ML Self Test")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "overrides"))
        p5 = Phase5(
            overrides_path=os.path.join(tmp,"overrides","active_overrides.json"),
            history_path  =os.path.join(tmp,"overrides","override_history.json"),
            ml_log_path   =os.path.join(tmp,"overrides","ml_log.json"),
        )
        p5.load_overrides()

        print("\n--- Test 1: Normal override ---")
        r = p5.add_override("uk_manchester","acceptance_criteria.ielts_min",
                            6.5, reason="IELTS raised to 6.5 for 2026 intake")
        print(f"  Category: {r['ml_category']} | Anomaly: {r['ml_is_anomaly']}")

        print("\n--- Test 2: Anomaly (IELTS 9.5) ---")
        r = p5.add_override("uk_oxford","acceptance_criteria.ielts_min",
                            9.5, reason="IELTS requirement increased")
        print(f"  Anomaly: {r['ml_is_anomaly']} | Warning: {r['ml_warning']}")

        print("\n--- Test 3: ML-3 Auto-suggestion ---")
        unis = [
            {"id":"uk_a","acceptance_criteria":{"ielts_min":6.5,"min_grade_point":3.3}},
            {"id":"uk_b","acceptance_criteria":{"ielts_min":6.5,"min_grade_point":3.5}},
            {"id":"uk_c","acceptance_criteria":{"ielts_min":9.9,"min_grade_point":3.3}},
        ]
        sug = p5.suggest_overrides(unis)
        print(f"  {len(sug)} suggestions found")

        print("\n--- ML Report ---")
        rpt = p5.get_ml_report()
        print(f"  Total overrides: {rpt['total_overrides']}")
        print(f"  Anomalies:       {rpt['anomaly_count']}")
        print(f"  Categories:      {rpt['category_distribution']}")

    print("\n✓ Phase 5 test complete")