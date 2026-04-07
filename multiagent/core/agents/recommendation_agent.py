"""
Recommendation Agent
====================
Provides ranked university recommendations by combining eligibility, financial feasibility,
application timelines, and risk/priority factors.

Features:
  - Prioritises universities by application deadlines and risk scores
  - Blends eligibility and financial feasibility into a single ranking
  - Flags high-risk / high-cost options for transparency
  - Supports "why" explanations to increase trust & reduce bias

Algorithm (hybrid: heuristic first, ML when real outcomes exist):
  1. Collect eligibility status (eligible/borderline/ineligible) from Eligibility Agent.
  2. Collect financial feasibility status (feasible/borderline/infeasible) from Financial Feasibility Agent.
  3. For each university:
      a. Compute heuristic score (+2 eligible, +1.5 feasible, rank bonus, risk notes).
      b. If a real-outcomes ML model is available, predict admit probability.
      c. Blend heuristic and ML score; keep heuristic dominant for stability.
      d. Add notes about visa risk and deadlines.
  4. Sort universities by combined score (and deadline urgency).
  5. Split into: recommended, backup options, and avoid lists.

Intended to be used by the application layer (UI/chatbot) to give students a clear,
explainable list of suggested next steps.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import json
import logging
import os
import urllib.request

try:
    from sklearn.ensemble import RandomForestClassifier as _RFC
except Exception:
    _RFC = None


logger = logging.getLogger(__name__)


@dataclass
class RecommendationEntry:
    university_id: str
    university_name: str
    country: str
    score: float
    reasons: List[str]
    deadline: Optional[str]
    visa_risk: Optional[str]
    financial_risk: Optional[str]
    note: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RecommendationReport:
    recommended: List[RecommendationEntry]
    backup_options: List[RecommendationEntry]
    avoid: List[RecommendationEntry]
    global_recommendations: List[str]

    def to_dict(self) -> dict:
        return {
            "recommended": [r.to_dict() for r in self.recommended],
            "backup_options": [r.to_dict() for r in self.backup_options],
            "avoid": [r.to_dict() for r in self.avoid],
            "global_recommendations": self.global_recommendations,
        }


class RecommendationAgent:
    """Ranks universities using eligibility, cost, deadlines, and risk factors."""

    BASE_VISA_RISK_BY_COUNTRY = {
        "UK": "medium",
        "Australia": "medium",
        "Singapore": "low",
    }

    # If no deadline is provided, we treat it as "later" (lower priority).
    DEFAULT_DEADLINE_PRIORITY = 1000

    def __init__(self, historical_outcomes_path: str = ""):
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.historical_outcomes_path = historical_outcomes_path or os.environ.get(
            "RECOMMENDATION_OUTCOMES_PATH",
            os.path.join(root, "data", "training", "historical_admissions_outcomes.jsonl"),
        )
        self.visa_risk_data_path = os.environ.get("VISA_RISK_DATA_PATH", "").strip()
        self.visa_risk_data_url = os.environ.get("VISA_RISK_DATA_URL", "").strip()
        self._live_visa_risk_map = self._load_live_visa_risk_map()
        # auto(default): use ML only when enough real outcomes exist.
        # off: force heuristic mode.
        self.ml_mode = (os.environ.get("RECOMMENDATION_ML_MODE", "auto") or "auto").strip().lower()
        self._ml_model = None
        self._ml_trained = False
        self._init_ml_model()

    def _normalize_country_label(self, value: str) -> str:
        raw = (value or "").strip().lower()
        aliases = {
            "uk": "UK",
            "u.k.": "UK",
            "united kingdom": "UK",
            "great britain": "UK",
            "gb": "UK",
            "australia": "Australia",
            "au": "Australia",
            "singapore": "Singapore",
            "sg": "Singapore",
            "sri lanka": "Sri Lanka",
            "lanka": "Sri Lanka",
            "india": "India",
            "pakistan": "Pakistan",
            "bangladesh": "Bangladesh",
            "nepal": "Nepal",
        }
        if raw in aliases:
            return aliases[raw]
        if raw.endswith("n") and raw[:-1] in aliases:
            return aliases[raw[:-1]]
        return (value or "").strip()

    def _extract_applicant_nationality(self, profile_data: Dict) -> str:
        for key in ("nationality", "passport_country", "citizenship", "country_of_citizenship", "country"):
            val = profile_data.get(key)
            if isinstance(val, str) and val.strip():
                return self._normalize_country_label(val)
        return ""

    def _load_live_visa_risk_map(self) -> Dict[str, Dict[str, str]]:
        """Load optional live visa-risk matrix from file/URL.

        Expected JSON format:
        {
          "UK": {"Sri Lanka": "high", "India": "medium", "_default": "medium"},
          "Australia": {"Sri Lanka": "medium", "_default": "medium"}
        }
        """
        source_payload = None

        if self.visa_risk_data_path and os.path.exists(self.visa_risk_data_path):
            try:
                with open(self.visa_risk_data_path, "r", encoding="utf-8") as handle:
                    source_payload = json.load(handle)
            except Exception as exc:
                logger.warning(f"RecommendationAgent: visa risk file load failed: {exc}")

        if source_payload is None and self.visa_risk_data_url:
            try:
                with urllib.request.urlopen(self.visa_risk_data_url, timeout=4) as resp:
                    source_payload = json.loads(resp.read().decode("utf-8"))
            except Exception as exc:
                logger.warning(f"RecommendationAgent: visa risk URL load failed: {exc}")

        if not isinstance(source_payload, dict):
            return {}

        clean: Dict[str, Dict[str, str]] = {}
        for destination, table in source_payload.items():
            dest = self._normalize_country_label(str(destination))
            if not isinstance(table, dict) or not dest:
                continue
            mapped: Dict[str, str] = {}
            for nat, risk in table.items():
                nat_key = "_default" if str(nat) == "_default" else self._normalize_country_label(str(nat))
                risk_val = str(risk).strip().lower()
                if risk_val in {"low", "medium", "high"}:
                    mapped[nat_key] = risk_val
            if mapped:
                clean[dest] = mapped
        return clean

    def _resolve_visa_risk(self, destination_country: str, applicant_nationality: str) -> str:
        destination = self._normalize_country_label(destination_country)
        nationality = self._normalize_country_label(applicant_nationality)

        if destination and nationality and destination == nationality:
            return "low"

        # Live/file-based matrix has highest priority when provided.
        if destination in self._live_visa_risk_map:
            table = self._live_visa_risk_map[destination]
            if nationality and nationality in table:
                return table[nationality]
            if "_default" in table:
                return table["_default"]

        base = self.BASE_VISA_RISK_BY_COUNTRY.get(destination, "unknown")

        # Nationality-aware adjustments (lightweight fallback when no live matrix).
        if nationality and destination in {"UK", "Australia", "Singapore"}:
            higher_scrutiny = {"Sri Lanka", "Pakistan", "Bangladesh", "Nepal"}
            if nationality in higher_scrutiny:
                if base == "low":
                    return "medium"
                if base == "medium":
                    return "high"

        return base

    def _init_ml_model(self) -> None:
        if self.ml_mode == "off":
            return
        if _RFC is None:
            logger.warning("RecommendationAgent: scikit-learn unavailable; using heuristic mode.")
            return

        rows = self._load_outcomes(self.historical_outcomes_path)
        X: List[List[float]] = []
        y: List[int] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            outcome = str(row.get("admission_outcome", "")).strip().lower()
            if outcome not in {"accepted", "rejected"}:
                continue
            X.append(self._features_from_record(row))
            y.append(1 if outcome == "accepted" else 0)

        # Safety gate: avoid unstable training with tiny datasets.
        if len(X) < 30 or len(set(y)) < 2:
            return

        try:
            self._ml_model = _RFC(n_estimators=120, random_state=42, class_weight="balanced")
            self._ml_model.fit(X, y)
            self._ml_trained = True
            logger.info("RecommendationAgent: ML ranker trained from real historical outcomes.")
        except Exception as exc:
            logger.warning(f"RecommendationAgent: ML train failed; fallback to heuristic. {exc}")
            self._ml_model = None
            self._ml_trained = False

    def _load_outcomes(self, path: str) -> list:
        if not path or not os.path.exists(path):
            return []
        records = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = (line or "").strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        continue
        except Exception as exc:
            logger.warning(f"RecommendationAgent: failed reading outcomes file {path}: {exc}")
            return []
        return records

    def _country_risk_value(self, country: str, nationality: str = "") -> float:
        risk = self._resolve_visa_risk(country, nationality)
        # Higher value = lower risk
        return {"low": 1.0, "medium": 0.6, "high": 0.2, "unknown": 0.5}.get(risk, 0.5)

    def _features_from_record(self, row: Dict) -> List[float]:
        gpa = float(row.get("gpa") or 0.0)
        uni_min = float(row.get("university_min_gpa") or 0.0)
        gpa_diff = row.get("gpa_diff")
        if gpa_diff is None:
            gpa_diff = gpa - uni_min
        gpa_diff = float(gpa_diff)

        uni_data = row.get("university_data") if isinstance(row.get("university_data"), dict) else {}
        qs = row.get("qs_world")
        if qs is None:
            qs = uni_data.get("qs")
        the = row.get("the_world")
        if the is None:
            the = uni_data.get("the")

        qs_score = max(0.0, (200.0 - float(qs)) / 200.0) if isinstance(qs, (int, float)) else 0.0
        the_score = max(0.0, (200.0 - float(the)) / 200.0) if isinstance(the, (int, float)) else 0.0

        align = row.get("alignment_label")
        align_val = float(align) if align in (0, 1) else 0.5

        return [
            gpa,
            uni_min,
            gpa_diff,
            qs_score,
            the_score,
            self._country_risk_value(str(row.get("country") or ""), str(row.get("nationality") or "")),
            align_val,
        ]

    def _features_for_university(
        self,
        uni: Dict,
        grade_point: float,
        applicant_nationality: str,
        is_eligible: bool,
        is_borderline: bool,
        is_feasible: bool,
        is_borderline_fin: bool,
    ) -> List[float]:
        acceptance = uni.get("acceptance_criteria") or {}
        uni_min = float(acceptance.get("min_grade_point") or 0.0)
        gpa_diff = grade_point - uni_min
        ranks = uni.get("rankings") or {}
        qs = ranks.get("qs_world")
        the = ranks.get("the_world")
        qs_score = max(0.0, (200.0 - float(qs)) / 200.0) if isinstance(qs, (int, float)) else 0.0
        the_score = max(0.0, (200.0 - float(the)) / 200.0) if isinstance(the, (int, float)) else 0.0
        eligibility_signal = 1.0 if is_eligible else (0.5 if is_borderline else 0.0)
        financial_signal = 1.0 if is_feasible else (0.5 if is_borderline_fin else 0.0)
        return [
            grade_point + (0.1 * eligibility_signal),
            uni_min,
            gpa_diff,
            qs_score,
            the_score,
            self._country_risk_value(str(uni.get("country") or ""), applicant_nationality),
            financial_signal,
        ]

    def _ml_probability(self, features: List[float]) -> Optional[float]:
        if not self._ml_trained or self._ml_model is None:
            return None
        try:
            return float(self._ml_model.predict_proba([features])[0][1])
        except Exception:
            return None

    def recommend(self,
                  universities: List[Dict],
                  profile_data: Dict,
                  eligibility_report: Optional[Dict] = None,
                  financial_report: Optional[Dict] = None) -> RecommendationReport:
        """Generate a recommendation report for a list of universities."""

        grade_point = 0.0
        applicant_nationality = self._extract_applicant_nationality(profile_data or {})
        if eligibility_report:
            try:
                grade_point = float(eligibility_report.get("grade_point") or 0.0)
            except Exception:
                grade_point = 0.0

        eligible_ids = set()
        borderline_ids = set()
        if eligibility_report:
            eligible_ids.update(u.get("university_id") for u in eligibility_report.get("eligible_universities", []))
            borderline_ids.update(u.get("university_id") for u in eligibility_report.get("borderline_universities", []))

        feasible_ids = set()
        borderline_feasible_ids = set()
        infeasible_ids = set()
        if financial_report:
            feasible_ids.update(u.get("university_id") for u in financial_report.get("feasible_universities", []))
            borderline_feasible_ids.update(u.get("university_id") for u in financial_report.get("borderline_universities", []))
            infeasible_ids.update(u.get("university_id") for u in financial_report.get("infeasible_universities", []))

        recommendations: List[RecommendationEntry] = []
        backup: List[RecommendationEntry] = []
        avoid: List[RecommendationEntry] = []

        for uni in universities:
            uni_id = uni.get("id") or uni.get("university_id")
            name = uni.get("name", "Unknown University")
            country = uni.get("country", "")

            score = 0.0
            reasons: List[str] = []
            visa_risk = self._resolve_visa_risk(country, applicant_nationality)
            financial_risk = None
            note = None

            # Eligibility contribution
            if uni_id in eligible_ids:
                score += 2.0
                reasons.append("Academic & English requirements met")
            elif uni_id in borderline_ids:
                score += 1.0
                reasons.append("Borderline eligibility — worth applying to some programs")
            else:
                reasons.append("Eligibility uncertain — consider foundation or pathway programs")

            # Financial contribution
            if uni_id in feasible_ids:
                score += 1.5
                financial_risk = "low"
                reasons.append("Falls within your budget")
            elif uni_id in borderline_feasible_ids:
                score += 0.5
                financial_risk = "medium"
                reasons.append("May stretch your budget; review scholarships")
            elif uni_id in infeasible_ids:
                score -= 1.0
                financial_risk = "high"
                reasons.append("Likely exceeds budget; consider alternatives")

            # Rankings & quality bonus
            qs = uni.get("rankings", {}).get("qs_world")
            the = uni.get("rankings", {}).get("the_world")
            if isinstance(qs, (int, float)):
                score += max(0, (100 - qs) / 100)
            if isinstance(the, (int, float)):
                score += max(0, (100 - the) / 100)

            # Deadline prioritization
            deadline = uni.get("application_deadline") or uni.get("deadline")
            if deadline:
                reasons.append(f"Application deadline: {deadline}")

            # Risk notes
            if visa_risk == "high":
                reasons.append("Higher visa risk — check government guidance")

            # ML contribution (optional): keep process stable by blending with heuristic score.
            is_eligible = uni_id in eligible_ids
            is_borderline = uni_id in borderline_ids
            is_feasible = uni_id in feasible_ids
            is_borderline_fin = uni_id in borderline_feasible_ids
            ml_prob = self._ml_probability(self._features_for_university(
                uni,
                grade_point,
                applicant_nationality,
                is_eligible,
                is_borderline,
                is_feasible,
                is_borderline_fin,
            ))
            if ml_prob is not None:
                score = (score * 0.7) + ((ml_prob * 4.0) * 0.3)
                reasons.append(f"ML admit likelihood: {round(ml_prob * 100)}%")

            # Final decisions
            entry = RecommendationEntry(
                university_id=uni_id,
                university_name=name,
                country=country,
                score=round(score, 2),
                reasons=reasons,
                deadline=deadline,
                visa_risk=visa_risk,
                financial_risk=financial_risk,
                note=note,
            )

            if uni_id in infeasible_ids:
                avoid.append(entry)
            elif uni_id in feasible_ids and uni_id in eligible_ids:
                recommendations.append(entry)
            else:
                backup.append(entry)

        # Sort lists by score (higher first) and deadline (earliest first when present)
        def sort_key(e: RecommendationEntry):
            dprio = self.DEFAULT_DEADLINE_PRIORITY
            if e.deadline:
                try:
                    # assume YYYY-MM-DD or similar; lexicographic works for ISO-like dates
                    dprio = int(str(e.deadline).replace("-", ""))
                except Exception:
                    dprio = self.DEFAULT_DEADLINE_PRIORITY
            return (-e.score, dprio)

        recommendations.sort(key=sort_key)
        backup.sort(key=sort_key)
        avoid.sort(key=sort_key)

        global_recs = [
            "Focus first on universities with the earliest deadlines.",
            "Apply to a mix of reach (high rank), match (eligibility), and safety (budget-friendly) options.",
            "Review visa requirements early to avoid last-minute delays.",
        ]

        return RecommendationReport(
            recommended=recommendations,
            backup_options=backup,
            avoid=avoid,
            global_recommendations=global_recs,
        )
