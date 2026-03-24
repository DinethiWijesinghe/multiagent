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

Algorithm (heuristic, non-ML):
  1. Collect eligibility status (eligible/borderline/ineligible) from Eligibility Agent.
  2. Collect financial feasibility status (feasible/borderline/infeasible) from Financial Feasibility Agent.
  3. For each university:
     a. Score +2 for eligible, +1 for borderline, 0 otherwise.
     b. Score +1.5 for financially feasible, +0.5 for borderline, -1 for infeasible.
     c. Add ranking bonus from QS/THE rank (higher rank → higher score).
     d. Add notes about visa risk and deadlines.
  4. Sort universities by combined score (and deadline urgency).
  5. Split into: recommended, backup options, and avoid lists.

Intended to be used by the application layer (UI/chatbot) to give students a clear,
explainable list of suggested next steps.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


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

    VISA_RISK_BY_COUNTRY = {
        "UK": "medium",
        "Australia": "medium",
        "Singapore": "low",
    }

    # If no deadline is provided, we treat it as "later" (lower priority).
    DEFAULT_DEADLINE_PRIORITY = 1000

    def recommend(self,
                  universities: List[Dict],
                  profile_data: Dict,
                  eligibility_report: Optional[Dict] = None,
                  financial_report: Optional[Dict] = None) -> RecommendationReport:
        """Generate a recommendation report for a list of universities."""

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
            visa_risk = self.VISA_RISK_BY_COUNTRY.get(country, "unknown")
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
