"""
Eligibility Calculator Processor
===================================
GPA normalisation across grading systems + dynamic eligibility assessment.
Called by app.py and wraps EligibilityVerificationAgent for session-state usage.
"""

from __future__ import annotations
from typing import Optional


# ─────────────────────────────────────────────
# GPA NORMALISATION
# ─────────────────────────────────────────────

def normalize_gpa(gpa_value, gpa_system: str) -> float:
    """Convert any grading system to a normalised 4.0 GPA."""
    try:
        if gpa_system == "GPA (4.0 scale)":
            return round(float(gpa_value), 2)

        elif gpa_system == "GPA (5.0 scale)":
            return round((float(gpa_value) / 5.0) * 4.0, 2)

        elif gpa_system == "UK Class (First/2:1/2:2)":
            conversion = {
                "First Class":           3.7,
                "Upper Second (2:1)":    3.3,
                "Lower Second (2:2)":    3.0,
                "Third Class":           2.5,
            }
            return conversion.get(str(gpa_value), 3.0)

        elif gpa_system == "Percentage":
            pct = float(gpa_value)
            if pct >= 90: return 4.0
            if pct >= 85: return 3.7
            if pct >= 80: return 3.5
            if pct >= 75: return 3.3
            if pct >= 70: return 3.0
            if pct >= 65: return 2.7
            if pct >= 60: return 2.5
            return 2.0

    except (ValueError, TypeError):
        pass

    return 3.0  # safe default


# ─────────────────────────────────────────────
# A/L-SPECIFIC ASSESSMENTS
# ─────────────────────────────────────────────

def assess_alevel_eligibility(document_data: dict, profile_data: dict) -> dict:
    """Eligibility assessment specifically for A/L qualifications."""
    GRADE_VALUES  = {"A": 4.0, "B": 3.0, "C": 2.0, "S": 1.0, "F": 0.0}
    SPECIAL_SUBJS = {"general english", "common general test", "cgt"}

    subjects = document_data.get("subjects", {})
    main_subjects = {
        s: g for s, g in subjects.items()
        if s.lower() not in SPECIAL_SUBJS
    }

    if not main_subjects:
        return {
            "eligible": False, "grade_point": 0.0,
            "eligibility_tier": "foundation",
            "eligible_countries": [], "recommended_programs": [],
            "notes": ["No main subject grades found."],
        }

    points = [GRADE_VALUES.get(str(g).upper(), 0.0) for g in main_subjects.values()]
    avg    = round(sum(points) / len(points), 2)

    if avg >= 3.7:
        return {
            "eligible": True, "grade_point": avg, "eligibility_tier": "top",
            "eligible_countries": ["UK", "Singapore", "Australia"],
            "recommended_programs": ["Engineering", "Computer Science", "Medicine"],
            "notes": ["Excellent grades! Competitive for top universities."],
        }
    if avg >= 3.3:
        return {
            "eligible": True, "grade_point": avg, "eligibility_tier": "good",
            "eligible_countries": ["UK", "Singapore", "Australia"],
            "recommended_programs": ["Engineering", "Business", "Science"],
            "notes": ["Good grades! Many universities available."],
        }
    if avg >= 3.0:
        return {
            "eligible": True, "grade_point": avg, "eligibility_tier": "average",
            "eligible_countries": ["UK", "Australia"],
            "recommended_programs": ["Business", "IT", "Science"],
            "notes": ["You qualify for several universities."],
        }
    return {
        "eligible": False, "grade_point": avg, "eligibility_tier": "foundation",
        "eligible_countries": [],
        "recommended_programs": ["Foundation Programme"],
        "notes": ["Consider foundation programmes to improve your qualifications."],
    }


def assess_bachelors_eligibility(document_data: dict, profile_data: dict) -> dict:
    """Eligibility for Bachelor's degree holders applying to postgrad."""
    gpa = float(document_data.get("gpa_normalized", 0.0))

    if gpa >= 3.5:
        return {
            "eligible": True, "grade_point": gpa, "eligibility_tier": "top",
            "eligible_countries": ["UK", "Singapore", "Australia"],
            "recommended_programs": ["Master's", "PhD", "MBA"],
            "notes": ["Excellent GPA! Competitive for top programmes."],
        }
    if gpa >= 3.0:
        return {
            "eligible": True, "grade_point": gpa, "eligibility_tier": "good",
            "eligible_countries": ["UK", "Singapore", "Australia"],
            "recommended_programs": ["Master's", "MBA"],
            "notes": ["Good GPA! Many programmes available."],
        }
    if gpa >= 2.5:
        return {
            "eligible": True, "grade_point": gpa, "eligibility_tier": "average",
            "eligible_countries": ["Australia", "UK"],
            "recommended_programs": ["Master's", "Graduate Diploma"],
            "notes": ["You qualify for several programmes."],
        }
    return {
        "eligible": False, "grade_point": gpa, "eligibility_tier": "foundation",
        "eligible_countries": [],
        "recommended_programs": ["Graduate Certificate"],
        "notes": ["Consider graduate certificate programmes as a pathway."],
    }


def assess_diploma_eligibility(document_data: dict, profile_data: dict) -> dict:
    """Eligibility for diploma holders."""
    gpa = float(document_data.get("gpa_normalized", 3.0))
    return {
        "eligible": True, "grade_point": gpa, "eligibility_tier": "average",
        "eligible_countries": ["Australia", "UK"],
        "recommended_programs": ["Bachelor's (Advanced Standing)", "Bachelor's (Year 2 Entry)"],
        "notes": ["Pathways available for diploma holders. Credit transfer possible."],
    }


# ─────────────────────────────────────────────
# MAIN DYNAMIC ASSESSMENT 
# ─────────────────────────────────────────────

def assess_eligibility_dynamic(profile_data: dict, document_data: dict) -> dict:
    """
    Route to the correct assessment function based on document_type.
    Returns a flat dict compatible with the existing app.py session state.

    Also attempts to use EligibilityVerificationAgent for richer results
    if UnifiedDataManager is available. Falls back gracefully.
    """
    doc_type = document_data.get("document_type", "")

    # ── Route to specific assessment ───────────────────────────────────
    if "A-Level" in doc_type or "GCE" in doc_type:
        base = assess_alevel_eligibility(document_data, profile_data)
    elif "Bachelor" in doc_type or "Degree" in doc_type:
        base = assess_bachelors_eligibility(document_data, profile_data)
    elif "Diploma" in doc_type:
        base = assess_diploma_eligibility(document_data, profile_data)
    else:
        # Generic fallback
        base = {
            "eligible": False, "grade_point": 0.0, "eligibility_tier": "unknown",
            "eligible_countries": [], "recommended_programs": [],
            "notes": [f"Unknown qualification type: '{doc_type}'. Please check your input."],
        }

    # ── Attempt enhanced assessment via EligibilityVerificationAgent ───
    try:
        import sys, os
        # Allow relative imports when run from project root
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        from core.database.manager import UnifiedDataManager
        from core.agents.eligibility_verification_agent import EligibilityVerificationAgent
        from core.agents.financial_feasibility_agent import FinancialFeasibilityAgent

        manager = UnifiedDataManager()
        unis    = manager.search_universities(
            country         = profile_data.get("country"),
            min_grade_point = base.get("grade_point", 0),
            program         = profile_data.get("program_interest"),
        )

        agent  = EligibilityVerificationAgent()
        report = agent.assess(profile_data, document_data, unis)

        # Merge rich report data back into the flat dict app.py expects
        base["_report"]               = report.to_dict()
        base["eligible_universities"] = [u.to_dict() for u in report.eligible_universities]
        base["borderline_universities"]  = [u.to_dict() for u in report.borderline_universities]

        # Financial feasibility assessment
        try:
            fin_agent = FinancialFeasibilityAgent()
            fin_report = fin_agent.assess(profile_data, unis)
            base["_financial_report"] = fin_report.to_dict()
            base["financial_feasible"] = fin_report.overall_feasible
            base["feasible_universities"] = [u.to_dict() for u in fin_report.feasible_universities]
            base["borderline_feasible_universities"] = [u.to_dict() for u in fin_report.borderline_universities]
            base["infeasible_universities"] = [u.to_dict() for u in fin_report.infeasible_universities]
            base["financial_recommendations"] = fin_report.global_recommendations
        except Exception as e:     
            base.setdefault("notes", []).append(f"Financial assessment unavailable: {e}")

        base["program_alignment"]     = report.program_alignment
        base["english_status"]        = report.english_status
        base["global_improvements"]   = report.global_improvements

        # Override eligible_countries from report if richer data available
        from_report = list({u.country for u in report.eligible_universities + report.borderline_universities})
        if from_report:
            base["eligible_countries"] = from_report

    except Exception as e:
        # Non-fatal — basic assessment already populated
        base.setdefault("notes", []).append(
            f"ℹ Enhanced eligibility check unavailable: {e}"
        )

    return base