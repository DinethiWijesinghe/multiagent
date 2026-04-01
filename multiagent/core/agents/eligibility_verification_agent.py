"""
Eligibility Verification Agent
================================
Multi-criteria eligibility checking against real university requirements.

Features:
  - University-specific requirement validation (not just generic thresholds)
  - Per-criterion pass/fail with detailed reasons
  - Personalized improvement recommendations
  - English proficiency checks
  - Program-stream alignment checks
  - Confidence scoring on assessment
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# ENUMS & DATA MODELS
# ─────────────────────────────────────────────

class EligibilityTier(Enum):
    TOP        = "top"
    GOOD       = "good"
    AVERAGE    = "average"
    FOUNDATION = "foundation"


class CriterionStatus(Enum):
    PASS    = "pass"
    FAIL    = "fail"
    WARNING = "warning"


@dataclass
class CriterionResult:
    criterion:    str
    status:       str
    label:        str
    your_value:   str
    required:     str
    message:      str
    weight:       float = 1.0


@dataclass
class UniversityEligibility:
    university_id:    str
    university_name:  str
    country:          str
    overall_eligible: bool
    criteria_results: list
    match_score:      float
    tier_match:       str
    improvements:     list
    notes:            list

    def to_dict(self) -> dict:
        d = asdict(self)
        d["criteria_results"] = [asdict(c) for c in self.criteria_results]
        return d


@dataclass
class EligibilityReport:
    applicant_summary:       dict
    overall_eligible:        bool
    grade_point:             float
    tier:                    str
    eligible_universities:   list
    borderline_universities: list
    ineligible_universities: list
    global_improvements:     list
    english_status:          dict
    program_alignment:       str

    def to_dict(self) -> dict:
        return {
            "applicant_summary":       self.applicant_summary,
            "overall_eligible":        self.overall_eligible,
            "grade_point":             self.grade_point,
            "tier":                    self.tier,
            "eligible_universities":   [u.to_dict() for u in self.eligible_universities],
            "borderline_universities": [u.to_dict() for u in self.borderline_universities],
            "ineligible_universities": [u.to_dict() for u in self.ineligible_universities],
            "global_improvements":     self.global_improvements,
            "english_status":          self.english_status,
            "program_alignment":       self.program_alignment,
        }

    # Compatibility: allow dict-style access for app.py
    def get(self, key, default=None):
        return self.to_dict().get(key, default)


# ─────────────────────────────────────────────
# ELIGIBILITY VERIFICATION AGENT
# ─────────────────────────────────────────────

class EligibilityVerificationAgent:
    """Multi-criteria eligibility checker with per-university pass/fail."""

    GRADE_MAP = {"A": 4.0, "B": 3.0, "C": 2.0, "S": 1.0, "F": 0.0}

    PROGRAM_MIN_GPA = {
        "Medicine":         3.9,
        "Law":              3.7,
        "Engineering":      3.3,
        "Computer Science": 3.3,
        "Business":         3.0,
        "Science":          3.0,
        "Arts":             2.8,
        "IT":               2.8,
        "Education":        2.8,
    }

    STREAM_PROGRAMS = {
        "Physical Science": ["Engineering", "Computer Science", "Science", "Medicine", "IT"],
        "Bio Science":      ["Medicine", "Science", "Pharmacy", "Biology", "Engineering"],
        "Commerce":         ["Business", "Economics", "Accountancy", "Law", "IT"],
        "Arts":             ["Arts", "Law", "Social Science", "Education", "Business"],
        "Technology":       ["Engineering", "IT", "Computer Science", "Design"],
    }

    ENGLISH_REQUIREMENTS = {
        "UK":        {"ielts": 6.5, "toefl": 90, "pte": 58},
        "Singapore": {"ielts": 6.0, "toefl": 85, "pte": 54},
        "Australia": {"ielts": 6.5, "toefl": 79, "pte": 58},
    }

    _DEFAULT_ENGLISH_REQUIREMENT = {"ielts": 6.5, "toefl": 90, "pte": 58}

    def __init__(self, training_data_path: str = ""):
        # Default: resolve relative to project root regardless of working directory
        if not training_data_path:
            _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            training_data_path = os.path.join(_root, "data", "training", "eligibility_training_data.json")
        self.training_data = self._load_training_data(training_data_path)
        if self.training_data:
            self._special_reqs        = self.training_data.get("university_specific_requirements", {})
            self._country_english_reqs = self.training_data.get("country_english_requirements", {})
            self._program_reqs        = self.training_data.get("program_specific_requirements", {})
            self._improvement_library = self.training_data.get("improvement_recommendations", {})
        else:
            self._special_reqs        = {}
            self._country_english_reqs = {}
            self._program_reqs        = {}
            self._improvement_library = {}

    # ── Public API ────────────────────────────────────────────────────────

    def assess(self,
               profile_data:  dict,
               document_data: dict,
               universities:  list) -> EligibilityReport:
        """Run full multi-criteria eligibility assessment."""

        grade_point = self._compute_grade_point(document_data)
        tier        = self._compute_tier(grade_point)

        english_status    = self._check_english(
            document_data,
            profile_data.get("country", "UK"),
            profile_data.get("program_interest", ""),
        )
        program_alignment = self._check_program_alignment(
            profile_data.get("stream", ""),
            profile_data.get("program_interest", "")
        )

        eligible_list   = []
        borderline_list = []
        ineligible_list = []

        for uni in universities:
            result = self._assess_university(uni, profile_data, document_data, grade_point, english_status)
            if result.overall_eligible:
                (eligible_list if result.match_score >= 0.8 else borderline_list).append(result)
            else:
                ineligible_list.append(result)

        eligible_list.sort(key=lambda x: x.match_score, reverse=True)
        borderline_list.sort(key=lambda x: x.match_score, reverse=True)

        global_improvements = self._global_recommendations(
            grade_point, english_status,
            profile_data.get("stream", ""),
            profile_data.get("program_interest", ""),
            document_data.get("document_type", "")
        )

        return EligibilityReport(
            applicant_summary={
                "name":             profile_data.get("full_name", "Applicant"),
                "qualification":    document_data.get("document_type", ""),
                "grade_point":      grade_point,
                "tier":             tier.value,
                "country_target":   profile_data.get("country", ""),
                "program_interest": profile_data.get("program_interest", ""),
                "english_ok":       english_status.get("overall_pass", False),
            },
            overall_eligible        = len(eligible_list) > 0 or len(borderline_list) > 0,
            grade_point             = grade_point,
            tier                    = tier.value,
            eligible_universities   = eligible_list,
            borderline_universities = borderline_list,
            ineligible_universities = ineligible_list,
            global_improvements     = global_improvements,
            english_status          = english_status,
            program_alignment       = program_alignment,
        )

    # ── Per-university ──────────────────────────────────────────────────── 

    def _assess_university(self, uni, profile, document, grade_point, english_status) -> UniversityEligibility:
        criteria, improvements, notes = [], [], []
        uni_id  = uni.get("id", "")
        country = uni.get("country", "")

        # Criterion 1 — Grade Point
        uni_min_gpa = uni.get("acceptance_criteria", {}).get("min_grade_point", 3.0)
        if uni_id in self._special_reqs:
            uni_min_gpa = self._special_reqs[uni_id].get("minimum_gpa_override", uni_min_gpa)

        gpa_gap  = grade_point - uni_min_gpa
        gpa_pass = gpa_gap >= 0
        gpa_warn = not gpa_pass and gpa_gap >= -0.2
        gpa_status = CriterionStatus.PASS if gpa_pass else (CriterionStatus.WARNING if gpa_warn else CriterionStatus.FAIL)

        criteria.append(CriterionResult(
            criterion="grade_point", status=gpa_status.value, label="Academic Grade Point",
            your_value=f"{grade_point:.2f} / 4.0", required=f"{uni_min_gpa:.1f} / 4.0 minimum",
            message=(
                "Your GPA meets the requirement." if gpa_pass
                else f"You are {abs(gpa_gap):.2f} pts below the minimum."
                     + (" Borderline — worth applying." if gpa_warn else "")
            ), weight=1.0,
        ))
        if not gpa_pass:
            improvements.append(f"Improve grades to {uni_min_gpa:.1f} GPA for {uni['name']}")

        target_program  = profile.get("program_interest", "")
        available_progs = uni.get("programs", [])
        prog_available  = target_program in available_progs
        program_min     = self.PROGRAM_MIN_GPA.get(target_program, 3.0)

        # Criterion 2 — English
        req       = self._resolve_english_requirements(uni, target_program, country)
        uni_ielts = req["ielts"]
        uni_toefl = req["toefl"]
        uni_pte   = req["pte"]
        ielts_val = english_status.get("ielts_score")
        toefl_val = english_status.get("toefl_score")
        pte_val   = english_status.get("pte_score")
        uni_eng_pass = (
            (ielts_val is not None and ielts_val >= uni_ielts) or
            (toefl_val is not None and toefl_val >= uni_toefl) or
            (pte_val is not None and pte_val >= uni_pte)
        )

        criteria.append(CriterionResult(
            criterion="english_proficiency",
            status=CriterionStatus.PASS.value if uni_eng_pass else CriterionStatus.FAIL.value,
            label="English Proficiency",
            your_value=english_status.get("score_summary", "Not provided"),
            required=f"IELTS {uni_ielts} or TOEFL {uni_toefl} or PTE {uni_pte}",
            message="English requirement met." if uni_eng_pass
                    else f"IELTS {uni_ielts}+ or TOEFL {uni_toefl}+ or PTE {uni_pte}+ required",
            weight=0.9,
        ))
        if not uni_eng_pass:
            improvements.append(
                f"Take IELTS (target {uni_ielts}+), TOEFL (target {uni_toefl}+), or PTE (target {uni_pte}+)."
            )

        # Criterion 3 — Program availability
        criteria.append(CriterionResult(
            criterion="program_availability",
            status=CriterionStatus.PASS.value if prog_available else CriterionStatus.FAIL.value,
            label=f"Program: {target_program}",
            your_value=target_program, required=f"Available at {uni['name']}",
            message=(f"{target_program} is offered here." if prog_available
                     else f"Not offered. Available: {', '.join(available_progs[:4])}"),
            weight=0.8,
        ))

        if prog_available and grade_point < program_min:
            criteria.append(CriterionResult(
                criterion="program_grade_requirement", status=CriterionStatus.FAIL.value,
                label=f"{target_program} Grade Requirement",
                your_value=f"{grade_point:.2f}", required=f"{program_min:.1f} min for {target_program}",
                message=f"{target_program} requires {program_min:.1f}+; you have {grade_point:.2f}",
                weight=0.9,
            ))
            improvements.append(f"{target_program} requires GPA {program_min:.1f}+")

        # Criterion 4 — Special requirements
        if uni_id in self._special_reqs:
            for req_note in self._special_reqs[uni_id].get("special_requirements", []):
                criteria.append(CriterionResult(
                    criterion="special_requirement", status=CriterionStatus.WARNING.value,
                    label="Additional Requirement", your_value="See notes",
                    required=req_note, message=f"⚠️ {req_note}", weight=0.5,
                ))
            if self._special_reqs[uni_id].get("notes"):
                notes.append(self._special_reqs[uni_id]["notes"])

        # Overall score
        score_map    = {"pass": 1.0, "warning": 0.5, "fail": 0.0}
        total_weight = sum(c.weight for c in criteria) or 1.0
        match_score  = sum(score_map[c.status] * c.weight for c in criteria) / total_weight

        hard = [c for c in criteria if c.weight >= 0.8]
        hard_pass   = all(c.status == "pass"    for c in hard)
        any_warn    = any(c.status == "warning"  for c in hard)
        overall_eligible = hard_pass or (any_warn and match_score >= 0.6)

        return UniversityEligibility(
            university_id=uni_id, university_name=uni["name"], country=country,
            overall_eligible=overall_eligible, criteria_results=criteria,
            match_score=round(match_score, 3),
            tier_match=self._tier_match_label(grade_point, uni_min_gpa),
            improvements=list(set(improvements)), notes=notes,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _compute_grade_point(self, document_data: dict) -> float:
        doc_type = document_data.get("document_type", "")
        if "A-Level" in doc_type:
            subjects = document_data.get("subjects", {})
            SPECIAL  = {"general english", "common general test", "cgt"}
            grades   = [
                self.GRADE_MAP.get(str(g).upper(), 0.0)
                for s, g in subjects.items()
                if s.lower() not in SPECIAL
            ]
            return round(sum(grades) / len(grades), 2) if grades else 0.0
        if "Bachelor" in doc_type or "Degree" in doc_type:
            return float(document_data.get("gpa_normalized", 0.0))
        if "Diploma" in doc_type:
            return float(document_data.get("gpa_normalized", 3.0))
        return 0.0

    def _compute_tier(self, gpa: float) -> EligibilityTier:
        if gpa >= 3.7: return EligibilityTier.TOP
        if gpa >= 3.3: return EligibilityTier.GOOD
        if gpa >= 3.0: return EligibilityTier.AVERAGE
        return EligibilityTier.FOUNDATION

    def _check_english(self, document_data: dict, country: str, program: str = "") -> dict:
        eng  = document_data.get("english_proficiency", {}) or {}
        req  = self._resolve_english_requirements(None, program, country)

        ielts = (eng.get("ielts") or {}).get("overall") if isinstance(eng.get("ielts"), dict) else eng.get("ielts")
        toefl = eng.get("toefl")
        pte   = eng.get("pte")

        ielts_ok = ielts is not None and float(ielts) >= req["ielts"]
        toefl_ok = toefl is not None and float(toefl) >= req["toefl"]
        pte_ok   = pte   is not None and float(pte)   >= req.get("pte", 58)

        parts = []
        if ielts is not None: parts.append(f"IELTS {ielts}")
        if toefl is not None: parts.append(f"TOEFL {toefl}")
        if pte   is not None: parts.append(f"PTE {pte}")

        return {
            "overall_pass":  ielts_ok or toefl_ok or pte_ok,
            "ielts_score":   float(ielts) if ielts is not None else None,
            "toefl_score":   float(toefl) if toefl is not None else None,
            "pte_score":     float(pte)   if pte   is not None else None,
            "ielts_ok": ielts_ok, "toefl_ok": toefl_ok, "pte_ok": pte_ok,
            "country_req":   req,
            "score_summary": " | ".join(parts) if parts else "Not provided",
        }

    def _resolve_english_requirements(
        self,
        uni: Optional[dict],
        program: str,
        country: str,
    ) -> dict:
        req = dict(self._DEFAULT_ENGLISH_REQUIREMENT)

        country_req = self._country_english_reqs.get(country) or self.ENGLISH_REQUIREMENTS.get(country)
        if isinstance(country_req, dict):
            req.update({k: v for k, v in country_req.items() if v is not None})

        program_req = self._program_reqs.get(program, {}).get("english_requirements", {})
        if isinstance(program_req, dict):
            req.update({k: v for k, v in program_req.items() if v is not None})

        if uni:
            acceptance = uni.get("acceptance_criteria", {}) or {}
            req.update({
                "ielts": acceptance.get("ielts_min", req["ielts"]),
                "toefl": acceptance.get("toefl_min", req["toefl"]),
                "pte": acceptance.get("pte_min", req["pte"]),
            })

            uni_id = uni.get("id", "")
            special_req = self._special_reqs.get(uni_id, {}).get("english_requirements", {})
            if isinstance(special_req, dict):
                req.update({k: v for k, v in special_req.items() if v is not None})

        return req

    def _check_program_alignment(self, stream: str, program: str) -> str:
        if not stream or not program:
            return "No stream/program information provided."
        compatible = self.STREAM_PROGRAMS.get(stream, [])
        if program in compatible:
            return f"✅ {stream} stream aligns well with {program}."
        return (
            f"⚠️ {stream} stream may not directly align with {program}. "
            f"Compatible: {', '.join(compatible[:4])}. Contact admissions for assessment."
        )

    def _tier_match_label(self, gpa: float, uni_min: float) -> str:
        diff = gpa - uni_min
        if diff >= 0.3:  return "strong_match"
        if diff >= 0.0:  return "meets_minimum"
        if diff >= -0.2: return "borderline"
        return "below_minimum"

    def _global_recommendations(self, gpa, english, stream, program, doc_type) -> list:
        recs = []
        if gpa < 3.0:
            recs += self._improvement_library.get("grade_below_minimum",
                ["Consider foundation/pathway programmes", "Retake exams to improve grades"])
        if not english.get("overall_pass"):
            recs += self._improvement_library.get("no_english_proficiency",
                ["Take IELTS Academic (min 6.5) or TOEFL iBT (min 79)"])
        if stream and program and program not in self.STREAM_PROGRAMS.get(stream, []):
            recs += self._improvement_library.get("wrong_stream",
                ["Foundation programmes can bridge stream gaps"])
        if 3.0 <= gpa < 3.4:
            recs += self._improvement_library.get("gpa_borderline",
                ["Write a strong personal statement", "Highlight extracurricular activities"])
        if "Diploma" in doc_type:
            recs.append("Request advanced standing credit assessment from target universities")
        seen, unique = set(), []
        for r in recs:
            if r not in seen:
                seen.add(r); unique.append(r)
        return unique

    def _load_training_data(self, path: str) -> Optional[dict]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Training data not found: {path}")
            return None
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return None