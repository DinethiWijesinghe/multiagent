"""
Financial Feasibility Agent
============================
Assesses financial viability for international study options.

Features:
  - Tuition and living cost calculations with exchange rate adjustments
  - Budget matching against university costs
  - Scholarship recommendations and affordable alternatives
  - Handles external factors: exchange rate fluctuations, limited loans/scholarships

Algorithm (heuristic, non-ML):
  1. Resolve applicant budget into a consistent currency (LKR/GBP/USD/SGD/AUD).
  2. For each university:
     a. Normalize tuition and living costs into the same currency using exchange rates.
     b. Compute total annual cost and compare to the applicant budget.
     c. Tag as "feasible", "borderline", or "infeasible" based on thresholds.
     d. Suggest scholarships and lower-cost alternatives when a gap exists.
  3. Produce a report listing feasible/borderline/infeasible options plus action items.

External factors covered:
  - High tuition & living expenses
  - Exchange rate fluctuation
  - Limited loans/scholarships
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# ENUMS & DATA MODELS
# ─────────────────────────────────────────────

class FeasibilityStatus(Enum):
    FEASIBLE     = "feasible"
    BORDERLINE   = "borderline"
    INFEASIBLE   = "infeasible"


@dataclass
class CostBreakdown:
    tuition_annual:      float
    living_costs_annual: float
    total_annual:        float
    currency:            str
    exchange_rate:       float
    total_in_budget_currency: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScholarshipOption:
    name:           str
    type:           str  # merit, need-based, country-specific
    coverage:       str  # percentage or amount
    eligibility:    str
    deadline:       str
    website:        str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class UniversityFeasibility:
    university_id:       str
    university_name:     str
    country:             str
    status:              str
    cost_breakdown:      CostBreakdown
    budget_gap:          float  # positive = shortfall, negative = surplus
    feasibility_score:   float  # 0-1 scale
    scholarships:        List[ScholarshipOption]
    alternatives:        List[str]  # cheaper alternatives or funding options
    recommendations:     List[str]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cost_breakdown"] = self.cost_breakdown.to_dict()
        d["scholarships"] = [s.to_dict() for s in self.scholarships]
        return d


@dataclass
class FeasibilityReport:
    applicant_budget:         dict  # total_budget, currency
    overall_feasible:         bool
    feasible_universities:    List[UniversityFeasibility]
    borderline_universities:  List[UniversityFeasibility]
    infeasible_universities:  List[UniversityFeasibility]
    global_recommendations:   List[str]
    exchange_rate_notes:      List[str]

    def to_dict(self) -> dict:
        return {
            "applicant_budget":         self.applicant_budget,
            "overall_feasible":         self.overall_feasible,
            "feasible_universities":    [u.to_dict() for u in self.feasible_universities],
            "borderline_universities":  [u.to_dict() for u in self.borderline_universities],
            "infeasible_universities":  [u.to_dict() for u in self.infeasible_universities],
            "global_recommendations":   self.global_recommendations,
            "exchange_rate_notes":      self.exchange_rate_notes,
        }


# ─────────────────────────────────────────────
# FINANCIAL FEASIBILITY AGENT
# ─────────────────────────────────────────────

class FinancialFeasibilityAgent:
    """Financial viability assessor for international education."""

    # Estimated annual living costs by country (in local currency)
    LIVING_COSTS = {
        "UK":         {"amount": 12000, "currency": "GBP"},  # London-based estimate
        "Singapore": {"amount": 15000, "currency": "SGD"},  # Singapore estimate
        "Australia": {"amount": 20000, "currency": "AUD"},  # Major cities estimate
    }

    # Exchange rates MUST be provided via environment or configuration.
    # Hardcoded rates become stale immediately and erode trust in financial recommendations.
    EXCHANGE_RATES = None  # Loaded from environment at runtime

    # Scholarship database
    SCHOLARSHIPS = {
        "UK": [
            {
                "name": "Chevening Scholarship",
                "type": "merit",
                "coverage": "Full tuition + living expenses",
                "eligibility": "Master's applicants, leadership potential",
                "deadline": "November",
                "website": "https://www.chevening.org"
            },
            {
                "name": "Commonwealth Scholarship",
                "type": "country-specific",
                "coverage": "Full funding",
                "eligibility": "Commonwealth citizens",
                "deadline": "December",
                "website": "https://cscuk.fcdo.gov.uk"
            }
        ],
        "Singapore": [
            {
                "name": "ASEAN Undergraduate Scholarship",
                "type": "country-specific",
                "coverage": "Full tuition + living allowance",
                "eligibility": "ASEAN citizens, academic excellence",
                "deadline": "March",
                "website": "https://www.moe.gov.sg"
            },
            {
                "name": "NUS Global Merit Scholarship",
                "type": "merit",
                "coverage": "Up to SGD 17,000",
                "eligibility": "International undergraduates",
                "deadline": "January",
                "website": "https://www.nus.edu.sg"
            }
        ],
        "Australia": [
            {
                "name": "Australia Awards",
                "type": "country-specific",
                "coverage": "Full funding",
                "eligibility": "Developing country citizens",
                "deadline": "April",
                "website": "https://australiaawards.org"
            },
            {
                "name": "University Merit Scholarships",
                "type": "merit",
                "coverage": "Partial tuition",
                "eligibility": "Academic achievement",
                "deadline": "Varies",
                "website": "University websites"
            }
        ]
    }

    def __init__(self, universities_db_path: str = ""):
        # Default: resolve relative to project root
        if not universities_db_path:
            _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            universities_db_path = os.path.join(_root, "data", "databases", "universities_database.json")
        self.universities_db = self._load_universities_db(universities_db_path)
        # Load exchange rates from environment; fail fast if missing (no stale hardcoded rates)
        self._load_exchange_rates()

    # ── Public API ────────────────────────────────────────────────────────

    def assess(self,
               profile_data:  dict,
               universities:  list) -> FeasibilityReport:
        """Run full financial feasibility assessment."""

        budget_info = profile_data.get("financial", {})
        total_budget = float(budget_info.get("total_budget", 0))
        budget_currency = budget_info.get("budget_currency", "LKR")

        feasible_list   = []
        borderline_list = []
        infeasible_list = []

        exchange_notes = []

        for uni in universities:
            result = self._assess_university_feasibility(uni, total_budget, budget_currency)
            if result.status == FeasibilityStatus.FEASIBLE.value:
                feasible_list.append(result)
            elif result.status == FeasibilityStatus.BORDERLINE.value:
                borderline_list.append(result)
            else:
                infeasible_list.append(result)

            # Collect exchange rate notes
            if result.cost_breakdown.exchange_rate != 1.0:
                exchange_notes.append(
                    f"{uni['name']}: {result.cost_breakdown.currency} to {budget_currency} "
                    f"at {result.cost_breakdown.exchange_rate:.4f}"
                )

        feasible_list.sort(key=lambda x: x.feasibility_score, reverse=True)
        borderline_list.sort(key=lambda x: x.feasibility_score, reverse=True)

        global_recs = self._global_recommendations(
            total_budget, budget_currency,
            len(feasible_list) + len(borderline_list)
        )

        return FeasibilityReport(
            applicant_budget={"total_budget": total_budget, "currency": budget_currency},
            overall_feasible=len(feasible_list) > 0 or len(borderline_list) > 0,
            feasible_universities=feasible_list,
            borderline_universities=borderline_list,
            infeasible_universities=infeasible_list,
            global_recommendations=global_recs,
            exchange_rate_notes=list(set(exchange_notes)),  # unique notes
        )

    # ── Per-university ────────────────────────────────────────────────────

    def _assess_university_feasibility(self, uni: dict, budget: float, budget_currency: str) -> UniversityFeasibility:
        country = uni.get("country", "")
        tuition_info = uni.get("tuition", {})
        currency = tuition_info.get("currency", "GBP")

        # Get tuition (assume undergraduate for now)
        tuition_annual = tuition_info.get("undergraduate_intl_gbp", 25000)  # fallback

        # Get living costs
        living_info = self.LIVING_COSTS.get(country, {"amount": 15000, "currency": "GBP"})
        living_annual = living_info["amount"]

        # Convert to common currency for calculation (use GBP as base)
        tuition_gbp = self._convert_currency(tuition_annual, currency, "GBP")
        living_gbp = self._convert_currency(living_annual, living_info["currency"], "GBP")
        total_gbp = tuition_gbp + living_gbp

        # Convert to budget currency
        exchange_rate = self._get_exchange_rate("GBP", budget_currency)
        total_budget_currency = total_gbp * exchange_rate

        cost_breakdown = CostBreakdown(
            tuition_annual=tuition_annual,
            living_costs_annual=living_annual,
            total_annual=total_gbp,
            currency="GBP",
            exchange_rate=exchange_rate,
            total_in_budget_currency=total_budget_currency
        )

        # Calculate gap
        budget_gap = total_budget_currency - budget

        # Feasibility score (0-1, higher is better)
        if budget <= 0:
            feasibility_score = 0.0
        else:
            coverage_ratio = min(budget / total_budget_currency, 2.0)  # cap at 200%
            feasibility_score = coverage_ratio / 2.0

        # Determine status
        if feasibility_score >= 0.8:
            status = FeasibilityStatus.FEASIBLE.value
        elif feasibility_score >= 0.5:
            status = FeasibilityStatus.BORDERLINE.value
        else:
            status = FeasibilityStatus.INFEASIBLE.value

        # Get scholarships
        scholarships = self._get_scholarships(country)

        # Generate alternatives and recommendations
        alternatives = []
        recommendations = []

        if budget_gap > 0:
            shortfall_pct = (budget_gap / total_budget_currency) * 100
            recommendations.append(f"Budget shortfall: {shortfall_pct:.1f}%")

            if shortfall_pct > 50:
                alternatives.append("Consider universities in your home country")
                alternatives.append("Look for fully-funded scholarships")
            else:
                alternatives.append("Apply for partial scholarships or education loans")
                alternatives.append("Consider part-time work opportunities")

        if scholarships:
            recommendations.append(f"Apply for {len(scholarships)} available scholarships")

        return UniversityFeasibility(
            university_id=uni.get("id", ""),
            university_name=uni["name"],
            country=country,
            status=status,
            cost_breakdown=cost_breakdown,
            budget_gap=budget_gap,
            feasibility_score=round(feasibility_score, 3),
            scholarships=scholarships,
            alternatives=alternatives,
            recommendations=recommendations,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _convert_currency(self, amount: float, from_curr: str, to_curr: str) -> float:
        """Convert amount from one currency to another.
        
        Raises:
            RuntimeError: If exchange rate not available. No silent fallback to 1:1;
                returning incorrect conversions would compromise financial feasibility assessment.
        """
        if from_curr == to_curr:
            return amount
        if from_curr in self.EXCHANGE_RATES and to_curr in self.EXCHANGE_RATES[from_curr]:
            return amount * self.EXCHANGE_RATES[from_curr][to_curr]
        raise RuntimeError(
            f"Exchange rate {from_curr}->{to_curr} not configured. "
            f"Cannot assess financial feasibility without accurate conversion rates. "
            f"Update EXCHANGE_RATES environment variable with missing rate pair."
        )

    def _get_exchange_rate(self, from_curr: str, to_curr: str) -> float:
        """Get exchange rate from currency A to B.
        
        Raises:
            RuntimeError: If exchange rate not available (same as _convert_currency).
        """
        if from_curr == to_curr:
            return 1.0
        if from_curr in self.EXCHANGE_RATES and to_curr in self.EXCHANGE_RATES[from_curr]:
            return self.EXCHANGE_RATES[from_curr][to_curr]
        raise RuntimeError(
            f"Exchange rate {from_curr}->{to_curr} not configured. "
            f"Cannot assess financial feasibility without accurate conversion rates. "
            f"Update EXCHANGE_RATES environment variable with missing rate pair."
        )

    def _get_scholarships(self, country: str) -> List[ScholarshipOption]:
        """Get available scholarships for a country."""
        if country not in self.SCHOLARSHIPS:
            return []
        return [ScholarshipOption(**s) for s in self.SCHOLARSHIPS[country]]

    def _global_recommendations(self, budget: float, currency: str, feasible_count: int) -> List[str]:
        """Generate global recommendations based on overall situation."""
        recs = []

        if budget <= 0:
            recs.append("Please provide your total budget for accurate assessment")
            return recs

        if feasible_count == 0:
            recs.append("Your budget may be insufficient for the selected universities")
            recs.append("Consider exploring scholarship opportunities")
            recs.append("Look into education loans or part-time work options")
            recs.append("Consider more affordable study destinations")
        elif feasible_count < 3:
            recs.append("Limited feasible options - explore additional funding sources")
            recs.append("Consider universities with lower tuition fees")
        else:
            recs.append("Good financial alignment with multiple universities")
            recs.append("Still research scholarships to reduce financial burden")

        # Currency-specific notes
        if currency == "LKR":
            recs.append("Monitor LKR exchange rate fluctuations closely")
            recs.append("Consider maintaining funds in stable currencies")

        return recs

    def _load_exchange_rates(self) -> None:
        """Load exchange rates from environment (JSON string) or fail explicitly.
        
        Format: EXCHANGE_RATES='{ "GBP": {"USD": 1.27}, "USD": {"GBP": 0.79}, ...}'
        
        Raises:
            RuntimeError: If EXCHANGE_RATES not set or invalid. This is intentional:
                stale exchange rates cause incorrect cost calculations that destroy user trust.
                Better to fail loudly at startup than produce wrong recommendations.
        """
        rates_str = os.environ.get("EXCHANGE_RATES", "").strip()
        if not rates_str:
            raise RuntimeError(
                "EXCHANGE_RATES environment variable not configured. "
                "Financial feasibility assessment requires current exchange rates. "
                "Set EXCHANGE_RATES as a JSON object with currency pairs and their conversion rates. "
                "Example: EXCHANGE_RATES='{\"GBP\": {\"USD\": 1.27}, \"USD\": {\"GBP\": 0.79}}' "
                "Use a live rate service (OpenExchangeRates, XE, ECB) to prevent stale data."
            )
        try:
            self.EXCHANGE_RATES = json.loads(rates_str)
            if not isinstance(self.EXCHANGE_RATES, dict):
                raise ValueError("EXCHANGE_RATES must be a JSON object")
            logger.info(f"Loaded exchange rates for {len(self.EXCHANGE_RATES)} source currencies.")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"EXCHANGE_RATES environment variable is not valid JSON: {e}")

    def _load_universities_db(self, path: str) -> Optional[dict]:
        """Load universities database."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Universities database not found: {path}")
            return None
        except Exception as e:
            logger.error(f"Failed to load universities database: {e}")
            return None