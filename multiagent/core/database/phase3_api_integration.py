"""
Phase 3: API Integration + ML Intelligence
============================================
Location: multiagent/core/phases/phase3.py

Combines:
  - APIIntegrator      → fetches data from 3 open-source APIs
  - Phase3MLEngine     → forecasts exchange rates, smart cache decisions

Open-Source Data Attribution:
  - Hipo University Domains List : github.com/Hipo/university-domains-list  (MIT)
  - REST Countries API           : restcountries.com                          (MPL 2.0)
  - Open Exchange Rates          : open.er-api.com                            (free tier)

Usage:
    from core.phases.phase3 import Phase3

    p3 = Phase3()
    p3.ml.train()

    # Fetch real data
    rates   = p3.api.fetch_exchange_rates()
    meta    = p3.api.fetch_country_metadata("UK")
    enriched = p3.api.enrich_universities(unis, "UK")

    # ML intelligence
    forecast = p3.ml.forecast_exchange_rate("GBP", months_ahead=6)
    refresh  = p3.ml.should_refresh_cache("exchange_rates", hours_since_last=14)
    impact   = p3.ml.budget_impact_analysis(5_000_000, "UK", program_years=3)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

try:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("[Phase3] scikit-learn not found — run: pip install scikit-learn")

logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────

_ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE_DIR = os.path.join(_ROOT, "data", "scraped")
_MODEL_DIR = os.path.join(_ROOT, "data", "ml_models")

# ─── Historical LKR rates (training data) ────────────────────────────────────

HISTORICAL_RATES_LKR = {
    "GBP": [
        (2022,1,370),(2022,3,380),(2022,6,395),(2022,9,420),(2022,12,445),
        (2023,3,455),(2023,6,460),(2023,9,458),(2023,12,462),(2024,3,468),
        (2024,6,475),(2024,9,480),(2024,12,488),(2025,3,495),(2025,6,502),
        (2025,9,510),(2025,12,515),(2026,1,518),(2026,2,520),(2026,3,522),
    ],
    "SGD": [
        (2022,1,210),(2022,3,215),(2022,6,220),(2022,9,228),(2022,12,235),
        (2023,3,238),(2023,6,242),(2023,9,240),(2023,12,243),(2024,3,248),
        (2024,6,252),(2024,9,255),(2024,12,258),(2025,3,262),(2025,6,266),
        (2025,9,270),(2025,12,272),(2026,1,274),(2026,2,275),(2026,3,276),
    ],
    "AUD": [
        (2022,1,200),(2022,3,205),(2022,6,208),(2022,9,215),(2022,12,220),
        (2023,3,222),(2023,6,225),(2023,9,223),(2023,12,226),(2024,3,230),
        (2024,6,234),(2024,9,237),(2024,12,240),(2025,3,243),(2025,6,247),
        (2025,9,250),(2025,12,252),(2026,1,253),(2026,2,254),(2026,3,255),
    ],
}


def _months_since_2020(year: int, month: int) -> int:
    return (year - 2020) * 12 + (month - 1)


# ══════════════════════════════════════════════════════════════════════════════
#  API INTEGRATOR
# ══════════════════════════════════════════════════════════════════════════════

class APIIntegrator:
    """
    Fetches data from 3 free/open-source APIs.
    All responses cached locally — app works offline after first run.
    """

    HIPO_BASE      = "http://universities.hipolabs.com"
    REST_COUNTRIES = "https://restcountries.com/v3.1"
    EXCHANGE_RATES = "https://open.er-api.com/v6/latest/USD"
    RATE_LIMIT_SEC = 1.0

    COUNTRY_NAMES = {
        "UK": "United Kingdom",
        "Singapore": "Singapore",
        "Australia": "Australia",
    }

    def __init__(self, cache_dir: str = _CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._rates: Dict[str, float] = {}
        self._stats = {
            "hipo_calls": 0, "countries_calls": 0, "exchange_calls": 0,
            "hipo_hits": 0,  "hipo_misses": 0,     "cache_loads": 0,
            "last_exchange_at": "never",
        }
        # Preload cached rates
        cached = self._load_cache("exchange_rates.json")
        if cached:
            self._rates = cached.get("rates", {})
            self._stats["last_exchange_at"] = cached.get("fetched_at", "cached")

    # ── Public ────────────────────────────────────────────────────────────────

    def enrich_universities(self, universities: List[Dict], country: str) -> List[Dict]:
        """Query Hipo API and attach domain/web data to each university."""
        country_name = self.COUNTRY_NAMES.get(country, country)
        print(f"\n  Phase3 API — Hipo enrichment for {country} ({len(universities)} universities)...")
        enriched = []
        for uni in universities:
            result = dict(uni)
            try:
                hit = self._hipo_lookup(uni["name"], country_name)
                if hit:
                    result["api_verified"] = True
                    result["domains"]      = hit.get("domains", [])
                    result["web_pages"]    = hit.get("web_pages", [])
                    self._stats["hipo_hits"] += 1
                else:
                    result["api_verified"] = False
                    self._stats["hipo_misses"] += 1
            except Exception as exc:
                logger.warning("Hipo lookup failed for %s: %s", uni.get("name"), exc)
                result["api_verified"] = False
            enriched.append(result)
            time.sleep(self.RATE_LIMIT_SEC)
        verified = sum(1 for u in enriched if u.get("api_verified"))
        print(f"   Phase3 Hipo: {verified}/{len(universities)} verified")
        return enriched

    def fetch_country_metadata(self, country: str) -> Optional[Dict]:
        """Fetch and cache country info from restcountries.com."""
        cache_key = f"country_{country}.json"
        cached    = self._load_cache(cache_key)
        if cached:
            return cached

        country_name = self.COUNTRY_NAMES.get(country, country)
        url  = f"{self.REST_COUNTRIES}/name/{country_name.replace(' ','%20')}?fullText=true"
        raw  = self._get_json(url)
        self._stats["countries_calls"] += 1

        if not raw or not isinstance(raw, list):
            return None

        e = raw[0]
        meta = {
            "country_key":   country,
            "official_name": e.get("name", {}).get("official", country_name),
            "common_name":   e.get("name", {}).get("common",   country_name),
            "capital":       (e.get("capital") or ["Unknown"])[0],
            "region":        e.get("region", ""),
            "languages":     list((e.get("languages") or {}).values()),
            "currencies":    [
                {"code": c, "name": i.get("name",""), "symbol": i.get("symbol","")}
                for c, i in (e.get("currencies") or {}).items()
            ],
            "flag_emoji":    e.get("flag", ""),
            "fetched_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._save_cache(meta, cache_key)
        return meta

    def fetch_exchange_rates(self) -> Dict[str, float]:
        """Fetch live USD exchange rates from open.er-api.com."""
        raw = self._get_json(self.EXCHANGE_RATES)
        self._stats["exchange_calls"] += 1

        if not raw or raw.get("result") != "success":
            cached = self._load_cache("exchange_rates.json")
            if cached:
                self._rates = cached.get("rates", {})
                return self._rates
            self._rates = {"GBP": 0.79, "SGD": 1.34, "AUD": 1.54, "LKR": 325.0}
            return self._rates

        rates = raw.get("rates", {})
        payload = {
            "base":       "USD",
            "rates":      rates,
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._save_cache(payload, "exchange_rates.json")
        self._rates = rates
        self._stats["last_exchange_at"] = payload["fetched_at"]
        return rates

    def normalize_tuition_to_usd(self, amount: float, currency: str) -> Optional[float]:
        """Convert amount in given currency to USD using cached rates."""
        if not self._rates:
            self.fetch_exchange_rates()
        rate = self._rates.get(currency)
        return round(amount / rate, 2) if rate and rate > 0 else None

    def get_statistics(self) -> Dict:
        return {**self._stats, "currencies_loaded": len(self._rates)}

    # ── Private ───────────────────────────────────────────────────────────────

    def _hipo_lookup(self, name: str, country_name: str) -> Optional[Dict]:
        safe_key   = "".join(c if c.isalnum() else "_" for c in name[:40]).lower()
        cache_file = f"hipo_{safe_key}.json"
        cached     = self._load_cache(cache_file)
        if cached is not None:
            self._stats["hipo_calls"] += 1
            return cached if cached else None

        params = f"name={name.replace(' ','+')}&country={country_name.replace(' ','+')}"
        url    = f"{self.HIPO_BASE}/search?{params}"
        self._stats["hipo_calls"] += 1
        data = self._get_json(url)
        if not data or not isinstance(data, list):
            self._save_cache({}, cache_file)
            return None
        name_lower = name.lower()
        for entry in data:
            if name_lower in entry.get("name", "").lower():
                self._save_cache(entry, cache_file)
                return entry
        if len(data) == 1:
            self._save_cache(data[0], cache_file)
            return data[0]
        self._save_cache({}, cache_file)
        return None

    def _get_json(self, url: str) -> Optional[Dict]:
        try:
            import urllib.request
            req = urllib.request.Request(
                url, headers={"User-Agent": "UniAssist/1.0 (academic project)"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.warning("GET %s failed: %s", url, exc)
            return None

    def _save_cache(self, data, filename: str):
        try:
            with open(os.path.join(self.cache_dir, filename), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as exc:
            logger.warning("Cache write failed (%s): %s", filename, exc)

    def _load_cache(self, filename: str):
        try:
            with open(os.path.join(self.cache_dir, filename), "r", encoding="utf-8") as f:
                self._stats["cache_loads"] += 1
                return json.load(f)
        except FileNotFoundError:
            return None
        except Exception as exc:
            logger.warning("Cache read failed (%s): %s", filename, exc)
            return None


# ══════════════════════════════════════════════════════════════════════════════
#  ML ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class Phase3MLEngine:
    """
    ML intelligence layer for Phase 3.
    Polynomial regression forecasts exchange rates and guides cache refresh.
    """

    def __init__(self):
        self.models  = {}
        self.trained = False
        self._try_load()

    def _try_load(self):
        if not ML_AVAILABLE:
            return
        path = os.path.join(_MODEL_DIR, "phase3_rate_models.pkl")
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self.models = pickle.load(f)
                self.trained = len(self.models) > 0
                print(f"[Phase3-ML] Loaded models: {list(self.models.keys())}")
            except Exception as e:
                print(f"[Phase3-ML] Load failed: {e}")

    def train(self, verbose=True) -> dict:
        if not ML_AVAILABLE:
            return {"error": "scikit-learn not available"}
        metrics = {}
        for currency, history in HISTORICAL_RATES_LKR.items():
            X = np.array([[_months_since_2020(y, m)] for y, m, _ in history])
            y = np.array([r for _, _, r in history])
            model = Pipeline([
                ("poly",   PolynomialFeatures(degree=2, include_bias=True)),
                ("scaler", StandardScaler()),
                ("reg",    Ridge(alpha=1.0)),
            ])
            model.fit(X, y)
            mae = mean_absolute_error(y, model.predict(X))
            self.models[currency] = model
            metrics[currency] = {"mae_lkr": round(mae, 2), "data_points": len(history)}
            if verbose:
                print(f"[Phase3-ML] {currency} trained | MAE: {mae:.1f} LKR")
        self.trained = True
        os.makedirs(_MODEL_DIR, exist_ok=True)
        with open(os.path.join(_MODEL_DIR, "phase3_rate_models.pkl"), "wb") as f:
            pickle.dump(self.models, f)
        return {"currencies": list(metrics.keys()), "metrics": metrics,
                "trained_at": datetime.now().isoformat()}

    def forecast_exchange_rate(self, currency: str, months_ahead: int = 6) -> dict:
        """Forecast LKR exchange rate N months ahead."""
        if not ML_AVAILABLE or not self.trained or currency not in self.models:
            return self._fallback_forecast(currency, months_ahead)
        now      = datetime.now()
        base_m   = _months_since_2020(now.year, now.month)
        current  = HISTORICAL_RATES_LKR[currency][-1][2]
        X_future = np.array([[m] for m in range(base_m+1, base_m+months_ahead+1)])
        y_future = self.models[currency].predict(X_future)
        forecast = float(y_future[-1])
        change   = (forecast - current) / current * 100
        monthly  = []
        for i, rate in enumerate(y_future):
            mi = base_m + 1 + i
            monthly.append((f"{2020 + mi//12}-{(mi%12)+1:02d}", round(float(rate), 0)))
        if change > 5:   risk, trend = "High",   "Weakening LKR (costs rising)"
        elif change > 2: risk, trend = "Medium",  "Moderate LKR weakening"
        elif change < -2:risk, trend = "Low",     "LKR Strengthening"
        else:            risk, trend = "Low",     "Stable exchange rate"
        return {
            "currency": currency, "current_lkr": current,
            "forecast_lkr": round(forecast, 0), "change_pct": round(change, 2),
            "trend": trend, "risk_level": risk, "months_ahead": months_ahead,
            "monthly_forecast": monthly,
        }

    def _fallback_forecast(self, currency: str, months_ahead: int) -> dict:
        history = HISTORICAL_RATES_LKR.get(currency, [])
        if len(history) < 2:
            return {"currency": currency, "error": "No data"}
        rates    = [r for _, _, r in history[-4:]]
        slope    = (rates[-1] - rates[0]) / max(len(rates)-1, 1)
        forecast = rates[-1] + slope * months_ahead
        change   = (forecast - rates[-1]) / rates[-1] * 100
        return {
            "currency": currency, "current_lkr": rates[-1],
            "forecast_lkr": round(forecast, 0), "change_pct": round(change, 2),
            "trend": "Weakening LKR" if change > 0 else "Stable",
            "risk_level": "Medium", "months_ahead": months_ahead, "ml_used": False,
        }

    def should_refresh_cache(self, cache_type: str, hours_since_last: float) -> dict:
        """Decide whether a cached API result should be refreshed."""
        base = {"exchange_rates": 24, "university_domains": 168, "country_metadata": 720}
        base_hours = base.get(cache_type, 48)
        priority   = "Medium"
        reason     = f"Standard {base_hours}h cycle"
        if cache_type == "exchange_rates" and self.trained:
            vol = abs(self.forecast_exchange_rate("GBP", 1).get("change_pct", 0))
            if vol > 5:   base_hours, priority, reason = 12, "High",   f"High volatility ({vol:.1f}%)"
            elif vol > 2: base_hours, priority, reason = 18, "Medium", f"Moderate volatility ({vol:.1f}%)"
            else:                     priority, reason =     "Low",    f"Stable rates ({vol:.1f}%)"
        return {
            "refresh":   hours_since_last >= base_hours,
            "priority":  priority,
            "reason":    reason,
            "recommended_interval_hours": base_hours,
        }

    def budget_impact_analysis(self, student_budget_lkr: float,
                                country: str, program_years: int = 3) -> dict:
        """Analyse currency risk on a student's budget over the program."""
        cmap    = {"UK": "GBP", "Singapore": "SGD", "Australia": "AUD"}
        cur     = cmap.get(country, "GBP")
        history = HISTORICAL_RATES_LKR.get(cur, [])
        current = history[-1][2] if history else 500
        fd      = self.forecast_exchange_rate(cur, months_ahead=program_years*12)
        frate   = fd.get("forecast_lkr", current)
        b_now   = student_budget_lkr / current
        b_fore  = student_budget_lkr / frate
        short   = max(0, (frate - current) * b_now)
        chg     = fd.get("change_pct", 0)
        if chg > 8:   risk, rec = "High Risk",   f"Add LKR {short:,.0f} buffer."
        elif chg > 3: risk, rec = "Medium Risk", f"Budget LKR {short:,.0f} contingency."
        else:         risk, rec = "Low Risk",    "Exchange rate stable. Standard budget applies."
        return {
            "currency": cur, "country": country,
            "budget_lkr": student_budget_lkr,
            "current_rate_lkr": current, "forecast_rate_lkr": round(frate, 0),
            "budget_current_foreign": round(b_now, 0),
            "budget_forecast_foreign": round(b_fore, 0),
            "shortfall_lkr": round(short, 0),
            "rate_change_pct": round(chg, 2),
            "program_years": program_years,
            "risk_assessment": risk, "recommendation": rec,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINED PHASE 3 ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

class Phase3:
    """
    Single import for all Phase 3 functionality.

    Usage:
        from core.phases.phase3 import Phase3
        p3 = Phase3()
        p3.ml.train()
        rates = p3.api.fetch_exchange_rates()
        forecast = p3.ml.forecast_exchange_rate("GBP", 6)
    """

    def __init__(self, cache_dir: str = _CACHE_DIR):
        self.api = APIIntegrator(cache_dir=cache_dir)
        self.ml  = Phase3MLEngine()
        if not self.ml.trained:
            self.ml.train(verbose=False)
        print("✅ Phase3 ready (API + ML)")

    def enrich_and_forecast(self, universities: List[Dict],
                             country: str) -> dict:
        """
        Single call: enrich universities + forecast currency risk.
        Returns {enriched_universities, currency_forecast, cache_advice}.
        """
        cmap = {"UK": "GBP", "Singapore": "SGD", "Australia": "AUD"}
        cur  = cmap.get(country, "GBP")
        enriched = self.api.enrich_universities(universities, country)
        forecast = self.ml.forecast_exchange_rate(cur, months_ahead=6)
        cache_advice = self.ml.should_refresh_cache(
            "exchange_rates",
            hours_since_last=24,
        )
        return {
            "enriched_universities": enriched,
            "currency_forecast":     forecast,
            "cache_advice":          cache_advice,
        }


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3 — API Integration + ML Self Test")
    print("=" * 60)

    p3 = Phase3()

    print("\n--- Exchange Rate Forecasts ---")
    for ccy in ["GBP", "SGD", "AUD"]:
        f = p3.ml.forecast_exchange_rate(ccy, months_ahead=6)
        print(f"  {ccy}: {f['current_lkr']} → {f['forecast_lkr']} LKR "
              f"({f['change_pct']:+.1f}%) | Risk: {f['risk_level']}")

    print("\n--- Budget Impact (LKR 5,000,000 for UK, 3 years) ---")
    b = p3.ml.budget_impact_analysis(5_000_000, "UK", 3)
    print(f"  In GBP today:    £{b['budget_current_foreign']:,.0f}")
    print(f"  In GBP in 3 yrs: £{b['budget_forecast_foreign']:,.0f}")
    print(f"  Risk: {b['risk_assessment']}")

    print("\n✓ Phase 3 test complete")