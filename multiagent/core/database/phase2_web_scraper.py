from __future__ import annotations

import json
import logging
import os
import pickle
import re
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("[Phase2] scikit-learn not found — run: pip install scikit-learn")

logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────

_ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE_DIR = os.path.join(_ROOT, "data", "scraped")
_MODEL_DIR = os.path.join(_ROOT, "data", "ml_models")

# ─── Baseline rankings (Phase 1 DB ground truth) ─────────────────────────────

KNOWN_RANKINGS = {
    "uk_cambridge":  {"qs": 2,   "the": 5},
    "uk_oxford":     {"qs": 3,   "the": 1},
    "uk_imperial":   {"qs": 6,   "the": 8},
    "uk_ucl":        {"qs": 9,   "the": 22},
    "uk_edinburgh":  {"qs": 27,  "the": 30},
    "uk_manchester": {"qs": 32,  "the": 54},
    "uk_kings":      {"qs": 40,  "the": 35},
    "uk_lse":        {"qs": 45,  "the": 27},
    "uk_bristol":    {"qs": 55,  "the": 91},
    "uk_warwick":    {"qs": 67,  "the": 81},
    "sg_nus":        {"qs": 8,   "the": 19},
    "sg_ntu":        {"qs": 15,  "the": 36},
    "sg_smu":        {"qs": 500, "the": None},
    "au_melbourne":  {"qs": 14,  "the": 33},
    "au_sydney":     {"qs": 18,  "the": 59},
    "au_anu":        {"qs": 30,  "the": 67},
    "au_unsw":       {"qs": 19,  "the": 71},
    "au_monash":     {"qs": 37,  "the": 95},
    "au_uq":         {"qs": 43,  "the": 79},
    "au_uwa":        {"qs": 72,  "the": 132},
}

# ─── QS/THE source URLs ───────────────────────────────────────────────────────

QS_URL  = "https://www.topuniversities.com/university-rankings/world-university-rankings/2025"
THE_URL = "https://www.timeshighereducation.com/world-university-rankings/2025/world-ranking"


# ══════════════════════════════════════════════════════════════════════════════
#  RANKING EXTRACTOR (NLP-style regex)
# ══════════════════════════════════════════════════════════════════════════════

class RankingExtractor:
    """Extract ranking numbers from raw scraped HTML/text using regex patterns."""

    QS_PATTERNS = [
        r"(?:QS\s*World\s*University\s*Rankings?\s*[\d]{4}[:\s#=]+)(\d{1,4})",
        r"(?:ranked?\s*#?)(\d{1,4})(?:\s*(?:in|by)\s*QS)",
        r"(?:QS\s*rank(?:ing)?\s*[:\s=]+)(\d{1,4})",
        r"(?:world\s*rank(?:ing)?\s*[:\s=]+)(\d{1,4})",
        r"(?:#|no\.\s*)(\d{1,4})\s*(?:in\s*the\s*world|globally|worldwide)",
    ]

    THE_PATTERNS = [
        r"(?:Times\s*Higher\s*Education\s*[\d]{4}[:\s#=]+)(\d{1,4})",
        r"(?:THE\s*World\s*University\s*Rankings?\s*[\d]{4}[:\s#=]+)(\d{1,4})",
        r"(?:THE\s*rank(?:ing)?\s*[:\s=]+)(\d{1,4})",
    ]

    TUITION_PATTERNS = {
        "GBP": [r"£\s*([\d,]+)(?:\s*per\s*year)?", r"([\d,]+)\s*(?:£|GBP)\s*(?:per\s*year|p\.a\.)?"],
        "AUD": [r"A\$\s*([\d,]+)(?:\s*per\s*year)?", r"([\d,]+)\s*(?:AUD|A\$)\s*(?:per\s*year)?"],
        "SGD": [r"S\$\s*([\d,]+)(?:\s*per\s*year)?", r"([\d,]+)\s*(?:SGD|S\$)\s*(?:per\s*year)?"],
    }

    def _extract_rank(self, text: str, patterns: list) -> Optional[int]:
        text_clean = text.replace(",","").lower()
        for p in patterns:
            m = re.search(p, text_clean, re.IGNORECASE)
            if m:
                try:
                    r = int(m.group(1))
                    if 1 <= r <= 2000: return r
                except: continue
        return None

    def extract_qs_rank(self, text: str) -> Optional[int]:
        return self._extract_rank(text, self.QS_PATTERNS)

    def extract_the_rank(self, text: str) -> Optional[int]:
        return self._extract_rank(text, self.THE_PATTERNS)

    def extract_tuition(self, text: str, currency: str = "GBP") -> Optional[int]:
        for p in self.TUITION_PATTERNS.get(currency, []):
            m = re.search(p, text, re.IGNORECASE)
            if m:
                try:
                    v = int(m.group(1).replace(",",""))
                    if 5000 <= v <= 200000: return v
                except: continue
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  QUALITY SCORER
# ══════════════════════════════════════════════════════════════════════════════

def score_scraped_record(record: dict) -> dict:
    """
    Score quality/confidence of a scraped record (0–100).
    Returns: {quality_score, confidence, issues, is_usable}
    """
    score, issues = 100, []
    qs = record.get("qs_rank")
    if qs is None:                  score -= 20; issues.append("Missing QS rank")
    elif not (1 <= qs <= 1500):     score -= 30; issues.append(f"QS rank out of range: {qs}")
    the = record.get("the_rank")
    if the is None:                 score -= 10; issues.append("Missing THE rank")
    elif not (1 <= the <= 1500):    score -= 15; issues.append(f"THE rank out of range: {the}")
    scraped_at = record.get("scraped_at")
    if scraped_at:
        try:
            age = (datetime.now() - datetime.fromisoformat(scraped_at)).days
            if age > 365:   score -= 15; issues.append(f"Data is {age} days old")
            elif age > 180: score -= 5;  issues.append(f"Data is {age} days old (>6mo)")
        except: score -= 5; issues.append("Invalid timestamp")
    else: score -= 10; issues.append("No scrape timestamp")
    if not record.get("university_name"): score -= 10; issues.append("Missing name")
    score = max(0, score)
    if score >= 80: conf = "High"
    elif score >= 60: conf = "Medium"
    elif score >= 40: conf = "Low"
    else: conf = "Unreliable"
    return {"quality_score": score, "confidence": conf, "issues": issues, "is_usable": score >= 50}


# ══════════════════════════════════════════════════════════════════════════════
#  WEB SCRAPER
# ══════════════════════════════════════════════════════════════════════════════

class WebScraper:
    """
    Scrapes QS and THE world university rankings.
    All results cached to data/scraped/ — works offline after first run.
    """

    RATE_LIMIT = 2.0  # polite delay between requests

    def __init__(self, cache_dir: str = _CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.extractor = RankingExtractor()
        self._stats = {"qs_scraped": 0, "the_scraped": 0, "cache_hits": 0, "errors": 0}

    def scrape_qs_rankings(self, use_cache: bool = True) -> Dict[str, int]:
        """
        Scrape QS World University Rankings.
        Returns {university_name: qs_rank} dict.
        Falls back to cache if scraping fails.
        """
        cache_file = "qs_rankings.json"
        if use_cache:
            cached = self._load_cache(cache_file)
            if cached:
                self._stats["cache_hits"] += 1
                print(f"  Phase2: QS rankings loaded from cache ({len(cached)} entries)")
                return cached

        print(f"  Phase2: Scraping QS rankings from {QS_URL}...")
        try:
            html = self._fetch(QS_URL)
            if html:
                rankings = self._parse_qs_html(html)
                if rankings:
                    self._save_cache({"rankings": rankings, "scraped_at": datetime.now().isoformat()}, cache_file)
                    self._stats["qs_scraped"] += len(rankings)
                    print(f"  Phase2: Scraped {len(rankings)} QS rankings")
                    return rankings
        except Exception as exc:
            logger.warning("QS scrape failed: %s", exc)
            self._stats["errors"] += 1

        # Return hardcoded baseline if scraping fails
        print("  Phase2: Scraping failed — using baseline rankings")
        return {uid: r["qs"] for uid, r in KNOWN_RANKINGS.items() if r["qs"]}

    def scrape_the_rankings(self, use_cache: bool = True) -> Dict[str, int]:
        """Scrape THE World University Rankings."""
        cache_file = "the_rankings.json"
        if use_cache:
            cached = self._load_cache(cache_file)
            if cached:
                self._stats["cache_hits"] += 1
                return cached.get("rankings", {})

        print(f"  Phase2: Scraping THE rankings from {THE_URL}...")
        try:
            html = self._fetch(THE_URL)
            if html:
                rankings = self._parse_the_html(html)
                if rankings:
                    self._save_cache({"rankings": rankings, "scraped_at": datetime.now().isoformat()}, cache_file)
                    self._stats["the_scraped"] += len(rankings)
                    return rankings
        except Exception as exc:
            logger.warning("THE scrape failed: %s", exc)
            self._stats["errors"] += 1

        return {uid: r["the"] for uid, r in KNOWN_RANKINGS.items() if r["the"]}

    def get_statistics(self) -> Dict:
        return {**self._stats, "cache_dir": self.cache_dir}

    def _fetch(self, url: str) -> Optional[str]:
        import urllib.request
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; UniAssist/1.0; academic research)"
        })
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read().decode("utf-8", errors="ignore")

    def _parse_qs_html(self, html: str) -> Dict[str, int]:
        """Parse QS rankings from HTML. Returns {uni_name_lower: rank}."""
        rankings = {}
        # Look for rank patterns in the HTML
        matches = re.findall(r'"rank":\s*"?(\d+)"?.*?"title":\s*"([^"]+)"', html)
        for rank, name in matches:
            try:
                rankings[name.lower()] = int(rank)
            except: pass
        return rankings

    def _parse_the_html(self, html: str) -> Dict[str, int]:
        """Parse THE rankings from HTML."""
        rankings = {}
        matches = re.findall(r'"rank":\s*"?(\d+)"?.*?"name":\s*"([^"]+)"', html)
        for rank, name in matches:
            try:
                rankings[name.lower()] = int(rank)
            except: pass
        return rankings

    def _save_cache(self, data, filename: str):
        try:
            with open(os.path.join(self.cache_dir, filename), "w") as f:
                json.dump(data, f, indent=2)
        except OSError as exc:
            logger.warning("Cache write failed: %s", exc)

    def _load_cache(self, filename: str):
        try:
            with open(os.path.join(self.cache_dir, filename)) as f:
                return json.load(f)
        except: return None


# ══════════════════════════════════════════════════════════════════════════════
#  ML ENGINE — Isolation Forest anomaly detection
# ══════════════════════════════════════════════════════════════════════════════

class Phase2AnomalyDetector:
    """
    Isolation Forest validates scraped rankings before they touch the database.

    What it adds:
      - Flag when scraped rank is statistically impossible (rank 32 → 320)
      - Detect when scrape returned garbage (HTML parse error)
      - Quality gate: only clean data reaches the database
      - Batch check all 30 universities in one call
    """

    def __init__(self):
        self.model   = None
        self.scaler  = StandardScaler() if ML_AVAILABLE else None
        self.trained = False
        self.extractor = RankingExtractor()
        self.model_metrics = {}
        self._try_load()

    def _try_load(self):
        mp = os.path.join(_MODEL_DIR, "phase2_anomaly_model.pkl")
        sp = os.path.join(_MODEL_DIR, "phase2_anomaly_scaler.pkl")
        if ML_AVAILABLE and os.path.exists(mp) and os.path.exists(sp):
            try:
                with open(mp,"rb") as f: self.model  = pickle.load(f)
                with open(sp,"rb") as f: self.scaler = pickle.load(f)
                self.trained = True
                print("[Phase2-ML] Loaded Isolation Forest model from disk")
            except Exception: pass

    def _build_baseline_samples(self, rng: np.random.Generator, samples_per_uni: int) -> np.ndarray:
        samples = []
        for ranks in KNOWN_RANKINGS.values():
            qs_b = ranks["qs"] or 500
            the_b = ranks["the"] or 200
            for _ in range(samples_per_uni):
                qv = max(1, int(qs_b * rng.uniform(0.75, 1.25)))
                tv = max(1, int(the_b * rng.uniform(0.75, 1.25)))
                samples.append([qv, tv, qv / max(tv, 1), np.log1p(qv), np.log1p(tv)])
        return np.array(samples, dtype=float)

    def _build_anomaly_samples(self, rng: np.random.Generator, samples_per_uni: int) -> np.ndarray:
        samples = []
        for ranks in KNOWN_RANKINGS.values():
            qs_b = ranks["qs"] or 500
            the_b = ranks["the"] or 200
            for _ in range(samples_per_uni):
                qv = max(1, int(qs_b * rng.uniform(1.8, 4.5)))
                tv = max(1, int(the_b * rng.uniform(0.1, 0.5)))
                if rng.random() > 0.5:
                    qv, tv = max(qv, 1200), min(tv, 25)
                samples.append([qv, tv, qv / max(tv, 1), np.log1p(qv), np.log1p(tv)])
        return np.array(samples, dtype=float)

    def _evaluate_detector(self) -> dict:
        if not ML_AVAILABLE:
            return {"available": False, "reason": "scikit-learn not available"}

        rng = np.random.default_rng(42)
        normal = self._build_baseline_samples(rng, samples_per_uni=30)
        anomalies = self._build_anomaly_samples(np.random.default_rng(7), samples_per_uni=8)
        order = rng.permutation(len(normal))
        normal = normal[order]
        split_index = max(int(len(normal) * 0.8), 1)
        train_normal = normal[:split_index]
        test_normal = normal[split_index:]
        if len(test_normal) == 0 or len(anomalies) == 0:
            return {"available": False, "reason": "Synthetic evaluation dataset is empty"}

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_normal)
        detector = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        detector.fit(train_scaled)

        X_test = np.vstack([test_normal, anomalies])
        y_true = np.concatenate([
            np.zeros(len(test_normal), dtype=int),
            np.ones(len(anomalies), dtype=int),
        ])
        X_test_scaled = scaler.transform(X_test)
        anomaly_scores = -detector.decision_function(X_test_scaled)
        y_pred = (detector.predict(X_test_scaled) == -1).astype(int)
        false_positive_rate = float(((y_pred == 1) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1))

        return {
            "available": True,
            "train_samples": int(len(train_normal)),
            "test_samples": int(len(X_test)),
            "synthetic_anomalies": int(len(anomalies)),
            "metrics": {
                "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
                "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
                "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(y_true, anomaly_scores)), 4),
                "false_positive_rate": round(false_positive_rate, 4),
            },
        }

    def fit_baseline(self, verbose: bool = True) -> dict:
        """Train Isolation Forest on known-good ranking baselines + variations."""
        if not ML_AVAILABLE:
            return {"error": "scikit-learn not available"}
        self.model_metrics = self._evaluate_detector()
        np.random.seed(42)
        X = []
        for ranks in KNOWN_RANKINGS.values():
            qs_b  = ranks["qs"] or 500
            the_b = ranks["the"] or 200
            for _ in range(50):
                qv = max(1, int(qs_b  * np.random.uniform(0.75, 1.25)))
                tv = max(1, int(the_b * np.random.uniform(0.75, 1.25)))
                X.append([qv, tv, qv/max(tv,1), np.log1p(qv), np.log1p(tv)])
        X = np.array(X)
        Xs = self.scaler.fit_transform(X)
        self.model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        self.model.fit(Xs)
        self.trained = True
        os.makedirs(_MODEL_DIR, exist_ok=True)
        with open(os.path.join(_MODEL_DIR,"phase2_anomaly_model.pkl"),"wb") as f: pickle.dump(self.model,f)
        with open(os.path.join(_MODEL_DIR,"phase2_anomaly_scaler.pkl"),"wb") as f: pickle.dump(self.scaler,f)
        if verbose: print(f"[Phase2-ML] Trained on {len(X)} ranking samples")
        return {
            "trained_on": len(X),
            "universities": len(KNOWN_RANKINGS),
            "trained_at": datetime.now().isoformat(),
            "evaluation": self.model_metrics,
        }

    def check_ranking(self, uni_id: str, new_qs: Optional[int], new_the: Optional[int]) -> dict:
        """Check if newly scraped ranking looks anomalous."""
        alerts = []
        baseline = KNOWN_RANKINGS.get(uni_id, {})
        bq, bt   = baseline.get("qs"), baseline.get("the")
        cq = ct  = 0.0

        if bq and new_qs:
            cq = abs(new_qs - bq) / bq * 100
            if cq > 50: alerts.append(f"⚠ QS changed {bq}→{new_qs} ({cq:.0f}%) — verify")
        if bt and new_the:
            ct = abs(new_the - bt) / bt * 100
            if ct > 50: alerts.append(f"⚠ THE changed {bt}→{new_the} ({ct:.0f}%) — verify")

        score, ml_anom = 0.0, False
        if ML_AVAILABLE and self.trained and new_qs and new_the:
            qv, tv  = max(1,new_qs), max(1,new_the)
            feat    = np.array([[qv,tv,qv/max(tv,1),np.log1p(qv),np.log1p(tv)]])
            feat_s  = self.scaler.transform(feat)
            pred    = self.model.predict(feat_s)[0]
            score   = float(self.model.decision_function(feat_s)[0])
            ml_anom = (pred == -1)
            if ml_anom: alerts.append(f"🤖 ML anomaly: QS={new_qs} THE={new_the} (score={score:.3f})")

        is_anom = len(alerts) > 0
        if is_anom:     rec = "Do NOT auto-update DB — flag for admin review (Phase 5 override)"
        elif cq>20 or ct>20: rec = "Moderate change — auto-update allowed, log the change"
        else:           rec = "Ranking looks normal — safe to auto-update"

        return {
            "is_anomaly": is_anom, "is_ml_anomaly": ml_anom,
            "anomaly_score": round(score,4),
            "change_qs_pct": round(cq,1), "change_the_pct": round(ct,1),
            "alerts": alerts, "recommendation": rec,
        }

    def batch_check_rankings(self, scraped: dict) -> dict:
        """Check all scraped rankings at once. Returns summary + flagged list."""
        results, flagged = {}, []
        for uid, data in scraped.items():
            check = self.check_ranking(uid, data.get("qs"), data.get("the"))
            check["quality"] = score_scraped_record({
                "qs_rank": data.get("qs"), "the_rank": data.get("the"),
                "university_name": uid, "scraped_at": data.get("scraped_at"),
            })
            results[uid] = check
            if check["is_anomaly"]: flagged.append(uid)
        return {
            "summary": {"total": len(scraped), "anomalies": len(flagged), "clean": len(scraped)-len(flagged)},
            "results": results, "flagged": flagged,
        }

    def extract_from_text(self, text: str, currency: str = "GBP") -> dict:
        """Extract and validate structured data from raw scraped text."""
        qs  = self.extractor.extract_qs_rank(text)
        the = self.extractor.extract_the_rank(text)
        tui = self.extractor.extract_tuition(text, currency)
        extracted = {"qs_rank": qs, "the_rank": the, "tuition": tui,
                     "extracted_at": datetime.now().isoformat()}
        extracted["quality"] = score_scraped_record({
            "qs_rank": qs, "the_rank": the,
            "scraped_at": datetime.now().isoformat(),
        })
        return extracted


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINED PHASE 2 ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

class Phase2:
    """
    Single import for all Phase 2 functionality.

    Usage:
        from core.phases.phase2 import Phase2

        p2 = Phase2()
        qs_data = p2.scraper.scrape_qs_rankings()
        check   = p2.ml.check_ranking("uk_manchester", new_qs=35, new_the=56)
        batch   = p2.ml.batch_check_rankings(all_scraped)
    """

    def __init__(self, cache_dir: str = _CACHE_DIR):
        self.scraper = WebScraper(cache_dir=cache_dir)
        self.ml      = Phase2AnomalyDetector()
        if not self.ml.trained:
            self.ml.fit_baseline(verbose=False)
        print("✅ Phase2 ready (Scraper + ML)")

    def scrape_and_validate(self, use_cache: bool = True) -> dict:
        """
        Scrape both QS and THE rankings, then ML-validate before returning.
        Returns only clean rankings safe to update in the database.
        """
        qs_raw  = self.scraper.scrape_qs_rankings(use_cache=use_cache)
        the_raw = self.scraper.scrape_the_rankings(use_cache=use_cache)

        # Build combined dict for batch check
        combined = {}
        for uid in KNOWN_RANKINGS:
            combined[uid] = {
                "qs":         qs_raw.get(uid),
                "the":        the_raw.get(uid),
                "scraped_at": datetime.now().isoformat(),
            }

        validation = self.ml.batch_check_rankings(combined)
        clean = {uid: data for uid, data in combined.items()
                 if uid not in validation["flagged"]}

        print(f"  Phase2: {len(clean)}/{len(combined)} rankings passed validation "
              f"({len(validation['flagged'])} flagged)")
        return {
            "clean_rankings": clean,
            "flagged":        validation["flagged"],
            "summary":        validation["summary"],
        }


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2 — Scraper + ML Self Test")
    print("=" * 60)

    p2 = Phase2()

    print("\n--- Anomaly check: normal change (Manchester QS 32→35) ---")
    r = p2.ml.check_ranking("uk_manchester", new_qs=35, new_the=56)
    print(f"  Anomaly: {r['is_anomaly']} | QS change: {r['change_qs_pct']}%")
    print(f"  {r['recommendation']}")

    print("\n--- Anomaly check: suspicious jump (Manchester QS 32→320) ---")
    r = p2.ml.check_ranking("uk_manchester", new_qs=320, new_the=56)
    print(f"  Anomaly: {r['is_anomaly']} | Alerts: {r['alerts']}")

    print("\n--- NLP text extraction ---")
    text = """
    The University of Manchester ranked 32nd in QS World University Rankings 2026.
    Times Higher Education ranking: 54th globally.
    International tuition fees: £19,000 per year.
    """
    extracted = p2.ml.extract_from_text(text, "GBP")
    print(f"  QS: {extracted['qs_rank']} | THE: {extracted['the_rank']} | "
          f"Tuition: £{extracted['tuition']} | Quality: {extracted['quality']['quality_score']}/100")

    print("\n--- Batch check ---")
    batch = {
        "uk_manchester": {"qs": 35,  "the": 56, "scraped_at": datetime.now().isoformat()},
        "uk_cambridge":  {"qs": 2,   "the": 5,  "scraped_at": datetime.now().isoformat()},
        "au_melbourne":  {"qs": 999, "the": 300,"scraped_at": datetime.now().isoformat()},
    }
    result = p2.ml.batch_check_rankings(batch)
    print(f"  Total: {result['summary']['total']} | "
          f"Clean: {result['summary']['clean']} | "
          f"Flagged: {result['flagged']}")

    print("\n✓ Phase 2 test complete")