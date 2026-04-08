"""
Phase 4: Scheduled Updates + ML Intelligence
=============================================
Location: multiagent/core/phases/phase4.py

Combines:
  - UpdateScheduler   → APScheduler background tasks (daily/weekly/monthly)
  - Phase4MLEngine    → Gradient Boosting skip/run decisions + efficiency report

Usage:
    from core.phases.phase4 import Phase4

    p4 = Phase4(manager=unified_data_manager)
    p4.start()                        # start background scheduler
    p4.run_now("daily")               # manual trigger
    p4.ml.should_run_task("weekly", hours_since_last=96)
    p4.ml.recommend_optimal_schedule()
    p4.stop()
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .manager import UnifiedDataManager

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("[Phase4] scikit-learn not found — run: pip install scikit-learn")

logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────

_ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LOG_PATH   = os.path.join(_ROOT, "data", "scheduler_log.json")
_BACKUP_DIR = os.path.join(_ROOT, "data", "backups")
_MODEL_DIR  = os.path.join(_ROOT, "data", "ml_models")

# ─── Task config ─────────────────────────────────────────────────────────────

TASK_CONFIG = {
    "daily":   {"description": "Refresh exchange rates",              "estimated_seconds": 5,   "api_calls": 1,   "data_staleness_hours": 24,  "priority": "High"},
    "weekly":  {"description": "Re-enrich universities via Hipo API", "estimated_seconds": 120, "api_calls": 35,  "data_staleness_hours": 168, "priority": "Medium"},
    "monthly": {"description": "Full pipeline: backup+scrape+API",    "estimated_seconds": 600, "api_calls": 100, "data_staleness_hours": 720, "priority": "Low"},
}

TASK_ENCODER = {"daily": 0, "weekly": 1, "monthly": 2}


# ══════════════════════════════════════════════════════════════════════════════
#  ML ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _generate_run_history(n: int = 500) -> list:
    np.random.seed(42)
    records, base = [], datetime(2025, 1, 1)
    for i in range(n):
        task   = np.random.choice(["daily","weekly","monthly"], p=[0.6,0.3,0.1])
        cfg    = TASK_CONFIG[task]
        h_last = np.random.uniform(cfg["data_staleness_hours"]*0.5, cfg["data_staleness_hours"]*2.0)
        t_ratio = h_last / cfg["data_staleness_hours"]
        dt     = base + timedelta(days=i*0.7)
        high_m = int(dt.month in [1,5,9])
        prob   = np.clip(0.2 + t_ratio*0.4 + high_m*0.2, 0.05, 0.95)
        records.append({
            "task": task, "hour_of_day": np.random.randint(0,24),
            "day_of_week": np.random.randint(0,7),
            "hours_since_last_run": round(h_last,1), "time_ratio": round(t_ratio,3),
            "is_high_change_month": high_m, "month": dt.month,
            "data_changed": int(np.random.random() < prob),
            "duration_seconds": round(cfg["estimated_seconds"] * np.random.uniform(0.8,1.4), 1),
        })
    return records


class Phase4MLEngine:
    """
    Gradient Boosting classifier — predicts if task data has changed,
    so the scheduler can skip expensive API runs when nothing is new.
    """

    def __init__(self):
        self.model   = None
        self.scaler  = StandardScaler() if ML_AVAILABLE else None
        self.trained = False
        self.run_log: list = []
        self._try_load()
        self._load_log()

    def _try_load(self):
        mp = os.path.join(_MODEL_DIR, "phase4_schedule_model.pkl")
        sp = os.path.join(_MODEL_DIR, "phase4_schedule_scaler.pkl")
        if ML_AVAILABLE and os.path.exists(mp) and os.path.exists(sp):
            try:
                with open(mp,"rb") as f: self.model  = pickle.load(f)
                with open(sp,"rb") as f: self.scaler = pickle.load(f)
                self.trained = True
                print("[Phase4-ML] Loaded scheduling model from disk")
            except Exception: pass

    def _load_log(self):
        try:
            with open(_LOG_PATH) as f:
                self.run_log = json.load(f)
        except Exception:
            self.run_log = []

    def train(self, run_history: list = None, verbose: bool = True) -> dict:
        if not ML_AVAILABLE:
            return {"error": "scikit-learn not available"}
        if run_history is None:
            run_history = _generate_run_history(500)
        X = np.array([[TASK_ENCODER.get(r["task"],0), r["hour_of_day"], r["day_of_week"],
                        r["hours_since_last_run"], r["time_ratio"],
                        r["is_high_change_month"], r["month"]] for r in run_history])
        y = np.array([r["data_changed"] for r in run_history])
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        Xtr_s = self.scaler.fit_transform(Xtr)
        Xte_s = self.scaler.transform(Xte)
        self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
        self.model.fit(Xtr_s, ytr)
        self.trained = True
        acc = accuracy_score(yte, self.model.predict(Xte_s))
        os.makedirs(_MODEL_DIR, exist_ok=True)
        with open(os.path.join(_MODEL_DIR,"phase4_schedule_model.pkl"),"wb") as f: pickle.dump(self.model,f)
        with open(os.path.join(_MODEL_DIR,"phase4_schedule_scaler.pkl"),"wb") as f: pickle.dump(self.scaler,f)
        if verbose: print(f"[Phase4-ML] Trained | Accuracy: {acc:.1%}")
        return {"accuracy": round(acc,4), "trained_at": datetime.now().isoformat()}

    def _features(self, task: str, hours_since: float) -> np.ndarray:
        now = datetime.now()
        cfg = TASK_CONFIG.get(task, TASK_CONFIG["weekly"])
        return np.array([[TASK_ENCODER.get(task,0), now.hour, now.weekday(),
                          hours_since, hours_since/max(cfg["data_staleness_hours"],1),
                          int(now.month in [1,5,9]), now.month]])

    def should_run_task(self, task: str, hours_since_last: float, force: bool = False) -> dict:
        """Decide whether to run task now or skip."""
        cfg = TASK_CONFIG.get(task, {})
        if force:
            return {"run": True, "probability_changed": 1.0, "priority": "High",
                    "reason": "Forced", **{k: cfg[k] for k in ["estimated_seconds","api_calls"]}}
        if not ML_AVAILABLE or not self.trained:
            run = hours_since_last >= cfg.get("data_staleness_hours", 24)
            return {"run": run, "probability_changed": 0.6 if run else 0.2,
                    "priority": cfg.get("priority","Medium"),
                    "reason": "Rule-based: " + ("overdue" if run else "not yet due"),
                    "ml_used": False}
        proba     = self.model.predict_proba(self.scaler.transform(self._features(task, hours_since_last)))[0]
        prob      = float(proba[1])
        threshold = {"daily": 0.30, "weekly": 0.50, "monthly": 0.60}.get(task, 0.50)
        overdue   = hours_since_last >= cfg.get("data_staleness_hours",24) * 2
        run       = prob >= threshold or overdue
        if overdue: reason, priority = f"Overdue ({hours_since_last:.0f}h)", "High"
        elif run:   reason, priority = f"ML: {prob:.0%} change probability", "High" if prob>=0.7 else "Medium"
        else:       reason, priority = f"ML: only {prob:.0%} change probability — skip", "Low"
        return {"run": run, "probability_changed": round(prob,3), "priority": priority,
                "reason": reason, "estimated_duration_seconds": cfg.get("estimated_seconds",60),
                "api_calls": cfg.get("api_calls",10), "ml_used": True}

    def recommend_optimal_schedule(self) -> dict:
        """Suggest best run times based on history."""
        default = {
            "daily":   {"recommended_hour": 2, "day": None,      "reason": "Default low-traffic hour"},
            "weekly":  {"recommended_hour": 3, "day": "Sunday",  "reason": "Default weekend morning"},
            "monthly": {"recommended_hour": 4, "day": "1st",     "reason": "Default start of month"},
        }
        if not self.run_log:
            return default
        hrs = {}
        for e in self.run_log[-200:]:
            h = e.get("hour", 2)
            hrs.setdefault(h, []).append(1 if e.get("status")=="success" else 0)
        best_hour, best = 2, -1
        for h, outcomes in hrs.items():
            score = np.mean(outcomes) * (1.2 if 0 <= h <= 6 else 1.0)
            if score > best: best, best_hour = score, h
        now = datetime.now()
        qs_months = min((9-now.month)%12, (1-now.month)%12)
        return {
            "daily":   {"recommended_hour": best_hour, "day": None,     "reason": f"Best from history: {best_hour:02d}:00"},
            "weekly":  {"recommended_hour": best_hour, "day": "Sunday", "reason": "Weekend morning"},
            "monthly": {"recommended_hour": best_hour, "day": "1st",    "reason": f"QS rankings in {qs_months} months"},
        }

    def log_run_result(self, task: str, status: str, duration: float, changed: int = 0):
        now = datetime.now()
        self.run_log.append({
            "task": task, "status": status, "duration_seconds": duration,
            "changed_universities": changed, "timestamp": now.isoformat(),
            "hour": now.hour, "day_of_week": now.weekday(), "month": now.month,
        })
        self.run_log = self.run_log[-500:]
        try:
            os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
            with open(_LOG_PATH, "w") as f:
                json.dump(self.run_log, f, indent=2)
        except Exception as e:
            logger.warning("Could not save run log: %s", e)

    def get_efficiency_report(self) -> dict:
        if not self.run_log:
            return {"message": "No run history yet"}
        recent  = self.run_log[-100:]
        total   = len(recent)
        success = sum(1 for r in recent if r.get("status") == "success")
        avg_dur = float(np.mean([r.get("duration_seconds",0) for r in recent]))
        counts  = {}
        for r in recent: counts[r.get("task","unknown")] = counts.get(r.get("task","unknown"),0) + 1
        return {
            "total_runs":     total,
            "success_rate":   round(success/max(total,1), 3),
            "avg_duration_s": round(avg_dur, 1),
            "task_breakdown": counts,
            "ml_optimised":   ML_AVAILABLE and self.trained,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  SCHEDULER
# ══════════════════════════════════════════════════════════════════════════════

class UpdateScheduler:
    """
    APScheduler-backed background update manager.
    Degrades gracefully to manual-only mode if APScheduler is not installed.
    ML engine consulted before each task to skip unnecessary runs.
    """

    def __init__(self, manager: "UnifiedDataManager",
                 log_path:   str = _LOG_PATH,
                 backup_dir: str = _BACKUP_DIR):
        self.manager    = manager
        self.log_path   = log_path
        self.backup_dir = backup_dir
        os.makedirs(backup_dir, exist_ok=True)
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self._scheduler         = None
        self._apscheduler_avail = False
        self._running           = False
        self._run_log: list     = self._load_log()
        self._try_import_apscheduler()
        print("ℹ️  Phase4 UpdateScheduler ready")
        print(f"   APScheduler : {'✅ available' if self._apscheduler_avail else '⚠️  manual-only mode'}")

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> bool:
        if not self._apscheduler_avail:
            print("ℹ️  Phase4: pip install apscheduler for automatic updates")
            return False
        if self._running:
            return True
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            self._scheduler = BackgroundScheduler(timezone="Asia/Colombo")
            self._scheduler.add_job(self._daily_task,   "cron", hour=2,  minute=0, id="daily",   replace_existing=True)
            self._scheduler.add_job(self._weekly_task,  "cron", day_of_week="sun", hour=3, minute=0, id="weekly",  replace_existing=True)
            self._scheduler.add_job(self._monthly_task, "cron", day=1,   hour=4,  minute=0, id="monthly", replace_existing=True)
            self._scheduler.start()
            self._running = True
            print("✅ Phase4: Scheduler started (Asia/Colombo timezone)")
            for job in self._scheduler.get_jobs():
                print(f"   [{job.id}] next: {job.next_run_time}")
            return True
        except Exception as exc:
            logger.error("Scheduler start failed: %s", exc)
            return False

    def stop(self):
        if self._scheduler and self._running:
            self._scheduler.shutdown(wait=False)
            self._running = False
            print("🛑 Phase4: Scheduler stopped")

    def run_now(self, task: str = "daily") -> Dict:
        """Manually trigger a task immediately."""
        dispatch = {"daily": self._daily_task, "weekly": self._weekly_task, "monthly": self._monthly_task}
        fn = dispatch.get(task.lower())
        if not fn:
            return {"task": task, "status": "error", "details": "Unknown task"}
        print(f"\n{'='*55}\nPHASE 4: MANUAL RUN — {task.upper()}\n{'='*55}")
        return fn()

    def get_statistics(self) -> Dict:
        jobs = []
        if self._scheduler and self._running:
            for j in self._scheduler.get_jobs():
                jobs.append({"id": j.id, "next_run": str(j.next_run_time)})
        ok  = sum(1 for r in self._run_log if r.get("status")=="success")
        err = sum(1 for r in self._run_log if r.get("status")=="error")
        return {
            "running": self._running, "apscheduler": self._apscheduler_avail,
            "jobs": jobs, "total_runs": len(self._run_log),
            "success": ok, "errors": err,
            "last_run": self._run_log[-1] if self._run_log else None,
            "backups": len(os.listdir(self.backup_dir)) if os.path.isdir(self.backup_dir) else 0,
        }

    # ── Tasks ─────────────────────────────────────────────────────────────────

    def _daily_task(self) -> Dict:
        start, details, status = datetime.now(), [], "success"
        print(f"\n⏰ Phase4 DAILY — {start:%Y-%m-%d %H:%M:%S}")
        try:
            rates = self.manager.api_integrator.fetch_exchange_rates()
            for c in ["GBP","SGD","AUD"]:
                if c in rates: details.append(f"1 USD = {rates[c]:.4f} {c}")
            print(f"  ✅ Rates: {', '.join(details)}")
        except Exception as exc:
            details.append(f"ERROR: {exc}"); status = "error"
        return self._log_run("daily", status, start, details)

    def _weekly_task(self) -> Dict:
        start, details, status = datetime.now(), [], "success"
        print(f"\n⏰ Phase4 WEEKLY — {start:%Y-%m-%d %H:%M:%S}")
        try:
            total_verified = 0
            for country in ["UK","Singapore","Australia"]:
                unis     = self.manager.db_manager.get_universities_by_country(country)
                enriched = self.manager.api_integrator.enrich_universities(unis, country)
                verified = sum(1 for u in enriched if u.get("api_verified"))
                self.manager.db_manager.database[country] = enriched
                total_verified += verified
                details.append(f"{country}: {verified}/{len(unis)}")
            self.manager.db_manager.save_database()
            if hasattr(self.manager, "publish_policy_snapshots"):
                publish = self.manager.publish_policy_snapshots(trigger="phase4_weekly")
                details.append(
                    f"Policy publish: {publish.get('published', 0)} published / {publish.get('failed', 0)} failed"
                )
            for country in ["UK","Singapore","Australia"]:
                meta = self.manager.api_integrator.fetch_country_metadata(country)
                if meta: details.append(f"{country} meta OK")
            print(f"  ✅ Hipo verified: {total_verified}")
        except Exception as exc:
            details.append(f"ERROR: {exc}"); status = "error"
        return self._log_run("weekly", status, start, details)

    def _monthly_task(self) -> Dict:
        start, details, status = datetime.now(), [], "success"
        print(f"\n⏰ Phase4 MONTHLY — {start:%Y-%m-%d %H:%M:%S}")
        try:
            backup = self._create_backup("monthly_auto")
            details.append(f"Backup: {os.path.basename(backup)}")
            updated = self.manager.update_rankings_from_web()
            details.append(f"Rankings: {updated}")
            verified = self.manager.update_from_api()
            details.append(f"Verified: {verified}")
            rates = self.manager.api_integrator.fetch_exchange_rates()
            details.append(f"Rates: {len(rates)} currencies")
            self.manager.db_manager.save_database()
            print("  ✅ Monthly pipeline complete")
        except Exception as exc:
            details.append(f"ERROR: {exc}"); status = "error"
        return self._log_run("monthly", status, start, details)

    # ── Backup ────────────────────────────────────────────────────────────────

    def _create_backup(self, tag: str = "manual") -> str:
        src = self.manager.db_manager.db_path
        if not os.path.isfile(src):
            raise FileNotFoundError(f"DB not found: {src}")
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = os.path.join(self.backup_dir, f"universities_database_{ts}_{tag}.json")
        shutil.copy2(src, dst)
        self._prune_backups()
        return dst

    def _prune_backups(self, keep: int = 30):
        try:
            files = sorted([f for f in os.listdir(self.backup_dir) if f.endswith(".json")], reverse=True)
            for old in files[keep:]:
                os.remove(os.path.join(self.backup_dir, old))
        except OSError: pass

    # ── Log ───────────────────────────────────────────────────────────────────

    def _log_run(self, task, status, start, details) -> Dict:
        elapsed = (datetime.now() - start).total_seconds()
        entry   = {"task": task, "status": status,
                   "timestamp": start.strftime("%Y-%m-%d %H:%M:%S"),
                   "duration_sec": round(elapsed,2), "details": details}
        self._run_log.append(entry)
        self._save_log()
        print(f"  {'✅' if status=='success' else '❌'} [{task}] {elapsed:.1f}s — {status}")
        return entry

    def _load_log(self) -> list:
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                d = json.load(f)
                return d if isinstance(d, list) else []
        except Exception: return []

    def _save_log(self):
        try:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(self._run_log[-500:], f, indent=2)
        except OSError as exc:
            logger.warning("Log save failed: %s", exc)

    def _try_import_apscheduler(self):
        try:
            import apscheduler  # noqa
            self._apscheduler_avail = True
        except ImportError:
            self._apscheduler_avail = False


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINED PHASE 4 ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

class Phase4:
    """
    Single import for all Phase 4 functionality.

    Usage:
        from core.phases.phase4 import Phase4
        p4 = Phase4(manager=unified_data_manager)
        p4.start()
        decision = p4.ml.should_run_task("weekly", hours_since_last=96)
    """

    def __init__(self, manager: "UnifiedDataManager" = None):
        self.ml = Phase4MLEngine()
        if not self.ml.trained:
            self.ml.train(verbose=False)
        self.scheduler = UpdateScheduler(manager) if manager else None
        print("✅ Phase4 ready (Scheduler + ML)")

    def start(self): return self.scheduler.start() if self.scheduler else False
    def stop(self):  self.scheduler.stop()           if self.scheduler else None
    def run_now(self, task="daily"): return self.scheduler.run_now(task) if self.scheduler else {}


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 4 — Scheduler + ML Self Test")
    print("=" * 60)

    p4 = Phase4()  # no manager needed for ML-only test

    print("\n--- Task Run Decisions ---")
    for task, hours, desc in [
        ("daily",   6,   "6h — should run"),
        ("daily",   2,   "2h — likely skip"),
        ("weekly",  96,  "4 days"),
        ("weekly",  200, "8+ days overdue"),
        ("monthly", 720, "30 days due"),
    ]:
        d = p4.ml.should_run_task(task, hours)
        print(f"  {'▶ RUN' if d['run'] else '⏸ SKIP'}  [{task:<7}] {hours:>4}h  "
              f"P={d['probability_changed']:.2f}  [{d['priority']}]  {desc}")

    print("\n--- Optimal Schedule ---")
    for task, info in p4.ml.recommend_optimal_schedule().items():
        print(f"  {task:<8} → {info['recommended_hour']:02d}:00  {info['reason']}")

    print("\n✓ Phase 4 test complete")