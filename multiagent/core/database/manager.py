"""
Unified Data Manager
======================
Orchestrates all 5 phases + Eligibility Verification Agent.

Phase 1: Curated Database (universities_database.json) — 30 real universities
Phase 2: Web Scraping    (QS/THE rankings via public endpoints)
Phase 3: API Integration (Hipo University Domains API — MIT License)
Phase 4: Scheduled Updates (APScheduler — daily/weekly/monthly)
Phase 5: Manual Overrides  (admin audit trail with JSON persistence)

Open-Source Data Attribution:
  - Hipo University Domains List: github.com/Hipo/university-domains-list (MIT)
  - QS World Rankings 2025:       topuniversities.com (public data)
  - THE World Rankings 2025:      timeshighereducation.com (public data)
  - REST Countries API:           restcountries.com (MPL 2.0)
  - Open Exchange Rates:          open.er-api.com (free tier)
"""

from __future__ import annotations
import json, os, shutil, urllib.request
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from typing import Dict, List, Optional

from .phase1_curated_db       import CuratedDatabaseManager
from .phase2_web_scraper      import WebScraper
from .phase3_api_integration  import APIIntegrator
from .phase4_scheduler        import UpdateScheduler
from .phase5_override_manager_ml import Phase5 as OverrideManager


class UnifiedDataManager:
    def __init__(self, start_scheduler: bool = False):
        print("🚀 Initialising Unified Data Manager...")
        print("=" * 60)
        self.db_manager       = CuratedDatabaseManager()
        self.scraper          = WebScraper()
        self.api_integrator   = APIIntegrator()
        self.override_manager = OverrideManager()
        self.scheduler        = UpdateScheduler(manager=self)
        self._root_dir        = Path(__file__).resolve().parents[2]
        self.policy_dir       = self._root_dir / "data" / "policy_snapshots"
        self.policy_log_path  = self.policy_dir / "publish_log.json"
        self.policy_dir.mkdir(parents=True, exist_ok=True)

        if start_scheduler:
            self.scheduler.start()

        print("\n✅ All phases initialised!")
        print("   Phase 1: Curated DB        ✅")
        print("   Phase 2: Web Scraper        ✅")
        print("   Phase 3: API Integration    ✅  [Hipo, REST Countries, Open Exchange Rates]")
        print("   Phase 4: Scheduler          ✅  [APScheduler — start with start_scheduler=True]")
        print("   Phase 5: Override Manager   ✅")

    # ── Phase 1 ───────────────────────────────────────────────────────────

    def get_universities(self, country: Optional[str] = None, **filters) -> List[Dict]:
        if country:
            unis = self.db_manager.get_universities_by_country(country)
        else:
            unis = []
            for c in ["UK", "Singapore", "Australia"]:
                unis.extend(self.db_manager.get_universities_by_country(c))
        if filters:
            unis = self.db_manager.search_universities(country=country, **filters)
        return [self.override_manager.apply_overrides(u) for u in unis]

    def get_university_by_id(self, university_id: str) -> Optional[Dict]:
        uni = self.db_manager.get_university_by_id(university_id)
        return self.override_manager.apply_overrides(uni) if uni else None

    def search_universities(self,
                             country:         Optional[str]   = None,
                             min_grade_point: Optional[float] = None,
                             program:         Optional[str]   = None,
                             max_qs_rank:     Optional[int]   = None) -> List[Dict]:
        unis = self.db_manager.search_universities(
            country=country,
            min_grade_point=min_grade_point,
            program=program,
            max_qs_rank=max_qs_rank,
        )
        return [self.override_manager.apply_overrides(u) for u in unis]

    # ── Phase 2 ───────────────────────────────────────────────────────────

    def update_rankings_from_web(self) -> int:
        """Scrape QS + THE rankings and update database in memory + disk."""
        print("\n" + "=" * 60 + "\nPHASE 2: UPDATE RANKINGS\n" + "=" * 60)
        qs  = self.scraper.scrape_qs_rankings()
        the = self.scraper.scrape_the_rankings()
        count = 0
        for c in ["UK", "Singapore", "Australia"]:
            for uni in self.db_manager.get_universities_by_country(c):
                if uni["name"] in qs:
                    uni["rankings"]["qs_world"]  = qs[uni["name"]];  count += 1
                if uni["name"] in the:
                    uni["rankings"]["the_world"] = the[uni["name"]]; count += 1
        self.db_manager.save_database()
        print(f"\n✅ Updated {count} ranking entries")
        return count

    # ── Phase 3 ───────────────────────────────────────────────────────────

    def update_from_api(self) -> int:
        """
        Enrich database using Hipo University Domains API.
        Returns count of universities with API-verified data.
        """
        print("\n" + "=" * 60 + "\nPHASE 3: API ENRICHMENT\n" + "=" * 60)
        total_matched = 0
        for country in ["UK", "Singapore", "Australia"]:
            unis     = self.db_manager.get_universities_by_country(country)
            enriched = self.api_integrator.enrich_universities(unis, country)
            matched  = sum(1 for u in enriched if u.get("api_verified"))
            total_matched += matched
            # Patch back into database
            self.db_manager.database[country] = enriched
        self.db_manager.save_database()
        publish_result = self.publish_policy_snapshots(trigger="phase3_api")
        print(
            "\n✅ Policy snapshot publish: "
            f"{publish_result.get('published', 0)} published, {publish_result.get('failed', 0)} failed"
        )
        print(f"\n✅ Phase 3: {total_matched} universities enriched via API")
        return total_matched

    def get_exchange_rates(self) -> Dict[str, float]:
        """Fetch live exchange rates (GBP/SGD/AUD vs USD)."""
        all_rates = self.api_integrator.fetch_exchange_rates()
        return {k: all_rates[k] for k in ["GBP", "SGD", "AUD"] if k in all_rates}

    def get_country_metadata(self, country: str) -> Optional[Dict]:
        """Fetch country metadata from REST Countries API."""
        return self.api_integrator.fetch_country_metadata(country)

    # ── Phase 4 ───────────────────────────────────────────────────────────

    def start_scheduler(self) -> bool:
        """Start background scheduled updates. Returns True if APScheduler available."""
        return self.scheduler.start()

    def stop_scheduler(self):
        """Stop background scheduler gracefully."""
        self.scheduler.stop()

    def run_scheduled_task(self, task: str = "daily"):
        """
        Manually trigger a scheduled task.
        task: "daily" | "weekly" | "monthly"
        """
        self.scheduler.run_now(task)

    def get_schedule_status(self) -> Dict:
        """Return scheduler status and next run times."""
        return self.scheduler.get_statistics()

    def run_monthly_update(self):
        """Convenience: run the full monthly update pipeline."""
        print("\n" + "=" * 60 + "\nPHASE 4: MONTHLY UPDATE\n" + "=" * 60)
        self.scheduler.run_now("monthly")

    def create_backup(self, tag: str = "manual") -> str:
        """Create a timestamped backup of the database."""
        return self.scheduler._create_backup(tag)

    # ── Phase 5 ───────────────────────────────────────────────────────────

    def add_override(self, uni_id: str, field: str, value,
                     updated_by: str = "admin", reason: str = "") -> bool:
        return self.override_manager.add_override(uni_id, field, value, updated_by, reason)

    def remove_override(self, uni_id: str, field: Optional[str] = None,
                        updated_by: str = "admin", reason: str = "") -> bool:
        return self.override_manager.remove_override(uni_id, field, updated_by, reason)

    def list_overrides(self, uni_id: Optional[str] = None):
        self.override_manager.print_overrides(uni_id)

    def get_override_history(self, uni_id: Optional[str] = None) -> list:
        hist = self.override_manager.override_history
        if uni_id:
            hist = [e for e in hist if e.get("university_id") == uni_id]
        return hist

    # ── Statistics / Validation ───────────────────────────────────────────

    def get_statistics(self) -> Dict:
        policy_stats = self.get_policy_publish_statistics()
        return {
            "database":     self.db_manager.get_statistics(),
            "overrides":    self.override_manager.get_statistics(),
            "scheduler":    self.scheduler.get_statistics(),
            "api":          self.api_integrator.get_statistics(),
            "policy_publish": policy_stats,
            "data_sources": {
                "phase1_curated":   True,
                "phase2_scraping":  True,
                "phase3_api":       True,   # ✅ fully implemented
                "phase4_scheduler": True,   # ✅ fully implemented
                "phase5_overrides": True,
            },
            "open_source_licenses": {
                "hipo_domains":     "MIT — github.com/Hipo/university-domains-list",
                "qs_rankings":      "Public data — topuniversities.com",
                "the_rankings":     "Public data — timeshighereducation.com",
                "rest_countries":   "MPL 2.0 — restcountries.com",
                "open_exchange":    "Free tier — open.er-api.com",
                "apscheduler":      "MIT — apscheduler.readthedocs.io",
            },
        }

    def print_statistics(self):
        s = self.get_statistics()
        print("\n" + "=" * 60 + "\nSYSTEM STATISTICS\n" + "=" * 60)

        db = s["database"]
        print(f"\n📚 Database: {db['total_universities']} universities (v{db['version']})")
        for c, n in db["by_country"].items():
            print(f"   {c}: {n}")
        print(f"   Last updated: {db['last_updated']}")

        ov = s["overrides"]
        print(f"\n✏️  Overrides: {ov['total_active_overrides']} active / "
              f"{ov['total_history_entries']} history entries")

        sc = s["scheduler"]
        status = "running" if sc.get("scheduler_running") else "manual-only"
        print(f"\n⏰ Scheduler: {status}")
        print(f"   Runs logged: {sc.get('total_runs_logged', 0)} "
              f"({sc.get('successful_runs', 0)} success)")
        print(f"   Backups: {sc.get('backup_count', 0)}")

        api = s["api"]
        print(f"\n🌐 API Sources:")
        print(f"   Hipo Domains API:   {api['hipo_api']}")
        print(f"   REST Countries:     {api['rest_countries']}")
        print(f"   Exchange Rates:     {api['exchange_rates']}")
        print(f"   Currencies loaded:  {api['currencies_loaded']}")

        pp = s.get("policy_publish", {})
        print("\n🧭 Policy Publish:")
        print(f"   Total runs: {pp.get('total_runs', 0)}")
        print(f"   Last run id: {pp.get('last_run_id', 'n/a')}")
        print(f"   Last trigger: {pp.get('last_trigger', 'n/a')}")
        print(f"   Last status: {pp.get('last_status', 'n/a')}")

        print("\n📋 Data Sources:")
        for src, ok in s["data_sources"].items():
            print(f"   {'✅' if ok else '❌'} {src}")

        print("\n⚖️  Open Source Licenses:")
        for src, lic in s["open_source_licenses"].items():
            print(f"   {src}: {lic}")

    def validate_system(self) -> bool:
        print("\n" + "=" * 60 + "\nSYSTEM VALIDATION\n" + "=" * 60)
        return self.db_manager.validate_database()

    def get_all_programs(self, country: Optional[str] = None) -> List[str]:
        return self.db_manager.get_all_programs(country)

    def normalize_tuition_to_usd(self, uni: Dict) -> Optional[float]:
        """Convert a university's tuition to USD using live exchange rates."""
        tuition = uni.get("tuition", {})
        country = uni.get("country", "")
        currency_map = {"UK": ("undergraduate_intl_gbp", "GBP"),
                        "Singapore": ("undergraduate_intl_sgd", "SGD"),
                        "Australia": ("undergraduate_intl_aud", "AUD")}
        if country in currency_map:
            field, currency = currency_map[country]
            amount = tuition.get(field)
            if amount:
                return self.api_integrator.normalize_tuition_to_usd(float(amount), currency)
        return None

    # ── Policy Publishing Orchestration ───────────────────────────────────

    def publish_policy_snapshots(self, trigger: str = "manual") -> Dict:
        """Publish policy payload snapshots from configured sources with run metadata."""
        run_id = uuid4().hex
        run_at = datetime.utcnow().isoformat() + "Z"

        policy_specs = [
            {
                "policy_key": "visa_risk_matrix",
                "url": (os.environ.get("POLICY_VISA_RISK_URL") or os.environ.get("VISA_RISK_DATA_URL") or "").strip(),
                "confidence": self._safe_float(os.environ.get("POLICY_VISA_RISK_CONFIDENCE"), 0.90),
                "validator": self._validate_visa_risk_payload,
            },
            {
                "policy_key": "living_costs",
                "url": (os.environ.get("POLICY_LIVING_COSTS_URL") or "").strip(),
                "confidence": self._safe_float(os.environ.get("POLICY_LIVING_COSTS_CONFIDENCE"), 0.85),
                "validator": self._validate_living_costs_payload,
            },
            {
                "policy_key": "scholarships",
                "url": (os.environ.get("POLICY_SCHOLARSHIPS_URL") or "").strip(),
                "confidence": self._safe_float(os.environ.get("POLICY_SCHOLARSHIPS_CONFIDENCE"), 0.85),
                "validator": self._validate_scholarships_payload,
            },
            {
                "policy_key": "eligibility_thresholds",
                "url": (os.environ.get("POLICY_ELIGIBILITY_THRESHOLDS_URL") or "").strip(),
                "confidence": self._safe_float(os.environ.get("POLICY_ELIGIBILITY_THRESHOLDS_CONFIDENCE"), 0.90),
                "validator": self._validate_eligibility_thresholds_payload,
            },
        ]

        entries = []
        published = 0
        failed = 0
        for spec in policy_specs:
            entry = {
                "run_id": run_id,
                "trigger": trigger,
                "run_at": run_at,
                "policy_key": spec["policy_key"],
                "source": spec["url"] or "",
                "confidence": spec["confidence"],
                "status": "skipped",
                "record_count": 0,
                "snapshot_file": None,
                "error": None,
            }
            try:
                if not spec["url"]:
                    entry["status"] = "skipped_no_url"
                    entries.append(entry)
                    continue

                raw_payload = self._fetch_json_from_url(spec["url"])
                normalized_payload, record_count = spec["validator"](raw_payload)

                snapshot_path = self.policy_dir / f"{spec['policy_key']}.payload.json"
                with open(snapshot_path, "w", encoding="utf-8") as handle:
                    json.dump(normalized_payload, handle, indent=2, ensure_ascii=False)

                entry["status"] = "published"
                entry["record_count"] = record_count
                entry["snapshot_file"] = str(snapshot_path)
                published += 1
            except Exception as exc:
                entry["status"] = "failed"
                entry["error"] = str(exc)
                failed += 1
            entries.append(entry)

        self._append_policy_publish_log(entries)
        self._trigger_api_policy_ingestion(entries)

        return {
            "run_id": run_id,
            "trigger": trigger,
            "published": published,
            "failed": failed,
            "entries": entries,
        }

    def get_policy_publish_statistics(self) -> Dict:
        logs = self._load_policy_publish_log()
        if not logs:
            return {
                "total_runs": 0,
                "last_run_id": None,
                "last_trigger": None,
                "last_status": "none",
            }

        run_ids = []
        seen = set()
        for item in logs:
            rid = item.get("run_id")
            if rid and rid not in seen:
                seen.add(rid)
                run_ids.append(rid)

        last = logs[-1]
        return {
            "total_runs": len(run_ids),
            "last_run_id": last.get("run_id"),
            "last_trigger": last.get("trigger"),
            "last_status": last.get("status"),
        }

    def _fetch_json_from_url(self, url: str) -> Dict:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "UniAssist-PolicyPublisher/1.0"},
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"Policy source must return JSON object: {url}")
        return payload

    def _validate_visa_risk_payload(self, payload: Dict) -> tuple[Dict, int]:
        clean = {}
        for destination, table in payload.items():
            if not isinstance(table, dict):
                continue
            mapped = {}
            for nationality, risk in table.items():
                level = str(risk).strip().lower()
                if level in {"low", "medium", "high"}:
                    mapped[str(nationality).strip()] = level
            if mapped:
                clean[str(destination).strip()] = mapped
        if not clean:
            raise RuntimeError("Visa risk payload contains no valid rows")
        return clean, sum(len(v) for v in clean.values())

    def _validate_living_costs_payload(self, payload: Dict) -> tuple[Dict, int]:
        clean = {}
        for country, row in payload.items():
            if not isinstance(row, dict):
                continue
            try:
                amount = float(row.get("amount"))
            except Exception:
                continue
            currency = str(row.get("currency") or "").strip().upper()
            if amount <= 0 or not currency:
                continue
            clean[str(country).strip()] = {"amount": amount, "currency": currency}
        if not clean:
            raise RuntimeError("Living costs payload contains no valid rows")
        return clean, len(clean)

    def _validate_scholarships_payload(self, payload: Dict) -> tuple[Dict, int]:
        clean = {}
        count = 0
        for country, rows in payload.items():
            if not isinstance(rows, list):
                continue
            normalized_rows = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                item = {
                    "name": str(row.get("name") or "").strip(),
                    "type": str(row.get("type") or "").strip(),
                    "coverage": str(row.get("coverage") or "").strip(),
                    "eligibility": str(row.get("eligibility") or "").strip(),
                    "deadline": str(row.get("deadline") or "").strip(),
                    "website": str(row.get("website") or "").strip(),
                }
                if not item["name"]:
                    continue
                normalized_rows.append(item)
            if normalized_rows:
                clean[str(country).strip()] = normalized_rows
                count += len(normalized_rows)
        if not clean:
            raise RuntimeError("Scholarships payload contains no valid rows")
        return clean, count

    def _validate_eligibility_thresholds_payload(self, payload: Dict) -> tuple[Dict, int]:
        out = {
            "program_min_gpa": {},
            "english_requirements": {},
            "default_english_requirement": {},
        }

        pmg = payload.get("program_min_gpa")
        if isinstance(pmg, dict):
            for program, value in pmg.items():
                try:
                    out["program_min_gpa"][str(program).strip()] = float(value)
                except Exception:
                    continue

        er = payload.get("english_requirements")
        if isinstance(er, dict):
            for country, row in er.items():
                if not isinstance(row, dict):
                    continue
                normalized = {}
                for key in ("ielts", "toefl", "pte"):
                    try:
                        normalized[key] = float(row.get(key))
                    except Exception:
                        continue
                if normalized:
                    out["english_requirements"][str(country).strip()] = normalized

        default_row = payload.get("default_english_requirement")
        if isinstance(default_row, dict):
            for key in ("ielts", "toefl", "pte"):
                try:
                    out["default_english_requirement"][key] = float(default_row.get(key))
                except Exception:
                    continue

        total_rows = len(out["program_min_gpa"]) + len(out["english_requirements"])
        if total_rows == 0 and not out["default_english_requirement"]:
            raise RuntimeError("Eligibility thresholds payload contains no valid rows")
        return out, total_rows

    def _load_policy_publish_log(self) -> List[Dict]:
        if not self.policy_log_path.exists():
            return []
        try:
            with open(self.policy_log_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload if isinstance(payload, list) else []
        except Exception:
            return []

    def _append_policy_publish_log(self, entries: List[Dict]) -> None:
        history = self._load_policy_publish_log()
        history.extend(entries)
        history = history[-1000:]
        with open(self.policy_log_path, "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2, ensure_ascii=False)

    def _trigger_api_policy_ingestion(self, entries: List[Dict]) -> None:
        ingest_url = (os.environ.get("POLICY_PUBLISH_API_INGEST_URL") or "").strip()
        ingest_token = (os.environ.get("POLICY_PUBLISH_ADMIN_TOKEN") or "").strip()
        if not ingest_url or not ingest_token:
            return

        published_count = sum(1 for e in entries if e.get("status") == "published")
        if published_count == 0:
            return

        try:
            request = urllib.request.Request(
                ingest_url,
                method="POST",
                headers={
                    "Authorization": f"Bearer {ingest_token}",
                    "Content-Type": "application/json",
                    "User-Agent": "UniAssist-PolicyPublisher/1.0",
                },
                data=b"{}",
            )
            with urllib.request.urlopen(request, timeout=20) as response:
                response.read()
        except Exception:
            pass

    @staticmethod
    def _safe_float(value: Optional[str], default: float) -> float:
        try:
            if value is None:
                return default
            return float(str(value).strip())
        except Exception:
            return default