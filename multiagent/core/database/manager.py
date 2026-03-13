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
import json, os, shutil
from datetime import datetime
from typing import Dict, List, Optional

from .phase1_curated_db       import CuratedDatabaseManager
from .phase2_web_scraper      import WebScraper
from .phase3_api_integration  import APIIntegrator
from .phase4_scheduler        import UpdateScheduler
from .phase5_override_manager import OverrideManager


class UnifiedDataManager:
    def __init__(self, start_scheduler: bool = False):
        print("🚀 Initialising Unified Data Manager...")
        print("=" * 60)
        self.db_manager       = CuratedDatabaseManager()
        self.scraper          = WebScraper()
        self.api_integrator   = APIIntegrator()
        self.override_manager = OverrideManager()
        self.scheduler        = UpdateScheduler(manager=self)

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
        return {
            "database":     self.db_manager.get_statistics(),
            "overrides":    self.override_manager.get_statistics(),
            "scheduler":    self.scheduler.get_statistics(),
            "api":          self.api_integrator.get_statistics(),
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