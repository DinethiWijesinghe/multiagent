from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


@dataclass(frozen=True)
class TestCase:
    case_id: str
    question: str
    expected_intents: list[str]
    expected_factors: list[str]
    expected_actions: list[str]
    meaning_keywords: list[str]


TEST_CASES: list[TestCase] = [
    TestCase(
        case_id="EF-01",
        question="My family budget is very low. Can I still apply to UK universities?",
        expected_intents=["financial"],
        expected_factors=["financial_constraints"],
        expected_actions=["scholarships", "budget-feasible", "fee waiver", "annual budget"],
        meaning_keywords=["budget", "affordable", "scholarship", "cost"],
    ),
    TestCase(
        case_id="EF-02",
        question="Why did you recommend University A over University B? I want transparent reasons.",
        expected_intents=["recommendation"],
        expected_factors=["trust_transparency", "reliable_information"],
        expected_actions=["ranking", "eligibility", "compare", "evidence"],
        meaning_keywords=["reason", "eligibility", "cost", "deadline"],
    ),
    TestCase(
        case_id="EF-03",
        question="I have A/L results with mixed grades. Do I need a foundation pathway?",
        expected_intents=["eligibility"],
        expected_factors=["educational_background", "language_proficiency"],
        expected_actions=["eligibility", "pathway", "program fit"],
        meaning_keywords=["foundation", "pathway", "eligible", "direct"],
    ),
    TestCase(
        case_id="EF-04",
        question="My IELTS is 5.5. Can I apply now or should I retake?",
        expected_intents=["eligibility"],
        expected_factors=["language_proficiency"],
        expected_actions=["language", "retake", "requirements"],
        meaning_keywords=["ielts", "score", "retake", "requirement"],
    ),
    TestCase(
        case_id="EF-05",
        question="I live in a rural area with weak internet. Can I complete this process online?",
        expected_intents=["document"],
        expected_factors=["geographic_socioeconomic"],
        expected_actions=["remote", "upload", "saved progress", "low-travel"],
        meaning_keywords=["online", "remote", "upload", "progress"],
    ),
    TestCase(
        case_id="EF-06",
        question="I am very anxious and overwhelmed. I feel I will miss everything.",
        expected_intents=["emotional"],
        expected_factors=["psychological_emotional", "time_deadlines"],
        expected_actions=["small steps", "one next action", "prioritize"],
        meaning_keywords=["step", "support", "deadline", "manage"],
    ),
    TestCase(
        case_id="EF-07",
        question="What visa documents do I need for Canada student permit?",
        expected_intents=["visa"],
        expected_factors=["visa_immigration", "global_external"],
        expected_actions=["visa", "checklist", "lower-risk"],
        meaning_keywords=["visa", "document", "proof", "timeline"],
    ),
    TestCase(
        case_id="EF-08",
        question="Deadlines are this month. Which application should I submit first?",
        expected_intents=["recommendation"],
        expected_factors=["time_deadlines", "trust_transparency"],
        expected_actions=["earliest deadline", "sequence", "documents"],
        meaning_keywords=["deadline", "first", "priority", "sequence"],
    ),
    TestCase(
        case_id="EF-09",
        question="I heard there may be policy changes and travel restrictions. Should I keep backups?",
        expected_intents=["visa"],
        expected_factors=["global_external", "trust_transparency"],
        expected_actions=["risk", "backup", "hybrid", "online"],
        meaning_keywords=["policy", "risk", "backup", "travel"],
    ),
    TestCase(
        case_id="EF-10",
        question="Consultants gave me different advice. Which source should I trust?",
        expected_intents=["general", "recommendation"],
        expected_factors=["reliable_information", "trust_transparency"],
        expected_actions=["official", "compare", "evidence"],
        meaning_keywords=["official", "evidence", "compare", "trust"],
    ),
    TestCase(
        case_id="EF-11",
        question="Exchange rate keeps changing. How do I plan my tuition payments safely?",
        expected_intents=["financial"],
        expected_factors=["financial_constraints", "global_external"],
        expected_actions=["tuition", "living costs", "scholarships", "buffer"],
        meaning_keywords=["exchange", "currency", "budget", "risk"],
    ),
    TestCase(
        case_id="EF-12",
        question="I want fast recommendations but also need visa-safe and affordable options.",
        expected_intents=["recommendation"],
        expected_factors=["financial_constraints", "visa_immigration", "time_deadlines", "trust_transparency"],
        expected_actions=["shortlist", "lower-risk", "earliest deadlines", "budget-feasible"],
        meaning_keywords=["affordable", "visa", "deadline", "shortlist"],
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time chatbot accuracy checks against live API.")
    parser.add_argument("--base-url", default=os.getenv("API_BASE_URL", "http://127.0.0.1:8000"), help="API base URL")
    parser.add_argument("--email", default=os.getenv("ACCURACY_TEST_EMAIL"), help="Student account email")
    parser.add_argument("--password", default=os.getenv("ACCURACY_TEST_PASSWORD"), help="Student account password")
    parser.add_argument("--token", default=os.getenv("ACCURACY_TEST_TOKEN"), help="Optional existing bearer token")
    parser.add_argument("--quick", action="store_true", help="Run only 5 critical release checks")
    parser.add_argument("--timeout", type=float, default=20.0, help="Request timeout in seconds")
    return parser.parse_args()


def _auth_token(base_url: str, timeout: float, token: str | None, email: str | None, password: str | None) -> str:
    if token:
        return token
    if not email or not password:
        raise SystemExit("Provide --token or both --email and --password")

    response = requests.post(
        f"{base_url}/auth/login",
        json={"email": email, "password": password},
        timeout=timeout,
    )
    if response.status_code != 200:
        raise SystemExit(f"Login failed: {response.status_code} {response.text}")

    payload = response.json() if response.content else {}
    token_value = payload.get("token")
    if not token_value:
        raise SystemExit("Login succeeded but token missing in response")
    return str(token_value)


def _extract_factor_ids(agent_data: dict[str, Any]) -> list[str]:
    factors = agent_data.get("external_factors")
    if not isinstance(factors, list):
        return []
    ids: list[str] = []
    for factor in factors:
        if isinstance(factor, dict) and factor.get("id"):
            ids.append(str(factor["id"]))
    return ids


def _contains_any(haystack_values: list[str], needles: list[str]) -> bool:
    haystack = " ".join(haystack_values).lower()
    return any(needle.lower() in haystack for needle in needles)


def _score_case(case: TestCase, result_payload: dict[str, Any]) -> dict[str, Any]:
    intent = str(result_payload.get("intent") or "").strip().lower()
    response_text = str(result_payload.get("response") or "")
    actions = [str(item) for item in (result_payload.get("actions") or []) if isinstance(item, str)]
    agent_data = result_payload.get("agent_data") if isinstance(result_payload.get("agent_data"), dict) else {}
    factors = _extract_factor_ids(agent_data)

    intent_pass = intent in [x.lower() for x in case.expected_intents]
    factors_pass = set(case.expected_factors).issubset(set(factors))
    actions_pass = _contains_any(actions, case.expected_actions)

    keyword_hits = sum(1 for key in case.meaning_keywords if key.lower() in response_text.lower())
    required_hits = 2 if len(case.meaning_keywords) >= 3 else 1
    meaning_pass = keyword_hits >= required_hits

    checks = {
        "intent": intent_pass,
        "external_factors": factors_pass,
        "actions": actions_pass,
        "meaning": meaning_pass,
    }
    passed = all(checks.values())

    return {
        "case_id": case.case_id,
        "question": case.question,
        "passed": passed,
        "checks": checks,
        "predicted_intent": intent,
        "expected_intents": case.expected_intents,
        "observed_external_factors": factors,
        "expected_external_factors": case.expected_factors,
        "observed_actions": actions,
        "response_preview": response_text[:300],
    }


def _default_context() -> dict[str, Any]:
    return {
        "profile_data": {
            "country": "Sri Lanka",
            "target_country": "UK",
            "financial": {"total_budget": 20000},
            "program_interest": "Computer Science",
            "english_test": {"type": "IELTS", "score": 5.5},
        },
        "document_data": {
            "documents": [
                {"type": "alevel", "grade_summary": "A, B, C"},
                {"type": "passport", "status": "available"},
            ]
        },
    }


def _select_cases(quick: bool) -> list[TestCase]:
    if not quick:
        return TEST_CASES
    quick_ids = {"EF-01", "EF-06", "EF-07", "EF-08", "EF-12"}
    return [c for c in TEST_CASES if c.case_id in quick_ids]


def _write_reports(results: list[dict[str, Any]], summary: dict[str, Any]) -> tuple[Path, Path]:
    reports_dir = Path(__file__).resolve().parents[1] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    json_path = reports_dir / f"realtime_accuracy_{stamp}.json"
    md_path = reports_dir / f"realtime_accuracy_{stamp}.md"

    json_path.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Real-Time Accuracy Check",
        "",
        f"Generated at: {summary['generated_at']}",
        "",
        f"Base URL: {summary['base_url']}",
        f"Cases Run: {summary['cases_run']}",
        f"Cases Passed: {summary['cases_passed']}",
        f"Accuracy: {summary['accuracy_pct']}%",
        "",
        "## Case Results",
        "",
        "| Case ID | PASS | Intent | Factors | Actions | Meaning |",
        "|---|---|---|---|---|---|",
    ]

    for row in results:
        checks = row["checks"]
        lines.append(
            f"| {row['case_id']} | {'YES' if row['passed'] else 'NO'} | "
            f"{'OK' if checks['intent'] else 'NO'} | {'OK' if checks['external_factors'] else 'NO'} | "
            f"{'OK' if checks['actions'] else 'NO'} | {'OK' if checks['meaning'] else 'NO'} |"
        )

    failed = [r for r in results if not r["passed"]]
    if failed:
        lines.extend(["", "## Failed Cases", ""])
        for row in failed:
            lines.append(f"- {row['case_id']}: intent={row['predicted_intent']}, factors={row['observed_external_factors']}")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")

    token = _auth_token(
        base_url=base_url,
        timeout=args.timeout,
        token=args.token,
        email=args.email,
        password=args.password,
    )

    headers = {"Authorization": f"Bearer {token}"}
    cases = _select_cases(args.quick)
    context = _default_context()

    results: list[dict[str, Any]] = []
    for case in cases:
        payload = {
            "user_message": case.question,
            "context": context,
        }
        response = requests.post(
            f"{base_url}/chat/respond",
            json=payload,
            headers=headers,
            timeout=args.timeout,
        )
        if response.status_code != 200:
            results.append(
                {
                    "case_id": case.case_id,
                    "question": case.question,
                    "passed": False,
                    "checks": {
                        "intent": False,
                        "external_factors": False,
                        "actions": False,
                        "meaning": False,
                    },
                    "predicted_intent": "http_error",
                    "expected_intents": case.expected_intents,
                    "observed_external_factors": [],
                    "expected_external_factors": case.expected_factors,
                    "observed_actions": [],
                    "response_preview": response.text[:300],
                }
            )
            continue

        scored = _score_case(case, response.json())
        results.append(scored)

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    accuracy_pct = round((passed / total) * 100, 2) if total else 0.0

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "cases_run": total,
        "cases_passed": passed,
        "accuracy_pct": accuracy_pct,
        "mode": "quick" if args.quick else "full",
    }

    json_report, md_report = _write_reports(results, summary)

    print("Real-time accuracy check completed")
    print(f"Mode: {summary['mode']}")
    print(f"Cases: {total}")
    print(f"Passed: {passed}")
    print(f"Accuracy: {accuracy_pct}%")
    print(f"JSON report: {json_report}")
    print(f"MD report: {md_report}")


if __name__ == "__main__":
    main()
