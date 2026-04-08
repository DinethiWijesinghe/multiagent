"""Trigger policy ingestion jobs via admin API.

Usage:
  set API_BASE_URL=http://127.0.0.1:8000
  set ADMIN_TOKEN=<token>
  d:/Multiagent/.venv311/Scripts/python.exe scripts/ingest_policy_snapshots.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request


def _post_json(url: str, token: str) -> dict:
    request = urllib.request.Request(
        url=url,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        data=b"{}",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        body = response.read().decode("utf-8")
        return json.loads(body or "{}")


def main() -> int:
    api_base = (os.environ.get("API_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")
    admin_token = (os.environ.get("ADMIN_TOKEN") or "").strip()

    if not admin_token:
        print("ADMIN_TOKEN is required")
        return 1

    url = f"{api_base}/admin/policies/ingest/all"
    try:
        payload = _post_json(url, admin_token)
        print(json.dumps(payload, indent=2))
        return 0
    except Exception as exc:
        print(f"Failed to ingest policy snapshots: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
