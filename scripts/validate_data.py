#!/usr/bin/env python3
"""Simple data validation for repo JSON/JSONL files.

Checks parseability and basic schema constraints for
`multiagent/data/databases/universities_database.json` and
JSON/JSONL files under `multiagent/data/training/`.
"""
from pathlib import Path
import json
import sys


def validate_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)


def validate_jsonl(path: Path):
    errors = []
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                count += 1
            except Exception as e:
                errors.append(f"line {i}: {e}")
    return count, errors


def check_universities_schema(data, path: Path):
    errors = []
    if not isinstance(data, dict):
        errors.append("root must be a JSON object/dict")
        return errors

    metadata = data.get("metadata")
    if not metadata or not isinstance(metadata, dict):
        errors.append("missing or invalid 'metadata' object")
    else:
        if "total" in metadata and isinstance(metadata.get("total"), int):
            declared = metadata["total"]
            # sum country lists
            actual = sum(len(v) for k, v in data.items() if k != "metadata")
            if declared != actual:
                errors.append(f"metadata.total ({declared}) != actual entries ({actual})")

    countries = metadata.get("countries") if metadata else None
    if countries and isinstance(countries, list):
        for c in countries:
            if c not in data:
                errors.append(f"country '{c}' listed in metadata.countries but no top-level key present")

    # minimal per-entry checks
    for k, entries in data.items():
        if k == "metadata":
            continue
        if not isinstance(entries, list):
            errors.append(f"top-level key '{k}' is not a list")
            continue
        for idx, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                errors.append(f"{k}[{idx}] not an object")
                continue
            for field in ("id", "name", "country"):
                if field not in entry:
                    errors.append(f"{k}[{idx}] missing '{field}'")
            ac = entry.get("acceptance_criteria")
            if ac and isinstance(ac, dict):
                for sub in ("ielts_min", "toefl_min"):
                    if sub not in ac:
                        errors.append(f"{k}[{idx}].acceptance_criteria missing '{sub}'")
            else:
                errors.append(f"{k}[{idx}] missing or invalid 'acceptance_criteria' object")

    return errors


def main():
    repo_root = Path(__file__).resolve().parents[1]
    db_dir = repo_root / "multiagent" / "data" / "databases"
    training_dir = repo_root / "multiagent" / "data" / "training"

    failed = False

    # universities database
    uni_path = db_dir / "universities_database.json"
    if uni_path.exists():
        print(f"Validating {uni_path}")
        data, err = validate_json(uni_path)
        if err:
            print(f"ERROR: failed to parse {uni_path}: {err}")
            failed = True
        else:
            issues = check_universities_schema(data, uni_path)
            if issues:
                print("Schema issues found:")
                for it in issues:
                    print(" - ", it)
                failed = True
            else:
                print("OK: universities database looks consistent")
    else:
        print(f"Warning: {uni_path} not found")

    # training files (jsonl/json)
    if training_dir.exists():
        for p in sorted(training_dir.iterdir()):
            if p.suffix.lower() == ".jsonl":
                print(f"Checking JSONL {p}")
                count, errors = validate_jsonl(p)
                if errors:
                    print(f"ERROR: parse errors in {p}:")
                    for e in errors[:10]:
                        print(" - ", e)
                    failed = True
                else:
                    print(f"OK: {p} ({count} records)")
            elif p.suffix.lower() == ".json":
                print(f"Checking JSON {p}")
                _, err = validate_json(p)
                if err:
                    print(f"ERROR: failed to parse {p}: {err}")
                    failed = True
                else:
                    print(f"OK: {p}")
    else:
        print(f"Warning: training dir {training_dir} not found")

    if failed:
        print("One or more checks failed")
        sys.exit(1)
    else:
        print("All checks passed")


if __name__ == "__main__":
    main()
