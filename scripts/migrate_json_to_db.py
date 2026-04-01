from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Column, DateTime, String, Text, create_engine, func
from sqlalchemy.orm import declarative_base, sessionmaker


Base = declarative_base()


class DBUser(Base):
    __tablename__ = "users"

    email = Column(String(255), primary_key=True, index=True)
    data = Column(Text, nullable=False, default="{}")


class DBChatHistory(Base):
    __tablename__ = "chat_history"

    user_id = Column(String(255), primary_key=True, index=True)
    data = Column(Text, nullable=False, default="{}")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class DBUserState(Base):
    __tablename__ = "user_state"

    user_id = Column(String(255), primary_key=True, index=True)
    data = Column(Text, nullable=False, default="{}")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


def normalize_database_url(raw_url: str) -> str:
    url = (raw_url or "").strip()
    if url.startswith("postgresql://"):
        return f"postgresql+psycopg://{url[len('postgresql://'):]}"
    if url.startswith("postgres://"):
        return f"postgresql+psycopg://{url[len('postgres://'):]}"
    return url


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        print(f"[warn] Could not read {path}: {exc}")
        return default


def snapshot_sources(source_paths: list[Path], backup_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = backup_root / f"json_to_db_{stamp}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    for source in source_paths:
        if not source.exists():
            continue
        target = snapshot_dir / source.name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    return snapshot_dir


def cleanup_json_files(source_paths: list[Path]) -> None:
    for source in source_paths:
        if not source.exists():
            continue
        if source.is_dir():
            for json_file in source.glob("*.json"):
                json_file.unlink(missing_ok=True)
        else:
            source.unlink(missing_ok=True)


def collect_users(users_path: Path) -> dict[str, Any]:
    data = load_json(users_path, {"users": {}})
    users = data.get("users", {}) if isinstance(data, dict) else {}
    return users if isinstance(users, dict) else {}


def collect_chat_records(chat_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not chat_dir.exists():
        return records

    for path in sorted(chat_dir.glob("*.json")):
        raw = load_json(path, {"messages": [], "agent_data": {}})
        if isinstance(raw, list):
            payload = {"messages": [item for item in raw if isinstance(item, dict)], "agent_data": {}}
        elif isinstance(raw, dict):
            messages = raw.get("messages", [])
            agent_data = raw.get("agent_data", {})
            payload = {
                "messages": [item for item in messages if isinstance(item, dict)] if isinstance(messages, list) else [],
                "agent_data": agent_data if isinstance(agent_data, dict) else {},
            }
        else:
            payload = {"messages": [], "agent_data": {}}
        records[path.stem] = payload

    return records


def collect_user_state(state_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not state_dir.exists():
        return records

    for path in sorted(state_dir.glob("*.json")):
        raw = load_json(path, {})
        records[path.stem] = raw if isinstance(raw, dict) else {}

    return records


def upsert_users(session, users: dict[str, Any]) -> int:
    count = 0
    for email, user_data in users.items():
        row = session.get(DBUser, email)
        payload = user_data if isinstance(user_data, dict) else {}
        if row is None:
            session.add(DBUser(email=email, data=json.dumps(payload, ensure_ascii=False)))
        else:
            row.data = json.dumps(payload, ensure_ascii=False)
        count += 1
    return count


def upsert_chat_history(session, records: dict[str, dict[str, Any]]) -> int:
    count = 0
    for user_id, payload in records.items():
        row = session.get(DBChatHistory, user_id)
        data = json.dumps(payload, ensure_ascii=False)
        if row is None:
            session.add(DBChatHistory(user_id=user_id, data=data))
        else:
            row.data = data
        count += 1
    return count


def upsert_user_state(session, records: dict[str, dict[str, Any]]) -> int:
    count = 0
    for user_id, payload in records.items():
        row = session.get(DBUserState, user_id)
        data = json.dumps(payload, ensure_ascii=False)
        if row is None:
            session.add(DBUserState(user_id=user_id, data=data))
        else:
            row.data = data
        count += 1
    return count


def parse_args() -> argparse.Namespace:
    default_data_dir = repo_root() / "multiagent" / "data"
    default_backup_dir = default_data_dir / "backups"
    parser = argparse.ArgumentParser(
        description="Migrate multiagent JSON auth/chat/state data into the current PostgreSQL schema."
    )
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL", ""))
    parser.add_argument("--data-dir", default=str(default_data_dir))
    parser.add_argument("--backup-dir", default=str(default_backup_dir))
    parser.add_argument(
        "--cleanup-json",
        action="store_true",
        help="Delete source JSON files after a successful backup and migration.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    database_url = normalize_database_url(args.database_url)
    if not database_url:
        print("[error] DATABASE_URL is required. Pass --database-url or export DATABASE_URL.")
        return 1

    data_dir = Path(args.data_dir).resolve()
    backup_dir = Path(args.backup_dir).resolve()

    users_path = data_dir / "users" / "users.json"
    chat_dir = data_dir / "chat_history"
    state_dir = data_dir / "user_state"
    source_paths = [users_path, chat_dir, state_dir]

    users = collect_users(users_path)
    chat_records = collect_chat_records(chat_dir)
    state_records = collect_user_state(state_dir)

    print(f"[info] Users found: {len(users)}")
    print(f"[info] Chat histories found: {len(chat_records)}")
    print(f"[info] User states found: {len(state_records)}")

    snapshot_dir = snapshot_sources(source_paths, backup_dir)
    print(f"[info] Backup snapshot created at: {snapshot_dir}")

    engine = create_engine(database_url, pool_pre_ping=True)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    with SessionLocal() as session:
        migrated_users = upsert_users(session, users)
        migrated_chat = upsert_chat_history(session, chat_records)
        migrated_state = upsert_user_state(session, state_records)
        session.commit()

        db_users = session.query(DBUser).count()
        db_chat = session.query(DBChatHistory).count()
        db_state = session.query(DBUserState).count()

    print(f"[ok] Migrated users: {migrated_users} | users table rows: {db_users}")
    print(f"[ok] Migrated chat histories: {migrated_chat} | chat_history rows: {db_chat}")
    print(f"[ok] Migrated user states: {migrated_state} | user_state rows: {db_state}")

    if args.cleanup_json:
        cleanup_json_files(source_paths)
        print("[ok] Source JSON files removed after successful migration.")
    else:
        print("[info] Source JSON files were left in place. Re-run with --cleanup-json after verification.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())