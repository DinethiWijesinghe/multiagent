from __future__ import annotations

import argparse

from multiagent import api_server


DEFAULT_STUDENT_EMAIL = "student@example.com"
DEFAULT_STUDENT_PASSWORD = "Student@123"
DEFAULT_STUDENT_NAME = "Demo Student"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seed a development student user into the PostgreSQL/Neon auth database.",
    )
    parser.add_argument("--student-email", default=DEFAULT_STUDENT_EMAIL)
    parser.add_argument("--student-password", default=DEFAULT_STUDENT_PASSWORD)
    parser.add_argument("--student-name", default=DEFAULT_STUDENT_NAME)
    return parser


def _seed_user(email: str, password: str, name: str, role: str) -> dict[str, str]:
    record = api_server._upsert_user_account(email, password, name, role)
    if not record:
        raise RuntimeError(f"Failed to seed {role} account for {email}.")
    return {
        "email": record.get("email") or email,
        "name": record.get("name") or name,
        "role": record.get("role") or role,
        "password": password,
    }


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    api_server._initialize_database()

    seeded = [
        _seed_user(args.student_email, args.student_password, args.student_name, "student")
    ]

    print("Seeded development users:")
    for item in seeded:
        print(f"- {item['role']}: {item['email']} | password: {item['password']} | name: {item['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())