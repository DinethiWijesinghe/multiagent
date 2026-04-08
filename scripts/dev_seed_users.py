from __future__ import annotations

import argparse

from multiagent import api_server


DEFAULT_ADMIN_EMAIL = "admin@example.com"
DEFAULT_ADMIN_PASSWORD = "Admin@123"
DEFAULT_ADMIN_NAME = "System Admin"

DEFAULT_ADVISOR_EMAIL = "advisor@example.com"
DEFAULT_ADVISOR_PASSWORD = "Advisor@123"
DEFAULT_ADVISOR_NAME = "System Advisor"

DEFAULT_STUDENT_EMAIL = "student@example.com"
DEFAULT_STUDENT_PASSWORD = "Student@123"
DEFAULT_STUDENT_NAME = "Demo Student"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seed development users into the PostgreSQL/Neon auth database.",
    )
    parser.add_argument("--admin-email", default=DEFAULT_ADMIN_EMAIL)
    parser.add_argument("--admin-password", default=DEFAULT_ADMIN_PASSWORD)
    parser.add_argument("--admin-name", default=DEFAULT_ADMIN_NAME)

    parser.add_argument("--advisor-email", default=DEFAULT_ADVISOR_EMAIL)
    parser.add_argument("--advisor-password", default=DEFAULT_ADVISOR_PASSWORD)
    parser.add_argument("--advisor-name", default=DEFAULT_ADVISOR_NAME)

    parser.add_argument(
        "--include-student",
        action="store_true",
        help="Also seed a demo student account for UI testing.",
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
        _seed_user(args.admin_email, args.admin_password, args.admin_name, "admin"),
        _seed_user(args.advisor_email, args.advisor_password, args.advisor_name, "advisor"),
    ]

    if args.include_student:
        seeded.append(
            _seed_user(args.student_email, args.student_password, args.student_name, "student")
        )

    print("Seeded development users:")
    for item in seeded:
        print(f"- {item['role']}: {item['email']} | password: {item['password']} | name: {item['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())