import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from multiagent import api_server


class AuthSecurityControlsTests(unittest.TestCase):
    def test_authenticate_token_rejects_expired_session(self):
        token = "expired-token"
        old_created_at = (datetime.now(timezone.utc) - timedelta(hours=api_server.SESSION_TTL_HOURS + 2)).isoformat()
        sessions = {
            token: {
                "email": "user@example.com",
                "created_at": old_created_at,
            }
        }

        with patch("multiagent.api_server._load_sessions", return_value=sessions.copy()), patch(
            "multiagent.api_server._save_sessions"
        ) as save_sessions:
            with self.assertRaises(HTTPException) as ctx:
                api_server._authenticate_token(f"Bearer {token}")

        self.assertEqual(ctx.exception.status_code, 401)
        save_sessions.assert_called_once()

    def test_authenticate_token_accepts_unexpired_session(self):
        token = "valid-token"
        sessions = {
            token: {
                "email": "user@example.com",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        }

        with patch("multiagent.api_server._load_sessions", return_value=sessions):
            email = api_server._authenticate_token(f"Bearer {token}")

        self.assertEqual(email, "user@example.com")

    def test_register_forces_student_role_by_default(self):
        payload = api_server.RegisterPayload(
            name="Admin Candidate",
            email="admin.candidate@example.com",
            password="Secret123!",
            role="admin",
        )
        users_written = {}

        def _capture_save(users):
            users_written.update(users)

        with patch("multiagent.api_server._load_users", return_value={}), patch(
            "multiagent.api_server._save_users", side_effect=_capture_save
        ), patch("multiagent.api_server.ALLOW_PRIVILEGED_SELF_REGISTRATION", False):
            result = api_server.register(payload)

        self.assertTrue(result["success"])
        self.assertEqual(result["user"]["role"], "student")
        self.assertEqual(users_written[payload.email.lower()]["role"], "student")

    def test_validate_password_complexity_when_enabled(self):
        with patch("multiagent.api_server.PASSWORD_REQUIRE_COMPLEXITY", True):
            with self.assertRaises(HTTPException):
                api_server._validate_password_policy("passwordonly")


class ApiEndpointTests(unittest.TestCase):
    def test_login_throttle_hits_429(self):
        """After AUTH_MAX_LOGIN_ATTEMPTS wrong passwords the key is throttled to 429."""
        email = "throttle@example.com"
        users = {
            email: {
                "name": "Throttle User",
                "email": email,
                "password_hash": api_server._hash_password("correct-pass"),
                "role": "student",
            }
        }
        mock_request = MagicMock()
        mock_request.client.host = "127.0.0.1"

        key = api_server._login_rate_limit_key(email, "127.0.0.1")
        api_server._LOGIN_ATTEMPTS.pop(key, None)

        payload = api_server.LoginPayload(email=email, password="wrong-pass")
        try:
            with patch("multiagent.api_server._load_users", return_value=users):
                for _ in range(api_server.AUTH_MAX_LOGIN_ATTEMPTS):
                    with self.assertRaises(HTTPException) as ctx:
                        api_server.login(payload, mock_request)
                    self.assertEqual(ctx.exception.status_code, 401)

                # One more attempt should now be throttled
                with self.assertRaises(HTTPException) as ctx:
                    api_server.login(payload, mock_request)
                self.assertEqual(ctx.exception.status_code, 429)
        finally:
            api_server._LOGIN_ATTEMPTS.pop(key, None)

    def test_get_metrics_returns_summary(self):
        """get_metrics() returns the collector summary when METRICS_PUBLIC is True."""
        mock_collector = MagicMock()
        mock_collector.get_summary.return_value = {"total_queries": 0, "agents": {}}

        with patch("multiagent.api_server._metrics_collector", mock_collector), \
                patch("multiagent.api_server.METRICS_PUBLIC", True):
            result = api_server.get_metrics(authorization=None)

        self.assertIn("total_queries", result)
        mock_collector.get_summary.assert_called_once()

    def test_get_metrics_flows_returns_flows_and_count(self):
        """get_metrics_flows() returns both 'flows' and 'count' keys."""
        mock_collector = MagicMock()
        mock_collector.get_recent_flows.return_value = []

        with patch("multiagent.api_server._metrics_collector", mock_collector), \
                patch("multiagent.api_server.METRICS_PUBLIC", True):
            result = api_server.get_metrics_flows(limit=10, authorization=None)

        self.assertIn("flows", result)
        self.assertIn("count", result)
        self.assertEqual(result["count"], 0)
        mock_collector.get_recent_flows.assert_called_once_with(limit=10)

    def test_register_with_api_payload_returns_student_role(self):
        """register() called via API payload downgrades admin → student by default."""
        payload = api_server.RegisterPayload(
            name="Sneaky Admin",
            email="sneaky@example.com",
            password="ValidPass1!",
            role="advisor",
        )

        with patch("multiagent.api_server._load_users", return_value={}), \
                patch("multiagent.api_server._save_users"), \
                patch("multiagent.api_server.ALLOW_PRIVILEGED_SELF_REGISTRATION", False):
            result = api_server.register(payload)

        self.assertTrue(result["success"])
        self.assertEqual(result["user"]["role"], "student")


if __name__ == "__main__":
    unittest.main()
