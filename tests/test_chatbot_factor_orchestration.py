import unittest

from multiagent.core.agents.chatbot_agent import ChatbotAgent


class DummyReport:
    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return self._payload


class DummyEligibilityAgent:
    def assess(self, profile_data, document_data, universities):
        payload = {
            "tier": "good",
            "eligible_universities": [{"university_id": "u1"}],
            "borderline_universities": [{"university_id": "u2"}],
            "global_improvements": ["Improve IELTS to 6.5+"],
            "english_status": {"overall_pass": True},
            "program_alignment": "compatible",
        }
        return DummyReport(payload)


class DummyFinancialAgent:
    def assess(self, profile_data, universities):
        payload = {
            "feasible_universities": [{"university_id": "u1"}],
            "borderline_universities": [{"university_id": "u2"}],
            "infeasible_universities": [{"university_id": "u3"}],
            "global_recommendations": ["Apply early for scholarships"],
        }
        return DummyReport(payload)


class DummyRecommendationReport:
    def __init__(self):
        self.recommended = [{"university_id": "u1"}]
        self.backup_options = [{"university_id": "u2"}]
        self.global_recommendations = ["Apply to a mix of reach and safety options"]

    def to_dict(self):
        return {
            "recommended": self.recommended,
            "backup_options": self.backup_options,
            "avoid": [{"university_id": "u3"}],
            "global_recommendations": self.global_recommendations,
        }


class DummyRecommendationAgent:
    def recommend(self, universities, profile_data, eligibility_report=None, financial_report=None):
        assert isinstance(eligibility_report, dict)
        assert isinstance(financial_report, dict)
        return DummyRecommendationReport()


class ChatbotFactorOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.chatbot = ChatbotAgent(
            eligibility_agent=DummyEligibilityAgent(),
            financial_agent=DummyFinancialAgent(),
            recommendation_agent=DummyRecommendationAgent(),
            rag_system=None,
        )
        self.context = {
            "profile_data": {
                "country": "UK",
                "program_interest": "Computer Science",
                "financial": {
                    "total_budget": 4500000,
                    "budget_currency": "LKR",
                },
            },
            "document_data": {
                "document_type": "A-Level Results",
                "subjects": {"Mathematics": "A", "Physics": "B"},
            },
            "universities": [
                {"id": "u1", "name": "Uni One", "country": "UK"},
                {"id": "u2", "name": "Uni Two", "country": "Australia"},
                {"id": "u3", "name": "Uni Three", "country": "Singapore"},
            ],
        }

    def test_detects_multiple_external_factors_for_budget_deadline_visa_query(self):
        result = self.chatbot.process_message(
            "My family budget is low and deadlines are close. What visa documents do I need?",
            self.context,
        )

        factors = result.get("agent_data", {}).get("external_factors", [])
        factor_ids = {factor.get("id") for factor in factors}

        self.assertIn("financial_constraints", factor_ids)
        self.assertIn("time_deadlines", factor_ids)
        self.assertIn("visa_immigration", factor_ids)

        actions = result.get("actions", [])
        self.assertTrue(any("deadline" in action.lower() for action in actions))
        self.assertTrue(any("visa" in action.lower() for action in actions))

    def test_recommendation_orchestrates_eligibility_and_financial_agents(self):
        result = self.chatbot.process_message(
            "Can you recommend universities for me?",
            self.context,
        )

        agent_calls = result.get("agent_calls", [])
        self.assertIn("EligibilityVerificationAgent", agent_calls)
        self.assertIn("FinancialFeasibilityAgent", agent_calls)
        self.assertIn("RecommendationAgent", agent_calls)

        agent_results = result.get("agent_data", {}).get("agent_results", {})
        self.assertIn("recommendation", agent_results)
        self.assertIn("eligibility", agent_results)
        self.assertIn("financial", agent_results)

        recommendation_payload = agent_results["recommendation"]
        self.assertIsInstance(recommendation_payload.get("eligibility_report"), dict)
        self.assertIsInstance(recommendation_payload.get("financial_report"), dict)


if __name__ == "__main__":
    unittest.main()
