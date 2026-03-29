"""
Chatbot Agent
==============
Handles conversational interactions with users, providing guidance on eligibility,
financial feasibility, recommendations, and general study abroad advice.

Features:
  - Parses user queries to identify intent (eligibility, financial, recommendation, etc.)
  - Orchestrates calls to other agents (Eligibility, Financial, Recommendation) for personalized responses
  - Provides step-by-step guidance, reassurance, and explanations
  - Supports multiple languages (basic English/Sinhala fallback)
  - Reduces stress through clear, empathetic responses

Algorithm (heuristic, non-ML):
  1. Analyze user message for keywords/intent (e.g., "eligibility" → call Eligibility Agent).
  2. If intent matches an agent:
     a. Gather required data (profile, documents, universities).
     b. Call the agent and format its output into conversational response.
  3. If general query, use canned responses or escalate to human support.
  4. Add empathetic language, next steps, and reminders.

Intended to be integrated into the UI (app.jsx) or API for real-time chat.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    from ..rag_system import RAGSystem
except Exception:  # pragma: no cover - optional dependency at runtime
    RAGSystem = None


class ChatbotAgent:
    """Conversational agent that guides users through study abroad decisions."""

    # Keyword mappings for intent detection
    INTENT_KEYWORDS = {
        "eligibility": ["eligible", "qualification", "requirements", "gpa", "ielts", "toefl", "a/l", "diploma", "bachelor", "master"],
        "financial": ["cost", "fee", "tuition", "budget", "scholarship", "loan", "money", "afford", "expense", "living cost"],
        "recommendation": ["recommend", "suggest", "which university", "best for me", "options", "rank", "deadline"],
        "document": ["upload", "document", "ocr", "scan", "transcript", "certificate"],
        "visa": ["visa", "immigration", "permit", "travel", "entry"],
        "general": ["help", "how to", "what is", "guide", "start", "begin"],
        "emotional": ["worried", "stress", "fear", "confused", "pressure", "reassure"],
    }

    # Canned responses for general/emotional support
    GENERAL_RESPONSES = {
        "greeting": "Hello! I'm here to help you navigate your study abroad journey. What would you like to know about eligibility, costs, or university recommendations?",
        "eligibility_help": "To check eligibility, please upload your academic documents (A/L results, degrees, IELTS/TOEFL scores) and fill in your profile. I'll analyze them against university requirements.",
        "financial_help": "For financial planning, tell me your budget and preferred country. I'll calculate costs, suggest scholarships, and find affordable options.",
        "recommendation_help": "I can recommend universities based on your eligibility and budget. Share your profile and preferences, and I'll provide personalized suggestions.",
        "document_help": "Upload your documents in the Documents section. I can process transcripts, certificates, and test scores automatically.",
        "visa_help": "Visa processes vary by country. I can explain requirements and timelines once you select a university.",
        "stress_relief": "I understand applying abroad can be stressful. Take it one step at a time — start with eligibility checks, then finances. You're not alone in this!",
        "deadline_reminder": "Don't forget application deadlines! I can prioritize universities by earliest deadlines to help you stay on track.",
        "trust": "All my recommendations are based on verified university data and your personal details. No hidden agendas — just transparent guidance.",
        "global_advice": "For global uncertainties like pandemics or politics, I suggest flexible options like online programs or countries with stable policies.",
        "fallback": "I'm not sure about that specific question. Could you rephrase or ask about eligibility, costs, or recommendations?",
    }

    def __init__(self, eligibility_agent=None, financial_agent=None, recommendation_agent=None, document_agent=None, rag_system=None):
        self.eligibility_agent = eligibility_agent
        self.financial_agent = financial_agent
        self.recommendation_agent = recommendation_agent
        self.document_agent = document_agent
        self.rag_system = rag_system

    def set_rag_system(self, rag_system) -> None:
        """Attach or replace a RAG system at runtime."""
        self.rag_system = rag_system

    def process_message(self, user_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a user message and return a response.

        Args:
            user_message: The user's input text.
            context: Dict with user profile, documents, universities, etc.

        Returns:
            Dict with 'response' (str), 'intent' (str), 'actions' (list of next steps), 'agent_calls' (list of called agents).
        """
        intent = self._detect_intent(user_message.lower())
        response = ""
        actions = []
        agent_calls = []
        agent_data: Dict[str, Any] = {
            "intent": intent,
            "agent_results": {},
            "processed_at": datetime.utcnow().isoformat() + "Z",
        }

        if intent == "eligibility":
            response, agent_calls, agent_result = self._handle_eligibility(context)
            if agent_result:
                agent_data["agent_results"]["eligibility"] = agent_result
            actions = ["Upload academic documents", "Complete profile", "Check specific university requirements"]
        elif intent == "financial":
            response, agent_calls, agent_result = self._handle_financial(context)
            if agent_result:
                agent_data["agent_results"]["financial"] = agent_result
            actions = ["Provide budget details", "Explore scholarships", "Compare costs"]
        elif intent == "recommendation":
            response, agent_calls, agent_result = self._handle_recommendation(context)
            if agent_result:
                agent_data["agent_results"]["recommendation"] = agent_result
            actions = ["Review recommended universities", "Apply to top choices", "Prepare documents"]
        elif intent == "document":
            response = self.GENERAL_RESPONSES["document_help"]
            actions = ["Upload documents", "Verify OCR results"]
        elif intent == "visa":
            response = self.GENERAL_RESPONSES["visa_help"]
            actions = ["Research visa requirements", "Contact embassy"]
        elif intent == "emotional":
            response = self.GENERAL_RESPONSES["stress_relief"]
            actions = ["Take a break", "Focus on one step", "Seek support"]
        elif intent == "general":
            response = self.GENERAL_RESPONSES["greeting"]
            actions = ["Start with eligibility check", "Explore options"]
        else:
            response = self.GENERAL_RESPONSES["fallback"]
            actions = ["Rephrase question", "Ask about specific topics"]

        # Use RAG for general or unresolved queries when available.
        if self._should_use_rag(intent=intent, response=response):
            rag_response = self._handle_rag(user_message=user_message, context=context)
            if rag_response:
                response = rag_response
                agent_calls.append("RAGSystem")
                agent_data["agent_results"]["rag"] = {
                    "used": True,
                    "query": user_message,
                }
                if "Review cited information" not in actions:
                    actions.append("Review cited information")

        # Add empathetic wrapper
        response = self._add_empathy(response, intent)

        return {
            "response": response,
            "intent": intent,
            "actions": actions,
            "agent_calls": agent_calls,
            "agent_data": agent_data,
        }

    def _detect_intent(self, message: str) -> str:
        """Simple keyword-based intent detection."""
        for intent, keywords in self.INTENT_KEYWORDS.items():
            if any(re.search(r'\b' + re.escape(kw) + r'\b', message) for kw in keywords):
                return intent
        return "general"

    def _handle_eligibility(self, context: Dict) -> tuple[str, List[str], Dict[str, Any]]:
        """Call Eligibility Agent if available."""
        agent_calls = []
        agent_result: Dict[str, Any] = {}
        if self.eligibility_agent and context.get("profile_data") and context.get("document_data"):
            try:
                unis = context.get("universities", [])
                report = self.eligibility_agent.assess(context["profile_data"], context["document_data"], unis)
                response = f"Based on your documents, your eligibility tier is '{report.tier}'. "
                if report.eligible_universities:
                    response += f"You qualify for {len(report.eligible_universities)} universities. "
                    response += f"Global improvements: {', '.join(report.global_improvements[:2])}."
                else:
                    response += "You may need pathway programs. Check the improvements section."
                agent_calls.append("EligibilityVerificationAgent")
                agent_result = report.to_dict() if hasattr(report, "to_dict") else dict(report)
            except Exception as e:
                response = f"Eligibility check failed: {e}. Please ensure your profile and documents are complete."
        else:
            response = self.GENERAL_RESPONSES["eligibility_help"]
        return response, agent_calls, agent_result

    def _handle_financial(self, context: Dict) -> tuple[str, List[str], Dict[str, Any]]:
        """Call Financial Feasibility Agent if available."""
        agent_calls = []
        agent_result: Dict[str, Any] = {}
        if self.financial_agent and context.get("profile_data"):
            try:
                unis = context.get("universities", [])
                report = self.financial_agent.assess(context["profile_data"], unis)
                feasible_count = len(report.feasible_universities)
                response = f"With your budget, {feasible_count} universities are feasible. "
                if report.global_recommendations:
                    response += f"Recommendations: {', '.join(report.global_recommendations[:2])}."
                agent_calls.append("FinancialFeasibilityAgent")
                agent_result = report.to_dict() if hasattr(report, "to_dict") else dict(report)
            except Exception as e:
                response = f"Financial assessment failed: {e}. Please provide your budget details."
        else:
            response = self.GENERAL_RESPONSES["financial_help"]
        return response, agent_calls, agent_result

    def _handle_recommendation(self, context: Dict) -> tuple[str, List[str], Dict[str, Any]]:
        """Call Recommendation Agent if available."""
        agent_calls = []
        agent_result: Dict[str, Any] = {}
        if self.recommendation_agent and context.get("universities"):
            try:
                eligibility_report = context.get("eligibility_report")
                financial_report = context.get("financial_report")
                report = self.recommendation_agent.recommend(
                    context["universities"], context.get("profile_data", {}),
                    eligibility_report, financial_report
                )
                rec_count = len(report.recommended)
                response = f"I recommend {rec_count} universities for you. "
                if report.global_recommendations:
                    response += f"Tips: {', '.join(report.global_recommendations[:2])}."
                agent_calls.append("RecommendationAgent")
                agent_result = report.to_dict() if hasattr(report, "to_dict") else dict(report)
            except Exception as e:
                response = f"Recommendation failed: {e}. Please complete eligibility and financial checks first."
        else:
            response = self.GENERAL_RESPONSES["recommendation_help"]
        return response, agent_calls, agent_result

    def _add_empathy(self, response: str, intent: str) -> str:
        """Add empathetic language based on intent."""
        if intent in ["emotional", "general"]:
            return f"I'm here to help! {response}"
        elif intent == "eligibility":
            return f"Don't worry, eligibility is just the first step. {response}"
        elif intent == "financial":
            return f"Finances can be tricky, but we can find solutions. {response}"
        else:
            return response

    def _should_use_rag(self, intent: str, response: str) -> bool:
        """Decide whether to use RAG grounding for this turn."""
        if not self.rag_system:
            return False

        if intent == "general":
            return True

        if response == self.GENERAL_RESPONSES["fallback"]:
            return True

        return False

    def _handle_rag(self, user_message: str, context: Dict[str, Any]) -> Optional[str]:
        """Retrieve context and generate an answer using RAG + Gemini."""
        if not self.rag_system:
            return None

        try:
            result = self.rag_system.answer_with_context(query=user_message, context=context)
            text = (result or {}).get("response", "").strip()
            if text:
                return text
            return None
        except Exception:
            return None