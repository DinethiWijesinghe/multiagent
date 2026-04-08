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
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

try:
    from ..rag_system import RAGSystem
except Exception:  # pragma: no cover - optional dependency at runtime
    RAGSystem = None


logger = logging.getLogger(__name__)


class ChatbotAgent:
    """Conversational agent that guides users through study abroad decisions."""

    EXTERNAL_FACTOR_CONFIG = {
        "financial_constraints": {
            "label": "Financial Constraints",
            "keywords": [
                "budget", "cost", "tuition", "fee", "semester fee", "scholarship", "loan", "afford", "living cost",
                "exchange rate", "family income", "bank statement", "fund proof", "financial statement", "part-time work",
            ],
            "actions": [
                "Set your annual budget in profile", "Review tuition and living costs", "Check scholarships and fee waivers",
                "Shortlist budget-feasible universities",
            ],
        },
        "reliable_information": {
            "label": "Access to Reliable Information",
            "keywords": [
                "information", "advice", "trust", "reliable", "compare", "transparent", "explain", "why",
                "official source", "verified", "consultant", "agent", "authentic",
            ],
            "actions": [
                "Review grounded recommendations", "Check official university links", "Compare evidence before deciding",
            ],
        },
        "educational_background": {
            "label": "Educational Background Differences",
            "keywords": [
                "qualification", "gpa", "grades", "a/l", "ol", "diploma", "hnd", "bachelor", "master", "transcript",
                "pathway", "foundation", "credit transfer", "recognition",
            ],
            "actions": [
                "Upload transcripts and certificates", "Run eligibility check", "Check program fit and pathway options",
            ],
        },
        "language_proficiency": {
            "label": "Language Proficiency Challenges",
            "keywords": [
                "ielts", "toefl", "pte", "english", "language", "score", "overall band", "writing score", "speaking score",
                "retake", "english requirement",
            ],
            "actions": [
                "Check language-score requirements", "Compare your score with program thresholds", "Plan retake if needed",
            ],
        },
        "geographic_socioeconomic": {
            "label": "Geographic and Socio-Economic Factors",
            "keywords": [
                "remote", "rural", "travel", "distance", "mobile", "access", "online guidance", "internet", "data package",
                "device", "transport",
            ],
            "actions": [
                "Use remote document upload", "Prioritize low-travel application steps", "Keep all progress saved in your account",
            ],
        },
        "psychological_emotional": {
            "label": "Psychological and Emotional Factors",
            "keywords": [
                "stress", "worried", "fear", "confused", "pressure", "anxious", "reassure", "panic", "overwhelmed",
                "family pressure", "fear of rejection",
            ],
            "actions": [
                "Break the process into small steps", "Focus on one next action", "Start with realistic options first",
            ],
        },
        "visa_immigration": {
            "label": "Visa and Immigration Uncertainty",
            "keywords": [
                "visa", "permit", "embassy", "immigration", "entry", "travel document", "visa rejection", "visa refusal",
                "sop", "gte", "cas", "offer letter", "fund proof",
            ],
            "actions": [
                "Review visa requirements early", "Prepare visa document checklist", "Prioritize lower-risk destinations",
            ],
        },
        "time_deadlines": {
            "label": "Time Constraints and Deadlines",
            "keywords": [
                "deadline", "timeline", "urgent", "when", "date", "schedule", "submission", "intake", "jan intake",
                "may intake", "sep intake", "late application", "this month",
            ],
            "actions": [
                "Prioritize earliest deadlines", "Sequence applications by intake", "Prepare required documents first",
            ],
        },
        "trust_transparency": {
            "label": "Trust and Transparency Issues",
            "keywords": [
                "why", "reason", "trust", "bias", "transparent", "explain recommendation", "proof", "evidence",
                "fair", "agent scam", "consultant bias",
            ],
            "actions": [
                "Review ranking reasons", "Check eligibility and cost evidence", "Compare recommendation with backups",
            ],
        },
        "global_external": {
            "label": "Global External Factors",
            "keywords": [
                "pandemic", "restriction", "politics", "political", "instability", "hybrid", "online", "global risk",
                "war", "currency crisis", "policy change", "travel ban",
            ],
            "actions": [
                "Check country-risk updates", "Consider hybrid or online fallback options", "Keep backup destinations ready",
            ],
        },
    }

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
            "processed_at": datetime.now(timezone.utc).isoformat(),
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
                if isinstance(agent_result.get("eligibility_report"), dict):
                    agent_data["agent_results"]["eligibility"] = agent_result["eligibility_report"]
                if isinstance(agent_result.get("financial_report"), dict):
                    agent_data["agent_results"]["financial"] = agent_result["financial_report"]
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

        external_factors = self._identify_external_factors(
            message=user_message,
            intent=intent,
            context=context,
            agent_results=agent_data["agent_results"],
        )
        agent_data["external_factors"] = external_factors
        actions = self._merge_actions(actions, self._actions_for_external_factors(external_factors))

        # Use RAG for every intent when available, enriched with specialised analysis and factor context.
        if self._should_use_rag(intent=intent, response=response):
            rag_response = self._handle_rag(
                user_message=user_message,
                context=context,
                agent_results=agent_data.get("agent_results") or {},
                external_factors=external_factors,
            )
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
                logger.exception("Eligibility agent error")
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
                logger.exception("Financial agent error")
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
                if (
                    eligibility_report is None
                    and self.eligibility_agent
                    and context.get("profile_data")
                    and context.get("document_data")
                ):
                    derived_eligibility = self.eligibility_agent.assess(
                        context["profile_data"],
                        context["document_data"],
                        context["universities"],
                    )
                    eligibility_report = (
                        derived_eligibility.to_dict()
                        if hasattr(derived_eligibility, "to_dict")
                        else dict(derived_eligibility)
                    )
                    agent_calls.append("EligibilityVerificationAgent")

                financial_report = context.get("financial_report")
                if (
                    financial_report is None
                    and self.financial_agent
                    and context.get("profile_data")
                ):
                    derived_financial = self.financial_agent.assess(
                        context["profile_data"],
                        context["universities"],
                    )
                    financial_report = (
                        derived_financial.to_dict()
                        if hasattr(derived_financial, "to_dict")
                        else dict(derived_financial)
                    )
                    agent_calls.append("FinancialFeasibilityAgent")

                report = self.recommendation_agent.recommend(
                    context["universities"], context.get("profile_data", {}),
                    eligibility_report, financial_report
                )
                rec_count = len(report.recommended)
                backup_count = len(report.backup_options)
                response = f"I recommend {rec_count} universities for you, with {backup_count} backup options. "
                if report.global_recommendations:
                    response += f"Tips: {', '.join(report.global_recommendations[:2])}."
                if eligibility_report and eligibility_report.get("tier"):
                    response += f" Your current eligibility tier is {eligibility_report.get('tier')}."
                if financial_report and isinstance(financial_report.get("feasible_universities"), list):
                    response += f" {len(financial_report.get('feasible_universities', []))} options are currently budget-feasible."
                if "RecommendationAgent" not in agent_calls:
                    agent_calls.append("RecommendationAgent")
                report_dict = report.to_dict() if hasattr(report, "to_dict") else dict(report)
                agent_result = {
                    **report_dict,
                    "eligibility_report": eligibility_report,
                    "financial_report": financial_report,
                }
            except Exception as e:
                logger.exception("Recommendation agent error")
                response = f"Recommendation failed: {e}. Please complete eligibility and financial checks first."
        else:
            response = self.GENERAL_RESPONSES["recommendation_help"]
        return response, agent_calls, agent_result

    def _merge_actions(self, base_actions: List[str], factor_actions: List[str]) -> List[str]:
        seen = set()
        merged: List[str] = []
        for action in [*(base_actions or []), *(factor_actions or [])]:
            if not action or action in seen:
                continue
            seen.add(action)
            merged.append(action)
        return merged

    def _identify_external_factors(
        self,
        message: str,
        intent: str,
        context: Dict[str, Any],
        agent_results: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        message_lower = (message or "").lower()
        matched: List[str] = []

        for factor_id, config in self.EXTERNAL_FACTOR_CONFIG.items():
            if any(keyword in message_lower for keyword in config["keywords"]):
                matched.append(factor_id)

        default_factors_by_intent = {
            "eligibility": ["educational_background", "language_proficiency"],
            "financial": ["financial_constraints"],
            "recommendation": ["reliable_information", "time_deadlines", "trust_transparency"],
            "document": ["educational_background", "geographic_socioeconomic"],
            "visa": ["visa_immigration", "global_external"],
            "emotional": ["psychological_emotional"],
            "general": ["reliable_information"],
        }
        matched.extend(default_factors_by_intent.get(intent, []))

        profile_data = context.get("profile_data") or {}
        if (profile_data.get("financial") or {}).get("total_budget"):
            matched.append("financial_constraints")
        if context.get("document_data"):
            matched.append("educational_background")

        eligibility_result = agent_results.get("eligibility") or {}
        if eligibility_result.get("english_status") or eligibility_result.get("program_alignment"):
            matched.append("language_proficiency")

        recommendation_result = agent_results.get("recommendation") or {}
        if recommendation_result.get("recommended") or recommendation_result.get("backup_options"):
            matched.extend(["trust_transparency", "time_deadlines"])

        if agent_results.get("financial"):
            matched.append("financial_constraints")

        ordered: List[Dict[str, str]] = []
        seen = set()
        for factor_id in matched:
            if factor_id in seen or factor_id not in self.EXTERNAL_FACTOR_CONFIG:
                continue
            seen.add(factor_id)
            ordered.append({
                "id": factor_id,
                "label": self.EXTERNAL_FACTOR_CONFIG[factor_id]["label"],
            })
        return ordered

    def _actions_for_external_factors(self, external_factors: List[Dict[str, str]]) -> List[str]:
        actions: List[str] = []
        for factor in external_factors or []:
            config = self.EXTERNAL_FACTOR_CONFIG.get(factor.get("id") or "")
            if config:
                actions.extend(config.get("actions", []))
        return actions

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
        # Use RAG for every intent to ground responses with knowledge-base context
        return True

    def _handle_rag(
        self,
        user_message: str,
        context: Dict[str, Any],
        agent_results: Optional[Dict[str, Any]] = None,
        external_factors: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[str]:
        """Retrieve context and generate an answer using RAG + Gemini."""
        if not self.rag_system:
            return None

        try:
            conversation_history = context.get("conversation_history") or []
            enriched_context = {
                **context,
                "agent_results": agent_results or {},
                "external_factors": external_factors or [],
            }
            result = self.rag_system.answer_with_context(
                query=user_message,
                context=enriched_context,
                k=4,
                conversation_history=conversation_history,
            )
            text = (result or {}).get("response", "").strip()
            urls = (result or {}).get("relevant_urls") or []
            options = (result or {}).get("options") or []

            extras: List[str] = []
            if options:
                extras.append("Suggested options:")
                for item in options[:3]:
                    name = (item or {}).get("name") or "Option"
                    country = (item or {}).get("country") or ""
                    option_line = f"- {name}"
                    if country:
                        option_line += f" ({country})"
                    extras.append(option_line)

            if urls:
                extras.append("Relevant links:")
                for url in urls[:3]:
                    extras.append(f"- {url}")

            if text:
                if extras:
                    text = f"{text}\n\n" + "\n".join(extras)
                return text
            return None
        except Exception:
            logger.exception("RAG handling failed")
            return None