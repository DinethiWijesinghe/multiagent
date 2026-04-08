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

import json
import os
import re
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta

try:
    from ..rag_system import RAGSystem
except Exception:  # pragma: no cover - optional dependency at runtime
    RAGSystem = None


logger = logging.getLogger(__name__)


class ChatbotAgent:
    """Conversational agent that guides users through study abroad decisions."""
    DEFAULT_CONFIG_RELATIVE_PATH = os.path.join("data", "config", "chatbot_agent_config.json")

    def __init__(self, eligibility_agent=None, financial_agent=None, recommendation_agent=None, document_agent=None, rag_system=None):
        self.eligibility_agent = eligibility_agent
        self.financial_agent = financial_agent
        self.recommendation_agent = recommendation_agent
        self.document_agent = document_agent
        self.rag_system = rag_system
        self.external_factor_config: Dict[str, Dict[str, Any]] = {}
        self.intent_keywords: Dict[str, List[str]] = {}
        self.general_responses: Dict[str, str] = {}
        self.default_factors_by_intent: Dict[str, List[str]] = {}
        self.distress_signal_weights: Dict[str, int] = {}
        self._load_runtime_config()

    def _load_runtime_config(self) -> None:
        """Load chatbot behavior settings from JSON config."""
        module_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        default_path = os.path.join(module_root, self.DEFAULT_CONFIG_RELATIVE_PATH)
        config_path = os.environ.get("CHATBOT_AGENT_CONFIG_PATH", default_path)

        with open(config_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            raise RuntimeError("Chatbot agent config must be a JSON object.")

        required_sections = [
            "external_factor_config",
            "intent_keywords",
            "general_responses",
            "default_factors_by_intent",
            "distress_signal_weights",
        ]
        for key in required_sections:
            if key not in payload:
                raise RuntimeError(f"Missing required chatbot config section: {key}")

        self.external_factor_config = payload["external_factor_config"]
        self.intent_keywords = payload["intent_keywords"]
        self.general_responses = payload["general_responses"]
        self.default_factors_by_intent = payload["default_factors_by_intent"]
        self.distress_signal_weights = payload["distress_signal_weights"]

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
            response = self.general_responses.get("document_help", "Please upload your documents for processing.")
            actions = ["Upload documents", "Verify OCR results"]
        elif intent == "visa":
            response = self.general_responses.get("visa_help", "I can explain visa requirements once your target country is selected.")
            actions = ["Research visa requirements", "Contact embassy"]
        elif intent == "emotional":
            response = self.general_responses.get("stress_relief", "Take it one step at a time. We can do this together.")
            actions = ["Take a break", "Focus on one step", "Seek support"]
        elif intent == "general":
            response = self.general_responses.get("greeting", "I'm here to help with your study abroad plan.")
            actions = ["Start with eligibility check", "Explore options"]
        else:
            response = self.general_responses.get("fallback", "Could you rephrase your question?")
            actions = ["Rephrase question", "Ask about specific topics"]

        external_factors = self._identify_external_factors(
            message=user_message,
            intent=intent,
            context=context,
            agent_results=agent_data["agent_results"],
        )
        agent_data["external_factors"] = external_factors

        deadline_plan = self._build_deadline_plan(
            message=user_message,
            intent=intent,
            context=context,
            agent_results=agent_data["agent_results"],
        )
        if deadline_plan.get("items"):
            agent_data["deadline_plan"] = deadline_plan

        emotional_support = self._build_emotional_support_plan(
            message=user_message,
            intent=intent,
            context=context,
            agent_results=agent_data["agent_results"],
        )
        if emotional_support.get("level") != "none":
            agent_data["emotional_support"] = emotional_support

        actions = self._merge_actions(actions, self._actions_for_external_factors(external_factors))

        if deadline_plan.get("items"):
            actions = self._merge_actions(
                actions,
                [
                    "Review your generated deadline plan",
                    "Set calendar reminders for checkpoint dates",
                ],
            )

        if emotional_support.get("next_steps"):
            actions = self._merge_actions(actions, emotional_support.get("next_steps", []))

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
        response = self._add_empathy(response, intent, emotional_support)

        return {
            "response": response,
            "intent": intent,
            "actions": actions,
            "agent_calls": agent_calls,
            "agent_data": agent_data,
        }

    def _detect_intent(self, message: str) -> str:
        """Simple keyword-based intent detection."""
        for intent, keywords in self.intent_keywords.items():
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
            response = self.general_responses.get("eligibility_help", "Upload your academic details so I can check eligibility.")
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
            response = self.general_responses.get("financial_help", "Share your budget and preferred country to begin financial checks.")
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
                policy_meta = (report_dict.get("policy_metadata") or {}).get("visa_risk") or {}
                policy_source = policy_meta.get("source")
                policy_updated = policy_meta.get("updated_at")
                if policy_source:
                    trust_line = f" Visa risk policy source: {policy_source}"
                    if policy_updated:
                        trust_line += f" (updated: {policy_updated})."
                    else:
                        trust_line += "."
                    response += trust_line
                agent_result = {
                    **report_dict,
                    "eligibility_report": eligibility_report,
                    "financial_report": financial_report,
                }
            except Exception as e:
                logger.exception("Recommendation agent error")
                response = f"Recommendation failed: {e}. Please complete eligibility and financial checks first."
        else:
            response = self.general_responses.get("recommendation_help", "Share your profile to get personalized recommendations.")
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

        for factor_id, config in self.external_factor_config.items():
            if any(keyword in message_lower for keyword in config["keywords"]):
                matched.append(factor_id)

        matched.extend(self.default_factors_by_intent.get(intent, []))

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
            if factor_id in seen or factor_id not in self.external_factor_config:
                continue
            seen.add(factor_id)
            ordered.append({
                "id": factor_id,
                "label": self.external_factor_config[factor_id]["label"],
            })
        return ordered

    def _actions_for_external_factors(self, external_factors: List[Dict[str, str]]) -> List[str]:
        actions: List[str] = []
        for factor in external_factors or []:
            config = self.external_factor_config.get(factor.get("id") or "")
            if config:
                actions.extend(config.get("actions", []))
        return actions

    def _add_empathy(self, response: str, intent: str, emotional_support: Optional[Dict[str, Any]] = None) -> str:
        """Add empathetic language based on intent."""
        level = (emotional_support or {}).get("level")
        if level == "high":
            return (
                "I can see this feels heavy right now, and we can handle it one step at a time. "
                + response
                + " Start with the smallest next step in your action list."
            )
        if level == "moderate":
            return "This is manageable with a clear plan. " + response

        if intent in ["emotional", "general"]:
            return f"I'm here to help! {response}"
        elif intent == "eligibility":
            return f"Don't worry, eligibility is just the first step. {response}"
        elif intent == "financial":
            return f"Finances can be tricky, but we can find solutions. {response}"
        else:
            return response

    def _build_deadline_plan(
        self,
        message: str,
        intent: str,
        context: Dict[str, Any],
        agent_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a concrete checkpoint plan from available university deadlines."""
        message_l = (message or "").lower()
        recommendation_data = agent_results.get("recommendation") or {}
        candidates = []

        for entry in (recommendation_data.get("recommended") or []) + (recommendation_data.get("backup_options") or []):
            if isinstance(entry, dict):
                candidates.append(entry)

        if not candidates:
            for uni in context.get("universities") or []:
                if isinstance(uni, dict):
                    candidates.append(
                        {
                            "university_name": uni.get("name") or uni.get("university_name") or "Unknown University",
                            "deadline": uni.get("application_deadline") or uni.get("deadline"),
                        }
                    )

        plan_items = []
        now = datetime.now(timezone.utc)
        for item in candidates:
            deadline_raw = item.get("deadline")
            deadline_dt = self._parse_deadline(deadline_raw)
            if not deadline_dt:
                continue

            days_left = (deadline_dt - now).days
            if days_left < 0:
                continue

            checkpoints = []
            for offset in (30, 14, 7, 2):
                checkpoint = deadline_dt - timedelta(days=offset)
                if checkpoint > now:
                    checkpoints.append(checkpoint.date().isoformat())

            urgency = "normal"
            if days_left <= 14:
                urgency = "high"
            elif days_left <= 30:
                urgency = "medium"

            plan_items.append(
                {
                    "university": item.get("university_name") or item.get("name") or "Unknown University",
                    "deadline": deadline_dt.date().isoformat(),
                    "days_left": days_left,
                    "urgency": urgency,
                    "checkpoints": checkpoints,
                }
            )

        plan_items.sort(key=lambda p: p["days_left"])

        if not plan_items:
            should_force = intent == "recommendation" or any(k in message_l for k in ["deadline", "timeline", "intake", "urgent"])
            if should_force:
                return {
                    "items": [],
                    "note": "No dated deadlines found in current data. Add university deadline dates to generate reminders.",
                }
            return {"items": []}

        return {
            "items": plan_items[:5],
            "note": "Use checkpoint dates as reminder triggers in calendar/email systems.",
        }

    def _build_emotional_support_plan(
        self,
        message: str,
        intent: str,
        context: Dict[str, Any],
        agent_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Estimate support intensity and return practical emotional next steps."""
        msg = (message or "").lower()
        score = 0

        for keyword, weight in self.distress_signal_weights.items():
            if keyword in msg:
                score += weight

        profile_data = context.get("profile_data") or {}
        if not profile_data:
            score += 1
        if not context.get("document_data"):
            score += 1

        eligibility_result = agent_results.get("eligibility") or {}
        financial_result = agent_results.get("financial") or {}

        if isinstance(eligibility_result.get("ineligible_universities"), list) and not eligibility_result.get("eligible_universities"):
            score += 2
        if isinstance(financial_result.get("feasible_universities"), list) and len(financial_result.get("feasible_universities", [])) == 0:
            score += 2

        if intent == "emotional":
            score += 2

        if score >= 5:
            level = "high"
            next_steps = [
                "Choose one immediate task only (profile, document, or shortlist)",
                "Set a 20-minute focused session and pause",
                "Share your plan with a trusted advisor or family member",
            ]
            check_in_hours = 24
        elif score >= 3:
            level = "moderate"
            next_steps = [
                "Split tasks into eligibility, finances, and deadlines",
                "Complete the easiest task first for momentum",
                "Review progress at end of day",
            ]
            check_in_hours = 48
        elif score >= 1:
            level = "low"
            next_steps = [
                "Keep a simple checklist for next actions",
                "Confirm one completed step before starting another",
            ]
            check_in_hours = 72
        else:
            level = "none"
            next_steps = []
            check_in_hours = None

        return {
            "level": level,
            "score": score,
            "next_steps": next_steps,
            "check_in_hours": check_in_hours,
        }

    def _parse_deadline(self, value: Any) -> Optional[datetime]:
        if not value:
            return None
        raw = str(value).strip()
        if not raw:
            return None

        for fmt in (
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%b %d, %Y",
            "%B %d, %Y",
        ):
            try:
                parsed = datetime.strptime(raw, fmt)
                return parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        match = re.search(r"(\d{4})-(\d{2})-(\d{2})", raw)
        if match:
            try:
                parsed = datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
                return parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                return None
        return None

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