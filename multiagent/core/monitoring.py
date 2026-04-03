"""
Agent Monitoring & Metrics Collection
======================================
Tracks multi-agent interaction flows, latency, intent accuracy, and RAG relevance.

Provides:
  - Per-agent execution timing
  - Intent detection confidence
  - RAG source quality metrics
  - Agent call sequences
  - Real-time metrics export (JSON)
"""

import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class AgentMetric:
    """Single agent execution capture."""
    agent_name: str
    duration_ms: float
    success: bool
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class IntentMetric:
    """Intent detection capture."""
    detected_intent: str
    confidence: float  # 0.0-1.0
    user_message: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    keywords_matched: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class RAGMetric:
    """RAG retrieval capture."""
    query: str
    source_count: int
    relevance_level: str  # "high", "low", "unknown"
    llm_provider: str  # "gemini", "none"
    relevance_score: Optional[float] = None
    duration_ms: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self):
        return asdict(self)


@dataclass
class QueryFlowMetric:
    """Complete query processing flow."""
    query_id: str
    user_message: str
    intent: str
    agents_called: List[str] = field(default_factory=list)
    total_duration_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    intent_metric: Optional[Dict[str, Any]] = None
    agent_metrics: List[Dict[str, Any]] = field(default_factory=list)
    rag_metric: Optional[Dict[str, Any]] = None

    def to_dict(self):
        d = asdict(self)
        d['timestamp'] = self.timestamp
        return d


class MetricsCollector:
    """Centralized metrics collection with in-memory storage."""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.query_flows: List[QueryFlowMetric] = []
        self.agent_metrics: List[AgentMetric] = []
        self.intent_metrics: List[IntentMetric] = []
        self.rag_metrics: List[RAGMetric] = []
        self._lock = None  # For thread safety if needed

    def record_intent(self, detected_intent: str, confidence: float, user_message: str, keywords: Optional[List[str]] = None):
        """Record intent detection."""
        metric = IntentMetric(
            detected_intent=detected_intent,
            confidence=confidence,
            user_message=user_message,
            keywords_matched=keywords or []
        )
        self.intent_metrics.append(metric)
        self._trim_history(self.intent_metrics)
        logger.debug(f"[METRICS] Intent detected: {detected_intent} (conf={confidence:.2f})")

    def record_agent(self, agent_name: str, duration_ms: float, success: bool, error: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Record single agent execution."""
        metric = AgentMetric(
            agent_name=agent_name,
            duration_ms=duration_ms,
            success=success,
            error=error,
            metadata=metadata or {}
        )
        self.agent_metrics.append(metric)
        self._trim_history(self.agent_metrics)
        status = "✓" if success else "✗"
        logger.debug(f"[METRICS] Agent '{agent_name}': {status} {duration_ms:.1f}ms")

    def record_rag(self, query: str, source_count: int, relevance_level: str, llm_provider: str, relevance_score: Optional[float] = None, duration_ms: Optional[float] = None):
        """Record RAG retrieval."""
        metric = RAGMetric(
            query=query,
            source_count=source_count,
            relevance_level=relevance_level,
            llm_provider=llm_provider,
            relevance_score=relevance_score,
            duration_ms=duration_ms
        )
        self.rag_metrics.append(metric)
        self._trim_history(self.rag_metrics)
        logger.debug(f"[METRICS] RAG: sources={source_count}, relevance={relevance_level}, provider={llm_provider}")

    def start_query_flow(self, query_id: str, user_message: str, intent: str) -> 'QueryFlowContext':
        """Context manager for tracking full query flow."""
        return QueryFlowContext(self, query_id, user_message, intent)

    def _trim_history(self, history_list: List):
        """Maintain max history size."""
        if len(history_list) > self.max_history:
            del history_list[:-self.max_history]

    def get_summary(self) -> Dict[str, Any]:
        """Get aggregated metrics summary."""
        total_queries = len(self.query_flows)
        
        intent_distribution = defaultdict(int)
        for flow in self.query_flows:
            intent_distribution[flow.intent] += 1

        agent_stats = defaultdict(lambda: {"calls": 0, "total_ms": 0.0, "successes": 0})
        for metric in self.agent_metrics:
            stats = agent_stats[metric.agent_name]
            stats["calls"] += 1
            stats["total_ms"] += metric.duration_ms
            if metric.success:
                stats["successes"] += 1

        rag_stats = {
            "total_queries": len(self.rag_metrics),
            "avg_sources": sum(m.source_count for m in self.rag_metrics) / len(self.rag_metrics) if self.rag_metrics else 0,
            "high_relevance_rate": sum(1 for m in self.rag_metrics if m.relevance_level == "high") / len(self.rag_metrics) if self.rag_metrics else 0,
            "provider_distribution": defaultdict(int),
        }
        for metric in self.rag_metrics:
            rag_stats["provider_distribution"][metric.llm_provider] += 1

        # Convert intent distribution to dict
        intent_dist_dict = dict(intent_distribution)
        
        # Convert agent stats to dict with averages
        agent_stats_dict = {}
        for agent_name, stats in agent_stats.items():
            agent_stats_dict[agent_name] = {
                "calls": stats["calls"],
                "avg_duration_ms": stats["total_ms"] / stats["calls"] if stats["calls"] > 0 else 0,
                "success_rate": stats["successes"] / stats["calls"] if stats["calls"] > 0 else 0,
            }

        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_queries": total_queries,
            "intent_distribution": intent_dist_dict,
            "agent_stats": agent_stats_dict,
            "rag_stats": rag_stats,
            "recent_queries": [flow.to_dict() for flow in self.query_flows[-10:]],  # Last 10
        }

    def get_recent_flows(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent query flows."""
        return [flow.to_dict() for flow in self.query_flows[-limit:]]

    def export_json(self) -> str:
        """Export all metrics as JSON."""
        return json.dumps(self.get_summary(), indent=2, default=str)


class QueryFlowContext:
    """Context manager for tracking query flow."""

    def __init__(self, collector: MetricsCollector, query_id: str, user_message: str, intent: str):
        self.collector = collector
        self.query_id = query_id
        self.user_message = user_message
        self.intent = intent
        self.flow = QueryFlowMetric(
            query_id=query_id,
            user_message=user_message,
            intent=intent
        )
        self.start_time = time.time()

    def add_agent_call(self, agent_name: str, duration_ms: float, success: bool = True, metadata: Optional[Dict[str, Any]] = None):
        """Record agent execution within this flow."""
        self.flow.agents_called.append(agent_name)
        self.flow.agent_metrics.append(AgentMetric(
            agent_name=agent_name,
            duration_ms=duration_ms,
            success=success,
            metadata=metadata or {}
        ).to_dict())

    def set_intent_metric(self, detected_intent: str, confidence: float, keywords: Optional[List[str]] = None):
        """Set intent detection result."""
        self.flow.intent_metric = IntentMetric(
            detected_intent=detected_intent,
            confidence=confidence,
            user_message=self.user_message,
            keywords_matched=keywords or []
        ).to_dict()

    def set_rag_metric(self, query: str, source_count: int, relevance_level: str, llm_provider: str, relevance_score: Optional[float] = None, duration_ms: Optional[float] = None):
        """Set RAG metric within this flow."""
        self.flow.rag_metric = RAGMetric(
            query=query,
            source_count=source_count,
            relevance_level=relevance_level,
            llm_provider=llm_provider,
            relevance_score=relevance_score,
            duration_ms=duration_ms
        ).to_dict()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finalize and store query flow."""
        self.flow.total_duration_ms = (time.time() - self.start_time) * 1000
        self.collector.query_flows.append(self.flow)
        self.collector._trim_history(self.collector.query_flows)
        logger.info(f"[METRICS] Query {self.query_id}: {self.flow.total_duration_ms:.1f}ms | Intent: {self.intent} | Agents: {len(self.flow.agents_called)}")


# Global singleton
_metrics_collector = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def reset_metrics():
    """Reset all metrics (useful for testing)."""
    global _metrics_collector
    _metrics_collector = MetricsCollector()
