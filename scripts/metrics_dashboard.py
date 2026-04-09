
"""
Agent Metrics Dashboard
=======================
Real-time visualization of multi-agent system performance.

Usage:
  python scripts/metrics_dashboard.py

Displays:
  - Intent distribution (pie chart)
  - Agent latency (bar chart)
  - RAG statistics
  - Recent query flows (table)
"""

import requests
import json
import time
from datetime import datetime
from collections import defaultdict

API_BASE = "http://127.0.0.1:8000"


def get_metrics():
    """Fetch current metrics."""
    try:
        response = requests.get(f"{API_BASE}/metrics", timeout=5)
        return response.json()
    except Exception as e:
        print(f"❌ Error fetching metrics: {e}")
        return None


def get_recent_flows(limit=10):
    """Fetch recent query flows."""
    try:
        response = requests.get(f"{API_BASE}/metrics/flows?limit={limit}", timeout=5)
        return response.json()
    except Exception as e:
        print(f"❌ Error fetching flows: {e}")
        return None


def print_header(title):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_intent_distribution(metrics):
    """Display intent distribution."""
    print_header("INTENT DISTRIBUTION")
    intent_dist = metrics.get("intent_distribution", {})
    total = sum(intent_dist.values())
    
    if not intent_dist:
        print("  No queries yet.")
        return
    
    for intent, count in sorted(intent_dist.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 5)
        print(f"  {intent:15} │ {count:3} │ {pct:5.1f}% {bar}")


def print_agent_stats(metrics):
    """Display agent performance stats."""
    print_header("AGENT PERFORMANCE")
    agent_stats = metrics.get("agent_stats", {})
    
    if not agent_stats:
        print("  No agent calls yet.")
        return
    
    print(f"  {'Agent':<25} {'Calls':>6} {'Avg (ms)':>10} {'Success':>8}")
    print(f"  {'-'*60}")
    
    for agent, stats in sorted(agent_stats.items()):
        calls = stats.get("calls", 0)
        avg_ms = stats.get("avg_duration_ms", 0)
        success_rate = stats.get("success_rate", 0)
        success_pct = f"{success_rate*100:.1f}%"
        print(f"  {agent:<25} {calls:>6} {avg_ms:>10.1f} {success_pct:>8}")


def print_rag_stats(metrics):
    """Display RAG statistics."""
    print_header("RAG RETRIEVAL STATISTICS")
    rag = metrics.get("rag_stats", {})
    
    if rag.get("total_queries", 0) == 0:
        print("  No RAG queries yet.")
        return
    
    print(f"  Total queries:        {rag.get('total_queries', 0)}")
    print(f"  Avg sources found:    {rag.get('avg_sources', 0):.2f}")
    print(f"  High relevance rate:  {rag.get('high_relevance_rate', 0)*100:.1f}%")
    
    providers = rag.get("provider_distribution", {})
    if providers:
        print(f"\n  Provider breakdown:")
        for provider, count in providers.items():
            print(f"    - {provider:<15} {count} queries")


def print_recent_flows(flows_data):
    """Display recent query flows."""
    print_header("RECENT QUERY FLOWS (Last 10)")
    flows = flows_data.get("flows", [])
    
    if not flows:
        print("  No query flows yet.")
        return
    
    for i, flow in enumerate(flows[-10:], 1):
        agents = flow.get("agents_called", [])
        duration_ms = flow.get("total_duration_ms", 0)
        intent = flow.get("intent", "unknown")
        
        print(f"\n  [{i}] Intent: {intent:<15} | Duration: {duration_ms:.1f}ms")
        print(f"      Agents: {', '.join(agents) if agents else 'None'}")
        print(f"      Query:  {flow.get('user_message', '')[:60]}...")


def print_summary(metrics):
    """Print overall summary."""
    print_header("SYSTEM SUMMARY")
    print(f"  Total queries processed: {metrics.get('total_queries', 0)}")
    print(f"  Timestamp:              {metrics.get('timestamp', 'N/A')}")


def main():
    """Main dashboard loop."""
    print("\n🚀 Agent Metrics Dashboard Starting...")
    print(f"   API: {API_BASE}")
    
    try:
        # Fetch and display metrics once
        print("\n⏳ Fetching metrics...")
        metrics = get_metrics()
        flows_data = get_recent_flows(10)
        
        if not metrics:
            print("❌ Could not connect to API. Is it running?")
            return
        
        print_summary(metrics)
        print_intent_distribution(metrics)
        print_agent_stats(metrics)
        print_rag_stats(metrics)
        
        if flows_data:
            print_recent_flows(flows_data)
        
        print(f"\n\n✅ Dashboard updated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n📊 Metrics endpoints:")
        print(f"   - Summary:    GET {API_BASE}/metrics")
        print(f"   - Flows:      GET {API_BASE}/metrics/flows?limit=20")
        print(f"   - Health:     GET {API_BASE}/health")
        
    except Exception as e:
        print(f"❌ Dashboard error: {e}")


if __name__ == "__main__":
    main()
