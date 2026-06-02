"""
RAG System — Latency Stress Test
=================================
Sends 20 diverse queries to the FastAPI /query endpoint and measures
response times in milliseconds. Reports min, max, avg, median, p95, p99.

Usage:
    # Against local server (default: http://127.0.0.1:8000)
    python test_latency.py

    # Against Hugging Face Spaces
    python test_latency.py --base-url https://saniyamihani-rag-system.hf.space

    # Custom number of queries
    python test_latency.py --num-queries 10
"""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Fix Windows console encoding for emoji/unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import requests

# ---------------------------------------------------------------------------
# Test queries — diverse complexity to stress different pipeline stages
# ---------------------------------------------------------------------------
TEST_QUERIES = [
    # Simple factual
    "What is the main topic of the document?",
    "Summarize the key points.",
    "What are the prerequisites mentioned?",
    # Medium complexity — multi-hop reasoning
    "How does the system handle errors and edge cases?",
    "What is the relationship between the components described?",
    "Compare the advantages and disadvantages mentioned in the document.",
    # Long queries — tests tokenization and embedding
    "Can you explain in detail the process described in the document, including all the steps involved and any specific requirements or conditions that need to be met?",
    "What are all the technical specifications, configurations, and parameters mentioned throughout the entire document?",
    # Short / vague queries
    "Why?",
    "Explain.",
    # Domain-specific
    "What algorithms or methods are used?",
    "What data formats are supported?",
    "How is performance measured?",
    "What security considerations are mentioned?",
    # Analytical
    "What are the limitations of the approach described?",
    "How does this compare to alternative approaches?",
    "What future improvements are suggested?",
    # Retrieval-heavy
    "List all the tools, libraries, or frameworks mentioned.",
    "What are the exact configuration values and their defaults?",
    # Edge-ish but valid
    "What is not covered in this document?",
]


def run_single_query(base_url: str, query: str, timeout: int = 120) -> dict:
    """Send a single query and return timing + status info."""
    url = f"{base_url.rstrip('/')}/query"
    payload = {"query": query}

    start = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if resp.status_code == 200:
            body = resp.json()
            return {
                "query": query,
                "status": "success",
                "status_code": resp.status_code,
                "latency_ms": round(elapsed_ms, 2),
                "confidence_score": body.get("confidence_score", None),
                "confidence_label": body.get("confidence_label", None),
                "answer_length": len(body.get("answer", "")),
                "num_sources": len(body.get("sources", [])),
                "pipeline_latency": body.get("debug", {}).get("latency_ms", {}),
                "failure_class": body.get("debug", {}).get("failure_class", ""),
            }
        else:
            return {
                "query": query,
                "status": "error",
                "status_code": resp.status_code,
                "latency_ms": round(elapsed_ms, 2),
                "error": resp.text[:200],
            }
    except requests.exceptions.Timeout:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "query": query,
            "status": "timeout",
            "status_code": None,
            "latency_ms": round(elapsed_ms, 2),
            "error": f"Request timed out after {timeout}s",
        }
    except requests.exceptions.ConnectionError as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "query": query,
            "status": "connection_error",
            "status_code": None,
            "latency_ms": round(elapsed_ms, 2),
            "error": str(e)[:200],
        }


def check_health(base_url: str) -> dict | None:
    """Hit /health to verify server is reachable."""
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/health", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def compute_stats(latencies: list[float]) -> dict:
    """Compute latency statistics."""
    if not latencies:
        return {}
    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    return {
        "count": n,
        "min_ms": round(min(sorted_lat), 2),
        "max_ms": round(max(sorted_lat), 2),
        "avg_ms": round(statistics.mean(sorted_lat), 2),
        "median_ms": round(statistics.median(sorted_lat), 2),
        "stdev_ms": round(statistics.stdev(sorted_lat), 2) if n > 1 else 0,
        "p95_ms": round(sorted_lat[int(n * 0.95)], 2) if n >= 5 else round(sorted_lat[-1], 2),
        "p99_ms": round(sorted_lat[int(n * 0.99)], 2) if n >= 10 else round(sorted_lat[-1], 2),
    }


def format_results_markdown(results: list[dict], stats: dict, base_url: str) -> str:
    """Format results into a shareable markdown report."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]

    md = []
    md.append("# 🔬 RAG System Stress Test Results\n")
    md.append(f"**Date:** {timestamp}  ")
    md.append(f"**Endpoint:** `{base_url}`  ")
    md.append(f"**Total Queries:** {len(results)}  ")
    md.append(f"**Successful:** {len(successful)} | **Failed:** {len(failed)}\n")

    md.append("---\n")
    md.append("## 1️⃣ Latency Test\n")
    md.append("### Summary Statistics\n")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| Total Queries | {stats.get('count', 0)} |")
    md.append(f"| ⬇️ Min Latency | {stats.get('min_ms', 'N/A')} ms |")
    md.append(f"| ⬆️ Max Latency | {stats.get('max_ms', 'N/A')} ms |")
    md.append(f"| 📊 Avg Latency | {stats.get('avg_ms', 'N/A')} ms |")
    md.append(f"| 📐 Median Latency | {stats.get('median_ms', 'N/A')} ms |")
    md.append(f"| 📏 Std Dev | {stats.get('stdev_ms', 'N/A')} ms |")
    md.append(f"| 🎯 P95 Latency | {stats.get('p95_ms', 'N/A')} ms |")
    md.append(f"| 🏁 P99 Latency | {stats.get('p99_ms', 'N/A')} ms |")
    md.append("")

    # Per-query breakdown table
    md.append("### Per-Query Breakdown\n")
    md.append("| # | Query (truncated) | Latency (ms) | Status | Confidence |")
    md.append("|---|-------------------|-------------|--------|------------|")
    for i, r in enumerate(results, 1):
        q = r["query"][:50] + ("…" if len(r["query"]) > 50 else "")
        lat = f"{r['latency_ms']:.0f}"
        status_icon = "✅" if r["status"] == "success" else "❌"
        conf = r.get("confidence_label", "N/A") or "N/A"
        conf_score = r.get("confidence_score")
        conf_str = f"{conf} ({conf_score:.2f})" if conf_score is not None else conf
        md.append(f"| {i} | {q} | {lat} | {status_icon} {r['status']} | {conf_str} |")
    md.append("")

    # Pipeline stage breakdown (if available)
    pipeline_latencies = [r.get("pipeline_latency", {}) for r in successful if r.get("pipeline_latency")]
    if pipeline_latencies:
        md.append("### Pipeline Stage Breakdown (avg across successful queries)\n")
        all_stages = set()
        for pl in pipeline_latencies:
            all_stages.update(pl.keys())

        md.append("| Stage | Avg (ms) | Min (ms) | Max (ms) |")
        md.append("|-------|----------|----------|----------|")
        for stage in sorted(all_stages):
            values = [pl.get(stage, 0) for pl in pipeline_latencies if stage in pl]
            if values:
                md.append(
                    f"| {stage} | {statistics.mean(values):.2f} | {min(values):.2f} | {max(values):.2f} |"
                )
        md.append("")

    # Failures section
    if failed:
        md.append("### ❌ Failed Queries\n")
        for r in failed:
            md.append(f"- **Query:** \"{r['query'][:80]}\"")
            md.append(f"  - **Status:** {r['status']} (HTTP {r.get('status_code', 'N/A')})")
            md.append(f"  - **Error:** {r.get('error', 'Unknown')}")
        md.append("")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="RAG System Latency Test")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the FastAPI server (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=20,
        help="Number of queries to run (default: 20, max: 20)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output markdown file (default: ../stress_test_results.md)",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else Path(__file__).parent.parent / "stress_test_results.md"
    num_queries = min(args.num_queries, len(TEST_QUERIES))
    queries = TEST_QUERIES[:num_queries]

    print("=" * 70)
    print("  🔬 RAG SYSTEM — LATENCY STRESS TEST")
    print("=" * 70)
    print(f"  Endpoint:    {args.base_url}")
    print(f"  Queries:     {num_queries}")
    print(f"  Timeout:     {args.timeout}s per request")
    print(f"  Output:      {output_path}")
    print("=" * 70)

    # Health check
    print("\n🏥 Checking server health...")
    health = check_health(args.base_url)
    if health:
        print(f"   ✅ Server healthy — {health.get('document_count', '?')} documents indexed")
        print(f"   📦 Collection: {health.get('collection_name', 'N/A')}")
        models = health.get("models_loaded", {})
        print(f"   🧠 Models: embedder={models.get('embedder')}, reranker={models.get('reranker')}, "
              f"llm={models.get('llm')} ({models.get('llm_provider', 'N/A')})")
    else:
        print("   ⚠️  Could not reach /health — server may be starting up or unreachable.")
        print("   Continuing anyway...\n")

    # Run queries sequentially
    results = []
    print(f"\n🚀 Running {num_queries} queries sequentially...\n")
    for i, query in enumerate(queries, 1):
        display_q = query[:60] + ("…" if len(query) > 60 else "")
        print(f"  [{i:2d}/{num_queries}] {display_q}")

        result = run_single_query(args.base_url, query, timeout=args.timeout)
        results.append(result)

        status_icon = "✅" if result["status"] == "success" else "❌"
        print(f"          {status_icon} {result['latency_ms']:.0f}ms — {result['status']}")

    # Compute stats
    success_latencies = [r["latency_ms"] for r in results if r["status"] == "success"]
    all_latencies = [r["latency_ms"] for r in results]

    stats = compute_stats(success_latencies)

    # Print summary
    print("\n" + "=" * 70)
    print("  📊 LATENCY TEST RESULTS")
    print("=" * 70)

    if stats:
        print(f"  Successful:  {stats['count']}/{len(results)} queries")
        print(f"  ⬇️  Min:      {stats['min_ms']} ms")
        print(f"  ⬆️  Max:      {stats['max_ms']} ms")
        print(f"  📊 Avg:      {stats['avg_ms']} ms")
        print(f"  📐 Median:   {stats['median_ms']} ms")
        print(f"  📏 Std Dev:  {stats['stdev_ms']} ms")
        print(f"  🎯 P95:      {stats['p95_ms']} ms")
        print(f"  🏁 P99:      {stats['p99_ms']} ms")
    else:
        print("  ❌ No successful queries — check server connection.")

    print("=" * 70)

    # Save markdown report
    markdown = format_results_markdown(results, stats, args.base_url)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"\n📄 Results saved to: {output_path}")

    # Also save raw JSON for programmatic analysis
    json_path = output_path.with_suffix(".json")
    raw_data = {
        "test_type": "latency",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "num_queries": num_queries,
        "stats": stats,
        "results": results,
    }
    json_path.write_text(json.dumps(raw_data, indent=2), encoding="utf-8")
    print(f"📊 Raw data saved to:  {json_path}")


if __name__ == "__main__":
    main()
