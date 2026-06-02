"""
RAG System — Concurrent Load Test
===================================
Tests how many simultaneous requests the system can handle before
degradation or failure. Uses asyncio + aiohttp for true concurrency.

Concurrency levels tested: 1, 5, 10, 20

Usage:
    # Against local server
    python test_concurrency.py

    # Against Hugging Face Spaces
    python test_concurrency.py --base-url https://saniyamihani-rag-system.hf.space

    # Custom concurrency levels
    python test_concurrency.py --levels 1 5 10 25 50
"""

import argparse
import asyncio
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

try:
    import aiohttp
except ImportError:
    aiohttp = None

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Queries used for load testing — varied complexity
# ---------------------------------------------------------------------------
LOAD_QUERIES = [
    "What is the main topic of the document?",
    "Summarize the key points.",
    "How does the system handle errors?",
    "What algorithms or methods are used?",
    "What are the prerequisites mentioned?",
    "List all tools and frameworks mentioned.",
    "What security considerations are discussed?",
    "Explain the architecture described.",
    "What data formats are supported?",
    "What are the limitations?",
    "Compare the approaches mentioned.",
    "What configuration options are available?",
    "How is performance measured?",
    "What are the design decisions?",
    "What future improvements are suggested?",
    "What is the deployment process?",
    "How does caching work?",
    "What monitoring is available?",
    "What are the system requirements?",
    "How does authentication work?",
]


def send_query_sync(base_url: str, query: str, timeout: int = 120) -> dict:
    """Send a single query synchronously (for ThreadPoolExecutor fallback)."""
    url = f"{base_url.rstrip('/')}/query"
    start = time.perf_counter()
    try:
        resp = requests.post(url, json={"query": query}, timeout=timeout)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "query": query,
            "status": "success" if resp.status_code == 200 else "error",
            "status_code": resp.status_code,
            "latency_ms": round(elapsed_ms, 2),
            "error": resp.text[:200] if resp.status_code != 200 else None,
        }
    except requests.exceptions.Timeout:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "query": query,
            "status": "timeout",
            "status_code": None,
            "latency_ms": round(elapsed_ms, 2),
            "error": f"Timed out after {timeout}s",
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "query": query,
            "status": "connection_error",
            "status_code": None,
            "latency_ms": round(elapsed_ms, 2),
            "error": str(e)[:200],
        }


async def send_query_async(session: "aiohttp.ClientSession", base_url: str, query: str, timeout: int = 120) -> dict:
    """Send a single query using aiohttp."""
    url = f"{base_url.rstrip('/')}/query"
    start = time.perf_counter()
    try:
        async with session.post(url, json={"query": query}, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            elapsed_ms = (time.perf_counter() - start) * 1000
            body = await resp.text()
            return {
                "query": query,
                "status": "success" if resp.status == 200 else "error",
                "status_code": resp.status,
                "latency_ms": round(elapsed_ms, 2),
                "error": body[:200] if resp.status != 200 else None,
            }
    except asyncio.TimeoutError:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "query": query,
            "status": "timeout",
            "status_code": None,
            "latency_ms": round(elapsed_ms, 2),
            "error": f"Timed out after {timeout}s",
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "query": query,
            "status": "connection_error",
            "status_code": None,
            "latency_ms": round(elapsed_ms, 2),
            "error": str(e)[:200],
        }


async def run_concurrent_async(base_url: str, concurrency: int, timeout: int) -> dict:
    """Run N concurrent queries using asyncio + aiohttp."""
    queries = [LOAD_QUERIES[i % len(LOAD_QUERIES)] for i in range(concurrency)]

    start_wall = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        tasks = [send_query_async(session, base_url, q, timeout) for q in queries]
        results = await asyncio.gather(*tasks)
    wall_time_ms = (time.perf_counter() - start_wall) * 1000

    return _summarize_level(concurrency, list(results), wall_time_ms)


def run_concurrent_threads(base_url: str, concurrency: int, timeout: int) -> dict:
    """Run N concurrent queries using ThreadPoolExecutor (fallback if aiohttp unavailable)."""
    queries = [LOAD_QUERIES[i % len(LOAD_QUERIES)] for i in range(concurrency)]

    start_wall = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(send_query_sync, base_url, q, timeout): q for q in queries}
        for future in as_completed(futures):
            results.append(future.result())
    wall_time_ms = (time.perf_counter() - start_wall) * 1000

    return _summarize_level(concurrency, results, wall_time_ms)


def _summarize_level(concurrency: int, results: list[dict], wall_time_ms: float) -> dict:
    """Summarize results for a single concurrency level."""
    success = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]

    success_latencies = [r["latency_ms"] for r in success]
    all_latencies = [r["latency_ms"] for r in results]

    summary = {
        "concurrency": concurrency,
        "total_requests": len(results),
        "successful": len(success),
        "failed": len(failed),
        "success_rate": round(len(success) / len(results) * 100, 1) if results else 0,
        "wall_time_ms": round(wall_time_ms, 2),
        "throughput_rps": round(len(results) / (wall_time_ms / 1000), 2) if wall_time_ms > 0 else 0,
    }

    if success_latencies:
        summary.update({
            "min_ms": round(min(success_latencies), 2),
            "max_ms": round(max(success_latencies), 2),
            "avg_ms": round(statistics.mean(success_latencies), 2),
            "median_ms": round(statistics.median(success_latencies), 2),
            "stdev_ms": round(statistics.stdev(success_latencies), 2) if len(success_latencies) > 1 else 0,
        })

    if failed:
        summary["failures"] = [
            {"query": r["query"][:60], "status": r["status"], "error": r.get("error", "")[:100]}
            for r in failed
        ]

    return summary


def format_concurrency_markdown(levels: list[dict], base_url: str) -> str:
    """Format concurrency test results as markdown."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    md = []
    md.append("## 2️⃣ Concurrent Load Test\n")
    md.append(f"**Date:** {timestamp}  ")
    md.append(f"**Endpoint:** `{base_url}`\n")

    # Summary table
    md.append("### Results by Concurrency Level\n")
    md.append("| Concurrent Users | Success Rate | Avg Latency (ms) | Median (ms) | Max (ms) | Throughput (req/s) | Wall Time (ms) |")
    md.append("|-----------------|-------------|-----------------|------------|---------|-------------------|----------------|")
    for lvl in levels:
        md.append(
            f"| {lvl['concurrency']} | "
            f"{lvl['success_rate']}% ({lvl['successful']}/{lvl['total_requests']}) | "
            f"{lvl.get('avg_ms', 'N/A')} | "
            f"{lvl.get('median_ms', 'N/A')} | "
            f"{lvl.get('max_ms', 'N/A')} | "
            f"{lvl['throughput_rps']} | "
            f"{lvl['wall_time_ms']:.0f} |"
        )
    md.append("")

    # Degradation analysis
    md.append("### Degradation Analysis\n")
    baseline = levels[0] if levels else None
    if baseline and baseline.get("avg_ms"):
        for lvl in levels[1:]:
            if lvl.get("avg_ms") and baseline.get("avg_ms"):
                slowdown = lvl["avg_ms"] / baseline["avg_ms"]
                emoji = "🟢" if slowdown < 1.5 else ("🟡" if slowdown < 3 else "🔴")
                md.append(
                    f"- {emoji} **{lvl['concurrency']} users**: "
                    f"{slowdown:.1f}x slower than baseline (1 user). "
                    f"Avg: {lvl['avg_ms']}ms vs {baseline['avg_ms']}ms"
                )
        md.append("")

    # Failures detail
    any_failures = any(lvl.get("failures") for lvl in levels)
    if any_failures:
        md.append("### ❌ Failures\n")
        for lvl in levels:
            if lvl.get("failures"):
                md.append(f"**{lvl['concurrency']} concurrent users:**")
                for f in lvl["failures"]:
                    md.append(f"  - `{f['query']}…` → {f['status']}: {f['error']}")
                md.append("")

    # Verdict
    md.append("### Verdict\n")
    max_clean = 0
    for lvl in levels:
        if lvl["success_rate"] == 100:
            max_clean = lvl["concurrency"]

    if max_clean >= 20:
        md.append("✅ **System handles 20+ concurrent users with 100% success rate.**\n")
    elif max_clean >= 10:
        md.append("🟡 **System handles up to 10 concurrent users cleanly. Degradation observed beyond that.**\n")
    elif max_clean >= 5:
        md.append("🟠 **System handles up to 5 concurrent users. Significant degradation at higher loads.**\n")
    else:
        md.append("🔴 **System struggles under concurrent load. Consider scaling or optimizing.**\n")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="RAG System Concurrent Load Test")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the FastAPI server",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=[1, 5, 10, 20],
        help="Concurrency levels to test (default: 1 5 10 20)",
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
        help="Output markdown file (default: ../stress_test_results.md — appends)",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else Path(__file__).parent.parent / "stress_test_results.md"
    use_async = aiohttp is not None

    print("=" * 70)
    print("  🏋️ RAG SYSTEM — CONCURRENT LOAD TEST")
    print("=" * 70)
    print(f"  Endpoint:      {args.base_url}")
    print(f"  Concurrency:   {args.levels}")
    print(f"  Engine:        {'asyncio + aiohttp' if use_async else 'ThreadPoolExecutor (install aiohttp for better results)'}")
    print(f"  Timeout:       {args.timeout}s per request")
    print(f"  Output:        {output_path}")
    print("=" * 70)

    # Health check
    print("\n🏥 Checking server health...")
    try:
        resp = requests.get(f"{args.base_url.rstrip('/')}/health", timeout=10)
        if resp.status_code == 200:
            print(f"   ✅ Server healthy")
        else:
            print(f"   ⚠️  Health check returned {resp.status_code}")
    except Exception:
        print("   ⚠️  Could not reach /health")

    # Run each concurrency level
    all_levels = []
    for level in args.levels:
        print(f"\n🚀 Testing with {level} concurrent user(s)...")

        if use_async:
            result = asyncio.run(run_concurrent_async(args.base_url, level, args.timeout))
        else:
            result = run_concurrent_threads(args.base_url, level, args.timeout)

        all_levels.append(result)

        # Print summary for this level
        sr = result["success_rate"]
        emoji = "✅" if sr == 100 else ("🟡" if sr >= 80 else "🔴")
        print(f"   {emoji} {result['successful']}/{result['total_requests']} succeeded "
              f"({sr}%) — Avg: {result.get('avg_ms', 'N/A')}ms, "
              f"Throughput: {result['throughput_rps']} req/s, "
              f"Wall: {result['wall_time_ms']:.0f}ms")

        # Brief cooldown between levels to let the server recover
        if level != args.levels[-1]:
            print("   ⏳ Cooling down 5s before next level...")
            time.sleep(5)

    # Print final summary
    print("\n" + "=" * 70)
    print("  📊 LOAD TEST SUMMARY")
    print("=" * 70)
    print(f"  {'Level':>8} | {'Success':>10} | {'Avg (ms)':>10} | {'Median (ms)':>12} | {'Throughput':>12}")
    print(f"  {'─' * 8} | {'─' * 10} | {'─' * 10} | {'─' * 12} | {'─' * 12}")
    for lvl in all_levels:
        print(f"  {lvl['concurrency']:>8} | "
              f"{lvl['success_rate']:>8}%  | "
              f"{lvl.get('avg_ms', 'N/A'):>10} | "
              f"{lvl.get('median_ms', 'N/A'):>12} | "
              f"{lvl['throughput_rps']:>10} rps")
    print("=" * 70)

    # Append to markdown
    markdown = format_concurrency_markdown(all_levels, args.base_url)

    if output_path.exists():
        existing = output_path.read_text(encoding="utf-8")
        output_path.write_text(existing + "\n---\n\n" + markdown, encoding="utf-8")
        print(f"\n📄 Results appended to: {output_path}")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
        print(f"\n📄 Results saved to: {output_path}")

    # Save raw JSON
    json_path = output_path.with_name("stress_test_concurrency.json")
    raw_data = {
        "test_type": "concurrency",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "levels": all_levels,
    }
    json_path.write_text(json.dumps(raw_data, indent=2), encoding="utf-8")
    print(f"📊 Raw data saved to:  {json_path}")


if __name__ == "__main__":
    main()
