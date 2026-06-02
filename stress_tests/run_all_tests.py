"""
RAG System — Full Stress Test Suite Runner
============================================
Runs all 4 stress tests in sequence and generates a unified report.

Usage:
    python run_all_tests.py
    python run_all_tests.py --base-url https://saniyamihani-rag-system.hf.space
    python run_all_tests.py --skip-concurrency  # skip slow load test
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Fix Windows console encoding for emoji/unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def run_test(script: str, base_url: str, extra_args: list[str] | None = None):
    """Run a test script as a subprocess."""
    cmd = [sys.executable, script, "--base-url", base_url]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'━' * 70}")
    print(f"  Running: {Path(script).name}")
    print(f"{'━' * 70}\n")

    result = subprocess.run(cmd, cwd=str(Path(script).parent))
    return result.returncode


def generate_final_header(base_url: str, output_path: Path):
    """Prepend a final summary header to the results file."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if output_path.exists():
        existing = output_path.read_text(encoding="utf-8")
    else:
        existing = ""

    # Only add the header if it hasn't been added yet
    if existing.startswith("# 🔬 RAG System Stress Test Results"):
        return  # Header already present from latency test

    # Otherwise, the latency test already wrote the header
    # We just need to make sure the file is clean


def main():
    parser = argparse.ArgumentParser(description="Run full RAG stress test suite")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--skip-concurrency", action="store_true", help="Skip the slow concurrency test")
    parser.add_argument("--skip-injection", action="store_true")
    parser.add_argument("--skip-edge-cases", action="store_true")
    args = parser.parse_args()

    test_dir = Path(__file__).parent
    start_time = time.time()

    print("=" * 70)
    print("  🔬 RAG SYSTEM — FULL STRESS TEST SUITE")
    print("=" * 70)
    print(f"  Endpoint: {args.base_url}")
    print(f"  Tests:    Latency → Concurrency → Prompt Injection → Edge Cases")
    print("=" * 70)

    # 1. Latency test (always first — creates the results file)
    print("\n\n" + "🔵 " * 20)
    print("  PHASE 1: LATENCY TEST")
    print("🔵 " * 20)
    run_test(str(test_dir / "test_latency.py"), args.base_url)

    # 2. Concurrency test
    if not args.skip_concurrency:
        print("\n\n" + "🟣 " * 20)
        print("  PHASE 2: CONCURRENT LOAD TEST")
        print("🟣 " * 20)
        run_test(str(test_dir / "test_concurrency.py"), args.base_url)
    else:
        print("\n⏭️ Skipping concurrency test")

    # 3. Prompt injection test
    if not args.skip_injection:
        print("\n\n" + "🔴 " * 20)
        print("  PHASE 3: PROMPT INJECTION TEST")
        print("🔴 " * 20)
        run_test(str(test_dir / "test_prompt_injection.py"), args.base_url)
    else:
        print("\n⏭️ Skipping prompt injection test")

    # 4. Edge case test
    if not args.skip_edge_cases:
        print("\n\n" + "🟠 " * 20)
        print("  PHASE 4: EDGE CASE TEST")
        print("🟠 " * 20)
        run_test(str(test_dir / "test_edge_cases.py"), args.base_url)
    else:
        print("\n⏭️ Skipping edge case test")

    # Final summary
    elapsed = time.time() - start_time
    output_path = test_dir.parent / "stress_test_results.md"

    print("\n\n" + "=" * 70)
    print("  ✅ ALL TESTS COMPLETE")
    print("=" * 70)
    print(f"  Total time:   {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Results:      {output_path}")
    print(f"  Raw data:     {test_dir.parent / 'stress_test_results.json'}")
    print(f"                {test_dir.parent / 'stress_test_concurrency.json'}")
    print(f"                {test_dir.parent / 'stress_test_injection.json'}")
    print(f"                {test_dir.parent / 'stress_test_edge_cases.json'}")
    print("=" * 70)
    print("\n📋 Share stress_test_results.md on GitHub and LinkedIn!")


if __name__ == "__main__":
    main()
