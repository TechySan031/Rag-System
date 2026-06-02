"""
RAG System — Edge Case Test
==============================
Tests retrieval quality and system robustness with 10 edge case queries:
empty queries, very long queries, ambiguous questions, multi-language,
and queries with no relevant documents.

Usage:
    python test_edge_cases.py
    python test_edge_cases.py --base-url https://saniyamihani-rag-system.hf.space
"""

import argparse
import json
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
# 10 edge case queries designed to stress retrieval quality
# ---------------------------------------------------------------------------
EDGE_CASE_QUERIES = [
    # ── Category 1: Empty / minimal input ──
    {
        "id": 1,
        "category": "Empty Input",
        "description": "Completely empty string",
        "query": "",
        "expected_behavior": "Should return 422 validation error (min_length=1 on QueryRequest) or a graceful error. Must NOT crash.",
    },
    {
        "id": 2,
        "category": "Minimal Input",
        "description": "Single character query",
        "query": "?",
        "expected_behavior": "Should handle gracefully — low confidence or 'insufficient information' response.",
    },
    # ── Category 2: Extremely long queries ──
    {
        "id": 3,
        "category": "Long Query",
        "description": "Extremely long query (>1500 chars)",
        "query": (
            "I need you to provide a comprehensive, detailed, and thorough analysis of "
            "every single topic, concept, methodology, framework, tool, technology, "
            "algorithm, data structure, design pattern, best practice, anti-pattern, "
            "security consideration, performance optimization, scalability concern, "
            "deployment strategy, monitoring approach, testing methodology, and "
            "documentation standard that is mentioned, referenced, implied, or even "
            "tangentially related to the content of the documents in your knowledge base. "
            "Please include specific examples, code snippets if applicable, comparisons "
            "with alternative approaches, historical context for why certain decisions "
            "were made, and forward-looking predictions about how these topics might "
            "evolve in the future. Additionally, please cross-reference any concepts "
            "that appear in multiple documents and highlight any contradictions or "
            "inconsistencies you find between different sources. "
        ) * 3,  # Repeat 3x to make it very long
        "expected_behavior": "Should either truncate/handle the long input gracefully or return a 422 error (max_length=2000). Must NOT hang or crash.",
    },
    {
        "id": 4,
        "category": "Long Query",
        "description": "Query at exact max_length boundary (2000 chars)",
        "query": "What is the main topic? " * 80,  # ~2000 chars
        "expected_behavior": "Should process or reject based on validation. Boundary test for max_length=2000.",
    },
    # ── Category 3: Ambiguous queries ──
    {
        "id": 5,
        "category": "Ambiguous",
        "description": "Vague, ambiguous question",
        "query": "How does it work?",
        "expected_behavior": "Should attempt retrieval but may return low confidence. 'It' has no antecedent — tests whether system handles ambiguity.",
    },
    {
        "id": 6,
        "category": "Ambiguous",
        "description": "Multiple possible interpretations",
        "query": "What is the best approach?",
        "expected_behavior": "Ambiguous without context — should either ask for clarification via answer or return low confidence results.",
    },
    # ── Category 4: No relevant documents ──
    {
        "id": 7,
        "category": "No Relevant Docs",
        "description": "Highly specific obscure topic",
        "query": "What is the mating ritual of the Antarctic deep-sea anglerfish Ceratias holboelli?",
        "expected_behavior": "Should return 'I don't have enough information' with retrieval_miss failure class. Must NOT hallucinate.",
    },
    {
        "id": 8,
        "category": "No Relevant Docs",
        "description": "Question in a completely different domain",
        "query": "Derive the Schrödinger equation from first principles and explain its implications for quantum tunneling in semiconductor devices.",
        "expected_behavior": "Should return 'I don't have enough information' — unless physics docs are in the knowledge base.",
    },
    # ── Category 5: Special formatting ──
    {
        "id": 9,
        "category": "Special Formatting",
        "description": "Query with SQL injection attempt",
        "query": "'; DROP TABLE documents; -- What is the main topic?",
        "expected_behavior": "Should treat as a text query, not execute SQL. ChromaDB is not SQL-based, but tests input sanitization.",
    },
    {
        "id": 10,
        "category": "Special Formatting",
        "description": "Query with HTML/script injection",
        "query": "<script>alert('xss')</script> What information is available about security?",
        "expected_behavior": "Should strip/ignore HTML tags and process the text query normally. Must NOT reflect raw HTML.",
    },
]


def run_edge_case(base_url: str, case: dict, timeout: int = 120) -> dict:
    """Send a single edge case query and analyze the response."""
    url = f"{base_url.rstrip('/')}/query"
    start = time.perf_counter()

    try:
        resp = requests.post(url, json={"query": case["query"]}, timeout=timeout)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if resp.status_code == 200:
            body = resp.json()
            answer = body.get("answer", "")
            return {
                **case,
                "status": "success",
                "status_code": 200,
                "latency_ms": round(elapsed_ms, 2),
                "answer": answer,
                "confidence_score": body.get("confidence_score"),
                "confidence_label": body.get("confidence_label"),
                "num_sources": len(body.get("sources", [])),
                "failure_class": body.get("debug", {}).get("failure_class", ""),
                "failure_reason": body.get("debug", {}).get("failure_reason", ""),
                "verdict": _classify_edge_case(case, body),
            }
        elif resp.status_code == 422:
            return {
                **case,
                "status": "validation_error",
                "status_code": 422,
                "latency_ms": round(elapsed_ms, 2),
                "answer": f"Validation Error: {resp.text[:300]}",
                "verdict": "✅ HANDLED" if case["category"] in ("Empty Input", "Long Query") else "⚠️ REVIEW",
            }
        else:
            return {
                **case,
                "status": "error",
                "status_code": resp.status_code,
                "latency_ms": round(elapsed_ms, 2),
                "answer": f"HTTP {resp.status_code}: {resp.text[:200]}",
                "verdict": "❌ ERROR",
            }
    except requests.exceptions.ConnectionError:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            **case,
            "status": "connection_error",
            "status_code": None,
            "latency_ms": round(elapsed_ms, 2),
            "answer": "Connection refused",
            "verdict": "❌ UNREACHABLE",
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            **case,
            "status": "exception",
            "status_code": None,
            "latency_ms": round(elapsed_ms, 2),
            "answer": str(e)[:200],
            "verdict": "❌ CRASH",
        }


def _classify_edge_case(case: dict, body: dict) -> str:
    """Heuristic verdict for edge case handling."""
    answer = body.get("answer", "").lower()
    confidence = body.get("confidence_label", "")
    failure_class = body.get("debug", {}).get("failure_class", "")

    cat = case["category"]

    if cat == "Empty Input":
        # Should have been caught by validation
        return "⚠️ REVIEW — empty input was accepted instead of rejected"

    if cat == "Minimal Input":
        if confidence in ("low",) or "don't have" in answer or failure_class == "retrieval_miss":
            return "✅ HANDLED — low confidence or appropriate fallback"
        return "⚠️ REVIEW — check if response is grounded"

    if cat == "Long Query":
        # If it got here with 200, it processed it
        return "✅ HANDLED — long query processed without crash"

    if cat == "Ambiguous":
        if confidence in ("low", "medium"):
            return "✅ HANDLED — appropriate confidence for ambiguous query"
        return "⚠️ REVIEW — high confidence on ambiguous query is suspicious"

    if cat == "No Relevant Docs":
        no_info_phrases = [
            "don't have enough", "no relevant", "cannot answer",
            "not found", "no information", "unable to",
            "insufficient", "don't have information",
        ]
        if any(p in answer for p in no_info_phrases) or failure_class == "retrieval_miss":
            return "✅ HANDLED — correctly identified no relevant documents"
        if confidence == "low":
            return "✅ HANDLED — low confidence indicates awareness of poor retrieval"
        return "❌ FAIL — generated confident answer with no relevant documents (hallucination risk)"

    if cat == "Special Formatting":
        if "<script>" in body.get("answer", "") or "alert(" in body.get("answer", ""):
            return "❌ FAIL — reflected raw HTML/script in response"
        if "drop table" in answer.lower():
            return "❌ FAIL — reflected SQL injection in response"
        return "✅ HANDLED — special characters processed safely"

    return "⚠️ REVIEW"


def format_edge_case_markdown(results: list[dict], base_url: str) -> str:
    """Format edge case test results as markdown."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    passed = sum(1 for r in results if "HANDLED" in r.get("verdict", "") or "PASS" in r.get("verdict", ""))
    failed = sum(1 for r in results if "FAIL" in r.get("verdict", "") or "CRASH" in r.get("verdict", ""))
    review = sum(1 for r in results if "REVIEW" in r.get("verdict", ""))

    md = []
    md.append("## 4️⃣ Edge Case Test\n")
    md.append(f"**Date:** {timestamp}  ")
    md.append(f"**Endpoint:** `{base_url}`  ")
    md.append(f"**Results:** ✅ {passed} Handled | ❌ {failed} Failed | ⚠️ {review} Review Needed\n")

    # Summary table
    md.append("### Results Summary\n")
    md.append("| # | Category | Description | Status | Verdict |")
    md.append("|---|----------|-------------|--------|---------|")
    for r in results:
        status_icon = {"success": "✅", "validation_error": "🛑", "error": "❌"}.get(r.get("status", ""), "❓")
        md.append(
            f"| {r['id']} | {r['category']} | {r['description']} | "
            f"{status_icon} {r.get('status', 'N/A')} | {r.get('verdict', 'N/A')} |"
        )
    md.append("")

    # Detailed results
    md.append("### Detailed Results\n")
    for r in results:
        md.append(f"#### {r['id']}. {r['description']} ({r['category']})\n")
        query_display = r["query"][:150] + ("…" if len(r["query"]) > 150 else "")
        if not query_display:
            query_display = "(empty string)"
        md.append(f"**Query:** `{query_display}`  ")
        md.append(f"**Expected:** {r['expected_behavior']}  ")
        answer = r.get("answer", "N/A")
        if len(answer) > 400:
            answer = answer[:400] + "… [truncated]"
        md.append(f"**Response:** {answer}  ")
        conf = r.get("confidence_label", "N/A")
        conf_score = r.get("confidence_score")
        if conf_score is not None:
            md.append(f"**Confidence:** {conf} ({conf_score:.2f})  ")
        md.append(f"**Failure Class:** {r.get('failure_class', 'N/A')}  ")
        md.append(f"**Verdict:** {r.get('verdict', 'N/A')}  ")
        md.append(f"**Latency:** {r.get('latency_ms', 'N/A')}ms\n")

    # Robustness score
    score = (passed / len(results) * 100) if results else 0
    md.append("### Robustness Scorecard\n")
    if score >= 90:
        md.append(f"🏆 **Score: {score:.0f}%** — Excellent edge case handling.\n")
    elif score >= 70:
        md.append(f"🟡 **Score: {score:.0f}%** — Good robustness with minor gaps.\n")
    elif score >= 50:
        md.append(f"🟠 **Score: {score:.0f}%** — Moderate robustness. Some edge cases not handled.\n")
    else:
        md.append(f"🔴 **Score: {score:.0f}%** — Significant robustness issues. Needs improvement.\n")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="RAG System Edge Case Test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else Path(__file__).parent.parent / "stress_test_results.md"

    print("=" * 70)
    print("  🧪 RAG SYSTEM — EDGE CASE TEST")
    print("=" * 70)
    print(f"  Endpoint:     {args.base_url}")
    print(f"  Test Cases:   {len(EDGE_CASE_QUERIES)}")
    print("=" * 70)

    results = []
    for case in EDGE_CASE_QUERIES:
        display_q = case["query"][:60] if case["query"] else "(empty)"
        display_q += "…" if len(case["query"]) > 60 else ""
        print(f"\n  [{case['id']:2d}/10] {case['category']} — {case['description']}")
        print(f"         Query: {display_q}")

        result = run_edge_case(args.base_url, case, args.timeout)
        results.append(result)

        print(f"         {result.get('verdict', 'N/A')}")
        if result.get("status") == "validation_error":
            print(f"         🛑 Rejected by validation (HTTP 422)")
        elif result.get("answer"):
            preview = result["answer"][:80]
            print(f"         Response: {preview}{'…' if len(result.get('answer', '')) > 80 else ''}")

    # Summary
    passed = sum(1 for r in results if "HANDLED" in r.get("verdict", "") or "PASS" in r.get("verdict", ""))
    failed = sum(1 for r in results if "FAIL" in r.get("verdict", "") or "CRASH" in r.get("verdict", ""))
    review = sum(1 for r in results if "REVIEW" in r.get("verdict", ""))

    print("\n" + "=" * 70)
    print("  🧪 EDGE CASE TEST SUMMARY")
    print("=" * 70)
    print(f"  ✅ Handled: {passed}/10")
    print(f"  ❌ Failed:  {failed}/10")
    print(f"  ⚠️ Review:  {review}/10")
    score = (passed / len(results) * 100) if results else 0
    print(f"  🏆 Score:   {score:.0f}%")
    print("=" * 70)

    # Append to markdown
    markdown = format_edge_case_markdown(results, args.base_url)
    if output_path.exists():
        existing = output_path.read_text(encoding="utf-8")
        output_path.write_text(existing + "\n---\n\n" + markdown, encoding="utf-8")
        print(f"\n📄 Results appended to: {output_path}")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
        print(f"\n📄 Results saved to: {output_path}")

    # Save raw JSON
    json_path = output_path.with_name("stress_test_edge_cases.json")
    raw_data = {
        "test_type": "edge_cases",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "results": [{k: v for k, v in r.items()} for r in results],
        "summary": {"passed": passed, "failed": failed, "review": review, "score_pct": score},
    }
    json_path.write_text(json.dumps(raw_data, indent=2, default=str), encoding="utf-8")
    print(f"📊 Raw data saved to:  {json_path}")


if __name__ == "__main__":
    main()
