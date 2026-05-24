"""
Regression evaluation runner with gating.

Runs the live pipeline on a typed evaluation dataset,
computes per-sample RAGAS metrics, compares with baseline,
and enforces soft/hard thresholds.

Features:
  - Per-sample evaluation results (not just aggregate)
  - Diff view: which queries regressed and why
  - Cached pipeline outputs to avoid recomputation
  - Soft (warn) + hard (block) thresholds
  - Minimum sample size validation before gating
  - Batch processing for evaluation dataset

Usage:
    python -m app.evaluation.regression
    python -m app.evaluation.regression --save-baseline
    python -m app.evaluation.regression --subset factual
"""
import json
import sys
import time
import hashlib
from pathlib import Path
from datetime import datetime, timezone

from app.config import (
    REGRESSION_THRESHOLDS_HARD,
    REGRESSION_THRESHOLDS_SOFT,
    REGRESSION_MIN_SAMPLES,
    EVAL_DATASET_PATH,
    BASELINE_PATH,
    LOG_DIR,
    REGRESSION_HISTORY_DIR,
    CONFIG_HASH,
)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: str | Path | None = None) -> list[dict]:
    """Load evaluation dataset from JSON file."""
    path = Path(path) if path else EVAL_DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("samples", [])


def filter_dataset(samples: list[dict], subset: str | None = None) -> list[dict]:
    """Filter samples by type (factual, multi_hop, out_of_scope, adversarial)."""
    if not subset:
        return samples
    return [s for s in samples if s.get("type") == subset]


# ---------------------------------------------------------------------------
# Pipeline runner (with output caching)
# ---------------------------------------------------------------------------

def _cache_path() -> Path:
    """Path for cached regression outputs."""
    p = LOG_DIR / "regression_cache"
    p.mkdir(exist_ok=True)
    return p


def _cache_key(question: str) -> str:
    """Deterministic cache key for a question."""
    return hashlib.sha256(question.encode()).hexdigest()[:16]


def run_pipeline_batch(
    samples: list[dict],
    use_cache: bool = True,
) -> list[dict]:
    """
    Run each sample through the live pipeline.
    Caches outputs to avoid recomputation on reruns.

    Returns list of dicts with: question, answer, contexts, sources, latency, etc.
    """
    # Late import to avoid circular dependency and keep decoupled
    from app.core.pipeline import run_query

    results = []
    cache_dir = _cache_path()

    for i, sample in enumerate(samples):
        question = sample["question"]
        key = _cache_key(question)
        cache_file = cache_dir / f"{key}.json"

        # Check cache
        if use_cache and cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            print(f"  [{i+1}/{len(samples)}] CACHE HIT: {question[:60]}...")
            results.append(cached)
            continue

        # Run live pipeline
        print(f"  [{i+1}/{len(samples)}] Running: {question[:60]}...")
        t = time.perf_counter()

        try:
            response = run_query(question)
            latency = (time.perf_counter() - t) * 1000

            # Extract contexts from sources
            contexts = [s.text for s in response.sources] if response.sources else []
            sources = [s.source for s in response.sources] if response.sources else []

            result = {
                "id": sample.get("id", i+1),
                "type": sample.get("type", "factual"),
                "question": question,
                "ground_truth": sample.get("ground_truth", ""),
                "expected_sources": sample.get("expected_sources", []),
                "answer": response.answer,
                "retrieved_contexts": contexts,
                "retrieved_sources": sources,
                "confidence_score": response.confidence_score,
                "confidence_label": response.confidence_label,
                "failure_class": response.debug.failure_class,
                "latency_ms": latency,
            }
        except Exception as e:
            result = {
                "id": sample.get("id", i+1),
                "type": sample.get("type", "factual"),
                "question": question,
                "ground_truth": sample.get("ground_truth", ""),
                "expected_sources": sample.get("expected_sources", []),
                "answer": f"ERROR: {str(e)}",
                "retrieved_contexts": [],
                "retrieved_sources": [],
                "confidence_score": 0.0,
                "confidence_label": "low",
                "failure_class": "pipeline_error",
                "latency_ms": 0.0,
            }

        # Cache result
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

def evaluate_results(results: list[dict]) -> dict:
    """
    Run RAGAS evaluation on pipeline results.
    Returns aggregate + per-sample metrics.
    """
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas import EvaluationDataset, SingleTurnSample
    except ImportError:
        print("WARNING: RAGAS not available. Returning empty metrics.")
        return {"aggregate": {}, "per_sample": [], "error": "ragas not installed"}

    samples = []
    for r in results:
        # Skip error results
        if r.get("failure_class") == "pipeline_error":
            continue

        samples.append(SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r.get("retrieved_contexts", []) or ["No context retrieved."],
            reference=r.get("ground_truth", ""),
        ))

    if not samples:
        return {"aggregate": {}, "per_sample": [], "error": "no valid samples"}

    dataset = EvaluationDataset(samples=samples)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    ragas_results = ragas_evaluate(dataset=dataset, metrics=metrics)

    # Aggregate
    aggregate = {}
    for name in metric_names:
        aggregate[name] = round(ragas_results.get(name, 0), 4)

    # Per-sample
    per_sample = []
    df = ragas_results.to_pandas()
    for idx, row in df.iterrows():
        entry = {"index": idx, "question": row.get("user_input", "")}
        for name in metric_names:
            entry[name] = round(row.get(name, 0), 4) if row.get(name) is not None else None
        per_sample.append(entry)

    return {"aggregate": aggregate, "per_sample": per_sample}


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def load_baseline(path: str | Path | None = None) -> dict | None:
    """Load saved baseline results."""
    path = Path(path) if path else BASELINE_PATH
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_baseline(eval_result: dict, path: str | Path | None = None):
    """Save current results as new baseline."""
    path = Path(path) if path else BASELINE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)
    print(f"Baseline saved to {path}")


def save_to_history(full_report: dict):
    """Persist regression run to timestamped history file."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = REGRESSION_HISTORY_DIR / f"regression_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    print(f"   History saved to {path}")


def load_history(limit: int = 10) -> list[dict]:
    """Load most recent regression history entries."""
    files = sorted(REGRESSION_HISTORY_DIR.glob("regression_*.json"), reverse=True)
    history = []
    for f in files[:limit]:
        with open(f, "r", encoding="utf-8") as fh:
            history.append(json.load(fh))
    return history


def compute_trend(history: list[dict]) -> dict:
    """
    Compute trend summary from regression history.
    Returns: {status, confidence_delta, failure_delta, per_metric}
    """
    if len(history) < 2:
        return {"status": "insufficient_data", "runs": len(history)}

    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    latest = history[0].get("evaluation", {}).get("aggregate", {})
    previous = history[1].get("evaluation", {}).get("aggregate", {})

    latest_summary = history[0].get("pipeline_results_summary", {})
    prev_summary = history[1].get("pipeline_results_summary", {})

    trends = {}
    for m in metric_names:
        cur = latest.get(m, 0)
        prev = previous.get(m, 0)
        delta = cur - prev
        if delta > 0.03:
            trends[m] = "improved"
        elif delta < -0.03:
            trends[m] = "degraded"
        else:
            trends[m] = "stable"

    # Overall
    degraded_count = sum(1 for t in trends.values() if t == "degraded")
    improved_count = sum(1 for t in trends.values() if t == "improved")
    if degraded_count >= 2:
        overall = "degrading"
    elif improved_count >= 2:
        overall = "improving"
    else:
        overall = "stable"

    # Compute confidence and failure deltas from pipeline results
    latest_ds = history[0].get("dataset_size", 1)
    prev_ds = history[1].get("dataset_size", 1)
    latest_fail = latest_summary.get("failure_count", 0)
    prev_fail = prev_summary.get("failure_count", 0)
    failure_delta = round((latest_fail / max(latest_ds, 1)) - (prev_fail / max(prev_ds, 1)), 4)

    # Avg confidence delta from evaluation metrics (use faithfulness as proxy)
    confidence_delta = round(latest.get("faithfulness", 0) - previous.get("faithfulness", 0), 4)

    return {
        "status": overall,
        "runs": len(history),
        "confidence_delta": confidence_delta,
        "failure_delta": failure_delta,
        "per_metric": trends,
        "latest_vs_previous": {
            m: {"current": latest.get(m, 0), "previous": previous.get(m, 0)}
            for m in metric_names
        },
    }


def compare_with_baseline(current: dict, baseline: dict) -> dict:
    """
    Compute metric deltas between current and baseline.
    Returns per-metric delta and per-sample regression diff.
    """
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    cur_agg = current.get("aggregate", {})
    base_agg = baseline.get("aggregate", {})

    deltas = {}
    for name in metric_names:
        cur_val = cur_agg.get(name, 0)
        base_val = base_agg.get(name, 0)
        delta = cur_val - base_val
        deltas[name] = {
            "current": cur_val,
            "baseline": base_val,
            "delta": round(delta, 4),
            "improved": delta > 0,
        }

    # Per-sample diff: find which queries regressed
    regressed_queries = []
    cur_samples = {s["question"]: s for s in current.get("per_sample", [])}
    base_samples = {s["question"]: s for s in baseline.get("per_sample", [])}

    for q, cur_s in cur_samples.items():
        if q in base_samples:
            base_s = base_samples[q]
            for name in metric_names:
                cur_v = cur_s.get(name)
                base_v = base_s.get(name)
                if cur_v is not None and base_v is not None and cur_v < base_v - 0.05:
                    regressed_queries.append({
                        "question": q[:80],
                        "metric": name,
                        "current": cur_v,
                        "baseline": base_v,
                        "delta": round(cur_v - base_v, 4),
                    })

    return {"deltas": deltas, "regressed_queries": regressed_queries}


# ---------------------------------------------------------------------------
# Regression gating
# ---------------------------------------------------------------------------

def gate_regression(
    eval_result: dict,
    comparison: dict | None = None,
) -> dict:
    """
    Enforce soft + hard thresholds on evaluation results.
    Returns structured report with PASS/FAIL/WARN status.

    - hard_fail: blocks deployment
    - soft_warn: logs warning but allows
    - Validates minimum sample size before gating
    """
    aggregate = eval_result.get("aggregate", {})
    per_sample = eval_result.get("per_sample", [])

    # Minimum sample size check
    if len(per_sample) < REGRESSION_MIN_SAMPLES:
        return {
            "status": "SKIP",
            "reason": f"Only {len(per_sample)} samples evaluated (minimum: {REGRESSION_MIN_SAMPLES})",
            "hard_failures": [],
            "soft_warnings": [],
            "metrics": aggregate,
        }

    hard_failures = []
    soft_warnings = []

    for metric, hard_threshold in REGRESSION_THRESHOLDS_HARD.items():
        value = aggregate.get(metric, 0)
        soft_threshold = REGRESSION_THRESHOLDS_SOFT.get(metric, hard_threshold)

        if value < hard_threshold:
            hard_failures.append({
                "metric": metric,
                "value": value,
                "hard_threshold": hard_threshold,
                "deficit": round(hard_threshold - value, 4),
            })
        elif value < soft_threshold:
            soft_warnings.append({
                "metric": metric,
                "value": value,
                "soft_threshold": soft_threshold,
                "gap": round(soft_threshold - value, 4),
            })

    # Status
    if hard_failures:
        status = "FAIL"
    elif soft_warnings:
        status = "WARN"
    else:
        status = "PASS"

    report = {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(per_sample),
        "metrics": aggregate,
        "hard_failures": hard_failures,
        "soft_warnings": soft_warnings,
    }

    if comparison:
        report["regressed_queries"] = comparison.get("regressed_queries", [])
        report["metric_deltas"] = comparison.get("deltas", {})

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_regression(
    subset: str | None = None,
    save_as_baseline: bool = False,
    use_cache: bool = True,
):
    """Full regression run: load → execute → evaluate → compare → gate → report."""
    print("=" * 70)
    print("  RAG Regression Evaluation")
    print("=" * 70)

    # Load dataset
    print(f"\n📂 Loading dataset from {EVAL_DATASET_PATH}...")
    samples = load_dataset()
    samples = filter_dataset(samples, subset)
    print(f"   {len(samples)} samples loaded" + (f" (subset: {subset})" if subset else ""))

    # Type distribution
    types = {}
    for s in samples:
        t = s.get("type", "unknown")
        types[t] = types.get(t, 0) + 1
    print(f"   Distribution: {types}")

    # Run pipeline
    print(f"\n🔄 Running pipeline on {len(samples)} samples...")
    results = run_pipeline_batch(samples, use_cache=use_cache)

    # Quick stats
    successes = sum(1 for r in results if r.get("failure_class") == "success")
    failures = len(results) - successes
    avg_lat = sum(r.get("latency_ms", 0) for r in results) / max(len(results), 1)
    print(f"   ✓ {successes} success, ✗ {failures} failures, avg latency: {avg_lat:.0f}ms")

    # Evaluate with RAGAS
    print("\n📊 Running RAGAS evaluation...")
    eval_result = evaluate_results(results)

    if eval_result.get("error"):
        print(f"   ⚠ {eval_result['error']}")
    else:
        print("   Aggregate metrics:")
        for m, v in eval_result.get("aggregate", {}).items():
            print(f"     {m:<25} {v:.4f}")

    # Compare with baseline
    comparison = None
    baseline = load_baseline()
    if baseline:
        print("\n📈 Comparing with baseline...")
        comparison = compare_with_baseline(eval_result, baseline)
        for m, d in comparison.get("deltas", {}).items():
            delta_str = f"+{d['delta']:.4f}" if d['delta'] >= 0 else f"{d['delta']:.4f}"
            icon = "✅" if d['improved'] else "❌"
            print(f"     {m:<25} {d['baseline']:.4f} → {d['current']:.4f}  {delta_str} {icon}")

        if comparison.get("regressed_queries"):
            print(f"\n   ⚠ {len(comparison['regressed_queries'])} queries regressed:")
            for rq in comparison["regressed_queries"][:5]:
                print(f"     • {rq['question']} [{rq['metric']}: {rq['baseline']:.3f}→{rq['current']:.3f}]")
    else:
        print("\n📈 No baseline found. Run with --save-baseline to create one.")

    # Gating
    print("\n🚦 Regression gating...")
    gate_report = gate_regression(eval_result, comparison)
    status = gate_report["status"]
    status_icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "SKIP": "⏭️"}[status]
    print(f"   {status_icon} Status: {status}")

    if gate_report.get("hard_failures"):
        print("   Hard failures (BLOCKING):")
        for f in gate_report["hard_failures"]:
            print(f"     • {f['metric']}: {f['value']:.4f} < {f['hard_threshold']:.4f} (deficit: {f['deficit']:.4f})")

    if gate_report.get("soft_warnings"):
        print("   Soft warnings:")
        for w in gate_report["soft_warnings"]:
            print(f"     • {w['metric']}: {w['value']:.4f} < {w['soft_threshold']:.4f}")

    # Save baseline if requested
    if save_as_baseline:
        save_baseline(eval_result)

    # Save full report
    report_path = LOG_DIR / "regression_report.json"
    full_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": CONFIG_HASH,
        "dataset_size": len(samples),
        "subset": subset,
        "pipeline_results_summary": {
            "success_count": successes,
            "failure_count": failures,
            "avg_latency_ms": round(avg_lat, 2),
        },
        "evaluation": eval_result,
        "comparison": comparison,
        "gate": gate_report,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    print(f"\n📄 Full report saved to {report_path}")

    # Persist to history
    save_to_history(full_report)

    # Trend summary
    history = load_history()
    trend = compute_trend(history)
    print(f"\n📊 Trend ({trend.get('runs', 0)} runs): {trend.get('status', 'unknown')}")
    if trend.get('per_metric'):
        for m, t in trend['per_metric'].items():
            icon = {'improved': '📈', 'degraded': '📉', 'stable': '➡️'}.get(t, '❓')
            print(f"     {icon} {m}: {t}")
    full_report["trend"] = trend

    print("\n" + "=" * 70)
    return full_report


if __name__ == "__main__":
    args = sys.argv[1:]
    save = "--save-baseline" in args
    subset_arg = None
    for a in args:
        if a.startswith("--subset"):
            idx = args.index(a)
            if idx + 1 < len(args):
                subset_arg = args[idx + 1]

    run_regression(subset=subset_arg, save_as_baseline=save)
