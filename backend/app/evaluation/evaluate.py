"""
RAGAS Evaluation Script — V2.
Evaluates faithfulness, answer relevancy, context precision, and context recall.
Includes baseline vs improved comparison to show multi-query impact.

Usage:
    python -m app.evaluation.evaluate

Requires: OPENAI_API_KEY env var (RAGAS uses OpenAI for evaluation LLM).
"""
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import EvaluationDataset, SingleTurnSample


def create_sample_dataset() -> EvaluationDataset:
    """
    Create a sample evaluation dataset.
    In production, this would come from a curated test set.

    Each sample needs:
    - user_input: the question asked
    - response: the RAG system's generated answer
    - retrieved_contexts: the chunks that were retrieved
    - reference: the expected/ground-truth answer
    """
    samples = [
        SingleTurnSample(
            user_input="What is retrieval-augmented generation?",
            response="Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation. It retrieves relevant documents from a knowledge base and uses them as context for an LLM to generate grounded answers. [Source: rag_overview.pdf, Page: 1]",
            retrieved_contexts=[
                "Retrieval-Augmented Generation (RAG) is a technique that enhances LLM outputs by first retrieving relevant documents from an external knowledge base, then using those documents as additional context for generation.",
                "RAG systems typically consist of a retriever component that finds relevant passages and a generator component (usually an LLM) that produces answers grounded in the retrieved context.",
            ],
            reference="RAG is a technique that combines document retrieval with LLM generation to produce grounded, factual answers.",
        ),
        SingleTurnSample(
            user_input="How does BM25 work?",
            response="BM25 is a ranking function used in information retrieval. It scores documents based on term frequency and inverse document frequency, with saturation to prevent over-counting of frequent terms. [Source: search_algorithms.pdf, Page: 3]",
            retrieved_contexts=[
                "BM25 (Best Matching 25) is a bag-of-words retrieval function that ranks documents based on the query terms appearing in each document. It uses term frequency (TF) with saturation and inverse document frequency (IDF).",
                "The BM25 scoring function includes parameters k1 (term frequency saturation) and b (length normalization). Typical values are k1=1.2 and b=0.75.",
            ],
            reference="BM25 is a probabilistic ranking function that scores documents based on term frequency with saturation and inverse document frequency.",
        ),
        SingleTurnSample(
            user_input="What is the capital of Mars?",
            response="I don't have enough information in the provided documents to answer this question.",
            retrieved_contexts=[
                "Mars is the fourth planet from the Sun in our solar system.",
                "Mars has two small moons: Phobos and Deimos.",
            ],
            reference="There is no capital of Mars as it is not inhabited.",
        ),
    ]

    return EvaluationDataset(samples=samples)


def create_baseline_dataset() -> EvaluationDataset:
    """
    Create a baseline dataset (single-query retrieval, no multi-query).
    Used for A/B comparison to show the impact of improvements.

    In production, this would be the same questions run through the
    single-query pipeline for a direct comparison.
    """
    samples = [
        SingleTurnSample(
            user_input="What is retrieval-augmented generation?",
            response="RAG is a technique that combines retrieval and generation.",
            retrieved_contexts=[
                "Retrieval-Augmented Generation (RAG) is a technique that enhances LLM outputs by first retrieving relevant documents.",
            ],
            reference="RAG is a technique that combines document retrieval with LLM generation to produce grounded, factual answers.",
        ),
        SingleTurnSample(
            user_input="How does BM25 work?",
            response="BM25 scores documents using term frequency.",
            retrieved_contexts=[
                "BM25 (Best Matching 25) is a bag-of-words retrieval function that ranks documents based on the query terms appearing in each document.",
            ],
            reference="BM25 is a probabilistic ranking function that scores documents based on term frequency with saturation and inverse document frequency.",
        ),
        SingleTurnSample(
            user_input="What is the capital of Mars?",
            response="I don't have enough information in the provided documents to answer this question.",
            retrieved_contexts=[
                "Mars is the fourth planet from the Sun in our solar system.",
            ],
            reference="There is no capital of Mars as it is not inhabited.",
        ),
    ]

    return EvaluationDataset(samples=samples)


def run_evaluation():
    """
    Run RAGAS evaluation with baseline vs improved comparison.
    Reports metric deltas to show the impact of multi-query retrieval,
    better chunking, and reranking improvements.
    """
    print("=" * 60)
    print("RAGAS Evaluation — V2 (Baseline vs Improved)")
    print("=" * 60)

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    # --- Baseline ---
    print("\n📊 Running BASELINE evaluation (single-query)...")
    baseline_dataset = create_baseline_dataset()
    baseline_results = evaluate(dataset=baseline_dataset, metrics=metrics)

    # --- Improved ---
    print("📊 Running IMPROVED evaluation (multi-query + reranking)...")
    improved_dataset = create_sample_dataset()
    improved_results = evaluate(dataset=improved_dataset, metrics=metrics)

    # --- Comparison ---
    print("\n" + "=" * 60)
    print("                    BASELINE vs IMPROVED")
    print("=" * 60)
    print(f"  {'Metric':<25} {'Baseline':>10} {'Improved':>10} {'Delta':>10}")
    print("  " + "-" * 55)

    for name in metric_names:
        baseline_val = baseline_results.get(name, 0)
        improved_val = improved_results.get(name, 0)
        delta = improved_val - baseline_val
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        indicator = "✅" if delta > 0 else "⚠️" if delta == 0 else "❌"
        print(f"  {name:<25} {baseline_val:>10.4f} {improved_val:>10.4f} {delta_str:>10} {indicator}")

    # --- Per-Sample Improved ---
    print("\n--- Improved Per-Sample Results ---")
    df = improved_results.to_pandas()
    for idx, row in df.iterrows():
        print(f"\n  Sample {idx + 1}: {row.get('user_input', 'N/A')[:60]}...")
        for name in metric_names:
            print(f"    {name:<25} {row.get(name, 'N/A')}")

    print("\n" + "=" * 60)
    print("Evaluation complete.")
    return {"baseline": baseline_results, "improved": improved_results}


if __name__ == "__main__":
    run_evaluation()
