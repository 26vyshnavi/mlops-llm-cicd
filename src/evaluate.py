"""
evaluate.py — ROUGE-based evaluation for summarization quality.

WHAT IS ROUGE?
  ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures
  how much text from the reference summary appears in the generated summary.

  Three variants:
    ROUGE-1: Overlap of unigrams (single words)
    ROUGE-2: Overlap of bigrams (word pairs) — captures phrasing
    ROUGE-L: Longest common subsequence — captures sentence structure

  Each returns Precision, Recall, and F1. We report F1 (the harmonic mean).
  Higher ROUGE = generated summary is closer to the reference.

  Typical XSum ROUGE-2 scores (F1):
    Extractive baseline:  ~17
    BART-large:           ~21
    Fine-tuned GPT-2:     ~10-14 (decoder-only models are harder to beat T5/BART
                                   on strict summarization, but still learn!)

WHY NOT BLEU?
  BLEU (used in machine translation) penalises length strongly and doesn't
  handle the high variance in valid summaries well. ROUGE-L is the standard
  for summarization benchmarks.
"""

from typing import List, Dict

from rouge_score import rouge_scorer as rs


def compute_rouge(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores for a list of predictions.

    Args:
        predictions: List of generated summaries.
        references:  List of ground-truth summaries (same length).

    Returns:
        Dict mapping metric name → mean F1 score (0–1 range).
        Example: {"rouge1": 0.32, "rouge2": 0.12, "rougeL": 0.28}

    Raises:
        ValueError: If predictions and references have different lengths.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) and references ({len(references)}) "
            "must have the same length."
        )

    scorer = rs.RougeScorer(
        rouge_types=["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,  # Porter stemmer: "running"→"run", reduces false negatives
    )

    accumulated: Dict[str, List[float]] = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        scores = scorer.score(target=ref, prediction=pred)
        for key in accumulated:
            accumulated[key].append(scores[key].fmeasure)

    # Return mean F1 across all examples
    return {key: sum(vals) / len(vals) for key, vals in accumulated.items()}


def evaluate_model_on_dataset(
    pipeline,          # SummarizationPipeline instance
    documents: List[str],
    references: List[str],
    max_new_tokens: int = 80,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Run inference on a list of documents and compute ROUGE scores.

    Args:
        pipeline:       A SummarizationPipeline instance.
        documents:      Input articles to summarize.
        references:     Ground-truth summaries.
        max_new_tokens: Max tokens to generate per summary.
        verbose:        Print a few examples if True.

    Returns:
        ROUGE scores dict.
    """
    predictions = [pipeline.summarize(doc, max_new_tokens=max_new_tokens) for doc in documents]

    if verbose:
        print("\n── Sample predictions ──────────────────────────────────")
        for i in range(min(3, len(documents))):
            print(f"\nDoc     : {documents[i][:120]}...")
            print(f"Reference: {references[i]}")
            print(f"Generated: {predictions[i]}")

    scores = compute_rouge(predictions, references)

    if verbose:
        print("\n── ROUGE Scores ────────────────────────────────────────")
        for metric, score in scores.items():
            print(f"  {metric}: {score:.4f}")

    return scores
