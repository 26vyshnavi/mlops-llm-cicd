"""
evaluate.py — ROUGE-based evaluation for summarization quality.
"""

# FIX I001: Dict before List alphabetically
from typing import Dict, List

from rouge_score import rouge_scorer as rs


def compute_rouge(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Args:
        predictions: Generated summaries.
        references:  Ground-truth summaries (same length).

    Returns:
        Dict mapping metric name → mean F1 (0–1 range).
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) and references ({len(references)}) "
            "must have the same length."
        )

    scorer = rs.RougeScorer(
        rouge_types=["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    accumulated: Dict[str, List[float]] = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        scores = scorer.score(target=ref, prediction=pred)
        for key in accumulated:
            accumulated[key].append(scores[key].fmeasure)

    return {key: sum(vals) / len(vals) for key, vals in accumulated.items()}
