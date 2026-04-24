"""
data.py — Dataset loading and preprocessing for DistilGPT-2 summarization.

We frame summarization as a completion task for the decoder-only model:
    "Article: {document}\nTL;DR: {summary}<|endoftext|>"
"""

# FIX I001: stdlib → third-party → local, each group alphabetically sorted
from typing import Any, Dict, Tuple

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.config import TrainingConfig

PROMPT_TEMPLATE = "Article: {document}\nTL;DR: {summary}"


def build_prompt(document: str, summary: str, eos_token: str, max_doc_chars: int = 800) -> str:
    """Construct a single training example string."""
    doc_truncated = document[:max_doc_chars]
    return PROMPT_TEMPLATE.format(document=doc_truncated, summary=summary) + eos_token


def tokenize_batch(
    examples: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> Dict[str, Any]:
    """
    Tokenize a batch of examples (called via dataset.map).

    For CausalLM, input_ids == labels — the Trainer shifts them internally.
    """
    prompts = [
        build_prompt(doc, summ, tokenizer.eos_token)
        for doc, summ in zip(examples["document"], examples["summary"])
    ]

    tokenized = tokenizer(
        prompts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors=None,
    )

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def load_and_preprocess(config: TrainingConfig) -> Tuple[DatasetDict, PreTrainedTokenizer]:
    """Load XSum, tokenize, and split into train/eval sets."""
    print(f"Loading '{config.dataset_name}' dataset ({config.dataset_split_size} samples)...")

    raw_dataset: Dataset = load_dataset(
        config.dataset_name,
        split=f"train[:{config.dataset_split_size}]",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenized_dataset = raw_dataset.map(
        lambda examples: tokenize_batch(examples, tokenizer, config.max_length),
        batched=True,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing",
    )

    split_dataset: DatasetDict = tokenized_dataset.train_test_split(
        test_size=config.eval_split,
        seed=42,
    )

    print(f"Train: {len(split_dataset['train'])} | Eval: {len(split_dataset['test'])}")
    return split_dataset, tokenizer
