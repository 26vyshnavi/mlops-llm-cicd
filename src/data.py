"""
data.py — Dataset loading and preprocessing for DistilGPT-2 summarization.

KEY CONCEPT — HOW WE FRAME SUMMARIZATION FOR A DECODER-ONLY MODEL:
  GPT-2 and its variants are causal language models: they predict the next
  token given all previous tokens. They are NOT encoder-decoder models like
  T5. To do summarization with a decoder-only model, we use a "prompt" trick:

      "Article: {document}\nTL;DR: {summary}<|endoftext|>"

  During training, the model sees the full string and learns that after
  "TL;DR:" comes the summary. During inference, we feed "Article: ...\nTL;DR:"
  and let the model complete the rest.

  This is exactly how GPT-3's few-shot summarization works — no architecture
  changes needed, just careful prompt engineering + fine-tuning.

WHY XSum?
  - BBC news articles paired with a 1-sentence summary
  - ~200k examples, but we sample a small subset for demo speed
  - Short summaries = fewer tokens = faster training
"""

from typing import Tuple, Dict, Any

# Module-level imports so patch("src.data.AutoTokenizer") works in tests.
# conftest.py stubs sys.modules so these succeed without the packages installed.
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.config import TrainingConfig


# ── Prompt template ──────────────────────────────────────────────────────────
# This template is the single most important design decision in this file.
# The model will learn to complete the "TL;DR:" part given the "Article:" part.
PROMPT_TEMPLATE = "Article: {document}\nTL;DR: {summary}"


def build_prompt(document: str, summary: str, eos_token: str, max_doc_chars: int = 800) -> str:
    """
    Construct a single training example string.

    We truncate the document at the CHARACTER level (not token level) first,
    as a quick pre-filter. The tokenizer will handle proper token truncation.

    Args:
        document:     The source article text.
        summary:      The target summary.
        eos_token:    The tokenizer's end-of-sequence token (e.g. "<|endoftext|>").
                      Appended at the end so the model learns WHEN to stop.
        max_doc_chars: Max characters to keep from the document before tokenising.
    """
    doc_truncated = document[:max_doc_chars]
    return PROMPT_TEMPLATE.format(document=doc_truncated, summary=summary) + eos_token


def tokenize_batch(
    examples: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> Dict[str, Any]:
    """
    Tokenize a batch of examples (called via dataset.map).

    HOW CAUSAL LM TRAINING WORKS:
      For CausalLM, input_ids == labels (the model predicts the next token at
      every position). The DataCollatorForLanguageModeling handles shifting
      labels by one position internally, so we just need to pass the same
      tokenised sequence as both input and output.

    Args:
        examples:   HuggingFace dataset batch dict with 'document' and 'summary'.
        tokenizer:  The tokenizer for the model.
        max_length: Maximum sequence length (tokens).

    Returns:
        Dict with 'input_ids' and 'attention_mask'.
    """
    prompts = [
        build_prompt(doc, summ, tokenizer.eos_token)
        for doc, summ in zip(examples["document"], examples["summary"])
    ]

    tokenized = tokenizer(
        prompts,
        max_length=max_length,
        truncation=True,       # Silently chop sequences that are too long
        padding="max_length",  # Pad shorter sequences to max_length
        return_tensors=None,   # Return plain Python lists, not tensors
    )

    # For causal LM, labels = input_ids (the Trainer handles -100 masking)
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def load_and_preprocess(config: TrainingConfig) -> Tuple[DatasetDict, PreTrainedTokenizer]:
    """
    Load the XSum dataset, tokenize it, and split into train/eval sets.

    Pipeline:
      1. Load a subset of XSum from HuggingFace Datasets (auto-downloaded & cached).
      2. Load the DistilGPT-2 tokenizer.
      3. Set pad_token = eos_token (GPT-2 has no pad token by default).
      4. Apply tokenize_batch to every row in parallel using .map().
      5. Split into train / eval.

    Returns:
        (DatasetDict with 'train'/'test' splits, fitted tokenizer)
    """
    print(f"Loading '{config.dataset_name}' dataset (first {config.dataset_split_size} samples)...")

    # The slice syntax "train[:N]" loads only the first N rows — no need to
    # download the whole dataset when we just want a quick demo.
    raw_dataset: Dataset = load_dataset(
        config.dataset_name,
        split=f"train[:{config.dataset_split_size}]",
        trust_remote_code=True,
    )

    print(f"Loaded {len(raw_dataset)} examples. Columns: {raw_dataset.column_names}")

    # ── Tokenizer setup ───────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # GPT-2 family has no padding token. We reuse the EOS token as PAD.
    # This is standard practice — the DataCollator will handle masking.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Tokenization ──────────────────────────────────────────────────────────
    print("Tokenizing dataset...")
    tokenized_dataset = raw_dataset.map(
        lambda examples: tokenize_batch(examples, tokenizer, config.max_length),
        batched=True,                          # Process in batches (faster)
        remove_columns=raw_dataset.column_names,  # Drop original text columns
        desc="Tokenizing",
    )

    # ── Train / eval split ────────────────────────────────────────────────────
    split_dataset: DatasetDict = tokenized_dataset.train_test_split(
        test_size=config.eval_split,
        seed=42,  # Reproducible split
    )

    print(f"Train: {len(split_dataset['train'])} | Eval: {len(split_dataset['test'])}")

    return split_dataset, tokenizer
