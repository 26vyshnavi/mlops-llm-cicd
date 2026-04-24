"""
config.py — Central configuration using Python dataclasses.

WHY DATACLASSES?
  All hyperparameters live in one place. You can override any field
  without touching scattered magic numbers. This also makes it easy
  to serialize config to JSON/YAML for experiment tracking.

WHY SEPARATE TRAINING vs INFERENCE CONFIGS?
  Training concerns (epochs, LR, dataset size) are completely separate
  from inference concerns (temperature, max tokens). Keeping them apart
  prevents accidental coupling and makes each config self-documenting.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    # ── Model ────────────────────────────────────────────────────────────────
    # DistilGPT-2 is a distilled version of GPT-2 (~82M params, ~300 MB).
    # It's small enough to fine-tune on a CPU or free-tier GPU in minutes,
    # while still being a real transformer that learns meaningful patterns.
    model_name: str = "distilgpt2"

    # ── Dataset ──────────────────────────────────────────────────────────────
    # XSum (Extreme Summarization) is a BBC news dataset where each article
    # has a single-sentence "extreme" summary. It's clean and well-structured,
    # making it great for a decoder-only prompt format.
    dataset_name: str = "xsum"

    # For CI/demo runs keep this small (500–1000). A full fine-tune would use
    # the entire 200k+ sample dataset, but that takes hours.
    dataset_split_size: int = 1000

    # Max token length for the combined prompt+summary string.
    # DistilGPT-2's context window is 1024 tokens; 256 is plenty for XSum.
    max_length: int = 256

    # ── Training hyperparameters ──────────────────────────────────────────────
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3

    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: str = "./model_output"

    # HF Hub repo to push the fine-tuned model to (e.g. "your-user/distilgpt2-xsum").
    # Leave None to skip pushing.
    hub_model_id: Optional[str] = None

    # Fraction of training data to use as validation
    eval_split: float = 0.1


@dataclass
class InferenceConfig:
    # Path to the fine-tuned model directory (or an HF Hub model ID)
    model_path: str = "./model_output"

    # How many NEW tokens to generate after the prompt
    max_new_tokens: int = 80

    # Temperature < 1.0 → more focused/deterministic outputs
    # Temperature > 1.0 → more creative/random outputs
    temperature: float = 0.7

    # Top-p (nucleus) sampling: only consider tokens whose cumulative
    # probability ≥ top_p. Keeps outputs coherent.
    top_p: float = 0.9

    # Number of beams for beam search (1 = greedy/sampling, >1 = beam search)
    num_beams: int = 1
