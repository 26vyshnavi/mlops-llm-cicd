"""
config.py — Central configuration using Python dataclasses.
"""

from dataclasses import dataclass  # FIX: removed unused `field`
from typing import Optional


@dataclass
class TrainingConfig:
    # DistilGPT-2 is ~82M params, fast to fine-tune on CPU
    model_name: str = "distilgpt2"
    # XSum: BBC news articles paired with 1-sentence summaries
    dataset_name: str = "xsum"
    # Use a small subset for CI speed; bump to 10k+ for real training
    dataset_split_size: int = 1000
    max_length: int = 256
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    output_dir: str = "./model_output"
    # Set to "your-username/distilgpt2-summarizer" to push to HF Hub
    hub_model_id: Optional[str] = None
    eval_split: float = 0.1


@dataclass
class InferenceConfig:
    # Path to fine-tuned model dir, or an HF Hub model ID
    model_path: str = "./model_output"
    max_new_tokens: int = 80
    # temperature < 1.0 → more focused; > 1.0 → more creative
    temperature: float = 0.7
    # Top-p (nucleus) sampling: keeps outputs coherent
    top_p: float = 0.9
    num_beams: int = 1
