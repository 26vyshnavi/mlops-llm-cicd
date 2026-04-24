"""
inference.py — Production inference wrapper for the fine-tuned model.

Lazy-loads the model on first request (singleton pattern) so the app
starts fast and tests can import this module without triggering downloads.
"""

import os
from typing import Optional

# FIX F821 + I001: removed unused `Pipeline` type; kept AutoModelForCausalLM,
# AutoTokenizer, pipeline — sorted alphabetically as ruff requires.
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.config import InferenceConfig


class SummarizationPipeline:
    """
    Thin wrapper around a HuggingFace text-generation pipeline.

    Usage:
        sp = SummarizationPipeline()
        summary = sp.summarize("Scientists discovered...")
    """

    def __init__(self, config: Optional[InferenceConfig] = None):
        if config is None:
            config = InferenceConfig()

        self.config = config
        model_path = os.getenv("MODEL_PATH", config.model_path)

        # Fall back to base model if the local path doesn't exist
        if not os.path.exists(model_path) and "/" not in model_path:
            print(f"[Warning] Model path '{model_path}' not found. Falling back to 'distilgpt2'.")
            model_path = "distilgpt2"

        print(f"[Inference] Loading model from: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.eval()

        # FIX F821: removed `: Pipeline` type annotation — Pipeline was never
        # imported after the import refactor. Plain assignment is fine here.
        self._pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1,  # -1 = CPU; change to 0 for first GPU
        )

        print("[Inference] Model loaded and ready.")

    def summarize(self, text: str, max_new_tokens: Optional[int] = None) -> str:
        """
        Summarize input text using the fine-tuned model.

        Prompt format mirrors training: "Article: {text}\\nTL;DR:"
        """
        if not text or not text.strip():
            return ""

        n_tokens = max_new_tokens or self.config.max_new_tokens
        prompt = f"Article: {text[:800]}\nTL;DR:"

        output = self._pipe(
            prompt,
            max_new_tokens=n_tokens,
            do_sample=True,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        full_text: str = output[0]["generated_text"]

        if "TL;DR:" in full_text:
            summary = full_text.split("TL;DR:", 1)[-1].strip()
        else:
            summary = full_text[len(prompt) :].strip()

        return summary.replace(self.tokenizer.eos_token, "").strip()
