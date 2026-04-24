"""
inference.py — Production inference wrapper around the fine-tuned model.

DESIGN PRINCIPLES:
  1. Singleton-style lazy loading: model is loaded ONCE on first call,
     not on every request. Critical for performance in a web server.

  2. Clean interface: callers pass plain strings, get plain strings back.
     All tokenisation/detokenisation is hidden inside this class.

  3. Stateless per-request: the model itself holds no request state.
     Multiple threads can call .summarize() concurrently (with the GIL
     being the only serialisation point for pure Python).

  4. Graceful fallback: if no fine-tuned model exists, fall back to the
     base DistilGPT-2. Useful during local development before training.

HOW TEXT GENERATION WORKS (quick primer):
  At inference time, we feed tokens [t1, t2, ..., tN] to the model.
  The model outputs a probability distribution over the vocabulary for
  the NEXT token. We sample from this distribution (or take the argmax),
  append the new token, and repeat until we hit max_new_tokens or EOS.

  Sampling strategies:
    - Greedy (do_sample=False): always pick the most probable token.
      Fast but repetitive / boring.
    - Temperature sampling: divide logits by T before softmax.
      T < 1.0 → sharper distribution (more focused).
      T > 1.0 → flatter distribution (more random).
    - Top-p (nucleus): only consider the top tokens summing to p of
      probability mass. Balances coherence and diversity.
"""

import os
from typing import Optional

# WHY MODULE-LEVEL IMPORTS HERE?
#   These names need to be in this module's namespace so that test patches like
#   patch("src.inference.AutoTokenizer") work correctly.
#   In tests, conftest.py stubs sys.modules['transformers'] so these imports
#   succeed without requiring the actual packages to be installed.
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.config import InferenceConfig


class SummarizationPipeline:
    """
    Thin wrapper around a HuggingFace text-generation pipeline for summarization.

    Usage:
        sp = SummarizationPipeline()
        summary = sp.summarize("Scientists have discovered...")
    """

    def __init__(self, config: Optional[InferenceConfig] = None):
        """
        Load the model and tokenizer from disk (or HF Hub).

        Args:
            config: InferenceConfig. Defaults are used if None.
                    You can also set the MODEL_PATH environment variable
                    to override config.model_path — handy in Docker/CI.
        """
        if config is None:
            config = InferenceConfig()

        # Allow env-var override so Docker/CI can inject the model path
        # without rebuilding the image or changing code.
        self.config = config
        model_path = os.getenv("MODEL_PATH", config.model_path)

        # If the configured path doesn't exist locally, fall back to the
        # base model. This lets the app start even before training runs.
        if not os.path.exists(model_path) and "/" not in model_path:
            print(f"[Warning] Model path '{model_path}' not found. Falling back to 'distilgpt2'.")
            model_path = "distilgpt2"

        print(f"[Inference] Loading model from: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # DistilGPT-2 has no pad token — set it to EOS to avoid warnings
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.eval()  # Disable dropout layers for deterministic inference

        # Build a HuggingFace pipeline — this handles batching, device placement,
        # and the generate() loop automatically.
        self._pipe: Pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1,  # -1 = CPU. Change to 0 for first GPU.
        )

        print("[Inference] Model loaded and ready.")

    def summarize(self, text: str, max_new_tokens: Optional[int] = None) -> str:
        """
        Summarize an input text using the fine-tuned model.

        The prompt format mirrors what we trained on:
            "Article: {text}\nTL;DR:"
        The model completes this prompt with the summary.

        Args:
            text:           The source text to summarize.
            max_new_tokens: Override the config value for this call.

        Returns:
            A string containing the generated summary.
        """
        if not text or not text.strip():
            return ""

        n_tokens = max_new_tokens or self.config.max_new_tokens

        # Truncate input at character level to avoid exceeding context window
        prompt = f"Article: {text[:800]}\nTL;DR:"

        output = self._pipe(
            prompt,
            max_new_tokens=n_tokens,
            do_sample=True,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            pad_token_id=self.tokenizer.eos_token_id,
            # Stop generation when the model produces EOS
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # The pipeline returns a list of dicts; we only sent one prompt.
        full_text: str = output[0]["generated_text"]

        # ── Extract just the summary ─────────────────────────────────────────
        # full_text = "Article: ...\nTL;DR: <summary>"
        # We split on "TL;DR:" and take everything after it.
        if "TL;DR:" in full_text:
            summary = full_text.split("TL;DR:", 1)[-1].strip()
        else:
            # Fallback: strip the prompt manually
            summary = full_text[len(prompt):].strip()

        # Clean up any EOS tokens that leaked into the output string
        summary = summary.replace(self.tokenizer.eos_token, "").strip()

        return summary
