"""
conftest.py — Shared pytest fixtures and ML dependency stubs.

WHY sys.modules STUBS AT THE TOP?
  torch, transformers, datasets, and gradio together weigh ~3 GB.
  Unit tests must never require them. We inject MagicMock objects into
  sys.modules BEFORE any test module imports src.*, so every
  `from transformers import X` returns a controllable mock attribute.
  Individual tests then use patch() to override specific behaviour.

  E402 (module-level import not at top) is suppressed in pyproject.toml
  for this file — the sys.modules manipulation MUST precede the imports,
  so the non-top-level placement of `import pytest` etc. is intentional.
"""

# ruff: noqa: E402  ← tells ruff this file intentionally has late imports
import sys
from unittest.mock import MagicMock

# FIX E402 + I001: stub heavy deps FIRST (required), then import test tools.
# FIX F401: removed unused `patch` import.
_PACKAGES_TO_STUB = [
    "torch",
    "torch.nn",
    "torch.utils",
    "torch.utils.data",
    "transformers",
    "transformers.trainer_utils",
    "datasets",
    "datasets.arrow_dataset",
    "accelerate",
    "huggingface_hub",
    "gradio",
    "gradio.themes",
]

for _pkg in _PACKAGES_TO_STUB:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = MagicMock()

# These imports intentionally come after sys.modules manipulation (E402 suppressed above)
import pytest  # noqa: E402

# ── Shared fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def mock_tokenizer():
    """Fake tokenizer — returns dummy token IDs, no real tokenisation."""
    tok = MagicMock()
    tok.pad_token = "[PAD]"
    tok.pad_token_id = 0
    tok.eos_token = "<|endoftext|>"
    tok.eos_token_id = 50256
    tok.return_value = {
        "input_ids": [[1, 2, 3, 4, 5]],
        "attention_mask": [[1, 1, 1, 1, 1]],
    }
    return tok


@pytest.fixture
def mock_model():
    """Fake CausalLM model."""
    model = MagicMock()
    model.eval.return_value = model
    return model


@pytest.fixture
def mock_pipeline_output():
    """Output format returned by HuggingFace text-generation pipeline."""
    return [{"generated_text": "Article: Some article text.\nTL;DR: A short summary."}]


@pytest.fixture
def sample_texts():
    return [
        "Scientists discovered a new species of deep-sea fish near the Mariana Trench.",
        "The stock market fell sharply on Friday amid fears of a global recession.",
        "A new study links excessive social media use to increased anxiety in teenagers.",
    ]


@pytest.fixture
def sample_summaries():
    return [
        "A new deep-sea fish species has been found near the Mariana Trench.",
        "Stock markets dropped sharply on recession fears.",
        "Social media overuse is linked to teen anxiety, a new study shows.",
    ]
