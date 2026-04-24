"""
conftest.py — Shared pytest fixtures and ML dependency stubs.

WHY STUB sys.modules?
  This project depends on torch, transformers, and datasets — packages that
  together weigh ~3 GB and take minutes to download. Unit tests should never
  require the full ML stack. The standard solution is to replace these packages
  with MagicMock objects in sys.modules BEFORE any test module imports them.

  When Python sees `from transformers import AutoTokenizer` and
  sys.modules['transformers'] is already a MagicMock, it returns a MagicMock
  attribute. This satisfies the import without downloading anything.

  Individual tests that need specific behaviour then use unittest.mock.patch
  to override specific attributes (AutoTokenizer, pipeline, etc.) for that
  test's duration.

  This pattern is standard in ML codebases — see huggingface/transformers,
  pytorch/pytorch, and many others.

ORDER MATTERS:
  sys.modules stubs must happen at the TOP of conftest.py, before any
  `import src.*` statements, because Python caches module imports.
"""

import sys
from unittest.mock import MagicMock

# ── Stub heavy ML packages ─────────────────────────────────────────────────────
# These stubs are installed into sys.modules so that any subsequent
# `from transformers import X` returns a MagicMock attribute.
# The stubs are only installed if the real packages are not already present,
# so this conftest works both with and without the ML packages installed.

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
    # Gradio is the UI framework — stub it so app tests don't need it installed
    "gradio",
    "gradio.themes",
]

for _pkg in _PACKAGES_TO_STUB:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = MagicMock()


# ── Now we can safely import project modules ───────────────────────────────────
import pytest
from unittest.mock import MagicMock, patch


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def mock_tokenizer():
    """
    A fake tokenizer that mimics HuggingFace tokenizer behaviour.
    Returns dummy token IDs instead of real tokenisation.
    """
    tok = MagicMock()
    tok.pad_token = "[PAD]"
    tok.pad_token_id = 0
    tok.eos_token = "<|endoftext|>"
    tok.eos_token_id = 50256

    # When called as a function (tokenizer(texts, ...)), return dummy encodings
    tok.return_value = {
        "input_ids": [[1, 2, 3, 4, 5]],
        "attention_mask": [[1, 1, 1, 1, 1]],
    }
    return tok


@pytest.fixture
def mock_model():
    """A fake CausalLM model. We just need it to not crash."""
    model = MagicMock()
    model.eval.return_value = model  # model.eval() returns self
    return model


@pytest.fixture
def mock_pipeline_output():
    """The output format returned by HuggingFace text-generation pipeline."""
    return [{"generated_text": "Article: Some article text.\nTL;DR: A short summary."}]


@pytest.fixture
def sample_texts():
    """A handful of short documents for testing data utilities."""
    return [
        "Scientists discovered a new species of deep-sea fish near the Mariana Trench.",
        "The stock market fell sharply on Friday amid fears of a global recession.",
        "A new study links excessive social media use to increased anxiety in teenagers.",
    ]


@pytest.fixture
def sample_summaries():
    """Corresponding reference summaries for sample_texts."""
    return [
        "A new deep-sea fish species has been found near the Mariana Trench.",
        "Stock markets dropped sharply on recession fears.",
        "Social media overuse is linked to teen anxiety, a new study shows.",
    ]
