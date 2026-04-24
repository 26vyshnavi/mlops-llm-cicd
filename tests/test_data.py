"""
test_data.py — Tests for data loading and preprocessing logic.

We patch 'load_dataset' and 'AutoTokenizer' so no real network calls are made.
The tests verify our LOGIC (prompt formatting, tokenization call signatures),
not HuggingFace's implementation.
"""

import pytest
from unittest.mock import patch, MagicMock, call

from src.config import TrainingConfig
from src.data import build_prompt, tokenize_batch, PROMPT_TEMPLATE


# ── Unit tests for pure functions (no mocking needed) ─────────────────────────

class TestBuildPrompt:

    def test_prompt_contains_article_and_tldr(self):
        """The prompt must include both the article and 'TL;DR:' separator."""
        prompt = build_prompt("Some article.", "Short summary.", eos_token="<EOS>")
        assert "Article:" in prompt
        assert "TL;DR:" in prompt
        assert "Some article." in prompt
        assert "Short summary." in prompt

    def test_eos_token_appended(self):
        """EOS token should appear at the end of every training example."""
        prompt = build_prompt("doc", "summary", eos_token="<|endoftext|>")
        assert prompt.endswith("<|endoftext|>")

    def test_document_truncation(self):
        """Long documents should be truncated to max_doc_chars characters."""
        long_doc = "A" * 2000
        prompt = build_prompt(long_doc, "summary", eos_token="<EOS>", max_doc_chars=100)
        # The document part (after "Article: ") should be at most 100 chars
        doc_part = prompt.split("TL;DR:")[0].replace("Article: ", "").strip()
        assert len(doc_part) <= 100

    def test_empty_document(self):
        """Empty document should still produce a valid (if useless) prompt."""
        prompt = build_prompt("", "summary", eos_token="<EOS>")
        assert "TL;DR: summary" in prompt

    def test_prompt_template_format(self):
        """Verify the exact prompt format matches the training expectation."""
        prompt = build_prompt("doc text", "the summary", eos_token="<EOS>")
        expected_start = "Article: doc text"
        assert prompt.startswith(expected_start)


# ── Tests for tokenize_batch (mocked tokenizer) ────────────────────────────────

class TestTokenizeBatch:

    def test_returns_labels_equal_to_input_ids(self, mock_tokenizer):
        """
        For causal LM training, labels must equal input_ids.
        The Trainer shifts them internally.
        """
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }
        batch = {"document": ["doc1"], "summary": ["sum1"]}
        result = tokenize_batch(batch, mock_tokenizer, max_length=64)

        assert "labels" in result
        assert result["labels"] == result["input_ids"]

    def test_tokenizer_called_with_correct_kwargs(self, mock_tokenizer):
        """Tokenizer must be called with truncation=True and padding='max_length'."""
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }
        batch = {"document": ["doc1"], "summary": ["sum1"]}
        tokenize_batch(batch, mock_tokenizer, max_length=128)

        # Check the tokenizer was called with the right keyword args
        call_kwargs = mock_tokenizer.call_args[1]
        assert call_kwargs["max_length"] == 128
        assert call_kwargs["truncation"] is True
        assert call_kwargs["padding"] == "max_length"

    def test_batch_of_multiple_examples(self, mock_tokenizer):
        """Batches with multiple examples should all be tokenised."""
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2], [3, 4], [5, 6]],
            "attention_mask": [[1, 1], [1, 1], [1, 1]],
        }
        batch = {
            "document": ["doc1", "doc2", "doc3"],
            "summary": ["sum1", "sum2", "sum3"],
        }
        result = tokenize_batch(batch, mock_tokenizer, max_length=64)
        assert len(result["input_ids"]) == 3
        assert len(result["labels"]) == 3


# ── Integration-style test for load_and_preprocess (fully mocked) ─────────────

class TestLoadAndPreprocess:

    @patch("src.data.AutoTokenizer")
    @patch("src.data.load_dataset")
    def test_load_and_preprocess_returns_dataset_and_tokenizer(
        self, mock_load_dataset, mock_tokenizer_class
    ):
        """load_and_preprocess should return a DatasetDict and tokenizer."""
        from src.data import load_and_preprocess

        # Mock the raw dataset
        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=100)
        mock_ds.column_names = ["document", "summary"]
        mock_ds.map.return_value = mock_ds
        mock_ds.train_test_split.return_value = {"train": mock_ds, "test": mock_ds}
        mock_load_dataset.return_value = mock_ds

        # Mock the tokenizer
        mock_tok = MagicMock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "<|endoftext|>"
        mock_tok.eos_token_id = 50256
        mock_tokenizer_class.from_pretrained.return_value = mock_tok

        config = TrainingConfig(dataset_split_size=100, model_name="distilgpt2")
        result_ds, result_tok = load_and_preprocess(config)

        # Dataset split should have been called
        mock_ds.train_test_split.assert_called_once()
        # Tokenizer should have been loaded
        mock_tokenizer_class.from_pretrained.assert_called_once_with("distilgpt2")
        # pad_token should have been set (since we returned None initially)
        assert mock_tok.pad_token is not None

    @patch("src.data.AutoTokenizer")
    @patch("src.data.load_dataset")
    def test_dataset_subset_slice_used(self, mock_load_dataset, mock_tokenizer_class):
        """Only a subset of the dataset should be loaded when split_size is set."""
        from src.data import load_and_preprocess

        mock_ds = MagicMock()
        mock_ds.column_names = ["document", "summary"]
        mock_ds.map.return_value = mock_ds
        mock_ds.train_test_split.return_value = {"train": mock_ds, "test": mock_ds}
        mock_load_dataset.return_value = mock_ds

        mock_tok = MagicMock()
        mock_tok.pad_token = "[PAD]"
        mock_tok.eos_token = "<EOS>"
        mock_tok.eos_token_id = 1
        mock_tokenizer_class.from_pretrained.return_value = mock_tok

        config = TrainingConfig(dataset_split_size=42)
        load_and_preprocess(config)

        # Verify that the split slice was passed to load_dataset
        call_args = mock_load_dataset.call_args
        assert "train[:42]" in call_args[1]["split"] or "42" in str(call_args)
