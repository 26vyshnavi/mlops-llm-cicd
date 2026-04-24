"""
test_data.py — Tests for data loading and preprocessing logic.

FIX F401: removed unused `import pytest`
FIX I001: imports sorted (stdlib → third-party → local, alphabetically within groups)
"""

from unittest.mock import MagicMock, patch

from src.config import TrainingConfig
from src.data import PROMPT_TEMPLATE, build_prompt, tokenize_batch


class TestBuildPrompt:
    def test_prompt_contains_article_and_tldr(self):
        prompt = build_prompt("Some article.", "Short summary.", eos_token="<EOS>")
        assert "Article:" in prompt
        assert "TL;DR:" in prompt
        assert "Some article." in prompt
        assert "Short summary." in prompt

    def test_eos_token_appended(self):
        prompt = build_prompt("doc", "summary", eos_token="<|endoftext|>")
        assert prompt.endswith("<|endoftext|>")

    def test_document_truncation(self):
        long_doc = "A" * 2000
        prompt = build_prompt(long_doc, "summary", eos_token="<EOS>", max_doc_chars=100)
        doc_part = prompt.split("TL;DR:")[0].replace("Article: ", "").strip()
        assert len(doc_part) <= 100

    def test_empty_document(self):
        prompt = build_prompt("", "summary", eos_token="<EOS>")
        assert "TL;DR: summary" in prompt

    def test_prompt_template_format(self):
        prompt = build_prompt("doc text", "the summary", eos_token="<EOS>")
        assert prompt.startswith("Article: doc text")

    def test_prompt_template_constant_used(self):
        assert "Article:" in PROMPT_TEMPLATE
        assert "TL;DR:" in PROMPT_TEMPLATE


class TestTokenizeBatch:
    def test_returns_labels_equal_to_input_ids(self, mock_tokenizer):
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }
        batch = {"document": ["doc1"], "summary": ["sum1"]}
        result = tokenize_batch(batch, mock_tokenizer, max_length=64)
        assert "labels" in result
        assert result["labels"] == result["input_ids"]

    def test_tokenizer_called_with_correct_kwargs(self, mock_tokenizer):
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }
        batch = {"document": ["doc1"], "summary": ["sum1"]}
        tokenize_batch(batch, mock_tokenizer, max_length=128)
        call_kwargs = mock_tokenizer.call_args[1]
        assert call_kwargs["max_length"] == 128
        assert call_kwargs["truncation"] is True
        assert call_kwargs["padding"] == "max_length"

    def test_batch_of_multiple_examples(self, mock_tokenizer):
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


class TestLoadAndPreprocess:
    @patch("src.data.AutoTokenizer")
    @patch("src.data.load_dataset")
    def test_returns_dataset_and_tokenizer(self, mock_load_dataset, mock_tokenizer_class):
        from src.data import load_and_preprocess

        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=100)
        mock_ds.column_names = ["document", "summary"]
        mock_ds.map.return_value = mock_ds
        mock_ds.train_test_split.return_value = {"train": mock_ds, "test": mock_ds}
        mock_load_dataset.return_value = mock_ds

        mock_tok = MagicMock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "<|endoftext|>"
        mock_tok.eos_token_id = 50256
        mock_tokenizer_class.from_pretrained.return_value = mock_tok

        config = TrainingConfig(dataset_split_size=100)
        result_ds, result_tok = load_and_preprocess(config)

        mock_ds.train_test_split.assert_called_once()
        mock_tokenizer_class.from_pretrained.assert_called_once_with("distilgpt2")
        assert mock_tok.pad_token is not None

    @patch("src.data.AutoTokenizer")
    @patch("src.data.load_dataset")
    def test_dataset_subset_slice_used(self, mock_load_dataset, mock_tokenizer_class):
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

        call_args = mock_load_dataset.call_args
        assert "42" in str(call_args)
