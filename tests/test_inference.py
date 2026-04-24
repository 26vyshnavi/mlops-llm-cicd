"""
test_inference.py — Tests for the SummarizationPipeline class.

TESTING STRATEGY:
  We patch three things at the module level of src.inference:
    1. AutoTokenizer.from_pretrained → returns mock_tokenizer
    2. AutoModelForCausalLM.from_pretrained → returns mock_model
    3. pipeline(...) → returns a callable mock that returns fake generated text

  This means SummarizationPipeline.__init__ runs real code (loading config,
  calling from_pretrained, etc.) but the heavy I/O is replaced by mocks.
  Tests run in milliseconds with zero network access.
"""

import os
import pytest
from unittest.mock import patch, MagicMock, call

from src.config import InferenceConfig


def make_pipeline(
    model_path="distilgpt2",
    generated_text="Article: test.\nTL;DR: Short summary here.",
):
    """
    Helper that constructs a SummarizationPipeline with all heavy deps mocked.
    Returns (pipeline_instance, mock_hf_pipe) so tests can inspect mock calls.
    """
    with patch("src.inference.AutoTokenizer") as mock_tok_cls, \
         patch("src.inference.AutoModelForCausalLM") as mock_mdl_cls, \
         patch("src.inference.pipeline") as mock_hf_pipe_fn:

        # Tokenizer mock
        tok = MagicMock()
        tok.pad_token = "[PAD]"
        tok.pad_token_id = 0
        tok.eos_token = "<|endoftext|>"
        tok.eos_token_id = 50256
        mock_tok_cls.from_pretrained.return_value = tok

        # Model mock
        mdl = MagicMock()
        mdl.eval.return_value = mdl
        mock_mdl_cls.from_pretrained.return_value = mdl

        # HF pipeline mock — returns preset generated_text
        hf_pipe = MagicMock(return_value=[{"generated_text": generated_text}])
        mock_hf_pipe_fn.return_value = hf_pipe

        # Now import and construct (patches are active during __init__)
        from src.inference import SummarizationPipeline
        config = InferenceConfig(model_path=model_path)
        sp = SummarizationPipeline(config=config)

        return sp, hf_pipe


class TestSummarizationPipelineInit:

    def test_tokenizer_loaded_from_model_path(self):
        """Tokenizer should be loaded using the configured model path."""
        with patch("src.inference.AutoTokenizer") as mock_tok_cls, \
             patch("src.inference.AutoModelForCausalLM"), \
             patch("src.inference.pipeline"), \
             patch("os.path.exists", return_value=True):  # Pretend the path exists
            tok = MagicMock()
            tok.pad_token = "[PAD]"
            tok.eos_token = "<|endoftext|>"
            tok.eos_token_id = 50256
            mock_tok_cls.from_pretrained.return_value = tok

            from src.inference import SummarizationPipeline
            config = InferenceConfig(model_path="my-model-path")
            SummarizationPipeline(config=config)

            mock_tok_cls.from_pretrained.assert_called_once_with("my-model-path")

    def test_pad_token_set_when_missing(self):
        """If tokenizer.pad_token is None, it should be set to eos_token."""
        with patch("src.inference.AutoTokenizer") as mock_tok_cls, \
             patch("src.inference.AutoModelForCausalLM"), \
             patch("src.inference.pipeline"):
            tok = MagicMock()
            tok.pad_token = None  # Simulate GPT-2 (no pad token)
            tok.eos_token = "<|endoftext|>"
            tok.eos_token_id = 50256
            mock_tok_cls.from_pretrained.return_value = tok

            from src.inference import SummarizationPipeline
            config = InferenceConfig(model_path="distilgpt2")
            SummarizationPipeline(config=config)

            # pad_token and pad_token_id should have been assigned
            assert tok.pad_token == "<|endoftext|>"
            assert tok.pad_token_id == 50256

    def test_env_var_overrides_model_path(self, monkeypatch):
        """MODEL_PATH environment variable should override config.model_path."""
        monkeypatch.setenv("MODEL_PATH", "/env/path/to/model")

        with patch("os.path.exists", return_value=True), \
             patch("src.inference.AutoTokenizer") as mock_tok_cls, \
             patch("src.inference.AutoModelForCausalLM"), \
             patch("src.inference.pipeline"):
            tok = MagicMock()
            tok.pad_token = "[PAD]"
            tok.eos_token = "<EOS>"
            tok.eos_token_id = 1
            mock_tok_cls.from_pretrained.return_value = tok

            from src.inference import SummarizationPipeline
            config = InferenceConfig(model_path="./model_output")
            SummarizationPipeline(config=config)

            # The env var path should have been used instead of config.model_path
            assert mock_tok_cls.from_pretrained.called
            actual_path = mock_tok_cls.from_pretrained.call_args[0][0]
            assert actual_path == "/env/path/to/model"


class TestSummarizationPipelineSummarize:

    def test_returns_string(self):
        """summarize() must always return a string."""
        sp, _ = make_pipeline(generated_text="Article: x.\nTL;DR: A summary.")
        result = sp.summarize("Some article text.")
        assert isinstance(result, str)

    def test_extracts_text_after_tldr(self):
        """Only the text after 'TL;DR:' should be returned."""
        sp, _ = make_pipeline(
            generated_text="Article: x.\nTL;DR: This is the extracted summary."
        )
        result = sp.summarize("Some article text.")
        assert "This is the extracted summary." in result
        assert "Article:" not in result

    def test_empty_string_returns_empty(self):
        """Empty input should return empty string without calling the model."""
        sp, hf_pipe = make_pipeline()
        result = sp.summarize("")
        assert result == ""
        hf_pipe.assert_not_called()  # Model should not be invoked

    def test_whitespace_only_returns_empty(self):
        """Whitespace-only input should also skip model inference."""
        sp, hf_pipe = make_pipeline()
        result = sp.summarize("   \n\t  ")
        assert result == ""
        hf_pipe.assert_not_called()

    def test_max_new_tokens_passed_to_pipeline(self):
        """The max_new_tokens arg should be forwarded to the HF pipeline."""
        sp, hf_pipe = make_pipeline(
            generated_text="Article: x.\nTL;DR: summary."
        )
        sp.summarize("Some text.", max_new_tokens=42)

        call_kwargs = hf_pipe.call_args[1]
        assert call_kwargs["max_new_tokens"] == 42

    def test_eos_token_stripped_from_output(self):
        """EOS tokens that leak into the generated text should be removed."""
        eos = "<|endoftext|>"
        sp, _ = make_pipeline(
            generated_text=f"Article: x.\nTL;DR: Clean summary.{eos}"
        )
        result = sp.summarize("Some article.")
        assert eos not in result
        assert "Clean summary." in result

    def test_do_sample_and_temperature_used(self):
        """Sampling parameters from config should be passed to the pipeline."""
        sp, hf_pipe = make_pipeline(
            generated_text="Article: x.\nTL;DR: summary."
        )
        sp.summarize("text")
        call_kwargs = hf_pipe.call_args[1]
        assert call_kwargs["do_sample"] is True
        assert "temperature" in call_kwargs
        assert "top_p" in call_kwargs


class TestEvaluate:
    """Tests for the evaluate module."""

    def test_compute_rouge_same_text_is_perfect(self):
        """Identical prediction and reference should give ROUGE scores near 1.0."""
        from src.evaluate import compute_rouge
        preds = ["The cat sat on the mat."]
        refs = ["The cat sat on the mat."]
        scores = compute_rouge(preds, refs)
        assert scores["rouge1"] == pytest.approx(1.0, abs=0.01)
        assert scores["rougeL"] == pytest.approx(1.0, abs=0.01)

    def test_compute_rouge_unrelated_text_is_low(self):
        """Completely unrelated texts should have low ROUGE scores."""
        from src.evaluate import compute_rouge
        preds = ["The weather is sunny and warm."]
        refs = ["Stock markets fell sharply on Friday."]
        scores = compute_rouge(preds, refs)
        assert scores["rouge2"] < 0.1

    def test_compute_rouge_mismatched_lengths_raises(self):
        """Mismatched input lengths should raise ValueError."""
        from src.evaluate import compute_rouge
        with pytest.raises(ValueError):
            compute_rouge(["pred1", "pred2"], ["ref1"])

    def test_compute_rouge_returns_all_metrics(self, sample_texts, sample_summaries):
        """Output should contain rouge1, rouge2, and rougeL."""
        from src.evaluate import compute_rouge
        scores = compute_rouge(sample_summaries, sample_summaries)  # self-ROUGE
        assert "rouge1" in scores
        assert "rouge2" in scores
        assert "rougeL" in scores
