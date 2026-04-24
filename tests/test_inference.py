"""
test_inference.py — Tests for the SummarizationPipeline class.

FIX I001: imports sorted (stdlib → third-party → local).
pytest is imported here because test_compute_rouge_same_text_is_perfect uses pytest.approx.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.config import InferenceConfig


def make_pipeline(
    model_path="distilgpt2",
    generated_text="Article: test.\nTL;DR: Short summary here.",
):
    """Construct a SummarizationPipeline with all heavy deps mocked."""
    with (
        patch("src.inference.AutoTokenizer") as mock_tok_cls,
        patch("src.inference.AutoModelForCausalLM") as mock_mdl_cls,
        patch("src.inference.pipeline") as mock_hf_pipe_fn,
    ):
        tok = MagicMock()
        tok.pad_token = "[PAD]"
        tok.pad_token_id = 0
        tok.eos_token = "<|endoftext|>"
        tok.eos_token_id = 50256
        mock_tok_cls.from_pretrained.return_value = tok

        mdl = MagicMock()
        mdl.eval.return_value = mdl
        mock_mdl_cls.from_pretrained.return_value = mdl

        hf_pipe = MagicMock(return_value=[{"generated_text": generated_text}])
        mock_hf_pipe_fn.return_value = hf_pipe

        from src.inference import SummarizationPipeline

        config = InferenceConfig(model_path=model_path)
        sp = SummarizationPipeline(config=config)
        return sp, hf_pipe


class TestSummarizationPipelineInit:
    def test_tokenizer_loaded_from_model_path(self):
        with (
            patch("src.inference.AutoTokenizer") as mock_tok_cls,
            patch("src.inference.AutoModelForCausalLM"),
            patch("src.inference.pipeline"),
            patch("os.path.exists", return_value=True),
        ):
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
        with (
            patch("src.inference.AutoTokenizer") as mock_tok_cls,
            patch("src.inference.AutoModelForCausalLM"),
            patch("src.inference.pipeline"),
        ):
            tok = MagicMock()
            tok.pad_token = None
            tok.eos_token = "<|endoftext|>"
            tok.eos_token_id = 50256
            mock_tok_cls.from_pretrained.return_value = tok

            from src.inference import SummarizationPipeline

            config = InferenceConfig(model_path="distilgpt2")
            SummarizationPipeline(config=config)
            assert tok.pad_token == "<|endoftext|>"
            assert tok.pad_token_id == 50256

    def test_env_var_overrides_model_path(self, monkeypatch):
        monkeypatch.setenv("MODEL_PATH", "/env/path/to/model")

        with (
            patch("os.path.exists", return_value=True),
            patch("src.inference.AutoTokenizer") as mock_tok_cls,
            patch("src.inference.AutoModelForCausalLM"),
            patch("src.inference.pipeline"),
        ):
            tok = MagicMock()
            tok.pad_token = "[PAD]"
            tok.eos_token = "<EOS>"
            tok.eos_token_id = 1
            mock_tok_cls.from_pretrained.return_value = tok

            from src.inference import SummarizationPipeline

            config = InferenceConfig(model_path="./model_output")
            SummarizationPipeline(config=config)
            assert mock_tok_cls.from_pretrained.called
            actual_path = mock_tok_cls.from_pretrained.call_args[0][0]
            assert actual_path == "/env/path/to/model"


class TestSummarizationPipelineSummarize:
    def test_returns_string(self):
        sp, _ = make_pipeline(generated_text="Article: x.\nTL;DR: A summary.")
        result = sp.summarize("Some article text.")
        assert isinstance(result, str)

    def test_extracts_text_after_tldr(self):
        sp, _ = make_pipeline(generated_text="Article: x.\nTL;DR: This is the extracted summary.")
        result = sp.summarize("Some article text.")
        assert "This is the extracted summary." in result
        assert "Article:" not in result

    def test_empty_string_returns_empty(self):
        sp, hf_pipe = make_pipeline()
        result = sp.summarize("")
        assert result == ""
        hf_pipe.assert_not_called()

    def test_whitespace_only_returns_empty(self):
        sp, hf_pipe = make_pipeline()
        result = sp.summarize("   \n\t  ")
        assert result == ""
        hf_pipe.assert_not_called()

    def test_max_new_tokens_passed_to_pipeline(self):
        sp, hf_pipe = make_pipeline(generated_text="Article: x.\nTL;DR: summary.")
        sp.summarize("Some text.", max_new_tokens=42)
        call_kwargs = hf_pipe.call_args[1]
        assert call_kwargs["max_new_tokens"] == 42

    def test_eos_token_stripped_from_output(self):
        eos = "<|endoftext|>"
        sp, _ = make_pipeline(generated_text=f"Article: x.\nTL;DR: Clean summary.{eos}")
        result = sp.summarize("Some article.")
        assert eos not in result
        assert "Clean summary." in result

    def test_do_sample_and_temperature_used(self):
        sp, hf_pipe = make_pipeline(generated_text="Article: x.\nTL;DR: summary.")
        sp.summarize("text")
        call_kwargs = hf_pipe.call_args[1]
        assert call_kwargs["do_sample"] is True
        assert "temperature" in call_kwargs
        assert "top_p" in call_kwargs


class TestEvaluate:
    def test_compute_rouge_same_text_is_perfect(self):
        from src.evaluate import compute_rouge

        preds = ["The cat sat on the mat."]
        refs = ["The cat sat on the mat."]
        scores = compute_rouge(preds, refs)
        assert scores["rouge1"] == pytest.approx(1.0, abs=0.01)
        assert scores["rougeL"] == pytest.approx(1.0, abs=0.01)

    def test_compute_rouge_unrelated_text_is_low(self):
        from src.evaluate import compute_rouge

        preds = ["The weather is sunny and warm."]
        refs = ["Stock markets fell sharply on Friday."]
        scores = compute_rouge(preds, refs)
        assert scores["rouge2"] < 0.1

    def test_compute_rouge_mismatched_lengths_raises(self):
        from src.evaluate import compute_rouge

        with pytest.raises(ValueError):
            compute_rouge(["pred1", "pred2"], ["ref1"])

    def test_compute_rouge_returns_all_metrics(self, sample_summaries):
        from src.evaluate import compute_rouge

        scores = compute_rouge(sample_summaries, sample_summaries)
        assert "rouge1" in scores
        assert "rouge2" in scores
        assert "rougeL" in scores
