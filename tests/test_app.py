"""
test_app.py — Tests for the Gradio app entrypoint.

CHALLENGE: app.py calls get_pipeline() at request time (not import time),
so we can safely import the module without loading a real model.
We patch get_pipeline() to return a mock pipeline for all tests here.
"""

import pytest
from unittest.mock import patch, MagicMock


def make_mock_pipeline(summary_return="Mock summary."):
    """Create a mock SummarizationPipeline that returns a preset summary."""
    mock = MagicMock()
    mock.summarize.return_value = summary_return
    return mock


class TestSummarizeFunction:
    """Tests for the app.summarize() function."""

    def test_empty_input_returns_warning(self):
        """Empty text should return a user-friendly warning, not crash."""
        with patch("app.app.get_pipeline") as mock_get:
            mock_get.return_value = make_mock_pipeline()
            from app.app import summarize

            result = summarize("", 80)

            assert "Please provide" in result or "provide" in result.lower()
            mock_get.return_value.summarize.assert_not_called()

    def test_whitespace_input_returns_warning(self):
        """Whitespace-only input should be treated the same as empty."""
        with patch("app.app.get_pipeline") as mock_get:
            mock_get.return_value = make_mock_pipeline()
            from app.app import summarize

            result = summarize("    \n\t  ", 80)

            assert len(result) > 0  # Returns something
            mock_get.return_value.summarize.assert_not_called()

    def test_valid_input_calls_pipeline(self):
        """Non-empty input should be forwarded to the pipeline."""
        with patch("app.app.get_pipeline") as mock_get:
            mock_pipe = make_mock_pipeline("Scientists discovered a fish.")
            mock_get.return_value = mock_pipe
            from app.app import summarize

            article = "Scientists have discovered a rare fish in the Pacific Ocean."
            result = summarize(article, 80)

            mock_pipe.summarize.assert_called_once_with(article, max_new_tokens=80)
            assert result == "Scientists discovered a fish."

    def test_max_tokens_forwarded(self):
        """max_tokens slider value should be passed to pipeline.summarize()."""
        with patch("app.app.get_pipeline") as mock_get:
            mock_pipe = make_mock_pipeline()
            mock_get.return_value = mock_pipe
            from app.app import summarize

            summarize("Some long article text here.", 150)

            call_kwargs = mock_pipe.summarize.call_args[1]
            assert call_kwargs["max_new_tokens"] == 150

    def test_pipeline_loaded_lazily(self):
        """get_pipeline() should NOT be called until summarize() is invoked."""
        with patch("app.app.get_pipeline") as mock_get:
            mock_get.return_value = make_mock_pipeline()

            # Just importing should not load the pipeline
            import app.app  # noqa: F401
            mock_get.assert_not_called()

    def test_returns_string(self):
        """summarize() must always return a string, not None or another type."""
        with patch("app.app.get_pipeline") as mock_get:
            mock_get.return_value = make_mock_pipeline("A string summary.")
            from app.app import summarize

            result = summarize("An article about something.", 80)
            assert isinstance(result, str)
