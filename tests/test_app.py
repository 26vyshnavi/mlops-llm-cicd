"""
test_app.py — Tests for the Gradio app entrypoint.

FIX F401: removed unused `import pytest`
FIX I001: imports sorted alphabetically
"""

from unittest.mock import MagicMock, patch


def make_mock_pipeline(summary_return="Mock summary."):
    mock = MagicMock()
    mock.summarize.return_value = summary_return
    return mock


class TestSummarizeFunction:
    def test_empty_input_returns_warning(self):
        with patch("app.app.get_pipeline") as mock_get:
            mock_get.return_value = make_mock_pipeline()
            from app.app import summarize

            result = summarize("", 80)
            assert "provide" in result.lower()
            mock_get.return_value.summarize.assert_not_called()

    def test_whitespace_input_returns_warning(self):
        with patch("app.app.get_pipeline") as mock_get:
            mock_get.return_value = make_mock_pipeline()
            from app.app import summarize

            result = summarize("    \n\t  ", 80)
            assert len(result) > 0
            mock_get.return_value.summarize.assert_not_called()

    def test_valid_input_calls_pipeline(self):
        with patch("app.app.get_pipeline") as mock_get:
            mock_pipe = make_mock_pipeline("Scientists discovered a fish.")
            mock_get.return_value = mock_pipe
            from app.app import summarize

            article = "Scientists have discovered a rare fish in the Pacific Ocean."
            result = summarize(article, 80)
            mock_pipe.summarize.assert_called_once_with(article, max_new_tokens=80)
            assert result == "Scientists discovered a fish."

    def test_max_tokens_forwarded(self):
        with patch("app.app.get_pipeline") as mock_get:
            mock_pipe = make_mock_pipeline()
            mock_get.return_value = mock_pipe
            from app.app import summarize

            summarize("Some long article text here.", 150)
            call_kwargs = mock_pipe.summarize.call_args[1]
            assert call_kwargs["max_new_tokens"] == 150

    def test_pipeline_loaded_lazily(self):
        with patch("app.app.get_pipeline") as mock_get:
            mock_get.return_value = make_mock_pipeline()
            import app.app  # noqa: F401

            mock_get.assert_not_called()

    def test_returns_string(self):
        with patch("app.app.get_pipeline") as mock_get:
            mock_get.return_value = make_mock_pipeline("A string summary.")
            from app.app import summarize

            result = summarize("An article about something.", 80)
            assert isinstance(result, str)
