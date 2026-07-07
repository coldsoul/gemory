"""Tests for :func:`src.llm.summarize_cluster`.

All API calls are mocked -- no real network requests are made.
"""

from unittest.mock import MagicMock

import pytest
from openai import AuthenticationError

from src.llm import summarize_cluster


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_openai(monkeypatch, return_content: str):
    """Patch ``gemory.llm.openai.OpenAI`` so that
    ``chat.completions.create`` returns *return_content*."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=return_content))],
    )
    monkeypatch.setattr("src.llm.openai.OpenAI", lambda **kw: mock_client)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSummarizeCluster:
    """Tests for :func:`src.llm.summarize_cluster`."""

    def test_summarize_cluster_returns_dict(self, monkeypatch) -> None:
        """A valid JSON response is parsed into a dict with label and summary."""
        _mock_openai(
            monkeypatch,
            '{"label": "Python development", "summary": "The user works with Python and builds projects with it."}',
        )

        result = summarize_cluster(["fact one", "fact two"])
        assert isinstance(result, dict)
        assert "label" in result
        assert "summary" in result
        assert result["label"] == "Python development"
        assert "Python" in result["summary"]

    def test_summarize_cluster_strips_code_fences(self, monkeypatch) -> None:
        """Response wrapped in ```json … ``` is still parsed."""
        _mock_openai(
            monkeypatch,
            "```json\n{\"label\": \"Theme\", \"summary\": \"A summary.\"}\n```",
        )

        result = summarize_cluster(["fact"])
        assert result["label"] == "Theme"
        assert result["summary"] == "A summary."

    def test_summarize_cluster_parse_failure_raises(self, monkeypatch) -> None:
        """When the response contains no parseable JSON, ValueError."""
        _mock_openai(monkeypatch, "This is not JSON at all.")

        with pytest.raises(ValueError, match="Could not parse"):
            summarize_cluster(["fact"])

    def test_summarize_cluster_auth_error_propagates(self, monkeypatch) -> None:
        """Authentication errors from the API propagate unchanged."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = AuthenticationError(
            "Incorrect API key",
            response=MagicMock(status_code=401),
            body=None,
        )
        monkeypatch.setattr("src.llm.openai.OpenAI", lambda **kw: mock_client)

        with pytest.raises(AuthenticationError):
            summarize_cluster(["fact"])
