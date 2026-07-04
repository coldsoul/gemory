"""Unit tests for :mod:`gemory.llm`.

All API calls are mocked -- no real network requests are made.
"""

from unittest.mock import MagicMock, patch

import pytest
from openai import AuthenticationError

from gemory.llm import embed, embed_batch, extract_facts


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_chat_client():
    """Patch ``openai.OpenAI`` in :mod:`gemory.llm` and yield the mock client
    instance returned by the constructor."""
    with patch("gemory.llm.openai.OpenAI") as mock_cls:
        client = mock_cls.return_value
        yield client


@pytest.fixture
def mock_embed_client():
    """Same as *mock_chat_client* but for embedding endpoints."""
    with patch("gemory.llm.openai.OpenAI") as mock_cls:
        client = mock_cls.return_value
        yield client


# ---------------------------------------------------------------------------
# extract_facts
# ---------------------------------------------------------------------------

class TestExtractFacts:
    """Tests for :func:`gemory.llm.extract_facts`."""

    def test_extract_facts_returns_list(self, mock_chat_client) -> None:
        """A standard JSON array response is parsed correctly."""
        mock_chat_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='["the user likes Python", "the user uses uv"]',
            ))],
        )

        result = extract_facts("I really like Python and use uv for projects.")
        assert result == ["the user likes Python", "the user uses uv"]

    def test_extract_facts_empty(self, mock_chat_client) -> None:
        """``[]`` is handled correctly -- returns an empty list."""
        mock_chat_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="[]"))],
        )

        result = extract_facts("Just a greeting, nothing to store.")
        assert result == []

    def test_extract_facts_strips_code_fences(self, mock_chat_client) -> None:
        """Response wrapped in ```json … ``` is still parsed."""
        mock_chat_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="```json\n[\"fact one\", \"fact two\"]\n```",
            ))],
        )

        result = extract_facts("Some transcript.")
        assert result == ["fact one", "fact two"]

    def test_extract_facts_strips_preamble(self, mock_chat_client) -> None:
        """Text before the JSON array (preamble) is discarded."""
        mock_chat_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content=(
                    "Here are the extracted facts:\n"
                    '["the user works remotely"]'
                ),
            ))],
        )

        result = extract_facts("I work from home.")
        assert result == ["the user works remotely"]

    def test_extract_facts_parse_failure_raises(self, mock_chat_client) -> None:
        """When the response contains no parseable JSON array, ValueError."""
        mock_chat_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="This is not JSON at all. Sorry.",
            ))],
        )

        with pytest.raises(ValueError, match="Could not parse"):
            extract_facts("some transcript")

    def test_extract_facts_auth_error_propagates(
        self, mock_chat_client,
    ) -> None:
        """Authentication errors from the API propagate unchanged."""
        mock_chat_client.chat.completions.create.side_effect = (
            AuthenticationError(
                "Incorrect API key",
                response=MagicMock(status_code=401),
                body=None,
            )
        )

        with pytest.raises(AuthenticationError):
            extract_facts("some transcript")


# ---------------------------------------------------------------------------
# embed / embed_batch
# ---------------------------------------------------------------------------

class TestEmbed:
    """Tests for :func:`gemory.llm.embed`."""

    def test_embed_returns_floats(self, mock_embed_client) -> None:
        """Single text embedding returns a list of floats."""
        mock_embed_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1, 0.2, 0.3], index=0)],
        )

        result = embed("hello world")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)
        assert result == [0.1, 0.2, 0.3]


class TestEmbedBatch:
    """Tests for :func:`gemory.llm.embed_batch`."""

    def test_embed_batch_returns_list_of_lists(self, mock_embed_client) -> None:
        """Batch embedding returns a list of vectors."""
        mock_embed_client.embeddings.create.return_value = MagicMock(
            data=[
                MagicMock(embedding=[0.1, 0.2], index=0),
                MagicMock(embedding=[0.3, 0.4], index=1),
            ],
        )

        result = embed_batch(["hello", "world"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]

    def test_embed_batch_handles_out_of_order_indices(
        self, mock_embed_client,
    ) -> None:
        """If the API returns results out of order, we sort by index."""
        mock_embed_client.embeddings.create.return_value = MagicMock(
            data=[
                MagicMock(embedding=[9.9, 8.8], index=1),
                MagicMock(embedding=[1.1, 2.2], index=0),
            ],
        )

        result = embed_batch(["first", "second"])
        assert result[0] == [1.1, 2.2]
        assert result[1] == [9.9, 8.8]

    def test_embed_batch_empty_input(self, mock_embed_client) -> None:
        """Empty input list returns an empty list without an API call."""
        result = embed_batch([])
        assert result == []
        mock_embed_client.embeddings.create.assert_not_called()
