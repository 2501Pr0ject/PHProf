"""Tests pour le module RAG."""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag import RetrievedChunk, is_index_available


class TestRetrievedChunk:
    """Tests pour la classe RetrievedChunk."""

    def test_retrieved_chunk_creation(self):
        """Test la création d'un chunk."""
        chunk = RetrievedChunk(
            content="Test content",
            source="Test Source",
            heading="Test Heading",
            score=0.85,
            url="https://example.com",
        )
        assert chunk.content == "Test content"
        assert chunk.source == "Test Source"
        assert chunk.heading == "Test Heading"
        assert chunk.score == 0.85
        assert chunk.url == "https://example.com"

    def test_retrieved_chunk_optional_url(self):
        """Test qu'un chunk peut être créé sans URL."""
        chunk = RetrievedChunk(
            content="Test",
            source="Source",
            heading="Heading",
            score=0.5,
        )
        assert chunk.url is None


class TestRAGAvailability:
    """Tests pour la disponibilité du RAG."""

    def test_is_index_available_returns_bool(self):
        """Test que is_index_available retourne un booléen."""
        result = is_index_available()
        assert isinstance(result, bool)
