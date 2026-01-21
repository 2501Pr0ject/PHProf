"""Tests pour le module de configuration."""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_project_root, load_config


class TestConfig:
    """Tests pour les fonctions de configuration."""

    def test_get_project_root(self):
        """Test que get_project_root retourne le bon chemin."""
        root = get_project_root()
        assert root.exists()
        assert (root / "app").exists()
        assert (root / "configs").exists()

    def test_load_config_model(self):
        """Test le chargement de la config modèle."""
        config = load_config("model_config")
        assert "model" in config
        assert "inference" in config
        assert "prompt" in config
        assert config["model"]["name"] == "qwen2.5-coder-1.5b-instruct"

    def test_load_config_rag(self):
        """Test le chargement de la config RAG."""
        config = load_config("rag_config")
        assert "sources" in config
        assert "embeddings" in config
        assert "index" in config
        assert "retrieval" in config

    def test_load_config_dataset(self):
        """Test le chargement de la config dataset."""
        config = load_config("dataset_config")
        assert "dataset" in config
        assert "categories" in config
        assert "topics" in config

    def test_load_config_invalid(self):
        """Test qu'une config invalide lève une exception."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config")
