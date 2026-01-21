"""Gestion de la configuration PHProf."""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_name: str) -> dict[str, Any]:
    """Charge un fichier de configuration YAML.

    Args:
        config_name: Nom du fichier de config (sans extension)

    Returns:
        Dictionnaire de configuration

    Raises:
        FileNotFoundError: Si le fichier n'existe pas
    """
    config_path = get_project_root() / "configs" / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration introuvable: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """Retourne le chemin racine du projet."""
    return Path(__file__).parent.parent


def get_model_path(finetuned: bool = False) -> Path:
    """Retourne le chemin vers le modèle GGUF.

    Args:
        finetuned: Si True, retourne le modèle fine-tuné

    Returns:
        Chemin vers le fichier .gguf
    """
    config = load_config("model_config")
    models_dir = get_project_root() / config["paths"]["models_dir"]

    if finetuned:
        return models_dir / config["model"]["finetuned_file"]
    return models_dir / config["model"]["file"]


def get_llama_cli_path() -> Path:
    """Retourne le chemin vers llama-cli."""
    config = load_config("model_config")
    return get_project_root() / config["paths"]["llama_cli"]


def get_index_path() -> Path:
    """Retourne le chemin vers l'index FAISS."""
    config = load_config("rag_config")
    return get_project_root() / config["index"]["path"]


def get_metadata_path() -> Path:
    """Retourne le chemin vers les métadonnées de l'index."""
    config = load_config("rag_config")
    return get_project_root() / config["index"]["metadata_path"]
