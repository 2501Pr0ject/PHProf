#!/usr/bin/env python3
"""Construction du dataset d'entraînement PHProf."""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_project_root, load_config


def load_raw_examples(raw_dir: Path) -> list[dict]:
    """Charge tous les exemples bruts.

    Args:
        raw_dir: Répertoire contenant les fichiers JSON/JSONL

    Returns:
        Liste des exemples
    """
    examples = []

    for file_path in raw_dir.glob("*.json"):
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                examples.extend(data)
            else:
                examples.append(data)

    for file_path in raw_dir.glob("*.jsonl"):
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))

    return examples


def convert_to_chat_format(example: dict) -> dict:
    """Convertit un exemple au format chat mlx-lm.

    Args:
        example: Exemple avec 'input' et 'output' ou 'messages'

    Returns:
        Exemple au format chat
    """
    # Si déjà au format messages, retourne tel quel
    if "messages" in example:
        return example

    # Convertit depuis input/output
    messages = []

    if "system" in example:
        messages.append({"role": "system", "content": example["system"]})

    if "input" in example:
        messages.append({"role": "user", "content": example["input"]})

    if "output" in example:
        messages.append({"role": "assistant", "content": example["output"]})

    return {
        "messages": messages,
        "metadata": example.get("metadata", {}),
    }


def split_dataset(
    examples: list[dict],
    train_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Divise le dataset en train/validation.

    Args:
        examples: Liste des exemples
        train_ratio: Proportion pour l'entraînement
        seed: Graine aléatoire

    Returns:
        Tuple (train, valid)
    """
    random.seed(seed)
    shuffled = examples.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def save_jsonl(examples: list[dict], file_path: Path) -> None:
    """Sauvegarde les exemples au format JSONL.

    Args:
        examples: Liste des exemples
        file_path: Chemin du fichier de sortie
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def main() -> None:
    """Point d'entrée principal."""
    print("PHProf - Construction du dataset\n")

    config = load_config("dataset_config")
    project_root = get_project_root()
    dataset_config = config["dataset"]

    raw_dir = project_root / dataset_config["raw_dir"]
    train_dir = project_root / dataset_config["train_dir"]

    # Charge les exemples bruts
    print(f"Chargement depuis {raw_dir}...")
    examples = load_raw_examples(raw_dir)

    if not examples:
        print("Aucun exemple trouvé dans data/raw/")
        print("   Ajoutez des fichiers JSON/JSONL avec vos exemples.")
        sys.exit(1)

    print(f"   {len(examples)} exemples chargés")

    # Convertit au format chat
    print("Conversion au format chat...")
    chat_examples = [convert_to_chat_format(ex) for ex in examples]

    # Split train/valid
    train_ratio = dataset_config["train_ratio"]
    seed = dataset_config["seed"]

    train_examples, valid_examples = split_dataset(
        chat_examples, train_ratio, seed
    )

    print(f"   Train: {len(train_examples)} exemples")
    print(f"   Valid: {len(valid_examples)} exemples")

    # Sauvegarde
    train_file = train_dir / dataset_config["train_file"]
    valid_file = train_dir / dataset_config["valid_file"]

    save_jsonl(train_examples, train_file)
    save_jsonl(valid_examples, valid_file)

    print(f"\nDataset créé!")
    print(f"   {train_file}")
    print(f"   {valid_file}")

    # Statistiques par catégorie
    categories = {}
    for ex in chat_examples:
        cat = ex.get("metadata", {}).get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print("\nRépartition par catégorie:")
    for cat, count in sorted(categories.items()):
        pct = count / len(chat_examples) * 100
        print(f"   {cat}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
