#!/usr/bin/env python3
"""Validation du dataset PHProf."""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_project_root, load_config


def check_french_language(text: str) -> bool:
    """Vérifie que le texte est principalement en français."""
    french_keywords = [
        "le", "la", "les", "un", "une", "des", "est", "sont", "dans",
        "pour", "avec", "que", "qui", "cette", "donc", "mais", "car",
    ]
    words = text.lower().split()
    if not words:
        return False

    french_count = sum(1 for word in words if word in french_keywords)
    return french_count >= len(words) * 0.05


def check_code_blocks(text: str) -> bool:
    """Vérifie la présence de blocs de code."""
    return "```" in text


def check_structured_format(text: str) -> list[str]:
    """Vérifie le format structuré et retourne les sections manquantes."""
    sections = ["TL;DR", "Problème", "Solution", "Explication", "À retenir"]
    missing = [s for s in sections if s not in text]
    return missing


def validate_example(example: dict, config: dict) -> list[str]:
    """Valide un exemple et retourne les erreurs.

    Args:
        example: Exemple à valider
        config: Configuration de validation

    Returns:
        Liste des erreurs trouvées
    """
    errors = []
    validation_config = config["validation"]

    # Vérifie les champs requis
    if "messages" not in example:
        errors.append("Champ 'messages' manquant")
        return errors

    messages = example["messages"]

    # Vérifie qu'il y a au moins user + assistant
    roles = [m.get("role") for m in messages]
    if "user" not in roles:
        errors.append("Message 'user' manquant")
    if "assistant" not in roles:
        errors.append("Message 'assistant' manquant")

    # Récupère la réponse assistant
    assistant_msg = next(
        (m["content"] for m in messages if m.get("role") == "assistant"),
        ""
    )

    if not assistant_msg:
        errors.append("Réponse assistant vide")
        return errors

    # Vérifie la longueur
    min_len = validation_config.get("min_response_length", 200)
    max_len = validation_config.get("max_response_length", 3000)

    if len(assistant_msg) < min_len:
        errors.append(f"Réponse trop courte ({len(assistant_msg)} < {min_len})")
    if len(assistant_msg) > max_len:
        errors.append(f"Réponse trop longue ({len(assistant_msg)} > {max_len})")

    # Vérifie la langue française
    if "french_language" in validation_config.get("quality_checks", []):
        if not check_french_language(assistant_msg):
            errors.append("Réponse pas en français")

    # Vérifie les blocs de code
    if "code_blocks_present" in validation_config.get("quality_checks", []):
        if not check_code_blocks(assistant_msg):
            errors.append("Pas de bloc de code")

    # Vérifie le format structuré
    if "structured_format" in validation_config.get("quality_checks", []):
        missing = check_structured_format(assistant_msg)
        if missing:
            errors.append(f"Sections manquantes: {', '.join(missing)}")

    # Vérifie les métadonnées
    if "metadata" in validation_config.get("required_fields", []):
        if "metadata" not in example:
            errors.append("Métadonnées manquantes")
        else:
            metadata = example["metadata"]
            for field in validation_config.get("metadata_fields", []):
                if field not in metadata:
                    errors.append(f"Métadonnée '{field}' manquante")

    return errors


def validate_file(file_path: Path, config: dict) -> dict:
    """Valide un fichier de dataset.

    Args:
        file_path: Chemin vers le fichier JSONL
        config: Configuration de validation

    Returns:
        Résultats de validation
    """
    results = {
        "file": str(file_path),
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "errors": [],
    }

    with open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue

            results["total"] += 1

            try:
                example = json.loads(line)
            except json.JSONDecodeError as e:
                results["invalid"] += 1
                results["errors"].append({
                    "line": i + 1,
                    "errors": [f"JSON invalide: {e}"],
                })
                continue

            errors = validate_example(example, config)

            if errors:
                results["invalid"] += 1
                results["errors"].append({
                    "line": i + 1,
                    "errors": errors,
                })
            else:
                results["valid"] += 1

    return results


def main() -> None:
    """Point d'entrée principal."""
    import argparse

    parser = argparse.ArgumentParser(description="Validation du dataset PHProf")
    parser.add_argument(
        "files",
        type=Path,
        nargs="*",
        help="Fichiers à valider (défaut: data/mlx_train/*.jsonl)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Affiche les détails des erreurs",
    )

    args = parser.parse_args()

    print("PHProf - Validation du dataset\n")

    config = load_config("dataset_config")
    project_root = get_project_root()

    # Détermine les fichiers à valider
    if args.files:
        files = args.files
    else:
        train_dir = project_root / config["dataset"]["train_dir"]
        files = list(train_dir.glob("*.jsonl"))

    if not files:
        print("Aucun fichier à valider")
        sys.exit(1)

    total_valid = 0
    total_invalid = 0
    all_errors = []

    for file_path in files:
        if not file_path.exists():
            print(f"Fichier introuvable: {file_path}")
            continue

        print(f"Validation de {file_path.name}...")
        results = validate_file(file_path, config)

        total_valid += results["valid"]
        total_invalid += results["invalid"]
        all_errors.extend(results["errors"])

        print(f"   ->{results['valid']} valides, [X]{results['invalid']} invalides")

    print("\n" + "=" * 50)
    print("RÉSULTATS DE VALIDATION")
    print("=" * 50)
    print(f"Total: {total_valid + total_invalid} exemples")
    print(f"Valides: {total_valid}")
    print(f"Invalides: {total_invalid}")

    if total_valid + total_invalid > 0:
        rate = total_valid / (total_valid + total_invalid) * 100
        print(f"Taux de validité: {rate:.1f}%")

    if args.verbose and all_errors:
        print("\nDÉTAIL DES ERREURS:")
        for error in all_errors[:20]:  # Limite à 20
            print(f"  Ligne {error['line']}:")
            for e in error["errors"]:
                print(f"    - {e}")

        if len(all_errors) > 20:
            print(f"  ... et {len(all_errors) - 20} autres erreurs")

    if total_invalid > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
