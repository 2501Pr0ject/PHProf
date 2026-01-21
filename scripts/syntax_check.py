#!/usr/bin/env python3
"""Vérification de la syntaxe PHP dans les réponses générées."""

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class PHPCheckResult:
    """Résultat de la vérification d'un bloc PHP."""

    code: str
    valid: bool
    error: str | None = None
    line: int | None = None


def extract_php_blocks(text: str) -> list[str]:
    """Extrait les blocs de code PHP d'un texte.

    Args:
        text: Texte contenant des blocs markdown

    Returns:
        Liste des blocs de code PHP
    """
    # Pattern pour les blocs ```php
    pattern = re.compile(r'```php\s*\n(.*?)```', re.DOTALL)
    matches = pattern.findall(text)
    return [m.strip() for m in matches if m.strip()]


def check_php_syntax(code: str) -> PHPCheckResult:
    """Vérifie la syntaxe d'un bloc de code PHP.

    Args:
        code: Code PHP à vérifier

    Returns:
        Résultat de la vérification
    """
    # Ajoute <?php si nécessaire
    if not code.strip().startswith("<?"):
        code = "<?php\n" + code

    # Crée un fichier temporaire
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".php",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            ["php", "-l", temp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            return PHPCheckResult(code=code, valid=True)

        # Parse l'erreur
        error_msg = result.stdout + result.stderr
        line_match = re.search(r'on line (\d+)', error_msg)
        line_num = int(line_match.group(1)) if line_match else None

        return PHPCheckResult(
            code=code,
            valid=False,
            error=error_msg.strip(),
            line=line_num,
        )

    except subprocess.TimeoutExpired:
        return PHPCheckResult(
            code=code,
            valid=False,
            error="Timeout lors de la vérification",
        )
    except FileNotFoundError:
        return PHPCheckResult(
            code=code,
            valid=False,
            error="PHP non installé ou non trouvé dans le PATH",
        )
    finally:
        Path(temp_path).unlink(missing_ok=True)


def check_responses_file(file_path: Path) -> dict:
    """Vérifie tous les blocs PHP dans un fichier de réponses.

    Args:
        file_path: Chemin vers le fichier JSON/JSONL des réponses

    Returns:
        Statistiques de vérification
    """
    results = {
        "total_responses": 0,
        "responses_with_code": 0,
        "total_blocks": 0,
        "valid_blocks": 0,
        "invalid_blocks": 0,
        "errors": [],
    }

    # Charge les réponses
    with open(file_path, encoding="utf-8") as f:
        if file_path.suffix == ".jsonl":
            responses = [json.loads(line) for line in f if line.strip()]
        else:
            responses = json.load(f)

    for i, response in enumerate(responses):
        results["total_responses"] += 1

        # Extrait le texte de la réponse
        if isinstance(response, dict):
            text = response.get("response", response.get("output", ""))
        else:
            text = str(response)

        blocks = extract_php_blocks(text)

        if blocks:
            results["responses_with_code"] += 1

        for j, block in enumerate(blocks):
            results["total_blocks"] += 1
            check_result = check_php_syntax(block)

            if check_result.valid:
                results["valid_blocks"] += 1
            else:
                results["invalid_blocks"] += 1
                results["errors"].append({
                    "response_index": i,
                    "block_index": j,
                    "error": check_result.error,
                    "code_preview": block[:200] + "..." if len(block) > 200 else block,
                })

    return results


def main() -> None:
    """Point d'entrée principal."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Vérification de la syntaxe PHP dans les réponses"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Fichier JSON/JSONL contenant les réponses",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Fichier de sortie pour le rapport (optionnel)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Affiche les détails des erreurs",
    )

    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Fichier introuvable: {args.input_file}")
        sys.exit(1)

    print(f"Vérification de {args.input_file}...\n")

    results = check_responses_file(args.input_file)

    # Affichage des résultats
    print("=" * 50)
    print("RÉSULTATS DE LA VÉRIFICATION SYNTAXIQUE PHP")
    print("=" * 50)
    print(f"Réponses analysées:      {results['total_responses']}")
    print(f"Réponses avec code:      {results['responses_with_code']}")
    print(f"Blocs PHP total:         {results['total_blocks']}")
    print(f"Blocs valides:           {results['valid_blocks']}")
    print(f"Blocs invalides:         {results['invalid_blocks']}")

    if results["total_blocks"] > 0:
        rate = results["valid_blocks"] / results["total_blocks"] * 100
        print(f"\nTaux de syntaxe valide: {rate:.1f}%")

    # Affichage des erreurs en mode verbose
    if args.verbose and results["errors"]:
        print("\n" + "=" * 50)
        print("DÉTAIL DES ERREURS")
        print("=" * 50)
        for error in results["errors"]:
            print(f"\nRéponse #{error['response_index']}, Bloc #{error['block_index']}:")
            print(f"  Erreur: {error['error']}")
            print(f"  Code: {error['code_preview']}")

    # Sauvegarde du rapport
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nRapport sauvegardé: {args.output}")


if __name__ == "__main__":
    main()
