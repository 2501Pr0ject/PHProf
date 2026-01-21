#!/usr/bin/env python3
"""Évaluation automatique des réponses PHProf."""

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_llama_cli_path, get_model_path, get_project_root, load_config


def load_prompts(prompts_file: Path) -> list[dict[str, Any]]:
    """Charge les prompts d'évaluation.

    Args:
        prompts_file: Chemin vers le fichier JSONL

    Returns:
        Liste des prompts avec métadonnées
    """
    prompts = []
    with open(prompts_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    return prompts


def run_inference(prompt: str, model_path: Path, config: dict[str, Any]) -> str:
    """Exécute l'inférence pour un prompt donné."""
    llama_cli = get_llama_cli_path()
    inference_config = config["inference"]

    # Construit le prompt complet
    full_prompt = config["prompt"]["template"].format(
        system=config["prompt"]["system"],
        user=prompt,
    )

    cmd = [
        str(llama_cli),
        "-m", str(model_path),
        "-p", full_prompt,
        "-n", str(inference_config["n_predict"]),
        "-c", str(inference_config["n_ctx"]),
        "--temp", str(inference_config["temperature"]),
        "--top-p", str(inference_config["top_p"]),
        "--top-k", str(inference_config["top_k"]),
        "--repeat-penalty", str(inference_config["repeat_penalty"]),
        "-ngl", str(inference_config["n_gpu_layers"]),
        "-s", str(inference_config["seed"]),
        "--no-display-prompt",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        cwd=get_project_root(),
    )

    return result.stdout.strip()


def score_language(response: str) -> float:
    """Évalue si la réponse est en français.

    Returns:
        Score entre 0 et 1
    """
    # Mots-clés français courants
    french_keywords = [
        "le", "la", "les", "un", "une", "des", "est", "sont", "dans",
        "pour", "avec", "que", "qui", "ce", "cette", "voici", "donc",
        "mais", "ou", "et", "car", "parce", "comme", "ainsi", "très",
        "peut", "faire", "être", "avoir", "fonction", "méthode", "classe",
        "permet", "utilise", "exemple", "code", "erreur", "solution",
    ]

    words = response.lower().split()
    if not words:
        return 0.0

    french_count = sum(1 for word in words if word in french_keywords)
    return min(1.0, french_count / (len(words) * 0.1))


def score_code_blocks(response: str) -> float:
    """Évalue la présence de blocs de code PHP.

    Returns:
        Score entre 0 et 1
    """
    php_blocks = re.findall(r'```php\s*\n.*?```', response, re.DOTALL)
    code_blocks = re.findall(r'```\w*\s*\n.*?```', response, re.DOTALL)

    if not code_blocks:
        return 0.0

    # Bonus si les blocs sont PHP
    php_ratio = len(php_blocks) / len(code_blocks) if code_blocks else 0
    return 0.5 + (0.5 * php_ratio)


def score_topic_coverage(
    response: str,
    expected_concepts: list[str],
) -> float:
    """Évalue la couverture des concepts attendus.

    Args:
        response: Réponse générée
        expected_concepts: Liste des concepts à couvrir

    Returns:
        Score entre 0 et 1
    """
    if not expected_concepts:
        return 1.0

    response_lower = response.lower()
    found = sum(
        1 for concept in expected_concepts
        if concept.lower() in response_lower
    )

    return found / len(expected_concepts)


def score_response_length(response: str) -> float:
    """Évalue la longueur de la réponse.

    Returns:
        Score entre 0 et 1 (optimal: 200-800 mots)
    """
    word_count = len(response.split())

    if word_count < 50:
        return 0.2
    elif word_count < 100:
        return 0.5
    elif word_count < 200:
        return 0.8
    elif word_count <= 800:
        return 1.0
    elif word_count <= 1200:
        return 0.8
    else:
        return 0.5


def score_structured_format(response: str) -> float:
    """Évalue le respect du format structuré.

    Returns:
        Score bonus entre 0 et 0.2
    """
    sections = ["TL;DR", "Problème", "Solution", "Explication", "À retenir"]
    found = sum(1 for section in sections if section in response)
    return (found / len(sections)) * 0.2


def evaluate_response(
    response: str,
    prompt_data: dict[str, Any],
) -> dict[str, float]:
    """Évalue une réponse sur tous les critères.

    Args:
        response: Réponse générée
        prompt_data: Données du prompt (avec concepts attendus)

    Returns:
        Dictionnaire des scores
    """
    expected_concepts = prompt_data.get("expected_concepts", [])

    scores = {
        "language": score_language(response),
        "code_blocks": score_code_blocks(response),
        "topic_coverage": score_topic_coverage(response, expected_concepts),
        "response_length": score_response_length(response),
    }

    # Score bonus pour le format
    format_bonus = score_structured_format(response)

    # Score global (moyenne pondérée)
    weights = {"language": 0.25, "code_blocks": 0.25, "topic_coverage": 0.25, "response_length": 0.25}
    weighted_score = sum(scores[k] * weights[k] for k in weights)
    scores["total"] = min(1.0, weighted_score + format_bonus)
    scores["format_bonus"] = format_bonus

    return scores


def run_evaluation(
    prompts_file: Path,
    output_file: Path,
    use_finetuned: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Exécute l'évaluation complète.

    Args:
        prompts_file: Fichier des prompts d'évaluation
        output_file: Fichier de sortie pour les résultats
        use_finetuned: Utiliser le modèle fine-tuné
        limit: Limite le nombre de prompts (pour les tests)

    Returns:
        Résultats agrégés
    """
    config = load_config("model_config")
    model_path = get_model_path(finetuned=use_finetuned)

    prompts = load_prompts(prompts_file)
    if limit:
        prompts = prompts[:limit]

    print(f"Évaluation de {len(prompts)} prompts")
    print(f"Modèle: {model_path.name}")

    results = {
        "model": model_path.name,
        "timestamp": datetime.now().isoformat(),
        "prompts_count": len(prompts),
        "responses": [],
        "scores_by_category": {},
    }

    total_scores = {
        "language": 0,
        "code_blocks": 0,
        "topic_coverage": 0,
        "response_length": 0,
        "total": 0,
    }

    for i, prompt_data in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] {prompt_data.get('category', 'unknown')}...")

        try:
            response = run_inference(prompt_data["prompt"], model_path, config)
            scores = evaluate_response(response, prompt_data)

            results["responses"].append({
                "prompt": prompt_data["prompt"],
                "category": prompt_data.get("category"),
                "response": response,
                "scores": scores,
            })

            for key in total_scores:
                total_scores[key] += scores.get(key, 0)

            # Scores par catégorie
            category = prompt_data.get("category", "unknown")
            if category not in results["scores_by_category"]:
                results["scores_by_category"][category] = {"count": 0, "total": 0}
            results["scores_by_category"][category]["count"] += 1
            results["scores_by_category"][category]["total"] += scores["total"]

        except Exception as e:
            print(f"    Erreur: {e}")
            results["responses"].append({
                "prompt": prompt_data["prompt"],
                "error": str(e),
            })

    # Calcul des moyennes
    n = len(prompts)
    results["average_scores"] = {k: v / n for k, v in total_scores.items()}

    for category in results["scores_by_category"]:
        cat_data = results["scores_by_category"][category]
        cat_data["average"] = cat_data["total"] / cat_data["count"]

    # Sauvegarde
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def main() -> None:
    """Point d'entrée principal."""
    import argparse

    parser = argparse.ArgumentParser(description="Évaluation PHProf")
    parser.add_argument(
        "-p", "--prompts",
        type=Path,
        default=get_project_root() / "eval" / "prompts_fr.jsonl",
        help="Fichier des prompts d'évaluation",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=get_project_root() / "reports" / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        help="Fichier de sortie",
    )
    parser.add_argument(
        "-f", "--finetuned",
        action="store_true",
        help="Utiliser le modèle fine-tuné",
    )
    parser.add_argument(
        "-l", "--limit",
        type=int,
        help="Limite le nombre de prompts",
    )

    args = parser.parse_args()

    if not args.prompts.exists():
        print(f"Fichier de prompts introuvable: {args.prompts}")
        sys.exit(1)

    print("PHProf - Évaluation automatique\n")

    results = run_evaluation(
        prompts_file=args.prompts,
        output_file=args.output,
        use_finetuned=args.finetuned,
        limit=args.limit,
    )

    # Affichage du résumé
    print("\n" + "=" * 50)
    print("📈 RÉSULTATS")
    print("=" * 50)

    avg = results["average_scores"]
    print(f"Score global:     {avg['total'] * 5:.2f}/5")
    print(f"Langue française: {avg['language'] * 100:.1f}%")
    print(f"Blocs de code:    {avg['code_blocks'] * 100:.1f}%")
    print(f"Couverture:       {avg['topic_coverage'] * 100:.1f}%")
    print(f"Longueur:         {avg['response_length'] * 100:.1f}%")

    print("\nPar catégorie:")
    for cat, data in sorted(
        results["scores_by_category"].items(),
        key=lambda x: x[1]["average"],
        reverse=True,
    ):
        print(f"  {cat}: {data['average'] * 5:.2f}/5 ({data['count']} prompts)")

    print(f"\nRésultats sauvegardés: {args.output}")


if __name__ == "__main__":
    main()
