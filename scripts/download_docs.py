#!/usr/bin/env python3
"""Téléchargement de la documentation pour le RAG PHProf."""

import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_project_root, load_config


def download_git_repo(
    url: str,
    local_path: Path,
    branch: str | None = None,
) -> bool:
    """Clone ou met à jour un dépôt Git.

    Args:
        url: URL du dépôt
        local_path: Chemin local de destination
        branch: Branche à cloner (optionnel)

    Returns:
        True si succès, False sinon
    """
    if local_path.exists():
        print(f"  Mise à jour de {local_path.name}...")
        try:
            subprocess.run(
                ["git", "pull", "--ff-only"],
                cwd=local_path,
                check=True,
                capture_output=True,
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"  [!] Erreur git pull: {e.stderr.decode()}")
            return False
    else:
        print(f"  Clonage de {url}...")
        local_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = ["git", "clone", "--depth", "1"]
        if branch:
            cmd.extend(["-b", branch])
        cmd.extend([url, str(local_path)])

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Erreur git clone: {e.stderr.decode()}")
            return False


def group_sources_by_repo(sources: dict) -> dict:
    """Regroupe les sources qui utilisent le même repository.

    Certaines sources (ex: symfony_docs, symfony_messenger, symfony_security)
    pointent vers le même repo. On ne le télécharge qu'une fois.

    Args:
        sources: Dict des sources depuis la config

    Returns:
        Dict groupé par URL de repo
    """
    repo_groups = defaultdict(list)

    for source_id, source_config in sources.items():
        url = source_config.get("url", "")
        repo_groups[url].append((source_id, source_config))

    return repo_groups


def create_symlinks_for_shared_repos(
    repo_groups: dict,
    project_root: Path,
) -> dict:
    """Crée des symlinks pour les sources partageant un repo.

    Args:
        repo_groups: Sources groupées par URL
        project_root: Racine du projet

    Returns:
        Dict mapping source_id -> actual_path
    """
    source_paths = {}

    for url, sources in repo_groups.items():
        if len(sources) == 1:
            # Une seule source pour ce repo
            source_id, config = sources[0]
            source_paths[source_id] = project_root / config["local_path"]
        else:
            # Plusieurs sources partagent ce repo
            # On utilise le premier comme référence
            primary_id, primary_config = sources[0]
            primary_path = project_root / primary_config["local_path"]
            source_paths[primary_id] = primary_path

            # Les autres pointent vers le même path (pas de symlink nécessaire,
            # le script build_index utilisera le bon chemin)
            for source_id, config in sources[1:]:
                # On garde le path configuré, build_index gérera les patterns
                source_paths[source_id] = primary_path

    return source_paths


def main() -> None:
    """Point d'entrée principal."""
    print("PHProf - Téléchargement de la documentation\n")

    config = load_config("rag_config")
    project_root = get_project_root()
    sources = config.get("sources", {})

    # Groupe les sources par repo pour éviter les téléchargements en double
    repo_groups = group_sources_by_repo(sources)

    print(f"{len(sources)} sources configurées, {len(repo_groups)} repos uniques\n")

    success_count = 0
    total_count = 0
    downloaded_repos = set()

    # Trie par priorité (les sources priorité 1 en premier)
    sorted_sources = sorted(
        sources.items(),
        key=lambda x: x[1].get("priority", 99)
    )

    for source_id, source_config in sorted_sources:
        source_name = source_config["name"]
        source_type = source_config.get("type", "git")
        url = source_config.get("url", "")
        priority = source_config.get("priority", 99)

        print(f"[{priority}] {source_name}")

        if source_type != "git":
            print(f"  [!] Type non supporté: {source_type}")
            continue

        # Vérifie si ce repo a déjà été téléchargé
        if url in downloaded_repos:
            print(f"  -> Repo déjà téléchargé (partagé)")
            success_count += 1
            total_count += 1
            continue

        total_count += 1
        local_path = project_root / source_config["local_path"]
        branch = source_config.get("branch")

        if download_git_repo(url, local_path, branch):
            success_count += 1
            downloaded_repos.add(url)

            # Compte les fichiers
            patterns = source_config.get("include_patterns", ["*.md", "*.rst"])
            file_count = 0
            for pattern in patterns:
                file_count += len(list(local_path.rglob(pattern)))

            print(f"  -> {file_count} fichiers trouvés")
        else:
            print(f"  [X] Échec du téléchargement")

    print(f"\n{'='*60}")
    print(f"RÉSUMÉ")
    print(f"{'='*60}")
    print(f"Sources configurées: {len(sources)}")
    print(f"Repos uniques:       {len(repo_groups)}")
    print(f"Téléchargements:     {success_count}/{total_count}")

    # Affiche la taille totale
    docs_dir = project_root / "rag" / "docs"
    if docs_dir.exists():
        total_size = sum(f.stat().st_size for f in docs_dir.rglob("*") if f.is_file())
        print(f"Taille totale:       {total_size / 1024 / 1024:.1f} MB")

    if success_count < total_count:
        print("\n[!] Certaines sources n'ont pas pu être téléchargées")
        sys.exit(1)
    else:
        print("\nToutes les sources ont été téléchargées!")
        print("\nProchaine étape: make build-index")


if __name__ == "__main__":
    main()
