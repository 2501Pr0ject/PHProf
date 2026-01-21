#!/usr/bin/env python3
"""Construction de l'index FAISS pour le RAG PHProf."""

import argparse
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Ajoute le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_project_root, load_config


def parse_rst_sections(content: str, source: str, file_path: Path) -> list[dict[str, Any]]:
    """Parse les sections d'un fichier RST.

    Args:
        content: Contenu du fichier RST
        source: Nom de la source
        file_path: Chemin du fichier

    Returns:
        Liste de chunks avec métadonnées
    """
    chunks = []

    # Pattern pour les titres RST (soulignés par =, -, ~, etc.)
    title_pattern = re.compile(r'^(.+)\n([=\-~^"\'`]+)\s*$', re.MULTILINE)

    # Trouve tous les titres
    matches = list(title_pattern.finditer(content))

    if not matches:
        # Pas de sections, retourne le contenu entier
        if content.strip():
            chunks.append({
                "content": content.strip(),
                "source": source,
                "heading": file_path.stem,
                "path": str(file_path),
            })
        return chunks

    # Découpe par sections
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)

        section_content = content[start:end].strip()
        if section_content:
            chunks.append({
                "content": section_content,
                "source": source,
                "heading": title,
                "path": str(file_path),
            })

    return chunks


def parse_markdown_sections(
    content: str, source: str, file_path: Path
) -> list[dict[str, Any]]:
    """Parse les sections d'un fichier Markdown/MDX.

    Args:
        content: Contenu du fichier Markdown
        source: Nom de la source
        file_path: Chemin du fichier

    Returns:
        Liste de chunks avec métadonnées
    """
    chunks = []

    # Nettoie le contenu MDX (retire les imports et composants React)
    content = clean_mdx_content(content)

    # Pattern pour les titres Markdown (# ## ###)
    title_pattern = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)

    matches = list(title_pattern.finditer(content))

    if not matches:
        if content.strip():
            chunks.append({
                "content": content.strip(),
                "source": source,
                "heading": file_path.stem,
                "path": str(file_path),
            })
        return chunks

    for i, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)

        section_content = content[start:end].strip()
        if section_content:
            # Génère un anchor pour l'URL
            anchor = re.sub(r'[^\w\s-]', '', title.lower())
            anchor = re.sub(r'[\s_]+', '-', anchor)

            chunks.append({
                "content": section_content,
                "source": source,
                "heading": title,
                "path": str(file_path),
                "anchor": anchor,
            })

    return chunks


def clean_mdx_content(content: str) -> str:
    """Nettoie le contenu MDX pour ne garder que le texte.

    Args:
        content: Contenu MDX brut

    Returns:
        Contenu nettoyé
    """
    # Retire les imports
    content = re.sub(r'^import\s+.*$', '', content, flags=re.MULTILINE)

    # Retire les exports
    content = re.sub(r'^export\s+.*$', '', content, flags=re.MULTILINE)

    # Retire les composants JSX auto-fermants <Component />
    content = re.sub(r'<[A-Z][a-zA-Z]*\s*[^>]*/>', '', content)

    # Retire les blocs de composants <Component>...</Component>
    content = re.sub(r'<[A-Z][a-zA-Z]*[^>]*>.*?</[A-Z][a-zA-Z]*>', '', content, flags=re.DOTALL)

    # Retire le frontmatter YAML
    content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)

    return content.strip()


def parse_php_xml_sections(
    content: str, source: str, file_path: Path
) -> list[dict[str, Any]]:
    """Parse les fichiers XML de la documentation PHP.

    Args:
        content: Contenu du fichier XML
        source: Nom de la source
        file_path: Chemin du fichier

    Returns:
        Liste de chunks avec métadonnées
    """
    chunks = []

    # Extrait le titre
    title_match = re.search(r'<title>([^<]+)</title>', content)
    title = title_match.group(1) if title_match else file_path.stem

    # Nettoie le XML pour extraire le texte
    text_content = clean_xml_content(content)

    if text_content and len(text_content) > 50:
        chunks.append({
            "content": text_content,
            "source": source,
            "heading": title,
            "path": str(file_path),
        })

    return chunks


def clean_xml_content(content: str) -> str:
    """Nettoie le contenu XML PHP pour extraire le texte.

    Args:
        content: Contenu XML brut

    Returns:
        Texte extrait
    """
    # Retire les entités XML communes
    content = re.sub(r'&[a-z]+;', ' ', content)

    # Extrait le contenu des balises <para>
    paras = re.findall(r'<para[^>]*>(.*?)</para>', content, re.DOTALL)

    # Extrait le contenu des balises <simpara>
    simparas = re.findall(r'<simpara[^>]*>(.*?)</simpara>', content, re.DOTALL)

    # Extrait les descriptions de fonctions
    descriptions = re.findall(r'<refsect1[^>]*role="description"[^>]*>(.*?)</refsect1>', content, re.DOTALL)

    # Extrait les exemples
    examples = re.findall(r'<example[^>]*>(.*?)</example>', content, re.DOTALL)

    all_text = paras + simparas + descriptions + examples

    # Nettoie les balises HTML/XML restantes
    cleaned_parts = []
    for text in all_text:
        # Retire toutes les balises XML
        text = re.sub(r'<[^>]+>', ' ', text)
        # Normalise les espaces
        text = re.sub(r'\s+', ' ', text).strip()
        if text and len(text) > 20:
            cleaned_parts.append(text)

    return '\n\n'.join(cleaned_parts)


def chunk_text(
    text: str,
    min_tokens: int,
    max_tokens: int,
    target_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Découpe un texte en chunks avec overlap.

    Args:
        text: Texte à découper
        min_tokens: Taille minimum (en mots approximatifs)
        max_tokens: Taille maximum
        target_tokens: Taille cible
        overlap_tokens: Nombre de tokens de chevauchement

    Returns:
        Liste de chunks
    """
    words = text.split()

    if len(words) <= max_tokens:
        return [text]

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + target_tokens, len(words))

        # Essaie de couper à une phrase
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        # Cherche la dernière phrase complète
        last_period = max(
            chunk_text.rfind(". "),
            chunk_text.rfind(".\n"),
            chunk_text.rfind("? "),
            chunk_text.rfind("! "),
        )

        if last_period > len(chunk_text) * 0.5:
            chunk_text = chunk_text[:last_period + 1]
            actual_words = len(chunk_text.split())
        else:
            actual_words = len(chunk_words)

        if len(chunk_text.split()) >= min_tokens:
            chunks.append(chunk_text.strip())

        # Avance d'au moins 1 mot pour éviter une boucle infinie
        advance = max(1, actual_words - overlap_tokens)
        start += advance

    return chunks


def load_documents(
    config: dict[str, Any],
    sources_filter: list[str] | None = None,
    skip_sources: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Charge tous les documents des sources configurées.

    Args:
        config: Configuration RAG
        sources_filter: Liste des sources à inclure (None = toutes)
        skip_sources: Liste des sources à ignorer

    Returns:
        Liste de tous les chunks avec métadonnées
    """
    project_root = get_project_root()
    all_chunks = []
    chunking_config = config["chunking"]

    # Trie les sources par priorité
    sorted_sources = sorted(
        config["sources"].items(),
        key=lambda x: x[1].get("priority", 99)
    )

    # Filtre les sources si demandé
    if sources_filter:
        sorted_sources = [(k, v) for k, v in sorted_sources if k in sources_filter]
    if skip_sources:
        sorted_sources = [(k, v) for k, v in sorted_sources if k not in skip_sources]

    for source_id, source_config in sorted_sources:
        source_name = source_config["name"]
        local_path = project_root / source_config["local_path"]
        priority = source_config.get("priority", 99)

        if not local_path.exists():
            print(f"[!] [{priority}] {source_name} - Non trouvé: {local_path}")
            continue

        print(f"[{priority}] {source_name}...")

        # Patterns d'inclusion
        include_patterns = source_config.get("include_patterns", ["*.md", "*.rst"])
        exclude_patterns = source_config.get("exclude_patterns", [])

        # Collecte tous les fichiers d'abord pour la barre de progression
        all_files = []
        for pattern in include_patterns:
            for file_path in local_path.rglob(pattern):
                relative_path = file_path.relative_to(local_path)
                if not any(relative_path.match(exc) for exc in exclude_patterns):
                    all_files.append(file_path)

        total_files = len(all_files)
        print(f"  {total_files} fichiers à traiter", flush=True)

        file_count = 0
        chunk_count = 0

        # Traite les fichiers avec affichage de progression
        for idx, file_path in enumerate(all_files):
            if idx % 100 == 0:
                print(f"    {idx}/{total_files}...", flush=True)
            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception:
                continue

            # Ignore les fichiers vides ou trop petits
            if len(content.strip()) < 100:
                continue

            # Parse selon le type de fichier
            if file_path.suffix == ".rst":
                sections = parse_rst_sections(content, source_name, file_path)
            elif file_path.suffix == ".xml":
                sections = parse_php_xml_sections(content, source_name, file_path)
            else:  # .md, .mdx
                sections = parse_markdown_sections(content, source_name, file_path)

            # Découpe les sections trop longues
            for section in sections:
                sub_chunks = chunk_text(
                    section["content"],
                    min_tokens=chunking_config["min_tokens"],
                    max_tokens=chunking_config["max_tokens"],
                    target_tokens=chunking_config["target_tokens"],
                    overlap_tokens=chunking_config["overlap_tokens"],
                )

                for sub_chunk in sub_chunks:
                    chunk_meta = section.copy()
                    chunk_meta["content"] = sub_chunk
                    chunk_meta["priority"] = priority
                    chunk_meta["source_id"] = source_id

                    # Génère l'URL si disponible
                    base_url = source_config.get("base_url")
                    if base_url:
                        anchor = chunk_meta.get("anchor", "")
                        rel_path = file_path.relative_to(local_path)
                        # Retire l'extension pour l'URL
                        url_path = str(rel_path).rsplit(".", 1)[0]
                        url = f"{base_url}/{url_path}"
                        if anchor:
                            url = f"{url}#{anchor}"
                        chunk_meta["url"] = url

                    all_chunks.append(chunk_meta)
                    chunk_count += 1

            file_count += 1

        print(f"  -> {file_count} fichiers, {chunk_count} chunks")

    return all_chunks


def build_index(
    chunks: list[dict[str, Any]],
    config: dict[str, Any],
    embedding_batch_size: int = 500,
) -> None:
    """Construit et sauvegarde l'index FAISS par batches.

    Args:
        chunks: Liste des chunks avec métadonnées
        config: Configuration RAG
        embedding_batch_size: Taille des batches pour les embeddings
    """
    if not chunks:
        print("[ERROR] Aucun chunk à indexer")
        return

    project_root = get_project_root()
    embeddings_config = config["embeddings"]
    index_config = config["index"]

    print(f"\nChargement du modèle {embeddings_config['model']}...")
    model = SentenceTransformer(
        embeddings_config["model"],
        device=embeddings_config.get("device", "cpu"),
    )

    print(f"Génération des embeddings pour {len(chunks)} chunks par batches de {embedding_batch_size}...")

    dimension = embeddings_config["dimension"]

    # Crée l'index vide
    if index_config["type"] == "IndexFlatIP":
        index = faiss.IndexFlatIP(dimension)
    else:
        index = faiss.IndexFlatL2(dimension)

    # Traite par batches
    all_embeddings = []
    num_batches = (len(chunks) + embedding_batch_size - 1) // embedding_batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * embedding_batch_size
        end_idx = min(start_idx + embedding_batch_size, len(chunks))

        batch_contents = [chunks[i]["content"] for i in range(start_idx, end_idx)]

        batch_embeddings = model.encode(
            batch_contents,
            normalize_embeddings=embeddings_config.get("normalize", True),
            show_progress_bar=False,
            batch_size=embeddings_config.get("batch_size", 64),
        )

        batch_embeddings = np.array(batch_embeddings, dtype=np.float32)
        all_embeddings.append(batch_embeddings)

        # Affiche la progression
        processed = end_idx
        print(f"  Batch {batch_idx + 1}/{num_batches}: {processed}/{len(chunks)} chunks traités", flush=True)

    # Concatène tous les embeddings
    print(f"\nConstruction de l'index {index_config['type']}...")
    embeddings = np.vstack(all_embeddings)
    index.add(embeddings)

    # Sauvegarde
    index_path = project_root / index_config["path"]
    metadata_path = project_root / index_config["metadata_path"]

    index_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Sauvegarde de l'index: {index_path}")
    faiss.write_index(index, str(index_path))

    print(f"Sauvegarde des métadonnées: {metadata_path}")
    with open(metadata_path, "wb") as f:
        pickle.dump(chunks, f)

    # Statistiques
    print(f"\n{'='*60}")
    print(f"INDEX CREE AVEC SUCCES")
    print(f"{'='*60}")
    print(f"Chunks indexés:    {len(chunks)}")
    print(f"Dimension:         {dimension}")
    print(f"Taille index:      {index_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Stats par source
    print(f"\nPar source:")
    source_counts = {}
    for chunk in chunks:
        source = chunk["source"]
        source_counts[source] = source_counts.get(source, 0) + 1

    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = count / len(chunks) * 100
        print(f"  {source}: {count} ({pct:.1f}%)")


def main() -> None:
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(description="Construction de l'index RAG PHProf")
    parser.add_argument(
        "--sources",
        nargs="+",
        help="Sources spécifiques à indexer (ex: symfony_docs php_manual)",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        help="Sources à ignorer (ex: php_manual pour exclure les 11k fichiers)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Mode rapide: ignore php_manual (11k fichiers)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Taille des batches pour les embeddings (défaut: 500)",
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="Liste les sources disponibles et quitte",
    )
    args = parser.parse_args()

    print("PHProf - Construction de l'index RAG\n")

    config = load_config("rag_config")

    # Liste les sources si demandé
    if args.list_sources:
        print("Sources disponibles:\n")
        sorted_sources = sorted(
            config["sources"].items(),
            key=lambda x: x[1].get("priority", 99)
        )
        for source_id, source_config in sorted_sources:
            priority = source_config.get("priority", 99)
            name = source_config["name"]
            print(f"  [{priority}] {source_id:30} - {name}")
        sys.exit(0)

    # Mode rapide = skip php_manual
    skip_sources = args.skip or []
    if args.quick:
        skip_sources.append("php_manual")
        print("Mode rapide: php_manual ignoré\n")

    # Charge les documents
    chunks = load_documents(
        config,
        sources_filter=args.sources,
        skip_sources=skip_sources if skip_sources else None,
    )

    if not chunks:
        print("\n[ERROR] Aucun document trouvé. Vérifiez que la documentation est téléchargée.")
        print("   Exécutez 'make download-docs' pour télécharger la documentation.")
        sys.exit(1)

    # Construit l'index par batches
    build_index(chunks, config, embedding_batch_size=args.batch_size)


if __name__ == "__main__":
    main()
