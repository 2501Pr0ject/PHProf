"""Système RAG (Retrieval-Augmented Generation) pour PHProf."""

import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_index_path, get_metadata_path, get_project_root, load_config


@dataclass
class RetrievedChunk:
    """Un chunk de documentation récupéré."""

    content: str
    source: str
    heading: str
    score: float
    url: str | None = None
    priority: int = 99


class RAGRetriever:
    """Retriever RAG utilisant FAISS et SentenceTransformers."""

    def __init__(self):
        """Initialise le retriever (chargement paresseux)."""
        self._config: dict[str, Any] | None = None
        self._model: SentenceTransformer | None = None
        self._index: faiss.Index | None = None
        self._metadata: list[dict[str, Any]] | None = None

    @property
    def config(self) -> dict[str, Any]:
        """Charge la configuration RAG."""
        if self._config is None:
            self._config = load_config("rag_config")
        return self._config

    @property
    def model(self) -> SentenceTransformer:
        """Charge le modèle d'embeddings (lazy loading)."""
        if self._model is None:
            model_name = self.config["embeddings"]["model"]
            device = self.config["embeddings"].get("device", "cpu")
            self._model = SentenceTransformer(model_name, device=device)
        return self._model

    @property
    def index(self) -> faiss.Index:
        """Charge l'index FAISS (lazy loading)."""
        if self._index is None:
            index_path = get_index_path()
            if not index_path.exists():
                raise FileNotFoundError(
                    f"Index FAISS introuvable: {index_path}\n"
                    "Exécutez 'make build-index' pour le créer."
                )
            self._index = faiss.read_index(str(index_path))
        return self._index

    @property
    def metadata(self) -> list[dict[str, Any]]:
        """Charge les métadonnées (lazy loading)."""
        if self._metadata is None:
            metadata_path = get_metadata_path()
            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"Métadonnées introuvables: {metadata_path}\n"
                    "Exécutez 'make build-index' pour les créer."
                )
            with open(metadata_path, "rb") as f:
                self._metadata = pickle.load(f)
        return self._metadata

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Recherche les chunks les plus pertinents pour une requête.

        Args:
            query: Question de l'utilisateur
            top_k: Nombre de résultats (défaut: config)

        Returns:
            Liste de chunks triés par score décroissant
        """
        retrieval_config = self.config["retrieval"]
        if top_k is None:
            top_k = retrieval_config["top_k"]

        # Encode la requête
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=self.config["embeddings"].get("normalize", True),
        )
        query_embedding = np.array([query_embedding], dtype=np.float32)

        # Recherche dans l'index
        initial_k = retrieval_config.get("initial_k", top_k * 3)
        scores, indices = self.index.search(query_embedding, initial_k)

        # Filtre par score minimum
        threshold = retrieval_config.get("score_threshold", 0.0)
        candidates = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < threshold:
                continue

            meta = self.metadata[idx]

            # Applique le boost de priorité
            adjusted_score = self._apply_priority_boost(score, meta)

            candidates.append(
                RetrievedChunk(
                    content=meta["content"],
                    source=meta["source"],
                    heading=meta.get("heading", ""),
                    score=float(adjusted_score),
                    url=meta.get("url"),
                    priority=meta.get("priority", 99),
                )
            )

        # Trie par score ajusté
        candidates.sort(key=lambda x: x.score, reverse=True)

        # Applique la diversification
        if retrieval_config.get("diversity", {}).get("enabled", False):
            candidates = self._diversify_results(candidates, top_k)
        else:
            candidates = candidates[:top_k]

        return candidates

    def _apply_priority_boost(self, score: float, meta: dict) -> float:
        """Applique un boost aux sources prioritaires.

        Args:
            score: Score original
            meta: Métadonnées du chunk

        Returns:
            Score ajusté
        """
        priority_config = self.config["retrieval"].get("priority_boost", {})
        if not priority_config.get("enabled", False):
            return score

        priority = meta.get("priority", 99)
        boost_factor = priority_config.get("boost_factor", 1.0)

        # Les sources priorité 1 reçoivent le boost complet
        # Les sources priorité 2 reçoivent la moitié du boost
        # Les autres n'ont pas de boost
        if priority == 1:
            return score * boost_factor
        elif priority == 2:
            return score * (1 + (boost_factor - 1) / 2)
        return score

    def _diversify_results(
        self, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Diversifie les résultats pour éviter trop de chunks de la même source.

        Args:
            candidates: Liste triée de candidats
            top_k: Nombre final de résultats

        Returns:
            Liste diversifiée
        """
        diversity_config = self.config["retrieval"].get("diversity", {})
        min_sources = diversity_config.get("min_sources", 2)
        max_per_source = diversity_config.get("max_per_source", 3)

        result = []
        source_counts: dict[str, int] = defaultdict(int)
        sources_seen: set[str] = set()

        # Premier passage : respecte max_per_source
        for chunk in candidates:
            if source_counts[chunk.source] < max_per_source:
                result.append(chunk)
                source_counts[chunk.source] += 1
                sources_seen.add(chunk.source)

                if len(result) >= top_k:
                    break

        # Vérifie qu'on a assez de sources différentes
        if len(sources_seen) < min_sources and len(result) < top_k:
            # Essaie d'ajouter des chunks d'autres sources
            for chunk in candidates:
                if chunk.source not in sources_seen and chunk not in result:
                    result.append(chunk)
                    sources_seen.add(chunk.source)

                    if len(sources_seen) >= min_sources or len(result) >= top_k:
                        break

        # Retrie par score
        result.sort(key=lambda x: x.score, reverse=True)
        return result[:top_k]

    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Formate les chunks en contexte pour le prompt.

        Args:
            chunks: Liste de chunks récupérés

        Returns:
            Contexte formaté avec balises de protection
        """
        if not chunks:
            return ""

        wrapper_tag = self.config["prompt_injection"]["wrapper_tag"]
        max_tokens = self.config["prompt_injection"].get("max_context_tokens", 2000)

        context_parts = []
        current_tokens = 0

        for i, chunk in enumerate(chunks, 1):
            source_info = f"[{chunk.source}]"
            if chunk.heading:
                source_info += f" {chunk.heading}"

            chunk_text = f"### Source {i} {source_info}\n{chunk.content}"

            # Estimation grossière des tokens (1 token ≈ 4 caractères)
            chunk_tokens = len(chunk_text) // 4

            if current_tokens + chunk_tokens > max_tokens:
                break

            context_parts.append(chunk_text)
            current_tokens += chunk_tokens

        context = "\n\n".join(context_parts)
        return f"<{wrapper_tag}>\n{context}\n</{wrapper_tag}>"

    def format_citations(self, chunks: list[RetrievedChunk]) -> str:
        """Formate les citations pour la réponse.

        Args:
            chunks: Liste de chunks utilisés

        Returns:
            Citations formatées en markdown
        """
        max_citations = self.config["retrieval"].get("max_citations", 4)
        citations = []
        seen_urls = set()

        for chunk in chunks[:max_citations]:
            # Évite les citations en double
            if chunk.url and chunk.url in seen_urls:
                continue

            citation = f"- **{chunk.source}**"
            if chunk.heading:
                citation += f": {chunk.heading}"
            if chunk.url:
                citation += f"\n  {chunk.url}"
                seen_urls.add(chunk.url)
            citations.append(citation)

        if citations:
            return "\n\n---\n**Sources:**\n" + "\n".join(citations)
        return ""

    def get_stats(self) -> dict[str, Any]:
        """Retourne des statistiques sur l'index.

        Returns:
            Dict avec les statistiques
        """
        source_counts: dict[str, int] = defaultdict(int)
        priority_counts: dict[int, int] = defaultdict(int)

        for meta in self.metadata:
            source_counts[meta["source"]] += 1
            priority_counts[meta.get("priority", 99)] += 1

        return {
            "total_chunks": len(self.metadata),
            "by_source": dict(source_counts),
            "by_priority": dict(priority_counts),
        }


def is_index_available() -> bool:
    """Vérifie si l'index RAG est disponible."""
    try:
        return get_index_path().exists() and get_metadata_path().exists()
    except Exception:
        return False
