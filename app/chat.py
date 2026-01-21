"""Interface de chat interactive pour PHProf."""

from pathlib import Path
from typing import Any

from llama_cpp import Llama
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from app.config import get_model_path, get_project_root, load_config
from app.rag import RAGRetriever, is_index_available

console = Console()

# Cache global pour le modèle
_llm_instance: Llama | None = None


def get_llm(model_path: Path, config: dict[str, Any]) -> Llama:
    """Charge le modèle LLM (avec cache).

    Args:
        model_path: Chemin vers le fichier GGUF
        config: Configuration d'inférence

    Returns:
        Instance Llama
    """
    global _llm_instance

    if _llm_instance is None:
        inference_config = config["inference"]
        console.print(f"[dim]Chargement du modèle {model_path.name}...[/dim]")

        _llm_instance = Llama(
            model_path=str(model_path),
            n_ctx=inference_config["n_ctx"],
            n_gpu_layers=inference_config["n_gpu_layers"],
            verbose=False,
        )

    return _llm_instance


def build_prompt(
    user_input: str,
    system_prompt: str,
    template: str,
    rag_context: str = "",
) -> str:
    """Construit le prompt complet au format ChatML.

    Args:
        user_input: Question de l'utilisateur
        system_prompt: Prompt système
        template: Template ChatML
        rag_context: Contexte RAG optionnel

    Returns:
        Prompt formaté
    """
    # Ajoute le contexte RAG au prompt système si disponible
    if rag_context:
        full_system = f"{system_prompt}\n\n{rag_context}"
    else:
        full_system = system_prompt

    return template.format(system=full_system, user=user_input)


def run_inference(
    prompt: str,
    model_path: Path,
    config: dict[str, Any],
) -> str:
    """Exécute l'inférence via llama-cpp-python.

    Args:
        prompt: Prompt complet
        model_path: Chemin vers le modèle GGUF
        config: Configuration d'inférence

    Returns:
        Réponse générée

    Raises:
        RuntimeError: Si le modèle n'est pas trouvé
    """
    if not model_path.exists():
        raise RuntimeError(
            f"Modèle introuvable: {model_path}\n"
            "Exécutez 'make download-model' pour le télécharger."
        )

    llm = get_llm(model_path, config)
    inference_config = config["inference"]

    output = llm(
        prompt,
        max_tokens=inference_config["n_predict"],
        temperature=inference_config["temperature"],
        top_p=inference_config["top_p"],
        top_k=inference_config["top_k"],
        repeat_penalty=inference_config["repeat_penalty"],
        stop=["<|im_end|>", "<|im_start|>"],
    )

    return output["choices"][0]["text"].strip()


def chat_loop(use_rag: bool = True, use_finetuned: bool = False) -> None:
    """Boucle de chat interactive.

    Args:
        use_rag: Utiliser le système RAG
        use_finetuned: Utiliser le modèle fine-tuné
    """
    config = load_config("model_config")
    model_path = get_model_path(finetuned=use_finetuned)

    # Initialisation du RAG
    retriever = None
    if use_rag:
        if is_index_available():
            retriever = RAGRetriever()
            console.print("[green]RAG activé[/green] - Documentation chargée")
        else:
            console.print(
                "[yellow]RAG indisponible[/yellow] - "
                "Exécutez 'make build-index' pour l'activer"
            )

    # Affichage de bienvenue
    console.print(
        Panel(
            "[bold blue]PHProf[/bold blue] - Assistant PHP/Symfony\n"
            "Tapez [bold]quit[/bold] ou [bold]exit[/bold] pour quitter\n"
            "Tapez [bold]clear[/bold] pour effacer l'écran",
            title="Bienvenue",
            border_style="blue",
        )
    )

    model_name = "fine-tuné" if use_finetuned else "baseline"
    console.print(f"Modèle: [cyan]{model_path.name}[/cyan] ({model_name})")

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]Vous[/bold green]")

            # Commandes spéciales
            if user_input.lower() in ("quit", "exit", "q"):
                console.print("[dim]Au revoir ![/dim]")
                break

            if user_input.lower() == "clear":
                console.clear()
                continue

            if not user_input.strip():
                continue

            # Récupération RAG
            rag_context = ""
            citations = ""
            if retriever:
                with console.status("[dim]Recherche dans la documentation...[/dim]"):
                    chunks = retriever.retrieve(user_input)
                    if chunks:
                        rag_context = retriever.format_context(chunks)
                        citations = retriever.format_citations(chunks)
                        console.print(
                            f"[dim]{len(chunks)} sources trouvées[/dim]"
                        )

            # Construction du prompt
            prompt = build_prompt(
                user_input=user_input,
                system_prompt=config["prompt"]["system"],
                template=config["prompt"]["template"],
                rag_context=rag_context,
            )

            # Inférence
            with console.status("[dim]Réflexion en cours...[/dim]"):
                response = run_inference(prompt, model_path, config)

            # Affichage de la réponse
            console.print("\n[bold blue]PHProf[/bold blue]")
            console.print(Markdown(response))

            # Affichage des citations
            if citations:
                console.print(Markdown(citations))

        except KeyboardInterrupt:
            console.print("\n[dim]Interruption - tapez 'quit' pour quitter[/dim]")
        except Exception as e:
            console.print(f"[red]Erreur:[/red] {e}")


def single_query(
    query: str,
    use_rag: bool = True,
    use_finetuned: bool = False,
) -> str:
    """Exécute une seule requête (mode non-interactif).

    Args:
        query: Question à poser
        use_rag: Utiliser le système RAG
        use_finetuned: Utiliser le modèle fine-tuné

    Returns:
        Réponse générée
    """
    config = load_config("model_config")
    model_path = get_model_path(finetuned=use_finetuned)

    # RAG
    rag_context = ""
    if use_rag and is_index_available():
        retriever = RAGRetriever()
        chunks = retriever.retrieve(query)
        if chunks:
            rag_context = retriever.format_context(chunks)

    # Prompt
    prompt = build_prompt(
        user_input=query,
        system_prompt=config["prompt"]["system"],
        template=config["prompt"]["template"],
        rag_context=rag_context,
    )

    return run_inference(prompt, model_path, config)
