"""Interface CLI pour PHProf."""

from typing import Optional

import typer
from rich.console import Console

from app import __version__
from app.chat import chat_loop, single_query

app = typer.Typer(
    name="phpprof",
    help="PHProf - Assistant pédagogique IA local pour PHP et Symfony",
    add_completion=False,
)
console = Console()


@app.command()
def chat(
    no_rag: bool = typer.Option(
        False,
        "--no-rag",
        help="Désactiver le système RAG",
    ),
    finetuned: bool = typer.Option(
        False,
        "--finetuned",
        "-f",
        help="Utiliser le modèle fine-tuné",
    ),
) -> None:
    """Lance le chat interactif avec PHProf."""
    chat_loop(use_rag=not no_rag, use_finetuned=finetuned)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question à poser"),
    no_rag: bool = typer.Option(
        False,
        "--no-rag",
        help="Désactiver le système RAG",
    ),
    finetuned: bool = typer.Option(
        False,
        "--finetuned",
        "-f",
        help="Utiliser le modèle fine-tuné",
    ),
) -> None:
    """Pose une question unique à PHProf."""
    response = single_query(
        query=question,
        use_rag=not no_rag,
        use_finetuned=finetuned,
    )
    console.print(response)


@app.command()
def version() -> None:
    """Affiche la version de PHProf."""
    console.print(f"PHProf v{__version__}")


def main() -> None:
    """Point d'entrée principal."""
    app()


if __name__ == "__main__":
    main()
