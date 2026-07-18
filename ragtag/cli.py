"""Provide operator commands for the ingestion workflow in PRD sections 6 and 8."""

import typer

from ragtag.rag.local import LocalRAG

app = typer.Typer(help="RAGtag pre-ingestion poisoning detector.")


@app.callback()
def root() -> None:
    """Manage the local RAGtag corpus and poisoning detector."""


@app.command()
def seed() -> None:
    """Build the local corpus index and persist its metadata cache."""

    rag = LocalRAG(rebuild=True)
    typer.echo(f"Seeded {rag.document_count} documents into {rag.chunk_count} chunks.")


def main() -> None:
    """Run the RAGtag command-line application."""

    app()


if __name__ == "__main__":
    main()
