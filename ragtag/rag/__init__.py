"""Select the configured TargetRAG adapter for PRD sections 6 and 8."""

from ragtag.config import Settings, settings
from ragtag.rag.base import TargetRAG


def create_target_rag(
    config: Settings | None = None,
    rebuild: bool = False,
) -> TargetRAG:
    """Create the configured local or OpenAI-compatible target adapter."""

    active = config or settings
    if active.rag_backend == "openai_compat":
        from ragtag.rag.openai_compat import OpenAICompatRAG

        return OpenAICompatRAG(active, rebuild=rebuild)

    from ragtag.rag.local import LocalRAG

    return LocalRAG(active, rebuild=rebuild)


__all__ = ["create_target_rag"]
