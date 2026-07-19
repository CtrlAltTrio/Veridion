"""Load RAGtag configuration for the pipeline described in PRD sections 5 and 6."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class SignalWeights(BaseModel):
    """Configured weights for the three detector signals."""

    anomaly: float
    injection: float
    influence: float


class Thresholds(BaseModel):
    """Configured fusion thresholds for verdict selection."""

    tau_low: float
    tau_high: float


class Paths(BaseModel):
    """Configured filesystem locations used by RAGtag."""

    corpus_dir: Path
    probes_file: Path
    attacks_dir: Path
    labelled_dir: Path
    cache_dir: Path
    private_key: Path


class Settings(BaseSettings):
    """Validated application settings loaded from YAML."""

    model_config = SettingsConfigDict(env_prefix="RAGTAG_", env_nested_delimiter="__")

    signal_weights: SignalWeights
    thresholds: Thresholds
    encoder_name: str
    ollama_model: str
    rag_backend: Literal["local", "openai_compat"] = "local"
    ollama_base_url: str = "http://localhost:11434"
    llm_timeout_seconds: float = 20.0
    openai_base_url: str = "http://localhost:8000"
    openai_api_key: str | None = None
    openai_chat_model: str = "chat-model"
    openai_embedding_model: str = "embedding-model"
    top_k: int
    paths: Paths


def load_settings(config_path: str | Path = "config.yaml") -> Settings:
    """Load and validate settings from ``config_path``."""

    path = Path(config_path)
    with path.open(encoding="utf-8") as config_file:
        values: dict[str, Any] = yaml.safe_load(config_file) or {}
    return Settings(**values)


settings = load_settings()
