"""Centralized configuration loader combining configs/config.yaml and .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_YAML_PATH = PROJECT_ROOT / "configs" / "config.yaml"
PROMPTS_YAML_PATH = PROJECT_ROOT / "configs" / "prompts.yaml"


class AppConfig(BaseModel):
    name: str = "ai-papers-agentic-rag"
    runtime_env: Literal["laptop", "kaggle"] = "laptop"
    max_query_chars: int = 2000
    conversation_window_turns: int = 6


class IngestionConfig(BaseModel):
    parser: Literal["docling", "pymupdf4llm"] = "docling"
    docling_do_ocr: bool = False
    docling_do_table_structure: bool = True
    exclude_sections: list[str] = Field(
        default_factory=lambda: [
            "references",
            "bibliography",
            "acknowledgments",
            "ethics statement",
        ]
    )
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    max_table_chunk_tokens: int = 700
    inject_contextual_header: bool = True


class QdrantConfig(BaseModel):
    mode: Literal["cloud", "local_path"] = "cloud"
    local_path: str = "data/qdrant_local"
    collection_name: str = "ai_papers_chunks"


class RetrievalConfig(BaseModel):
    backend: str = "fastembed"
    cache_dir: str = "./.cache/fastembed"
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    reranker_model_laptop: str = "jinaai/jina-reranker-v1-turbo-en"
    reranker_model_kaggle_heavy: str = "BAAI/bge-reranker-base"
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_k: int = 60
    rrf_candidates: int = 15
    rerank_top_k: int = 5
    min_rerank_score: float = 0.15


class MCPConfig(BaseModel):
    transport: Literal["stdio", "streamable-http"] = "streamable-http"
    host: str = "127.0.0.1"
    port: int = 8001
    sql_max_limit: int = 50
    sql_timeout_seconds: float = 2.0
    client_timeout_seconds: float = 10.0


class AgentConfig(BaseModel):
    max_rag_retries: int = 2
    max_sql_retries: int = 1
    max_gen_retries: int = 1
    max_tool_calls: int = 3
    enable_llm_hallucination_check: bool = True
    hitl_write_tools: list[str] = Field(default_factory=lambda: ["create_reading_note"])


class LLMModelsConfig(BaseModel):
    gemini_fast: str = "gemini-3.8-flash"
    gemini_main: str = "gemini-3.8-flash"
    groq_fast: str = "llama-3.3-70b-versatile"
    groq_main: str = "llama-3.3-70b-versatile"
    kaggle_openweight: str = "Qwen/Qwen2.5-7B-Instruct"


class LLMConfig(BaseModel):
    primary_provider: Literal["gemini", "groq", "openai_compatible"] = "gemini"
    fallback_order: list[str] = Field(default_factory=lambda: ["gemini", "groq"])
    temperature: float = 0.1
    models: LLMModelsConfig = Field(default_factory=LLMModelsConfig)


class EnvSettings(BaseSettings):
    """Environment variables loaded from .env or OS environment."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM API keys
    gemini_api_key: str = ""
    groq_api_key: str = ""
    openai_compat_base_url: str = ""
    openai_compat_api_key: str = "EMPTY"

    # Qdrant
    qdrant_mode: Literal["cloud", "local_path"] | None = None
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_local_path: str = "./data/qdrant_local"

    # Services
    mcp_transport: Literal["stdio", "streamable-http"] | None = None
    mcp_server_url: str = "http://127.0.0.1:8001/mcp"
    api_base_url: str = "http://127.0.0.1:8000"

    # Observability
    enable_tracing: bool = False
    phoenix_collector_endpoint: str = "http://127.0.0.1:6006/v1/traces"
    phoenix_api_key: str = ""

    # SQLite & Audit paths
    papers_db_path: str = "./data/sqlite/papers.db"
    checkpoints_db_path: str = "./data/sqlite/checkpoints.db"
    audit_log_path: str = "./logs/mcp_audit.jsonl"


class Settings(BaseModel):
    """Unified settings combining YAML config and environment secrets."""

    app: AppConfig = Field(default_factory=AppConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    env: EnvSettings = Field(default_factory=EnvSettings)

    @property
    def effective_qdrant_mode(self) -> Literal["cloud", "local_path"]:
        return self.env.qdrant_mode or self.qdrant.mode

    @property
    def effective_mcp_transport(self) -> Literal["stdio", "streamable-http"]:
        return self.env.mcp_transport or self.mcp.transport


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    raw: dict = {}
    if CONFIG_YAML_PATH.exists():
        with CONFIG_YAML_PATH.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    return Settings(**raw, env=EnvSettings())


@lru_cache(maxsize=1)
def get_prompts() -> dict[str, str]:
    with PROMPTS_YAML_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {k: str(v).strip() for k, v in data.items()}
