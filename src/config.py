"""
src/config.py
--------------
Centralized runtime configuration for the Procurement Assistant.

All secrets and tunables are read from environment variables (see .env.example).
We use pydantic-settings so that:
  * required keys fail fast with a clear error instead of a cryptic 401 deep
    inside a tool call three agents later.
  * every other module imports one `settings` object instead of scattering
    `os.getenv(...)` calls throughout the codebase.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = PROJECT_ROOT / ".cache"
TEMPLATE_DIR = PROJECT_ROOT / "src" / "templates"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider -----------------------------------------------------
    # CrewAI 1.x ships native SDKs for openai/anthropic/azure/bedrock/gemini,
    # and a generic OpenAI-compatible handler that covers 'ollama' (local,
    # free, no key) and 'dashscope' (Alibaba's hosted Qwen API) out of the
    # box -- no litellm install required for any of these. Default targets
    # Claude.
    llm_provider: Literal["anthropic", "openai", "groq", "ollama", "dashscope"] = Field(
        default="anthropic", alias="LLM_PROVIDER"
    )
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    dashscope_api_key: Optional[str] = Field(default=None, alias="DASHSCOPE_API_KEY")

    # Local Ollama server (e.g. running Qwen, Llama, Mistral, DeepSeek...).
    # No API key needed -- just Ollama itself installed and running.
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")

    manager_model: str = Field(default="anthropic/claude-sonnet-4-6", alias="MANAGER_MODEL")
    worker_model: str = Field(default="anthropic/claude-sonnet-4-6", alias="WORKER_MODEL")

    # --- Search / scraping tools -------------------------------------------
    tavily_api_key: Optional[str] = Field(default=None, alias="TAVILY_API_KEY")
    scrapegraph_api_key: Optional[str] = Field(default=None, alias="SCRAPEGRAPH_API_KEY")

    # --- Behaviour flags -----------------------------------------------------
    process_type: Literal["sequential", "hierarchical"] = Field(
        default="sequential", alias="CREW_PROCESS"
    )
    # Default OFF: `memory=True` pulls in chromadb + a local embedding model.
    # In testing this caused a hard process abort (SIGABRT) at interpreter
    # shutdown in some containerized environments, with any unflushed stdout
    # silently lost -- a genuinely bad failure mode for a CLI tool. Enable it
    # deliberately (CREW_MEMORY=true) once you've confirmed chromadb behaves
    # cleanly in your target environment.
    use_crew_memory: bool = Field(default=False, alias="CREW_MEMORY")
    max_products_per_source: int = Field(default=5, alias="MAX_PRODUCTS_PER_SOURCE")
    max_sources: int = Field(default=4, alias="MAX_SOURCES")
    request_cache_ttl_hours: int = Field(default=24, alias="CACHE_TTL_HOURS")
    verbose: bool = Field(default=True, alias="CREW_VERBOSE")

    # Off by default. Two independent reasons: (1) this crew handles a
    # company's real budget/vendor data -- telemetry should be an explicit
    # opt-in, not a default, for anything enterprise-shaped; (2) verified
    # during development that CrewAI's OpenTelemetry span flush at process
    # shutdown can stall for many seconds when the collector endpoint is
    # unreachable (e.g. restricted network egress), which is a bad default
    # for a CLI tool. Flip to true if you want CrewAI's usage analytics.
    disable_telemetry: bool = Field(default=True, alias="CREWAI_DISABLE_TELEMETRY")

    @field_validator(
        "anthropic_api_key", "openai_api_key", "groq_api_key",
        "dashscope_api_key", "tavily_api_key", "scrapegraph_api_key",
    )
    @classmethod
    def _blank_to_none(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v.strip() == "":
            return None
        return v

    def active_llm_key_present(self) -> bool:
        if self.llm_provider == "ollama":
            # Local server, no API key -- readiness depends on Ollama itself
            # actually running, which we can't check without a network call,
            # so we just don't block here; a clear connection error surfaces
            # on the first real request if it isn't.
            return True
        return {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "groq": self.groq_api_key,
            "dashscope": self.dashscope_api_key,
        }.get(self.llm_provider) is not None

    def export_to_environ(self) -> None:
        """CrewAI reads provider keys straight out of os.environ."""
        mapping = {
            "ANTHROPIC_API_KEY": self.anthropic_api_key,
            "OPENAI_API_KEY": self.openai_api_key,
            "GROQ_API_KEY": self.groq_api_key,
            "DASHSCOPE_API_KEY": self.dashscope_api_key,
            "TAVILY_API_KEY": self.tavily_api_key,
            # crewai_tools' ScrapegraphScrapeTool reads SCRAPEGRAPH_API_KEY;
            # the underlying scrapegraph-py SDK reads SGAI_API_KEY. We set
            # both so whichever version of the tool is installed works.
            "SCRAPEGRAPH_API_KEY": self.scrapegraph_api_key,
            "SGAI_API_KEY": self.scrapegraph_api_key,
        }
        for key, value in mapping.items():
            if value:
                os.environ[key] = value


settings = Settings()

# Set this immediately (not inside export_to_environ(), which callers invoke
# later) so it's in place before any `import crewai` elsewhere in the
# codebase can initialize CrewAI's telemetry singleton.
if settings.disable_telemetry:
    os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
    os.environ["OTEL_SDK_DISABLED"] = "true"

for _d in (OUTPUT_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
