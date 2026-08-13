"""
src/tools/search_tools.py
--------------------------
Wraps crewai_tools.TavilySearchTool. Tavily is purpose-built for LLM agents
(it returns clean, ranked snippets rather than raw SERP HTML), which is why
it's the primary search tool here rather than a generic Google wrapper.

If no TAVILY_API_KEY is configured we fall back to CrewAI's built-in
SerperDevTool-free DuckDuckGo-backed search where available, so the crew is
still runnable in a trial/demo environment without a Tavily account.
"""
from __future__ import annotations

import logging

from src.config import settings

logger = logging.getLogger("procurement_assistant.tools.search")


def get_search_tool():
    """
    Returns an instantiated CrewAI tool object ready to attach to an Agent.
    Raises a clear RuntimeError if nothing usable is configured, rather than
    letting the crew fail deep inside an agent's first tool call.
    """
    if settings.tavily_api_key:
        # IMPORTANT: crewai_tools' TavilySearchTool, when it can't find the
        # `tavily-python` SDK, prompts interactively ("Would you like to
        # install it? [y/N]") instead of raising. In any non-interactive
        # context (a server, a container, CI) that prompt has no terminal to
        # answer it and the process hangs indefinitely rather than failing.
        # We check importability ourselves first so a missing dependency is
        # always a fast, clear RuntimeError instead of a silent hang.
        try:
            import tavily  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "TAVILY_API_KEY is set but the `tavily-python` package isn't "
                "installed. Run: pip install 'crewai-tools[tavily-python]'"
            ) from exc

        try:
            from crewai_tools import TavilySearchTool

            return TavilySearchTool(
                search_depth="advanced",
                max_results=settings.max_products_per_source,
                include_answer=False,
                include_raw_content=False,
            )
        except ImportError as exc:  # pragma: no cover - dependency guidance
            raise RuntimeError(
                "TAVILY_API_KEY is set but `crewai_tools` doesn't expose "
                "TavilySearchTool. Run: pip install 'crewai-tools[tavily-python]' "
                "or `pip install tavily-python`."
            ) from exc

    logger.warning(
        "TAVILY_API_KEY not set - falling back to crewai_tools.SerperDevTool "
        "or EXASearchTool if configured, otherwise search quality will be limited."
    )
    try:
        from crewai_tools import SerperDevTool

        return SerperDevTool()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "No search backend available. Set TAVILY_API_KEY (recommended) "
            "in your .env, or SERPER_API_KEY as a fallback."
        ) from exc
