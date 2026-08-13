"""
src/tools/scraping_tools.py
-----------------------------
Wraps crewai_tools.ScrapegraphScrapeTool, which calls ScrapeGraph AI's
SmartScraper API. Unlike a plain HTML-to-text scraper, SmartScraper takes a
natural-language extraction prompt and returns structured data even when
product specs are scattered across tables, spec sheets, and JS-rendered
widgets -- exactly the shape of a real e-commerce/vendor product page.

Falls back to crewai_tools.ScrapeWebsiteTool (free, no API key, plain
readability-style extraction) when SCRAPEGRAPH_API_KEY isn't configured, so
the crew still runs end-to-end -- with the Scraping Agent doing the
structuring itself via its LLM reasoning instead of the SmartScraper API.
"""
from __future__ import annotations

import logging

from src.config import settings

logger = logging.getLogger("procurement_assistant.tools.scraping")

DEFAULT_EXTRACTION_PROMPT = (
    "Extract the product name, current price, currency, stock availability, "
    "warranty terms, estimated delivery/shipping time, and a dictionary of "
    "all listed technical specifications (dimensions, capacity, power, "
    "connectivity, materials, certifications, etc.) from this product page. "
    "Return null for any field you cannot find rather than guessing."
)


def get_scraping_tool(website_url: str | None = None):
    """
    Returns an instantiated CrewAI scraping tool.

    Passing `website_url` pins the tool to one page (used when we want a
    dedicated tool instance per candidate URL); leaving it None lets the
    agent supply a different URL on each tool call.
    """
    if settings.scrapegraph_api_key:
        # IMPORTANT (found via testing against the real package): crewai_tools'
        # ScrapegraphScrapeTool detects a missing/incompatible `scrapegraph-py`
        # SDK by trying `from scrapegraph_py import Client` and, on failure,
        # prompts interactively ("Would you like to install it? [y/N]")
        # instead of raising -- which hangs forever with no terminal attached.
        # This also fires on a *version mismatch*: scrapegraph-py 2.x renamed
        # `Client` to `client`, so an unpinned `pip install scrapegraph-py`
        # silently breaks this tool even though the package is technically
        # installed. We check for the exact symbol crewai_tools needs, first,
        # so both failure modes turn into a fast, clear RuntimeError instead.
        try:
            from scrapegraph_py import Client  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "SCRAPEGRAPH_API_KEY is set but a compatible `scrapegraph-py` "
                "SDK isn't installed (need scrapegraph-py>=1.9.0,<2 -- newer "
                "2.x releases renamed Client and are NOT compatible with the "
                "installed crewai_tools). Run: "
                "pip install 'crewai-tools[scrapegraph-py]'"
            ) from exc

        try:
            from crewai_tools import ScrapegraphScrapeTool

            kwargs = {"user_prompt": DEFAULT_EXTRACTION_PROMPT, "enable_logging": False}
            if website_url:
                kwargs["website_url"] = website_url
            return ScrapegraphScrapeTool(**kwargs)
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "SCRAPEGRAPH_API_KEY is set but `crewai_tools` doesn't expose "
                "ScrapegraphScrapeTool. Run: pip install 'crewai-tools[scrapegraph-py]' "
                "or `pip install scrapegraph-py`."
            ) from exc

    logger.warning(
        "SCRAPEGRAPH_API_KEY not set - falling back to crewai_tools.ScrapeWebsiteTool "
        "(no AI-structuring; the Scraping Agent must parse raw page text itself)."
    )
    from crewai_tools import ScrapeWebsiteTool

    return ScrapeWebsiteTool(website_url=website_url) if website_url else ScrapeWebsiteTool()
