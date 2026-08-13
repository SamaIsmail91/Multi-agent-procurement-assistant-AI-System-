"""
src/agents.py
--------------
Six specialized agents, each with a narrow mandate. This is deliberate:
a single "do everything" procurement agent tends to skip steps under time
pressure (skim search results, half-verify specs, rank from memory). Giving
each responsibility its own role with its own tools and its own "expected
output" contract forces every step to actually happen and be checked before
the next agent proceeds.

    Requirements Analyst  -> turns a vague request + company context into
                              concrete, searchable product criteria
    Market Researcher      -> Tavily web search for candidate products/vendors
    Web Scraper             -> ScrapeGraph AI visits each candidate URL and
                              extracts structured price/spec data
    Data Analyst            -> runs the deterministic scoring tool, ranks
                              and flags risks
    Compliance Officer       -> QA pass: budget check, hard-requirement
                              re-verification, red flags
    Report Writer            -> produces the final executive-ready narrative
"""
from __future__ import annotations

import src.config  # noqa: F401  (must run first: sets CREWAI_DISABLE_TELEMETRY before `import crewai` below)
from crewai import Agent, LLM

from src.company_context import CompanyProfile
from src.config import settings
from src.tools.scoring_tool import build_scoring_tool
from src.tools.scraping_tools import get_scraping_tool
from src.tools.search_tools import get_search_tool


def _llm(model: str) -> LLM:
    kwargs: dict = {"temperature": 0.2}
    if model.startswith("ollama"):
        # Local models (Ollama-served Qwen, Llama, Mistral, DeepSeek, ...)
        # need the server address explicitly -- crewai's OpenAI-compatible
        # handler defaults to localhost:11434 for the bare 'ollama/' prefix,
        # but we pass it explicitly so a custom OLLAMA_BASE_URL is honored
        # too (e.g. Ollama running on another machine on the network).
        kwargs["base_url"] = settings.ollama_base_url
    return LLM(model=model, **kwargs)


def build_agents(company: CompanyProfile) -> dict[str, Agent]:
    context_block = company.as_prompt_block()
    worker_llm = _llm(settings.worker_model)
    scoring_tool = build_scoring_tool(
        weights=company.priority_weights,
        budget_max=company.procurement_budget_max,
        risk_tolerance=company.risk_tolerance.value,
    )

    requirements_analyst = Agent(
        role="Procurement Requirements Analyst",
        goal=(
            "Translate a raw product request and the company's buying context into a "
            "precise requirements brief: a normalized product category, 3-5 concrete "
            "search queries, the non-negotiable hard requirements, and disqualifying "
            "criteria -- so downstream agents search for and evaluate the right thing."
        ),
        backstory=(
            "You are a senior procurement analyst who has seen countless projects go "
            "sideways because nobody wrote down what 'good' actually meant before the "
            "team started shopping. You are pedantic, in a useful way, about turning "
            "vague requests into checkable criteria.\n\n" + context_block
        ),
        llm=worker_llm,
        allow_delegation=False,
        verbose=settings.verbose,
    )

    market_researcher = Agent(
        role="Market Research Specialist",
        goal=(
            f"Use web search to find at least {settings.max_sources} distinct, "
            "currently-available product listings that plausibly match the "
            "requirements brief, across multiple vendors/retailers -- not just "
            "the first result."
        ),
        backstory=(
            "You are a market researcher who distrusts a single source. You "
            "deliberately diversify searches across manufacturer sites, major "
            "retailers, and regional distributors so the comparison that follows "
            "isn't accidentally biased toward whichever vendor has the best SEO.\n\n"
            + context_block
        ),
        tools=[get_search_tool()],
        llm=worker_llm,
        allow_delegation=False,
        verbose=settings.verbose,
    )

    web_scraper = Agent(
        role="Web Data Extraction Specialist",
        goal=(
            "Visit every candidate product URL and extract accurate, structured "
            "pricing and specification data. Never fabricate a spec or price you "
            "did not actually find on the page -- return null instead."
        ),
        backstory=(
            "You are meticulous about ground truth. You have been burned before by "
            "agents that 'remembered' a spec instead of checking the page, and it "
            "cost a real procurement decision. You would rather report a field as "
            "unknown than guess.\n\n" + context_block
        ),
        tools=[get_scraping_tool()],
        llm=worker_llm,
        allow_delegation=False,
        verbose=settings.verbose,
        max_iter=15,
    )

    data_analyst = Agent(
        role="Procurement Data Analyst",
        goal=(
            "Score and rank every scraped product using the procurement scoring "
            "calculator tool -- never rank by intuition. Surface the top "
            "recommendation, a credible runner-up, and clearly flag anything "
            "disqualified and why."
        ),
        backstory=(
            "You are a data analyst who believes rankings must be reproducible: if "
            "someone re-ran your numbers tomorrow, they should get the same "
            "ordering. You always call the scoring tool with complete, correctly "
            "formatted JSON rather than eyeballing which option looks best.\n\n"
            + context_block
        ),
        tools=[scoring_tool],
        llm=worker_llm,
        allow_delegation=False,
        verbose=settings.verbose,
    )

    compliance_officer = Agent(
        role="Procurement Compliance & Risk Officer",
        goal=(
            "Independently re-verify that the top recommendation actually respects "
            "the budget ceiling and every hard requirement, and flag any vendor, "
            "warranty, or supply-chain risk a procurement manager would want to know "
            "about before signing off."
        ),
        backstory=(
            "You are the last line of defense before a purchase order goes out. "
            "You have no stake in the recommendation looking good -- your job is to "
            "find the reason NOT to approve it, if one exists, and state it plainly.\n\n"
            + context_block
        ),
        llm=worker_llm,
        allow_delegation=False,
        verbose=settings.verbose,
    )

    report_writer = Agent(
        role="Procurement Report Writer",
        goal=(
            "Turn the ranked comparison and compliance review into a clear, "
            "professional executive summary and full structured recommendation "
            "that a procurement manager could act on without needing to re-read "
            "any of the raw agent output."
        ),
        backstory=(
            "You write for busy executives: lead with the recommendation and the "
            "'why', keep the language plain, and never bury the number that matters "
            "(total cost) in a paragraph.\n\n" + context_block
        ),
        llm=worker_llm,
        allow_delegation=False,
        verbose=settings.verbose,
    )

    return {
        "requirements_analyst": requirements_analyst,
        "market_researcher": market_researcher,
        "web_scraper": web_scraper,
        "data_analyst": data_analyst,
        "compliance_officer": compliance_officer,
        "report_writer": report_writer,
    }
