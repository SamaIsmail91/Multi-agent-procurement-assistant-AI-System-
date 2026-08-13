"""
src/flow.py
------------
OPTIONAL, PRODUCTION-HARDENING ALTERNATIVE to src/crew.py.

`ProcurementCrew` (crew.py) runs a fixed sequential pipeline -- simple,
predictable, and the right default. But it can't express conditional
logic like "if market research only turned up 1 usable lead, broaden the
search and try again before wasting a scraping pass on a starved
candidate set." A plain sequential Crew either runs every task once or
it doesn't; there's no branch point.

CrewAI Flows solve exactly this: they're event-driven graphs (@start /
@listen / @router) with typed, persistent state, built for the "this step
sometimes needs to loop back" cases that are common once a crew leaves the
demo stage and starts hitting real (flaky, sparse, rate-limited) web data.

This module re-uses the exact same agents and tasks from agents.py/tasks.py
-- it does not duplicate their prompts -- it only changes the *control flow*
around them:

    gather_requirements
            |
      research_market
            |
      [router] enough leads? --no--> broaden_and_retry (max 1x) --+
            |yes                                                   |
            +-----------------------------------------------------+
            |
      scrape_products
            |
      [router] scrape success rate ok? --no--> rescrape_with_fallback_tool
            |yes                                                    |
            +------------------------------------------------------+
            |
      score_and_compare
            |
      compliance_review
            |
      write_report  -> ProcurementRecommendation

Run with: `python main.py --engine flow`
"""
from __future__ import annotations

import logging

import src.config  # noqa: F401  (must run first: sets CREWAI_DISABLE_TELEMETRY before `import crewai` below)
from crewai import Crew, Process, Task
from crewai.flow.flow import Flow, listen, or_, router, start
from pydantic import BaseModel, Field

from src.agents import build_agents
from src.company_context import CompanyProfile
from src.config import settings
from src.models import (
    ComparisonOutput,
    ComplianceReview,
    MarketResearchOutput,
    ProcurementRecommendation,
    ScrapingOutput,
)
from src.tasks import (
    budget_guardrail,
    compliance_guardrail,
)

logger = logging.getLogger("procurement_assistant.flow")

MIN_ACCEPTABLE_LEADS = 2
MAX_SCRAPE_FAILURE_RATE = 0.5
MAX_RETRIES = 1


def _run_single_task(task: Task) -> "TaskOutputLike":
    """Executes exactly one Task via a throwaway single-agent Crew. Flows
    orchestrate *between* steps; each step still delegates the actual LLM
    work to a normal CrewAI Task so we keep one definition of each agent's
    prompt (in tasks.py) instead of forking it."""
    mini_crew = Crew(agents=[task.agent], tasks=[task], process=Process.sequential, verbose=settings.verbose)
    mini_crew.kickoff()
    return task.output


class ProcurementFlowState(BaseModel):
    product_request: str = ""
    company_name: str = ""
    retry_count: int = 0
    requirements_brief: str = ""
    research: dict = Field(default_factory=dict)
    scraping: dict = Field(default_factory=dict)
    comparison: dict = Field(default_factory=dict)
    compliance: dict = Field(default_factory=dict)
    final_report: dict = Field(default_factory=dict)


class ProcurementFlow(Flow[ProcurementFlowState]):
    """Instantiate with `ProcurementFlow(company=..., product_request=...)`."""

    def __init__(self, company: CompanyProfile, product_request: str, **kwargs):
        super().__init__(**kwargs)
        settings.export_to_environ()
        self.company = company
        self.agents = build_agents(company)
        self.state.product_request = product_request
        self.state.company_name = company.company_name

    # -- helpers to build one-off tasks bound to this flow's agents --------
    def _task(self, **kwargs) -> Task:
        return Task(**kwargs)

    @start()
    def gather_requirements(self):
        logger.info("[flow] gathering requirements")
        task = self._task(
            description=(
                f"A buyer has asked for: \"{self.state.product_request}\"\n\n"
                "Produce a requirements brief: normalized product category, "
                "3-5 concrete search queries, hard requirements, and clarifying "
                "assumptions."
            ),
            expected_output="A structured requirements brief.",
            agent=self.agents["requirements_analyst"],
        )
        output = _run_single_task(task)
        self.state.requirements_brief = output.raw
        return output.raw

    @listen(gather_requirements)
    def research_market(self, requirements_brief: str):
        logger.info("[flow] researching market (attempt %d)", self.state.retry_count + 1)
        task = self._task(
            description=(
                f"Requirements brief:\n{requirements_brief}\n\n"
                "Search the web for real, currently available product listings "
                "matching this brief across multiple vendors."
                + (
                    "\n\nNOTE: a prior search attempt returned too few usable "
                    "leads. Broaden the search: try more generic category terms, "
                    "include marketplace aggregators, and relax brand assumptions."
                    if self.state.retry_count > 0 else ""
                )
            ),
            expected_output="A MarketResearchOutput JSON object.",
            agent=self.agents["market_researcher"],
            output_pydantic=MarketResearchOutput,
        )
        output = _run_single_task(task)
        self.state.research = output.pydantic.model_dump() if output.pydantic else {"leads": []}
        return self.state.research

    @router(research_market)
    def check_lead_count(self):
        lead_count = len(self.state.research.get("leads", []))
        if lead_count < MIN_ACCEPTABLE_LEADS and self.state.retry_count < MAX_RETRIES:
            self.state.retry_count += 1
            logger.warning("[flow] only %d leads found, retrying with broadened search", lead_count)
            return "insufficient_leads"
        return "sufficient_leads"

    @listen("insufficient_leads")
    def broaden_and_retry(self):
        return self.research_market(self.state.requirements_brief)

    @listen(or_("sufficient_leads", broaden_and_retry))
    def scrape_products(self, *_):
        logger.info("[flow] scraping candidate product pages")
        task = self._task(
            description=(
                f"Requirements brief:\n{self.state.requirements_brief}\n\n"
                f"Candidate leads:\n{self.state.research}\n\n"
                "Visit every lead's source URL and extract structured product "
                "data (price, specs, warranty, delivery). Set "
                "meets_hard_requirements accordingly."
            ),
            expected_output="A ScrapingOutput JSON object.",
            agent=self.agents["web_scraper"],
            output_pydantic=ScrapingOutput,
        )
        output = _run_single_task(task)
        self.state.scraping = output.pydantic.model_dump() if output.pydantic else {
            "products": [], "sources_attempted": 0, "sources_failed": 0, "scraping_notes": "parse failure"
        }
        return self.state.scraping

    @router(scrape_products)
    def check_scrape_health(self):
        attempted = max(self.state.scraping.get("sources_attempted", 0), 1)
        failed = self.state.scraping.get("sources_failed", 0)
        if (failed / attempted) > MAX_SCRAPE_FAILURE_RATE and self.state.retry_count < MAX_RETRIES + 1:
            self.state.retry_count += 1
            logger.warning("[flow] scrape failure rate %.0f%% - retrying with fallback tool", 100 * failed / attempted)
            return "scraping_degraded"
        return "scraping_ok"

    @listen("scraping_degraded")
    def rescrape_with_fallback_tool(self):
        # Swap the scraper agent's tool for the no-API-key fallback and retry
        # once. This keeps the flow resilient to a ScrapeGraph outage/quota
        # limit instead of failing the whole pipeline outright.
        from crewai_tools import ScrapeWebsiteTool

        self.agents["web_scraper"].tools = [ScrapeWebsiteTool()]
        return self.scrape_products()

    @listen(or_("scraping_ok", rescrape_with_fallback_tool))
    def score_and_compare(self, *_):
        logger.info("[flow] scoring and ranking candidates")
        task = self._task(
            description=(
                f"Scraped products:\n{self.state.scraping}\n\n"
                "For each product meeting the hard requirements, call the "
                "procurement_scoring_calculator tool ONCE with the full JSON "
                "array of qualifying products, then populate ranked_products "
                "using the tool's exact output."
            ),
            expected_output="A ComparisonOutput JSON object.",
            agent=self.agents["data_analyst"],
            output_pydantic=ComparisonOutput,
            guardrail=budget_guardrail,
            guardrail_max_retries=3,
        )
        output = _run_single_task(task)
        self.state.comparison = output.pydantic.model_dump() if output.pydantic else {}
        return self.state.comparison

    @listen(score_and_compare)
    def compliance_review(self, comparison: dict):
        logger.info("[flow] running compliance review")
        task = self._task(
            description=(
                f"Ranked comparison:\n{comparison}\n\n"
                f"Company budget ceiling (total): {self.company.currency} "
                f"{self.company.procurement_budget_max * self.company.quantity_needed:,.2f}. "
                "Independently verify the #1 ranked product on budget and hard "
                "requirements. Set approved=True only if fully compliant."
            ),
            expected_output="A ComplianceReview JSON object.",
            agent=self.agents["compliance_officer"],
            output_pydantic=ComplianceReview,
            guardrail=compliance_guardrail,
            guardrail_max_retries=2,
        )
        output = _run_single_task(task)
        self.state.compliance = output.pydantic.model_dump() if output.pydantic else {}
        return self.state.compliance

    @listen(compliance_review)
    def write_report(self, compliance: dict):
        logger.info("[flow] writing final report")
        task = self._task(
            description=(
                f"Comparison:\n{self.state.comparison}\n\nCompliance review:\n{compliance}\n\n"
                f"Synthesize a complete ProcurementRecommendation for "
                f"{self.company.company_name}, requesting {self.company.quantity_needed} units."
            ),
            expected_output="A complete ProcurementRecommendation JSON object.",
            agent=self.agents["report_writer"],
            output_pydantic=ProcurementRecommendation,
        )
        output = _run_single_task(task)
        self.state.final_report = output.pydantic.model_dump() if output.pydantic else {}
        return output.pydantic

    def run(self) -> ProcurementRecommendation:
        result = self.kickoff()
        if isinstance(result, ProcurementRecommendation):
            return result
        return ProcurementRecommendation.model_validate(self.state.final_report)
