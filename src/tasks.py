"""
src/tasks.py
-------------
Wires the six agents into a sequential pipeline of six tasks. Each task:

  1. Has an `expected_output` precise enough that the agent knows exactly
     what "done" looks like.
  2. Uses `output_pydantic` so CrewAI validates (and retries on) malformed
     JSON before it ever reaches the next agent.
  3. Declares `context=[...]` explicitly rather than relying purely on
     sequential ordering, so it's obvious in the code which upstream outputs
     each task actually depends on.

Two tasks additionally carry a `guardrail` function -- deterministic Python
validation that runs *after* the LLM produces output and can force a retry
with feedback. This is where "advanced" earns its keep: guardrails catch
budget or requirement violations mechanically, instead of hoping the LLM
polices itself.
"""
from __future__ import annotations

import src.config  # noqa: F401  (must run first: sets CREWAI_DISABLE_TELEMETRY before `import crewai` below)
from crewai import Agent, Task
from crewai.tasks.task_output import TaskOutput

from src.company_context import CompanyProfile
from src.config import settings
from src.models import (
    ComparisonOutput,
    ComplianceReview,
    MarketResearchOutput,
    ProcurementRecommendation,
    ScrapingOutput,
)


# ---------------------------------------------------------------------------
# Guardrails: (bool, str_or_output) tuples. On failure, CrewAI feeds the
# string back to the agent as correction feedback and retries (up to
# `guardrail_max_retries`).
# ---------------------------------------------------------------------------

def budget_guardrail(output: TaskOutput):
    """Attached to the Data Analyst's comparison task: every ranked product
    must have a non-negative weighted score and rank 1 must actually be the
    highest score -- catches the class of bug where the tool ran but the
    LLM transcribed/re-sorted the results incorrectly in its final answer."""
    data: ComparisonOutput = output.pydantic
    if data is None:
        return False, "Output did not parse as ComparisonOutput. Return valid JSON matching the schema."

    if not data.ranked_products:
        return False, "ranked_products is empty. You must rank at least one product, or explain in methodology_note why none qualified."

    scores = [rp.score.weighted_total for rp in data.ranked_products]
    if scores != sorted(scores, reverse=True):
        return False, "ranked_products is not sorted by weighted_total descending. Re-sort using the scoring tool's output order."

    for rp in data.ranked_products:
        if rp.rank < 1:
            return False, f"Product '{rp.product.product_name}' has invalid rank {rp.rank}."

    return True, data


def compliance_guardrail(output: TaskOutput):
    """Attached to the Compliance Officer's task: an 'approved' review must
    also be budget_compliant -- an agent cannot approve something it flagged
    as over budget. Forces internal consistency instead of contradictory
    natural-language output."""
    data: ComplianceReview = output.pydantic
    if data is None:
        return False, "Output did not parse as ComplianceReview. Return valid JSON matching the schema."
    if data.approved and not data.budget_compliant:
        return False, (
            "Inconsistent review: approved=True but budget_compliant=False. "
            "A product that violates the budget ceiling cannot be approved. "
            "Set approved=False, or re-check the budget math."
        )
    return True, data


def build_tasks(agents: dict[str, Agent], company: CompanyProfile, product_request: str) -> list[Task]:
    requirements_task = Task(
        name="requirements_analysis",
        description=(
            f"A buyer has asked for: \"{product_request}\"\n\n"
            "Using the company context in your backstory, produce a requirements "
            "brief containing:\n"
            "1. A normalized product category name.\n"
            "2. 3-5 concrete, distinct web search queries likely to surface real "
            "current product listings (vary vendor type: manufacturer direct, "
            "major retailer, specialty distributor).\n"
            "3. The full list of hard requirements (from company context PLUS "
            "anything implied by the request itself).\n"
            "4. Any clarifying assumptions you had to make about the request."
        ),
        expected_output=(
            "A clearly structured brief with the four numbered sections above, "
            "written so the Market Researcher can copy the search queries directly."
        ),
        agent=agents["requirements_analyst"],
    )

    research_task = Task(
        name="market_research",
        description=(
            "Using the requirements brief, run web searches to find real, currently "
            "available product listings. For each promising result, capture the "
            "product name, vendor/retailer, exact source URL, any price visible in "
            "the snippet, and a 1-2 sentence note on why it's relevant. "
            f"Aim for at least {settings.max_sources} distinct vendors; "
            "do not return duplicate listings of the same product from the same vendor."
        ),
        expected_output=(
            "A MarketResearchOutput JSON object listing every candidate found, "
            "the exact queries used, and research notes on any gaps or caveats."
        ),
        agent=agents["market_researcher"],
        context=[requirements_task],
        output_pydantic=MarketResearchOutput,
    )

    scraping_task = Task(
        name="data_extraction",
        description=(
            "For every lead in the market research output, visit the source URL "
            "and extract structured product data: exact price, currency, stock "
            "status, warranty, estimated delivery, and every technical "
            "specification you can find. Cross-check each product against the "
            "company's hard requirements and set meets_hard_requirements "
            "accordingly, with a disqualification_reason when false. If a page "
            "cannot be scraped, note it in scraping_notes and skip it rather than "
            "fabricating data."
        ),
        expected_output=(
            "A ScrapingOutput JSON object with one ScrapedProduct entry per "
            "successfully scraped lead, plus counts of sources attempted/failed."
        ),
        agent=agents["web_scraper"],
        context=[requirements_task, research_task],
        output_pydantic=ScrapingOutput,
    )

    comparison_task = Task(
        name="scoring_and_comparison",
        description=(
            "Take every product from the extraction step that meets the hard "
            "requirements. For each, determine matched_required_specs / "
            "total_required_specs and matched_nice_to_have_specs / "
            "total_nice_to_have_specs by comparing its key_specifications against "
            "the company's requirements. Then call the procurement_scoring_calculator "
            "tool ONCE with the full JSON array of all qualifying products -- do not "
            "call it once per product, since price scoring is relative to the "
            "cheapest option in the whole set. Use the tool's exact output to "
            "populate each RankedProduct's score and rank; do not alter the numbers "
            "or reorder the list yourself. Add a one-line value_verdict and any "
            "risk_flags (e.g. single-vendor availability, no warranty, price outlier) "
            "per product. List disqualified products separately with their reason."
        ),
        expected_output=(
            "A ComparisonOutput JSON object: ranked_products sorted by weighted_total "
            "descending (exactly matching the scoring tool's output), "
            "disqualified_products with reasons, and a methodology_note."
        ),
        agent=agents["data_analyst"],
        context=[requirements_task, scraping_task],
        output_pydantic=ComparisonOutput,
        guardrail=budget_guardrail,
        guardrail_max_retries=3,
    )

    compliance_task = Task(
        name="compliance_review",
        description=(
            "Independently re-verify the #1 ranked product: does its price times "
            f"the requested quantity ({company.quantity_needed}) stay within the "
            f"total budget ceiling of {company.currency} "
            f"{company.procurement_budget_max * company.quantity_needed:,.2f}? Does it "
            "genuinely satisfy every hard requirement (re-check, don't just trust the "
            "prior step)? Flag any vendor, warranty, single-source, or supply-chain "
            "risk. Set approved=True only if budget_compliant AND all hard "
            "requirements are genuinely met."
        ),
        expected_output=(
            "A ComplianceReview JSON object: approved, budget_compliant, "
            "flagged_risks, and reviewer_notes explaining the decision."
        ),
        agent=agents["compliance_officer"],
        context=[requirements_task, comparison_task],
        output_pydantic=ComplianceReview,
        guardrail=compliance_guardrail,
        guardrail_max_retries=2,
    )

    report_task = Task(
        name="final_report",
        description=(
            "Synthesize everything into a final ProcurementRecommendation: a "
            "1-paragraph request_summary, the top_recommendation (rank #1 product "
            "with its full score breakdown), runner_up (rank #2, if one exists), "
            "the full_comparison (all ranked + disqualified products), the "
            "compliance_review verbatim, a 3-5 sentence executive_summary written "
            "for a procurement manager who has not read any prior output, and "
            f"estimated_total_cost = top pick's price * {company.quantity_needed} units. "
            f"Set generated_for to '{company.company_name}'."
        ),
        expected_output=(
            "A complete ProcurementRecommendation JSON object with every field "
            "populated -- this is the final deliverable, treat it as such."
        ),
        agent=agents["report_writer"],
        context=[requirements_task, comparison_task, compliance_task],
        output_pydantic=ProcurementRecommendation,
    )

    return [
        requirements_task,
        research_task,
        scraping_task,
        comparison_task,
        compliance_task,
        report_task,
    ]
