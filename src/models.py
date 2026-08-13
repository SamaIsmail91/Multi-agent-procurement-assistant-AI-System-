"""
src/models.py
--------------
Structured contracts passed between agents via `Task(output_pydantic=...)`.

Why this matters: the default failure mode of a multi-agent crew is agent 2
receiving a wall of agent 1's prose and having to re-parse it. By forcing
every hand-off through a Pydantic schema, CrewAI validates the LLM's JSON
against the model and retries automatically on a mismatch (see guardrails in
tasks.py), so bad data gets caught at the boundary instead of silently
corrupting the final report.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ProductLead(BaseModel):
    """A single candidate surfaced by market research, pre-scraping."""
    product_name: str
    vendor_or_retailer: str
    source_url: str
    listed_price: Optional[float] = Field(None, description="Price if visible in search snippet")
    currency: Optional[str] = "USD"
    snippet_summary: str = Field(description="1-2 sentence summary of why this looks relevant")


class MarketResearchOutput(BaseModel):
    query_used: List[str]
    leads: List[ProductLead]
    research_notes: str = Field(description="Caveats, gaps, or market observations")


class ScrapedProduct(BaseModel):
    """Enriched product record after the scraping agent visits each lead."""
    product_name: str
    vendor: str
    source_url: str
    price: Optional[float] = None
    currency: str = "USD"
    in_stock: Optional[bool] = None
    key_specifications: dict[str, str] = Field(
        default_factory=dict, description="spec name -> value, e.g. {'build_volume': '300x300x300mm'}"
    )
    warranty: Optional[str] = None
    estimated_delivery: Optional[str] = None
    meets_hard_requirements: bool = Field(
        description="Whether this product satisfies ALL of the company's required_specifications"
    )
    disqualification_reason: Optional[str] = Field(
        None, description="Set if meets_hard_requirements is False"
    )
    scrape_confidence: float = Field(
        0.8, ge=0, le=1, description="Agent's confidence the extracted data is accurate/complete"
    )


class ScrapingOutput(BaseModel):
    products: List[ScrapedProduct]
    sources_attempted: int
    sources_failed: int
    scraping_notes: str


class ScoreBreakdown(BaseModel):
    price_score: float = Field(ge=0, le=100)
    spec_score: float = Field(ge=0, le=100)
    vendor_score: float = Field(ge=0, le=100)
    delivery_score: float = Field(ge=0, le=100)
    weighted_total: float = Field(ge=0, le=100)


class RankedProduct(BaseModel):
    rank: int
    product: ScrapedProduct
    score: ScoreBreakdown
    value_verdict: str = Field(description="One-line verdict, e.g. 'Best value' / 'Premium option'")
    risk_flags: List[str] = Field(default_factory=list)


class ComparisonOutput(BaseModel):
    ranked_products: List[RankedProduct]
    disqualified_products: List[ScrapedProduct] = Field(default_factory=list)
    methodology_note: str = Field(
        description="Short explanation of how the weighted score was computed"
    )


class ComplianceReview(BaseModel):
    approved: bool
    budget_compliant: bool
    flagged_risks: List[str] = Field(default_factory=list)
    reviewer_notes: str


class ProcurementRecommendation(BaseModel):
    """Final structured payload the report generator turns into HTML."""
    request_summary: str
    top_recommendation: RankedProduct
    runner_up: Optional[RankedProduct] = None
    full_comparison: ComparisonOutput
    compliance_review: ComplianceReview
    executive_summary: str = Field(description="3-5 sentence summary for a procurement manager")
    estimated_total_cost: float
    currency: str
    generated_for: str = Field(description="Company name this report was generated for")
