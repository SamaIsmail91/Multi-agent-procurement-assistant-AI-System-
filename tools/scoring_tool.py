"""
src/tools/scoring_tool.py
---------------------------
The single most important design decision in this project: **ranking is not
left to LLM vibes.** An LLM asked to "rank these 8 products by value" will
produce plausible-sounding but non-reproducible, non-auditable orderings --
unacceptable for a procurement decision someone has to defend to Finance.

Instead, this module implements a deterministic, weighted multi-criteria
scoring function in plain Python. The Data Analyst agent calls it as a tool
(passing structured `ScrapedProduct` data extracted by the Scraping Agent);
the LLM's job is to gather and interpret the inputs, not to compute the math.
This is a common and important pattern in production agent systems: use the
LLM for judgment/extraction/language, use code for anything that must be
exact, reproducible, and explainable.

`compute_procurement_score` has zero CrewAI/LLM dependencies, so it is
unit-testable in isolation (see tests/test_scoring.py) -- it will keep
working correctly even if the rest of the crew's prompts drift.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.company_context import PriorityWeights


# ---------------------------------------------------------------------------
# Pure scoring logic (no CrewAI import at module scope -> safe to unit test)
# ---------------------------------------------------------------------------

def _price_score(price: Optional[float], budget_max: float, cheapest_in_set: float) -> float:
    """
    100 = at or below the cheapest comparable option.
    0   = at or above the budget ceiling.
    Linear interpolation in between, so a product exactly at budget scores 0
    and a product matching the cheapest competitor scores 100 -- this
    rewards being closer to the best price in the *actual candidate set*,
    not an arbitrary global scale.
    """
    if price is None or price <= 0:
        return 0.0
    if price <= cheapest_in_set:
        return 100.0
    if price >= budget_max:
        return 0.0
    span = budget_max - cheapest_in_set
    if span <= 0:
        return 100.0 if price <= budget_max else 0.0
    return max(0.0, min(100.0, 100.0 * (budget_max - price) / span))


def _spec_score(matched_required: int, total_required: int, matched_nice_to_have: int, total_nice_to_have: int) -> float:
    """
    Hard requirements dominate (70% of this sub-score); nice-to-haves add a
    bonus on top (30%). A product that fails a hard requirement should
    already have been filtered out upstream (meets_hard_requirements=False),
    but we still score defensively in case it slips through.
    """
    req_ratio = (matched_required / total_required) if total_required else 1.0
    nice_ratio = (matched_nice_to_have / total_nice_to_have) if total_nice_to_have else 0.0
    return round(100.0 * (0.70 * req_ratio + 0.30 * nice_ratio), 2)


VENDOR_RISK_KEYWORDS = {
    "no warranty": -25, "unverified seller": -20, "third-party marketplace": -10,
    "discontinued": -30, "no local support": -10, "grey import": -20,
}


def _vendor_score(vendor_reliability_hint: str, risk_tolerance: str, warranty_present: bool) -> float:
    """
    Starts from a neutral 70 and adjusts based on textual risk signals plus
    warranty presence. `risk_tolerance` softens or sharpens the penalty --
    a "high" risk-tolerance buyer is explicitly willing to accept more of
    this deduction in exchange for value elsewhere.
    """
    score = 70.0
    hint_lower = (vendor_reliability_hint or "").lower()
    for keyword, penalty in VENDOR_RISK_KEYWORDS.items():
        if keyword in hint_lower:
            score += penalty
    if warranty_present:
        score += 15
    tolerance_multiplier = {"low": 1.3, "medium": 1.0, "high": 0.6}.get(risk_tolerance, 1.0)
    if score < 70:
        score = 70 - (70 - score) * tolerance_multiplier
    return round(max(0.0, min(100.0, score)), 2)


def _delivery_score(estimated_delivery_days: Optional[int]) -> float:
    if estimated_delivery_days is None:
        return 50.0  # unknown -> neutral, not punished
    if estimated_delivery_days <= 3:
        return 100.0
    if estimated_delivery_days >= 30:
        return 0.0
    return round(100.0 * (30 - estimated_delivery_days) / 27, 2)


class ProductScoreInput(BaseModel):
    product_name: str
    price: Optional[float] = None
    matched_required_specs: int
    total_required_specs: int
    matched_nice_to_have_specs: int = 0
    total_nice_to_have_specs: int = 0
    vendor_reliability_hint: str = Field(
        "", description="Free text signals, e.g. 'authorized reseller, 2yr warranty'"
    )
    warranty_present: bool = False
    estimated_delivery_days: Optional[int] = None


def compute_procurement_score(
    products: list[ProductScoreInput],
    weights: PriorityWeights,
    budget_max: float,
    risk_tolerance: str = "medium",
) -> list[dict[str, Any]]:
    """
    Scores and ranks a batch of products together (batch, not one-at-a-time,
    because price scoring is relative to the cheapest option *within this
    candidate set* -- see `_price_score`).
    """
    w = weights.normalized()
    valid_prices = [p.price for p in products if p.price and p.price > 0]
    cheapest = min(valid_prices) if valid_prices else budget_max

    results = []
    for p in products:
        price_s = _price_score(p.price, budget_max, cheapest)
        spec_s = _spec_score(
            p.matched_required_specs, p.total_required_specs,
            p.matched_nice_to_have_specs, p.total_nice_to_have_specs,
        )
        vendor_s = _vendor_score(p.vendor_reliability_hint, risk_tolerance, p.warranty_present)
        delivery_s = _delivery_score(p.estimated_delivery_days)

        weighted_total = round(
            price_s * w.price
            + spec_s * w.specifications
            + vendor_s * w.vendor_reliability
            + delivery_s * w.delivery_speed,
            2,
        )
        results.append({
            "product_name": p.product_name,
            "price_score": round(price_s, 2),
            "spec_score": spec_s,
            "vendor_score": vendor_s,
            "delivery_score": delivery_s,
            "weighted_total": weighted_total,
        })

    results.sort(key=lambda r: r["weighted_total"], reverse=True)
    for i, r in enumerate(results, start=1):
        r["rank"] = i
    return results


# ---------------------------------------------------------------------------
# CrewAI tool wrapper. Imported lazily so tests / non-crew code paths never
# need the crewai package installed just to exercise the scoring math above.
# ---------------------------------------------------------------------------

def build_scoring_tool(weights: PriorityWeights, budget_max: float, risk_tolerance: str):
    from crewai.tools import BaseTool

    class ProcurementScoringInput(BaseModel):
        products_json: str = Field(
            description=(
                "JSON array of objects, one per product, each with keys: "
                "product_name, price, matched_required_specs, total_required_specs, "
                "matched_nice_to_have_specs, total_nice_to_have_specs, "
                "vendor_reliability_hint, warranty_present, estimated_delivery_days"
            )
        )

    class ProcurementScoringTool(BaseTool):
        name: str = "procurement_scoring_calculator"
        description: str = (
            "Deterministically scores and ranks a batch of candidate products "
            "on a 0-100 scale for price, specifications, vendor reliability, "
            "and delivery speed, then combines them into a single weighted "
            "total using this company's stated priority weights. ALWAYS use "
            "this tool to rank products -- never rank them by intuition, "
            "since the score must be reproducible and defensible to a "
            "procurement manager. Input must be a JSON array string."
        )
        args_schema: type[BaseModel] = ProcurementScoringInput

        def _run(self, products_json: str) -> str:
            try:
                raw_items = json.loads(products_json)
            except json.JSONDecodeError as e:
                return f"ERROR: products_json was not valid JSON ({e}). Re-check formatting and retry."

            try:
                parsed = [ProductScoreInput(**item) for item in raw_items]
            except Exception as e:  # noqa: BLE001
                return f"ERROR: product entries didn't match the required schema: {e}"

            scored = compute_procurement_score(
                products=parsed, weights=weights, budget_max=budget_max,
                risk_tolerance=risk_tolerance,
            )
            return json.dumps(scored, indent=2)

    return ProcurementScoringTool()
