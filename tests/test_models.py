"""
tests/test_models.py
----------------------
Sanity checks on the Pydantic contracts and company context rendering.
These matter more than they might look: `output_pydantic` in tasks.py means
CrewAI will validate every LLM response against these exact models, so a
typo here silently changes what the agents are allowed to return.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from pydantic import ValidationError

from src.company_context import DEFAULT_COMPANY, CompanyProfile, PriorityWeights
from src.models import ComparisonOutput, RankedProduct, ScoreBreakdown, ScrapedProduct


def _sample_scraped_product() -> ScrapedProduct:
    return ScrapedProduct(
        product_name="Test Printer",
        vendor="Test Vendor",
        source_url="https://example.com/product",
        price=999.0,
        currency="USD",
        meets_hard_requirements=True,
    )


def test_default_company_builds_valid_prompt_block():
    block = DEFAULT_COMPANY.as_prompt_block()
    assert "Northbridge Robotics" in block
    assert "COMPANY CONTEXT" in block
    assert "3,500.00" in block  # per-unit budget formatted with thousands separator


def test_priority_weights_reject_out_of_range():
    with pytest.raises(ValidationError):
        PriorityWeights(price=1.5, specifications=0.1, vendor_reliability=0.1, delivery_speed=0.1)


def test_company_profile_requires_core_fields():
    with pytest.raises(ValidationError):
        CompanyProfile(company_name="Acme")  # missing industry, size, location, budget, etc.


def test_score_breakdown_enforces_0_to_100_bounds():
    with pytest.raises(ValidationError):
        ScoreBreakdown(price_score=150, spec_score=50, vendor_score=50, delivery_score=50, weighted_total=50)


def test_ranked_product_roundtrips_through_json():
    rp = RankedProduct(
        rank=1,
        product=_sample_scraped_product(),
        score=ScoreBreakdown(price_score=80, spec_score=90, vendor_score=70, delivery_score=60, weighted_total=79.5),
        value_verdict="Best value",
    )
    dumped = rp.model_dump_json()
    reloaded = RankedProduct.model_validate_json(dumped)
    assert reloaded.product.product_name == "Test Printer"
    assert reloaded.score.weighted_total == 79.5


def test_comparison_output_defaults_empty_disqualified_list():
    co = ComparisonOutput(
        ranked_products=[],
        methodology_note="no candidates qualified",
    )
    assert co.disqualified_products == []
