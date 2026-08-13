"""
tests/test_scoring.py
-----------------------
Unit tests for the deterministic scoring engine. These import
`src.tools.scoring_tool` at the *function* level (not the CrewAI tool
wrapper), so they run with zero LLM/CrewAI dependency -- fast, fully
offline, and the right place to pin down the scoring math's behavior with
concrete numbers instead of eyeballing agent output.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.company_context import PriorityWeights
from src.tools.scoring_tool import (
    ProductScoreInput,
    _delivery_score,
    _price_score,
    _spec_score,
    _vendor_score,
    compute_procurement_score,
)


def test_price_score_at_cheapest_is_100():
    assert _price_score(price=100, budget_max=200, cheapest_in_set=100) == 100.0


def test_price_score_at_budget_ceiling_is_0():
    assert _price_score(price=200, budget_max=200, cheapest_in_set=100) == 0.0


def test_price_score_midpoint_is_50():
    assert _price_score(price=150, budget_max=200, cheapest_in_set=100) == 50.0


def test_price_score_missing_price_is_0():
    assert _price_score(price=None, budget_max=200, cheapest_in_set=100) == 0.0


def test_spec_score_all_requirements_met_no_nice_to_haves():
    assert _spec_score(3, 3, 0, 0) == 70.0  # 0.70 * 1.0 + 0.30 * 0.0


def test_spec_score_full_marks():
    assert _spec_score(3, 3, 2, 2) == 100.0


def test_spec_score_partial_requirements():
    # 0.70 * (2/3) + 0.30 * 0 = 46.67
    assert round(_spec_score(2, 3, 0, 2), 2) == 46.67


def test_vendor_score_no_signals_is_neutral():
    assert _vendor_score("", "medium", warranty_present=False) == 70.0


def test_vendor_score_warranty_boosts():
    assert _vendor_score("", "medium", warranty_present=True) == 85.0


def test_vendor_score_no_warranty_keyword_penalizes():
    score = _vendor_score("no warranty listed", "medium", warranty_present=False)
    assert score < 70.0


def test_vendor_score_high_risk_tolerance_softens_penalty():
    low = _vendor_score("grey import", "low", warranty_present=False)
    high = _vendor_score("grey import", "high", warranty_present=False)
    assert high > low  # same signal, less penalty when buyer tolerates more risk


def test_delivery_score_fast_is_100():
    assert _delivery_score(2) == 100.0


def test_delivery_score_slow_is_0():
    assert _delivery_score(45) == 0.0


def test_delivery_score_unknown_is_neutral():
    assert _delivery_score(None) == 50.0


def test_compute_procurement_score_ranks_cheaper_higher_when_specs_tied():
    weights = PriorityWeights(price=1.0, specifications=0.0, vendor_reliability=0.0, delivery_speed=0.0)
    products = [
        ProductScoreInput(product_name="Cheap", price=100, matched_required_specs=3, total_required_specs=3),
        ProductScoreInput(product_name="Expensive", price=190, matched_required_specs=3, total_required_specs=3),
    ]
    result = compute_procurement_score(products, weights, budget_max=200, risk_tolerance="medium")
    assert result[0]["product_name"] == "Cheap"
    assert result[0]["rank"] == 1
    assert result[1]["product_name"] == "Expensive"
    assert result[1]["rank"] == 2


def test_compute_procurement_score_is_sorted_descending():
    weights = PriorityWeights()
    products = [
        ProductScoreInput(product_name=f"P{i}", price=100 + i * 50,
                           matched_required_specs=3, total_required_specs=3,
                           warranty_present=bool(i % 2))
        for i in range(5)
    ]
    result = compute_procurement_score(products, weights, budget_max=1000, risk_tolerance="medium")
    totals = [r["weighted_total"] for r in result]
    assert totals == sorted(totals, reverse=True)
    assert [r["rank"] for r in result] == list(range(1, 6))


def test_weights_normalize_when_not_summing_to_one():
    w = PriorityWeights(price=0.6, specifications=0.6, vendor_reliability=0.0, delivery_speed=0.0)
    n = w.normalized()
    assert round(n.price + n.specifications + n.vendor_reliability + n.delivery_speed, 6) == 1.0
    assert n.price == 0.5
