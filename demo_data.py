"""
demo_data.py
-------------
Hand-built, realistic fixture data shaped exactly like what the crew would
produce for the DEFAULT_COMPANY scenario (Northbridge Robotics shopping for
enclosed-chamber 3D printers). This lets you:

  1. Verify report_generator.py's HTML/chart output looks right without
     spending a single API call.
  2. Verify the deterministic scoring engine end-to-end against numbers you
     can check by hand.
  3. Run `python main.py --demo` as a fast smoke test in CI, or when
     evaluating this project without Tavily/ScrapeGraph/LLM keys configured.

None of this is used by the real crew at runtime -- agents.py/tasks.py never
import this module.
"""
from __future__ import annotations

from src.company_context import DEFAULT_COMPANY
from src.models import (
    ComparisonOutput,
    ComplianceReview,
    ProcurementRecommendation,
    RankedProduct,
    ScoreBreakdown,
    ScrapedProduct,
)
from src.tools.scoring_tool import ProductScoreInput, compute_procurement_score

_RAW_CANDIDATES = [
    dict(
        product_name="Bambu Lab X1-Carbon Combo",
        vendor="Bambu Lab (Direct)",
        source_url="https://us.store.bambulab.com/products/x1-carbon-combo",
        price=1449.00,
        currency="USD",
        in_stock=True,
        key_specifications={
            "build_volume": "256 x 256 x 256 mm",
            "chamber": "Enclosed",
            "connectivity": "Wi-Fi, Ethernet, USB",
            "extrusion": "Single (AMS multi-material add-on available)",
            "bed_leveling": "Automatic",
        },
        warranty="12-month manufacturer warranty",
        estimated_delivery="5-7 business days",
        meets_hard_requirements=True,
        disqualification_reason=None,
        matched_required=3, total_required=3,
        matched_nice=2, total_nice=3,
        vendor_hint="authorized manufacturer direct, established brand, 12mo warranty",
    ),
    dict(
        product_name="Prusa CORE One",
        vendor="Prusa Research (Direct)",
        source_url="https://www.prusa3d.com/product/prusa-core-one/",
        price=1299.00,
        currency="USD",
        in_stock=True,
        key_specifications={
            "build_volume": "250 x 220 x 270 mm",
            "chamber": "Enclosed",
            "connectivity": "Wi-Fi, Ethernet",
            "extrusion": "Single",
            "bed_leveling": "Automatic",
        },
        warranty="24-month manufacturer warranty",
        estimated_delivery="10-14 business days",
        meets_hard_requirements=True,
        disqualification_reason=None,
        matched_required=3, total_required=3,
        matched_nice=1, total_nice=3,
        vendor_hint="authorized manufacturer direct, established brand, 24mo warranty",
    ),
    dict(
        product_name="Creality K2 Plus",
        vendor="Creality Official Store (via Amazon)",
        source_url="https://www.amazon.com/dp/example-k2plus",
        price=1899.00,
        currency="USD",
        in_stock=True,
        key_specifications={
            "build_volume": "350 x 350 x 320 mm",
            "chamber": "Enclosed",
            "connectivity": "Wi-Fi, Ethernet, USB",
            "extrusion": "Dual-material capable",
            "bed_leveling": "Automatic",
        },
        warranty="12-month warranty via marketplace seller",
        estimated_delivery="3-5 business days",
        meets_hard_requirements=True,
        disqualification_reason=None,
        matched_required=3, total_required=3,
        matched_nice=3, total_nice=3,
        vendor_hint="third-party marketplace listing, 12mo warranty",
    ),
    dict(
        product_name="Elegoo Neptune 4 Max",
        vendor="Elegoo Official (Direct)",
        source_url="https://www.elegoo.com/products/neptune-4-max",
        price=379.00,
        currency="USD",
        in_stock=True,
        key_specifications={
            "build_volume": "420 x 420 x 480 mm",
            "chamber": "Open frame",
            "connectivity": "Wi-Fi",
            "extrusion": "Single",
            "bed_leveling": "Automatic",
        },
        warranty="12-month manufacturer warranty",
        estimated_delivery="6-9 business days",
        meets_hard_requirements=False,
        disqualification_reason="Open-frame chassis does not satisfy the enclosed-chamber hard requirement.",
        matched_required=2, total_required=3,
        matched_nice=1, total_nice=3,
        vendor_hint="manufacturer direct, budget brand, 12mo warranty",
    ),
    dict(
        product_name="Anycubic Kobra S1",
        vendor="GreyMarket3DSupplies (Marketplace)",
        source_url="https://www.example-marketplace.com/kobra-s1",
        price=1099.00,
        currency="USD",
        in_stock=True,
        key_specifications={
            "build_volume": "230 x 230 x 260 mm",
            "chamber": "Enclosed",
            "connectivity": "Wi-Fi",
            "extrusion": "Single",
            "bed_leveling": "Automatic",
        },
        warranty="No warranty listed",
        estimated_delivery="18-25 business days (imported)",
        meets_hard_requirements=True,
        disqualification_reason=None,
        matched_required=3, total_required=3,
        matched_nice=0, total_nice=3,
        vendor_hint="grey import, unverified seller, no warranty",
    ),
]


def build_demo_recommendation() -> ProcurementRecommendation:
    company = DEFAULT_COMPANY
    qualifying = [c for c in _RAW_CANDIDATES if c["meets_hard_requirements"]]
    disqualified_raw = [c for c in _RAW_CANDIDATES if not c["meets_hard_requirements"]]

    score_inputs = [
        ProductScoreInput(
            product_name=c["product_name"],
            price=c["price"],
            matched_required_specs=c["matched_required"],
            total_required_specs=c["total_required"],
            matched_nice_to_have_specs=c["matched_nice"],
            total_nice_to_have_specs=c["total_nice"],
            vendor_reliability_hint=c["vendor_hint"],
            warranty_present="no warranty" not in c["warranty"].lower(),
            estimated_delivery_days={
                "5-7 business days": 6, "10-14 business days": 12,
                "3-5 business days": 4, "18-25 business days (imported)": 22,
            }.get(c["estimated_delivery"], 14),
        )
        for c in qualifying
    ]
    scored = compute_procurement_score(
        products=score_inputs,
        weights=company.priority_weights,
        budget_max=company.procurement_budget_max,
        risk_tolerance=company.risk_tolerance.value,
    )
    scored_by_name = {s["product_name"]: s for s in scored}

    def _to_scraped(c: dict) -> ScrapedProduct:
        return ScrapedProduct(
            product_name=c["product_name"], vendor=c["vendor"], source_url=c["source_url"],
            price=c["price"], currency=c["currency"], in_stock=c["in_stock"],
            key_specifications=c["key_specifications"], warranty=c["warranty"],
            estimated_delivery=c["estimated_delivery"],
            meets_hard_requirements=c["meets_hard_requirements"],
            disqualification_reason=c["disqualification_reason"],
            scrape_confidence=0.93,
        )

    risk_flag_map = {
        "Creality K2 Plus": ["Sold via third-party marketplace listing"],
        "Anycubic Kobra S1": ["No warranty listed", "Grey-market import", "Long estimated delivery (18-25 days)"],
    }
    verdict_map = {
        1: "Best overall value within budget",
        2: "Strong runner-up; longer lead time",
    }

    ranked_products = []
    for c in qualifying:
        s = scored_by_name[c["product_name"]]
        ranked_products.append(RankedProduct(
            rank=s["rank"],
            product=_to_scraped(c),
            score=ScoreBreakdown(
                price_score=s["price_score"], spec_score=s["spec_score"],
                vendor_score=s["vendor_score"], delivery_score=s["delivery_score"],
                weighted_total=s["weighted_total"],
            ),
            value_verdict=verdict_map.get(s["rank"], "Evaluated candidate"),
            risk_flags=risk_flag_map.get(c["product_name"], []),
        ))
    ranked_products.sort(key=lambda r: r.rank)

    disqualified_products = [_to_scraped(c) for c in disqualified_raw]

    comparison = ComparisonOutput(
        ranked_products=ranked_products,
        disqualified_products=disqualified_products,
        methodology_note=(
            "Each qualifying product was scored 0-100 on four dimensions (price, "
            "specification match, vendor reliability, delivery speed) and combined "
            "using Northbridge Robotics' stated priority weights. Price scores are "
            "relative to the cheapest qualifying option in this candidate set; "
            "products failing a hard requirement were excluded before scoring."
        ),
    )

    top = ranked_products[0]
    compliance = ComplianceReview(
        approved=True,
        budget_compliant=True,
        flagged_risks=[
            "Verify current AMS multi-material add-on pricing separately if multi-material printing becomes a requirement.",
        ],
        reviewer_notes=(
            f"Total cost for {company.quantity_needed} units of the {top.product.product_name} "
            f"is {company.currency} {top.product.price * company.quantity_needed:,.2f}, within the "
            f"{company.currency} {company.procurement_budget_max * company.quantity_needed:,.2f} ceiling. "
            "All three hard requirements (build volume, enclosed chamber, network connectivity) are met. "
            "No blocking vendor or supply-chain risks identified."
        ),
    )

    return ProcurementRecommendation(
        request_summary=(
            "Sourcing 6 enclosed-chamber, network-connected 3D printers with a minimum "
            "250x250x250mm build volume for Northbridge Robotics' prototyping floor, "
            "at a per-unit budget ceiling of USD 3,500."
        ),
        top_recommendation=top,
        runner_up=ranked_products[1] if len(ranked_products) > 1 else None,
        full_comparison=comparison,
        compliance_review=compliance,
        executive_summary=(
            f"The {top.product.product_name} at {company.currency} {top.product.price:,.2f}/unit is the "
            f"recommended purchase, scoring {top.score.weighted_total:.1f}/100 -- the strongest combination "
            "of price and specification match among all qualifying candidates, backed by a 12-month "
            "manufacturer warranty and direct-from-manufacturer availability. It comes in well under the "
            f"per-unit budget ceiling, freeing roughly {company.currency} "
            f"{(company.procurement_budget_max - top.product.price) * company.quantity_needed:,.0f} of the "
            "total allocated budget. The Prusa CORE One is a credible runner-up with a longer manufacturer "
            "warranty, at a modestly higher price-to-spec tradeoff. Two candidates were disqualified or "
            "flagged: the Elegoo Neptune 4 Max for lacking an enclosed chamber, and the Anycubic Kobra S1 "
            "for vendor-reliability and import-timeline risk despite its competitive price."
        ),
        estimated_total_cost=top.product.price * company.quantity_needed,
        currency=company.currency,
        generated_for=company.company_name,
    )
