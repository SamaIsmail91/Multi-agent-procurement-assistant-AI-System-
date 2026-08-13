"""
src/report_generator.py
-------------------------
Turns a validated `ProcurementRecommendation` (the Report Writer agent's
structured output) into the final polished HTML deliverable, via Jinja2 +
Chart.js. Deliberately kept separate from the crew/agent code: report
rendering is pure, deterministic, and template-driven, so it can be tested
and iterated on with mock data (see demo_data.py) without ever calling an
LLM or touching the network -- and if you swap CrewAI for a different
orchestration layer later, this module doesn't change at all.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.company_context import CompanyProfile
from src.config import TEMPLATE_DIR
from src.models import ProcurementRecommendation
from src.utils.currency import format_money


def _stamp_visuals(score: float, approved: bool) -> tuple[str, float]:
    """
    Maps a 0-100 score to an SVG stroke-dasharray value (circumference of
    the stamp ring is ~2*pi*56 ≈ 352; we scale so the arc length matches the
    score directly) and picks the stamp's ink color: teal for a compliance-
    approved top pick, brass/amber when it's a strong score but still
    pending sign-off, brick red when the score itself is weak.
    """
    circumference = 2 * 3.14159265 * 56
    dasharray = round(circumference * (max(0, min(100, score)) / 100), 1)
    if approved and score >= 60:
        color = "var(--teal)"
    elif score >= 40:
        color = "var(--brass)"
    else:
        color = "var(--brick)"
    return color, dasharray


def _radar_chart_json(report: ProcurementRecommendation) -> str:
    top_candidates = report.full_comparison.ranked_products[:3]
    labels = ["Price", "Specifications", "Vendor Reliability", "Delivery Speed"]
    palette = [
        ("rgba(46, 94, 170, 0.75)", "rgba(46, 94, 170, 0.15)"),
        ("rgba(184, 134, 59, 0.85)", "rgba(184, 134, 59, 0.15)"),
        ("rgba(91, 140, 104, 0.85)", "rgba(91, 140, 104, 0.15)"),
    ]
    datasets = []
    for i, rp in enumerate(top_candidates):
        border, fill = palette[i % len(palette)]
        datasets.append({
            "label": rp.product.product_name[:32],
            "data": [rp.score.price_score, rp.score.spec_score, rp.score.vendor_score, rp.score.delivery_score],
            "borderColor": border,
            "backgroundColor": fill,
            "borderWidth": 2,
            "pointRadius": 3,
        })
    return json.dumps({"labels": labels, "datasets": datasets})


def generate_html_report(
    report: ProcurementRecommendation,
    company: CompanyProfile,
    output_path: str | Path,
) -> Path:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report_template.html")

    top = report.top_recommendation
    stamp_color, stamp_dasharray = _stamp_visuals(
        top.score.weighted_total, report.compliance_review.approved
    )
    weights = company.priority_weights.normalized()

    html = template.render(
        report=report,
        top=top,
        weights=weights,
        quantity=company.quantity_needed,
        generated_date=datetime.now().strftime("%B %d, %Y at %H:%M"),
        total_cost_formatted=format_money(report.estimated_total_cost, report.currency),
        top_price_formatted=format_money(top.product.price, top.product.currency),
        stamp_color=stamp_color,
        stamp_dasharray=stamp_dasharray,
        radar_json=_radar_chart_json(report),
        format_money=format_money,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
