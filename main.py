#!/usr/bin/env python3
"""
main.py
--------
CLI entrypoint for the Multi-Agent Procurement Assistant.

Examples
--------
    # Full live run (needs ANTHROPIC_API_KEY + TAVILY_API_KEY at minimum)
    python main.py --request "enclosed-chamber 3D printer for a robotics lab"

    # Same, but orchestrated as an event-driven Flow with retry/branching
    # logic instead of a fixed sequential Crew (see src/flow.py)
    python main.py --request "..." --engine flow

    # Offline smoke test: renders the full HTML report from fixture data,
    # no API keys, no network calls, runs in under a second
    python main.py --demo

    # Wipe the local search/scrape cache
    python main.py --clear-cache
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.company_context import DEFAULT_COMPANY
from src.config import OUTPUT_DIR, settings
from src.report_generator import generate_html_report
from src.utils.logger import console, print_ranking_table, setup_logging, stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Agent Procurement Assistant (CrewAI)")
    parser.add_argument(
        "--request", "-r", type=str, default=None,
        help="The product/category to source, e.g. 'enclosed-chamber 3D printer'",
    )
    parser.add_argument(
        "--engine", choices=["crew", "flow"], default="crew",
        help="Orchestration engine: 'crew' = fixed sequential pipeline (default), "
             "'flow' = event-driven graph with retry/branching (src/flow.py)",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Skip all agents/tools entirely and render the report from bundled fixture data. "
             "No API keys or network access required.",
    )
    parser.add_argument(
        "--clear-cache", action="store_true", help="Delete cached search/scrape results and exit."
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output HTML path (default: outputs/procurement_report_<timestamp>.html)",
    )
    return parser.parse_args()


def _default_output_path() -> Path:
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"procurement_report_{stamp}.html"


def main() -> int:
    args = parse_args()
    logger = setup_logging(verbose=settings.verbose)

    if args.clear_cache:
        from src.tools.cache import clear_cache

        n = clear_cache()
        console.print(f"[green]Cleared {n} cached entries.[/green]")
        return 0

    company = DEFAULT_COMPANY
    output_path = Path(args.output) if args.output else _default_output_path()

    if args.demo:
        with stage("Rendering demo report from fixture data", "\U0001F9EA"):
            from demo_data import build_demo_recommendation

            recommendation = build_demo_recommendation()
    else:
        if not args.request:
            console.print(
                "[red]--request is required for a live run (or pass --demo). "
                "Example: python main.py --request \"enclosed-chamber 3D printer\"[/red]"
            )
            return 1
        if not settings.active_llm_key_present():
            console.print(
                f"[red]No API key found for LLM_PROVIDER='{settings.llm_provider}'. "
                "Set the matching *_API_KEY in your .env (see .env.example), "
                "or run with --demo to preview the system offline.[/red]"
            )
            return 1

        with stage(f"Running procurement crew ({args.engine} engine)", "\U0001F916"):
            if args.engine == "flow":
                from src.flow import ProcurementFlow

                recommendation = ProcurementFlow(company=company, product_request=args.request).run()
            else:
                from src.crew import ProcurementCrew

                recommendation = ProcurementCrew(company=company, product_request=args.request).run()

    with stage("Generating HTML report", "\U0001F4C4"):
        report_path = generate_html_report(recommendation, company, output_path)

    ranked = [
        {
            "rank": rp.rank,
            "product_name": rp.product.product_name,
            "price_score": rp.score.price_score,
            "spec_score": rp.score.spec_score,
            "vendor_score": rp.score.vendor_score,
            "delivery_score": rp.score.delivery_score,
            "weighted_total": rp.score.weighted_total,
        }
        for rp in recommendation.full_comparison.ranked_products
    ]
    print_ranking_table(ranked)

    console.print(f"\n[bold green]\u2713 Report ready:[/bold green] {report_path}")
    console.print(f"[dim]Top recommendation: {recommendation.top_recommendation.product.product_name} "
                   f"({recommendation.currency} {recommendation.estimated_total_cost:,.2f} total)[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
