"""
src/company_context.py
-----------------------
Every agent in this crew needs to know *who it's buying for*. Without a
grounded company context, a procurement crew will happily recommend a
$40,000 laser cutter to a 12-person design studio. This module defines that
context as a structured, validated object and renders it into the natural
language block that gets injected into every agent's backstory and every
task description.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class RiskTolerance(str, Enum):
    LOW = "low"          # only established, well-reviewed vendors
    MEDIUM = "medium"
    HIGH = "high"         # open to new/niche vendors for better value


class PriorityWeights(BaseModel):
    """
    How much the buyer cares about each factor, expressed as weights that
    sum to 1.0. The Data Analyst agent's scoring tool consumes this directly
    -- it is the single source of truth for "what does a good deal mean
    to this company", so procurement math and stated priorities never drift
    out of sync.
    """
    price: float = Field(0.40, ge=0, le=1)
    specifications: float = Field(0.30, ge=0, le=1)
    vendor_reliability: float = Field(0.20, ge=0, le=1)
    delivery_speed: float = Field(0.10, ge=0, le=1)

    def normalized(self) -> "PriorityWeights":
        total = self.price + self.specifications + self.vendor_reliability + self.delivery_speed
        if total == 0:
            raise ValueError("Priority weights cannot all be zero.")
        return PriorityWeights(
            price=self.price / total,
            specifications=self.specifications / total,
            vendor_reliability=self.vendor_reliability / total,
            delivery_speed=self.delivery_speed / total,
        )


class CompanyProfile(BaseModel):
    company_name: str
    industry: str
    company_size: str = Field(description="e.g. '50-person startup', '2,000-employee enterprise'")
    location: str = Field(description="Primary operating region, used for shipping/tax framing")
    currency: str = Field(default="USD", description="ISO currency code for all figures")

    procurement_budget_max: float = Field(description="Hard ceiling, per unit, in `currency`")
    quantity_needed: int = Field(default=1, ge=1)

    priority_weights: PriorityWeights = Field(default_factory=PriorityWeights)
    risk_tolerance: RiskTolerance = RiskTolerance.MEDIUM

    required_specifications: List[str] = Field(
        default_factory=list,
        description="Hard requirements a product MUST meet to be considered at all",
    )
    nice_to_have_specifications: List[str] = Field(default_factory=list)
    preferred_vendor_regions: List[str] = Field(default_factory=list)
    excluded_vendors: List[str] = Field(default_factory=list)
    sustainability_requirement: Optional[str] = Field(
        default=None, description="e.g. 'Prefer Energy Star certified or RoHS compliant'"
    )

    def as_prompt_block(self) -> str:
        """Render into the natural-language brief agents receive as context."""
        w = self.priority_weights.normalized()
        req = "\n".join(f"  - {r}" for r in self.required_specifications) or "  - (none specified)"
        nice = "\n".join(f"  - {r}" for r in self.nice_to_have_specifications) or "  - (none specified)"
        excluded = ", ".join(self.excluded_vendors) or "None"
        regions = ", ".join(self.preferred_vendor_regions) or "No preference"

        return f"""
COMPANY CONTEXT
================
Buyer: {self.company_name} ({self.company_size}, {self.industry})
Location: {self.location} | Reporting currency: {self.currency}

BUDGET
  Maximum unit price: {self.currency} {self.procurement_budget_max:,.2f}
  Quantity needed: {self.quantity_needed}
  Total budget ceiling: {self.currency} {self.procurement_budget_max * self.quantity_needed:,.2f}

DECISION PRIORITIES (weights, must guide ranking)
  Price:               {w.price:.0%}
  Specifications match: {w.specifications:.0%}
  Vendor reliability:   {w.vendor_reliability:.0%}
  Delivery speed:        {w.delivery_speed:.0%}

RISK TOLERANCE: {self.risk_tolerance.value} 
  (low = established vendors only, high = open to niche/new vendors for better value)

HARD REQUIREMENTS (a product failing any of these must be rejected):
{req}

NICE-TO-HAVE (boost score, not disqualifying if absent):
{nice}

VENDOR CONSTRAINTS
  Preferred regions: {regions}
  Excluded vendors: {excluded}
  Sustainability requirement: {self.sustainability_requirement or "None specified"}
""".strip()


# ---------------------------------------------------------------------------
# A ready-to-run example. Swap this out (or build one via CLI prompts in
# main.py) for a real engagement.
# ---------------------------------------------------------------------------
DEFAULT_COMPANY = CompanyProfile(
    company_name="Northbridge Robotics Ltd.",
    industry="Industrial Automation & Robotics",
    company_size="140-employee mid-market manufacturer",
    location="Austin, Texas, USA",
    currency="USD",
    procurement_budget_max=3500.00,
    quantity_needed=6,
    priority_weights=PriorityWeights(
        price=0.35, specifications=0.35, vendor_reliability=0.20, delivery_speed=0.10
    ),
    risk_tolerance=RiskTolerance.MEDIUM,
    required_specifications=[
        "Build volume of at least 250 x 250 x 250 mm",
        "Enclosed chamber (for ABS/engineering polymers)",
        "Wi-Fi or Ethernet network connectivity for fleet monitoring",
    ],
    nice_to_have_specifications=[
        "Multi-material / dual-extrusion capability",
        "Auto bed-leveling",
        "Available in the North American market with local support",
    ],
    preferred_vendor_regions=["North America", "European Union"],
    excluded_vendors=[],
    sustainability_requirement="Prefer RoHS-compliant hardware",
)
