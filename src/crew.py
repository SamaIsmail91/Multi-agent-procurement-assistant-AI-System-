"""
src/crew.py
------------
Assembles agents + tasks into a runnable CrewAI `Crew` and exposes a single
`ProcurementCrew.run(...)` entrypoint used by main.py.

Process choice: **sequential**, not hierarchical, by default. This isn't
laziness -- hierarchical crews add a manager-agent layer that decides
delegation at runtime, which is more flexible but less predictable and
harder to debug for a pipeline where the correct order of operations
(requirements -> research -> scrape -> score -> review -> report) is already
fully known upfront. We expose `process_type="hierarchical"` as an opt-in via
.env for experimentation, but sequential is the production-recommended path
for this workflow. See src/flow.py for the event-driven Flow variant, which
adds retry/branching logic sequential Crews can't express on their own.
"""
from __future__ import annotations

import logging

import src.config  # noqa: F401  (must run first: sets CREWAI_DISABLE_TELEMETRY before `import crewai` below)
from crewai import Crew, Process

from src.agents import build_agents
from src.company_context import CompanyProfile
from src.config import settings
from src.models import ProcurementRecommendation
from src.tasks import build_tasks

logger = logging.getLogger("procurement_assistant.crew")


class ProcurementCrew:
    def __init__(self, company: CompanyProfile, product_request: str):
        settings.export_to_environ()
        self.company = company
        self.product_request = product_request
        self.agents = build_agents(company)
        self.tasks = build_tasks(self.agents, company, product_request)
        self.crew = self._build_crew()

    def _build_crew(self) -> Crew:
        process = (
            Process.hierarchical if settings.process_type == "hierarchical" else Process.sequential
        )
        kwargs = dict(
            agents=list(self.agents.values()),
            tasks=self.tasks,
            process=process,
            memory=settings.use_crew_memory,
            cache=True,
            verbose=settings.verbose,
            planning=False,
        )
        if process is Process.hierarchical:
            # Hierarchical mode needs a manager model rather than a manager agent
            # from our own pool, so the manager isn't also trying to do the work
            # of one of the specialists it's meant to be delegating to.
            kwargs["manager_llm"] = settings.manager_model
        return Crew(**kwargs)

    def run(self) -> ProcurementRecommendation:
        logger.info(
            "Starting procurement crew for '%s' (budget %s %.2f x%d units)",
            self.product_request, self.company.currency,
            self.company.procurement_budget_max, self.company.quantity_needed,
        )
        result = self.crew.kickoff(
            inputs={
                "product_request": self.product_request,
                "company_name": self.company.company_name,
            }
        )

        final_task_output = self.crew.tasks[-1].output
        if final_task_output is not None and final_task_output.pydantic is not None:
            return final_task_output.pydantic

        # Fallback: some CrewAI versions surface the pydantic object on the
        # top-level kickoff result instead of the last task. Try both before
        # giving up so a minor version difference doesn't hard-crash the run.
        if hasattr(result, "pydantic") and result.pydantic is not None:
            return result.pydantic

        raise RuntimeError(
            "Crew finished but the final task did not return a valid "
            "ProcurementRecommendation. Raw output:\n" + str(result)
        )
