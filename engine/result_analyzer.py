"""Aggregate detector outputs into report-friendly summaries."""

from __future__ import annotations

from core.schemas import DetectorResult, ScenarioResult, Severity


SEVERITY_RANK = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


class ResultAnalyzer:
    """Summarize findings without coupling to UI or persistence."""

    def summarize_scenario(self, scenario: ScenarioResult) -> dict:
        vulnerable_results = [result for result in scenario.detector_results if result.vulnerable]
        top = self.highest(vulnerable_results or scenario.detector_results)
        return {
            "scenario_id": scenario.scenario_id,
            "scenario_name": scenario.scenario_name,
            "vulnerable": bool(vulnerable_results),
            "severity": top.severity.value if top else "INFO",
            "confidence": top.confidence if top else 0,
            "evidence": top.evidence if top else [],
            "turns": len(scenario.turns),
        }

    def highest(self, results: list[DetectorResult]) -> DetectorResult | None:
        if not results:
            return None
        return sorted(results, key=lambda result: (SEVERITY_RANK[result.severity], result.confidence), reverse=True)[0]

