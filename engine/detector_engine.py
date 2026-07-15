"""Detector engine that evaluates scenario results with dynamic plugins."""

from __future__ import annotations

from core.exceptions import DetectorError
from core.logging import get_logger
from core.schemas import DetectorResult, ScenarioResult
from engine.scenario_loader import PluginLoader


logger = get_logger(__name__)


class DetectorEngine:
    """Load detector modules and aggregate normalized findings."""

    def __init__(self, loader: PluginLoader | None = None) -> None:
        self.loader = loader or PluginLoader()

    async def evaluate(self, scenario_result: ScenarioResult) -> list[DetectorResult]:
        detectors = self.loader.discover_detectors(scenario_result.owasp_category)
        results: list[DetectorResult] = []
        for detector_id, detector in detectors.items():
            try:
                result = await detector.evaluate(scenario_result)
                results.append(result)
                logger.info(
                    "Detector completed.",
                    extra={
                        "scenario": scenario_result.scenario_id,
                        "detector": detector_id,
                        "vulnerable": result.vulnerable,
                        "confidence": result.confidence,
                    },
                )
            except Exception as exc:
                raise DetectorError(f"Detector {detector_id} failed: {exc}") from exc
        return self._deduplicate(results)

    def _deduplicate(self, results: list[DetectorResult]) -> list[DetectorResult]:
        ordered = sorted(results, key=lambda item: (item.vulnerable, item.confidence), reverse=True)
        return ordered

