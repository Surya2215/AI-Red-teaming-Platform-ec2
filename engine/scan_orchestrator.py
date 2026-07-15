"""End-to-end scan orchestration."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from core.config import ROOT_DIR
from core.logging import get_logger
from core.schemas import ScanRequest, ScanResult, ScanStatus
from database.repository import Repository
from engine.attack_engine import AttackEngine
from engine.detector_engine import DetectorEngine
from engine.report_generator import generate_enterprise_report
from engine.scenario_report_llm_analyzer import analyze_scenario_scan


logger = get_logger(__name__)

ProgressCallback = Callable[[dict], None]


class ScanOrchestrator:
    """Coordinate scenarios, detectors, persistence, and report output."""

    def __init__(
        self,
        attack_engine: AttackEngine | None = None,
        detector_engine: DetectorEngine | None = None,
        repository: Repository | None = None,
        report_dir: Path | None = None,
    ) -> None:
        self.attack_engine = attack_engine or AttackEngine()
        self.detector_engine = detector_engine or DetectorEngine()
        self.repository = repository
        self.report_dir = report_dir or ROOT_DIR / "reports"

    async def run_scan(self, request: ScanRequest, progress: ProgressCallback | None = None) -> ScanResult:
        started = datetime.now(UTC)
        result = ScanResult(
            scan_id=request.scan_id,
            scan_name=request.scan_name,
            target_name=request.target.name,
            status=ScanStatus.RUNNING,
            started_at=started,
        )
        if self.repository:
            await self.repository.upsert_target(request.target)
            await self.repository.create_scan(request)

        try:
            self._progress(progress, {"event": "scan_started", "scan_id": request.scan_id, "scan_name": request.scan_name})
            async for scenario_result in self.attack_engine.run(request):
                self._progress(progress, {"event": "scenario_completed", "scenario": scenario_result.scenario_id})
                detector_results = await self.detector_engine.evaluate(scenario_result)
                scenario_result.detector_results = detector_results
                result.scenario_results.append(scenario_result)
                self._progress(
                    progress,
                    {
                        "event": "detectors_completed",
                        "scenario": scenario_result.scenario_id,
                        "detectors": [detector.model_dump(mode="json") for detector in detector_results],
                    },
                )
            result.status = ScanStatus.COMPLETED
        except Exception as exc:
            logger.exception("Scan failed.", extra={"scan_id": request.scan_id, "target": request.target.name})
            result.status = ScanStatus.FAILED
            result.error = str(exc)
        finally:
            result.completed_at = datetime.now(UTC)
            if result.scenario_results:
                await self._attach_llm_analysis(result)
            if self.repository:
                await self.repository.complete_scan(result)
            self._write_report(result)
            self._progress(progress, {"event": "scan_finished", "status": result.status.value, "scan_id": request.scan_id})
        return result

    async def _attach_llm_analysis(self, result: ScanResult) -> None:
        """Run the LLM reporting agent on a completed scan and attach its analysis.

        Sets result.llm_analysis in place; engine/pdf_report_adapters.py prefers
        it over the deterministic EnterpriseReportGenerator output when present.
        Best-effort: MultiProviderLLMClient already falls back to a deterministic
        stub if no provider is configured or the call fails, but unexpected
        errors here must not break the scan result itself.
        """
        try:
            result.llm_analysis = await analyze_scenario_scan(result)
        except Exception:
            logger.exception("Failed to generate LLM scenario-scan analysis.", extra={"scan_id": result.scan_id})

    def _write_report(self, result: ScanResult) -> None:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        path = self.report_dir / f"{result.scan_id}.json"
        payload = result.model_dump(mode="json")
        structured_report, markdown_report = generate_enterprise_report(result)
        payload["enterprise_report"] = structured_report
        payload["enterprise_report_markdown"] = markdown_report
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _progress(self, progress: ProgressCallback | None, event: dict) -> None:
        if progress:
            progress(event)

