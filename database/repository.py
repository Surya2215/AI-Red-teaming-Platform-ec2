"""Persistence helpers for scans, targets, logs, and detector results."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.schemas import ScanRequest, ScanResult, ScanStatus, ScenarioResult, TargetConfig
from database.models import (
    AttackLogRecord,
    DetectorResultRecord,
    ScanFindingRecord,
    ScanJobRecord,
    ScanRecord,
    ScanResultRecord,
    ScanTargetRecord,
    TargetRecord,
)


class Repository:
    """Thin repository abstraction for easy PostgreSQL migration."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert_target(self, target: TargetConfig) -> TargetRecord:
        result = await self.session.execute(select(TargetRecord).where(TargetRecord.name == target.name))
        record = result.scalar_one_or_none()
        payload = target.model_dump(mode="json")
        if record is None:
            record = TargetRecord(**payload)
            self.session.add(record)
        else:
            for key, value in payload.items():
                setattr(record, key, value)
        await self.session.commit()
        return record

    async def list_targets(self) -> list[TargetRecord]:
        result = await self.session.execute(select(TargetRecord).order_by(TargetRecord.name))
        return list(result.scalars().all())

    async def create_scan(self, request: ScanRequest) -> ScanRecord:
        record = ScanRecord(
            scan_id=request.scan_id,
            scan_name=request.scan_name,
            target_name=request.target.name,
            status=ScanStatus.RUNNING.value,
            settings=request.settings.model_dump(),
        )
        self.session.add(record)
        await self.session.commit()
        return record

    async def complete_scan(self, result: ScanResult) -> None:
        scan = await self._get_scan(result.scan_id)
        scan.status = result.status.value
        scan.completed_at = result.completed_at
        scan.error = result.error
        for scenario in result.scenario_results:
            await self.add_scenario_result(result.scan_id, scenario)
        await self.session.commit()

    async def add_scenario_result(self, scan_id: str, scenario: ScenarioResult) -> None:
        result_record = ScanResultRecord(
            scan_id=scan_id,
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.scenario_name,
            owasp_category=scenario.owasp_category,
            result_json=scenario.model_dump(mode="json"),
        )
        self.session.add(result_record)
        await self.session.flush()

        for turn in scenario.turns:
            self.session.add(
                AttackLogRecord(
                    scan_id=scan_id,
                    scenario_id=scenario.scenario_id,
                    turn=turn.turn,
                    stage=turn.prompt.stage,
                    prompt=turn.prompt.prompt,
                    response=turn.response.body,
                    elapsed_ms=turn.response.elapsed_ms,
                )
            )

        for detector in scenario.detector_results:
            self.session.add(
                DetectorResultRecord(
                    scan_result_id=result_record.id,
                    detector_id=detector.detector_id,
                    vulnerable=detector.vulnerable,
                    confidence=detector.confidence,
                    severity=detector.severity.value,
                    reason=detector.reason,
                    evidence=detector.evidence,
                )
            )

    async def list_scans(self) -> list[ScanRecord]:
        result = await self.session.execute(select(ScanRecord).order_by(ScanRecord.started_at.desc()))
        return list(result.scalars().all())

    async def list_scan_results(self) -> list[ScanResultRecord]:
        result = await self.session.execute(select(ScanResultRecord).order_by(ScanResultRecord.created_at.desc()))
        return list(result.scalars().all())

    async def _get_scan(self, scan_id: str) -> ScanRecord:
        result = await self.session.execute(select(ScanRecord).where(ScanRecord.scan_id == scan_id))
        scan = result.scalar_one()
        return scan

    # -- Tool Scan jobs (async Garak/PyRIT/DeepTeam pipeline) ---------------------

    async def create_scan_job(self, job_id: str, tool_id: str, request_payload: dict[str, Any]) -> ScanJobRecord:
        record = ScanJobRecord(id=job_id, tool_id=tool_id, status="queued", request_payload=request_payload)
        self.session.add(record)
        await self.session.commit()
        return record

    async def get_scan_job(self, job_id: str) -> ScanJobRecord:
        result = await self.session.execute(select(ScanJobRecord).where(ScanJobRecord.id == job_id))
        return result.scalar_one()

    async def update_scan_job_status(
        self,
        job_id: str,
        status: str,
        *,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        error: str | None = None,
        report_paths: list[str] | None = None,
        llm_analysis: dict[str, Any] | None = None,
    ) -> ScanJobRecord:
        job = await self.get_scan_job(job_id)
        job.status = status
        if started_at is not None:
            job.started_at = started_at
        if completed_at is not None:
            job.completed_at = completed_at
        if error is not None:
            job.error = error
        if report_paths is not None:
            job.report_paths = report_paths
        if llm_analysis is not None:
            job.llm_analysis = llm_analysis
        await self.session.commit()
        return job

    async def list_scan_jobs(self, status: str | None = None) -> list[ScanJobRecord]:
        query = select(ScanJobRecord).order_by(ScanJobRecord.created_at.desc())
        if status is not None:
            query = query.where(ScanJobRecord.status == status)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def delete_scan_job(self, job_id: str) -> bool:
        # Loads the ORM object (rather than a bulk DELETE statement) so the
        # cascade="all, delete-orphan" relationship on ScanJobRecord.findings actually
        # fires - a bulk DELETE bypasses the ORM cascade and hits the findings FK.
        job = await self.session.get(ScanJobRecord, job_id)
        if job is None:
            return False
        await self.session.delete(job)
        await self.session.commit()
        return True

    async def insert_scan_findings(self, job_id: str, findings: list[dict[str, Any]]) -> None:
        for finding in findings:
            self.session.add(
                ScanFindingRecord(
                    job_id=job_id,
                    probe=finding["probe"],
                    detector=finding["detector"],
                    passed=finding["passed"],
                    score=finding.get("score"),
                    severity=finding.get("severity", "info"),
                    prompt=finding["prompt"],
                    response=finding["response"],
                    conversation_id=finding.get("conversation_id", ""),
                    turn=finding.get("turn", 0),
                )
            )
        await self.session.commit()

    async def list_scan_findings(self, job_id: str, offset: int = 0, limit: int = 100) -> list[ScanFindingRecord]:
        query = (
            select(ScanFindingRecord)
            .where(ScanFindingRecord.job_id == job_id)
            # conversation_id, turn keeps each conversation's turns contiguous and
            # ordered for the frontend's chat-transcript grouping, regardless of
            # insertion order.
            .order_by(ScanFindingRecord.conversation_id, ScanFindingRecord.turn, ScanFindingRecord.id)
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())

    # -- Saved Tool Scan targets ---------------------------------------------------

    async def create_scan_target(
        self, tool_id: str, target_type: str, name: str, encrypted_credentials: dict[str, str], target_name: str = ""
    ) -> ScanTargetRecord:
        record = ScanTargetRecord(
            tool_id=tool_id,
            target_type=target_type,
            name=name,
            encrypted_credentials=encrypted_credentials,
            target_name=target_name,
        )
        self.session.add(record)
        await self.session.commit()
        return record

    async def list_scan_targets(self, tool_id: str | None = None) -> list[ScanTargetRecord]:
        query = select(ScanTargetRecord).order_by(ScanTargetRecord.name)
        if tool_id is not None:
            query = query.where(ScanTargetRecord.tool_id == tool_id)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_scan_target(self, target_id: str) -> ScanTargetRecord:
        result = await self.session.execute(select(ScanTargetRecord).where(ScanTargetRecord.id == target_id))
        return result.scalar_one()

    async def delete_scan_target(self, target_id: str) -> bool:
        target = await self.session.execute(select(ScanTargetRecord).where(ScanTargetRecord.id == target_id))
        record = target.scalar_one_or_none()
        if record is None:
            return False
        await self.session.delete(record)
        await self.session.commit()
        return True
