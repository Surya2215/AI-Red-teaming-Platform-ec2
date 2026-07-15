"""The Celery task that executes one Tool Scan job. Runs on the Debian EC2 worker
process. Never imports garak/pyrit/deepteam directly - engine.tool_scan.run_tool_scan
invokes each tool's own venv interpreter via subprocess, exactly as it already does
for the synchronous flow this replaces (see engine/tool_scan.py::_get_tool_python).

Uses one persistent event loop for the worker process's whole lifetime (_run_async
below) rather than asyncio.run() per call. asyncpg connections are bound to the loop
that created them, but database.session.engine/AsyncSessionLocal are module-level
singletons shared across every task this process handles - asyncio.run() tears its
loop down on return, so a second call would hand out a pooled connection created on
an already-closed loop ("Event loop is closed" / "another operation is in progress").
This worker MUST run with `celery -A worker.celery_app worker --pool=solo`: prefork's
fork() would otherwise let child processes inherit the parent's already-open asyncpg
connections/loop, which is unsafe across a fork boundary."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import Any, TypeVar

from core.config import get_settings
from core.crypto import decrypt
from core.logging import get_logger
from database.repository import Repository
from database.session import AsyncSessionLocal
from engine.deepteam_catalog import DEEPTEAM_SECRET_CREDENTIAL_KEYS
from engine.tool_report_generator import analyze_tool_scan_findings
from engine.tool_scan import GARAK_SECRET_CREDENTIAL_KEYS, TOOL_DEFINITIONS, ToolScanRequest, run_tool_scan
from worker.celery_app import celery_app
from worker.result_parser import parse_deepteam_findings, parse_garak_findings, parse_pyrit_findings

logger = get_logger(__name__)

_T = TypeVar("_T")
_loop = asyncio.new_event_loop()


def _run_async(coro: Awaitable[_T]) -> _T:
    return _loop.run_until_complete(coro)


def _decrypt_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Decrypt the credential fields engine/tool_scan.py already treats as secret
    (GARAK_SECRET_CREDENTIAL_KEYS, plus pyrit_api_key - see _secret_values() in
    engine/tool_scan.py) immediately before reconstructing a ToolScanRequest."""

    decrypted = dict(payload)
    garak_credentials = decrypted.get("garak_credentials") or {}
    decrypted["garak_credentials"] = {
        key: (decrypt(value) if key in GARAK_SECRET_CREDENTIAL_KEYS and value else value)
        for key, value in garak_credentials.items()
    }
    if decrypted.get("pyrit_api_key"):
        decrypted["pyrit_api_key"] = decrypt(decrypted["pyrit_api_key"])
    deepteam_credentials = decrypted.get("deepteam_credentials") or {}
    decrypted["deepteam_credentials"] = {
        key: (decrypt(value) if key in DEEPTEAM_SECRET_CREDENTIAL_KEYS and value else value)
        for key, value in deepteam_credentials.items()
    }
    return decrypted


def _with_pyrit_job_label(request: ToolScanRequest, job_id: str) -> ToolScanRequest:
    """PyRIT writes every scan's results into one shared SQLite memory database (no
    per-scan report file - see worker/result_parser.py::parse_pyrit_findings), so the
    only way to find *this* scan's rows afterward is a memory label. Merges
    {"tool_scan_job_id": job_id} into whatever labels the request already carries."""

    labels: dict[str, Any] = {}
    if request.pyrit_memory_labels and request.pyrit_memory_labels.strip():
        try:
            labels = json.loads(request.pyrit_memory_labels)
        except json.JSONDecodeError:
            labels = {}
    labels["tool_scan_job_id"] = job_id
    return request.model_copy(update={"pyrit_memory_labels": json.dumps(labels)})


async def _load_job(job_id: str) -> tuple[str, dict[str, Any]]:
    async with AsyncSessionLocal() as session:
        job = await Repository(session).get_scan_job(job_id)
        return job.tool_id, job.request_payload


async def _mark_running(job_id: str) -> None:
    async with AsyncSessionLocal() as session:
        await Repository(session).update_scan_job_status(job_id, "running", started_at=datetime.now(UTC))


async def _mark_completed(job_id: str, *, report_paths: list[str], llm_analysis: dict[str, Any] | None) -> None:
    async with AsyncSessionLocal() as session:
        await Repository(session).update_scan_job_status(
            job_id,
            "completed",
            completed_at=datetime.now(UTC),
            report_paths=report_paths,
            llm_analysis=llm_analysis,
        )


async def _mark_failed(
    job_id: str,
    error: str,
    *,
    report_paths: list[str] | None = None,
    llm_analysis: dict[str, Any] | None = None,
) -> None:
    async with AsyncSessionLocal() as session:
        await Repository(session).update_scan_job_status(
            job_id,
            "failed",
            completed_at=datetime.now(UTC),
            error=error,
            report_paths=report_paths,
            llm_analysis=llm_analysis,
        )


async def _store_findings(job_id: str, findings: list[dict[str, Any]]) -> None:
    if findings:
        async with AsyncSessionLocal() as session:
            await Repository(session).insert_scan_findings(job_id, findings)


_STDERR_EXCERPT_CHARS = 1000
_STDOUT_EXCERPT_CHARS = 500


def _failure_error_message(tool_id: str, result: Any) -> str:
    """Build a diagnosable error string for a failed scan.

    Most non-garak failures leave result.error as None (see
    engine/tool_scan.py::_completed_status - only garak's missing-env-var case sets a
    specific message), and scan_jobs.error was the *only* place a failure reason ever
    reached the operator - stdout/stderr are captured on ToolScanResult but never
    persisted anywhere, so a generic "<tool> scan failed." was a dead end that could
    only be diagnosed by SSHing into the worker and manually reproducing the exact
    scan. Fold in the tail of stderr/stdout (where the actual traceback/exit reason
    lives) so the job record itself is enough to diagnose most failures."""

    if result.error:
        return result.error
    parts = [f"{tool_id} scan failed (exit code {result.return_code})."]
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    if stderr:
        parts.append("--- stderr (tail) ---\n" + stderr[-_STDERR_EXCERPT_CHARS:])
    elif stdout:
        parts.append("--- stdout (tail) ---\n" + stdout[-_STDOUT_EXCERPT_CHARS:])
    return "\n".join(parts)


@celery_app.task(name="run_scan_job")
def run_scan_job(job_id: str) -> None:
    try:
        tool_id, payload = _run_async(_load_job(job_id))
        request = ToolScanRequest.model_validate(_decrypt_payload(payload))
        if tool_id == "pyrit":
            request = _with_pyrit_job_label(request, job_id)
        _run_async(_mark_running(job_id))

        result = run_tool_scan(request)

        # Parse regardless of overall status - verified empirically that a "failed"
        # pyrit_scan process (non-zero exit, e.g. one attack erroring out) can still
        # have written plenty of real attack/conversation rows into its memory DB
        # before it died; the same can happen with garak's report.jsonl. A parsing
        # bug must never flip an otherwise-successful scan to failed.
        findings: list[dict[str, Any]] = []
        try:
            if tool_id == "garak":
                findings = parse_garak_findings(result.report_paths)
            elif tool_id == "pyrit":
                findings = parse_pyrit_findings(job_id, get_settings().pyrit_memory_db_path)
            elif tool_id == "deepteam":
                findings = parse_deepteam_findings(result.report_paths)
            _run_async(_store_findings(job_id, findings))
        except Exception:
            logger.exception("Failed to parse scan findings; scan result is unaffected.", extra={"scan_id": job_id})

        # Reporting agent: analyze the structured findings (not raw stdout) so the
        # LLM reasons over actual per-vulnerability/per-attack pass-fail results and
        # produces real remediation - this runs for every tool, including deepteam,
        # which never got an analysis before (engine/tool_scan.py used to skip it on
        # the assumption its own risk_assessment JSON was self-explanatory - it isn't
        # once the PDF template only ever reads job.llm_analysis). Best-effort: a
        # broken/unreachable LLM provider must not flip an otherwise-successful scan
        # to failed, matching the pre-existing invariant asserted by
        # tests/test_tool_scan_isolation.py::test_run_tool_scan_llm_report_failure_does_not_break_scan_result.
        llm_analysis = None
        try:
            tool_name = TOOL_DEFINITIONS[tool_id].name if tool_id in TOOL_DEFINITIONS else tool_id
            llm_analysis = analyze_tool_scan_findings(
                tool_name=tool_name,
                tool_id=tool_id,
                scan_id=job_id,
                status=result.status,
                findings=findings,
                error=result.error,
            )
        except Exception:
            logger.exception("Failed to generate LLM findings analysis; scan result is unaffected.", extra={"scan_id": job_id})

        if result.status == "COMPLETED":
            _run_async(_mark_completed(job_id, report_paths=result.report_paths, llm_analysis=llm_analysis))
        else:
            # Even on failure, run_tool_scan may have produced report artifacts and/or
            # findings worth keeping instead of discarding.
            _run_async(
                _mark_failed(
                    job_id,
                    _failure_error_message(tool_id, result),
                    report_paths=result.report_paths,
                    llm_analysis=llm_analysis,
                )
            )
    except Exception as exc:
        logger.exception("Unhandled error running scan job.", extra={"scan_id": job_id})
        try:
            _run_async(_mark_failed(job_id, str(exc)))
        except Exception:
            logger.exception("Failed to record job failure - job row may be missing.", extra={"scan_id": job_id})
