"""Parses Garak's *.report.jsonl output, PyRIT's SQLite memory database, and
DeepTeam's saved risk_assessment JSON into structured scan_findings rows.

Nothing in this codebase parsed any of these tools' output before this - engine/
tool_report_generator.py only LLM-summarizes raw stdout text. All three schemas below
were confirmed empirically (no credential-free way to verify this from docs alone):
Garak by running `garak --target_type test.Blank ...` and inspecting the emitted
report.jsonl; PyRIT by inspecting a real completed scan's rows in
~/.local/share/dbdata/pyrit.db on the deployed EC2 worker; DeepTeam by reading its
installed source (deepteam/red_teamer/risk_assessment.py's RiskAssessment.save() and
deepteam/test_case/test_case.py's RTTestCase) on the same worker (deepteam==1.0.7).

Every finding carries a conversation_id + turn so the UI can render it as a chat
transcript (see ToolScanFindings/ChatTranscript in the frontend) - Garak's attempts
are always exactly one turn (one conversation_id per attempt, one per generation - see
parse_garak_findings); PyRIT and DeepTeam attacks can be genuinely multi-turn (e.g.
crescendo), with all turns of one attack sharing a conversation_id.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

EVAL_THRESHOLD = 0.5
CRITICAL_THRESHOLD = 0.8


def _deepteam_severity_for(passed: bool) -> str:
    # DeepTeam's score is a binary pass/fail verdict (0.0 or 1.0), not a continuous
    # "how bad" gradient like garak's detector scores - reusing _severity_for's
    # score>=CRITICAL_THRESHOLD logic here would misclassify every failure as
    # "medium" (0.0 < 0.8), so failures always get "critical": a successful
    # jailbreak/attack IS the critical finding, there's no lesser gradient to derive.
    return "info" if passed else "critical"


def _severity_for(passed: bool, score: float) -> str:
    if passed:
        return "info"
    return "critical" if score >= CRITICAL_THRESHOLD else "medium"


def _prompt_text(attempt: dict[str, Any]) -> str:
    turns = (attempt.get("prompt") or {}).get("turns") or []
    if not turns:
        return ""
    return str((turns[-1].get("content") or {}).get("text", ""))


def parse_garak_findings(report_paths: list[str]) -> list[dict[str, Any]]:
    """Extract one finding per (attempt output x detector) pair from Garak's
    *.report.jsonl file. Returns [] if no report.jsonl is present among report_paths
    (e.g. a scan that failed before producing any output)."""

    report_jsonl = next((path for path in report_paths if path.endswith(".report.jsonl")), None)
    if report_jsonl is None or not Path(report_jsonl).exists():
        return []

    findings: list[dict[str, Any]] = []
    with open(report_jsonl, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("entry_type") != "attempt" or entry.get("status") != 2:
                continue

            probe = str(entry.get("probe_classname", ""))
            prompt_text = _prompt_text(entry)
            outputs = entry.get("outputs") or []
            detector_results = entry.get("detector_results") or {}

            attempt_id = str(entry.get("uuid") or f"garak-{len(findings)}")
            for detector, scores in detector_results.items():
                for index, raw_score in enumerate(scores):
                    response_text = ""
                    if index < len(outputs):
                        response_text = str((outputs[index] or {}).get("text", ""))
                    score = float(raw_score)
                    passed = score < EVAL_THRESHOLD
                    # One conversation per (attempt, generation) - NOT per attempt alone.
                    # Garak's --generations N resamples the SAME single-turn prompt N
                    # independent times (not a multi-turn dialogue); grouping all of them
                    # under attempt_id made repeated independent samples of one prompt
                    # render as a fake "N-turn conversation" with the prompt appearing to
                    # repeat verbatim. Multiple detectors scoring the SAME generation still
                    # share one conversation_id here, since that's genuinely one exchange.
                    findings.append(
                        {
                            "probe": probe,
                            "detector": str(detector),
                            "passed": passed,
                            "score": score,
                            "severity": _severity_for(passed, score),
                            "prompt": prompt_text,
                            "response": response_text,
                            "conversation_id": f"{attempt_id}-gen{index}",
                            "turn": 0,
                        }
                    )
    return findings


def parse_pyrit_findings(job_id: str, memory_db_path: str) -> list[dict[str, Any]]:
    """Extract one finding per (user, assistant) turn pair from every PyRIT attack
    tagged with this job's id, from PyRIT's shared SQLite memory database.

    PyRIT has no per-scan report file - every scan writes into the same
    ~/.local/share/dbdata/pyrit.db, so scans are correlated via a `tool_scan_job_id`
    memory label injected by worker/tasks.py before the scan runs (see
    _inject_job_label). Returns [] if the memory DB doesn't exist yet or no attacks
    are tagged with this job_id (e.g. the scan failed before PyRIT wrote anything)."""

    path = Path(memory_db_path).expanduser()
    if not path.exists():
        return []

    findings: list[dict[str, Any]] = []
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, conversation_id, objective, outcome, last_score_id FROM AttackResultEntries "
            "WHERE json_extract(labels, '$.tool_scan_job_id') = ?",
            (job_id,),
        )
        attacks = cur.fetchall()

        for attack in attacks:
            passed = attack["outcome"] != "success"
            score_value: float | None = None
            detector = "pyrit_scorer"
            if attack["last_score_id"]:
                cur.execute(
                    "SELECT score_value, score_category FROM ScoreEntries WHERE id = ?", (attack["last_score_id"],)
                )
                score_row = cur.fetchone()
                if score_row:
                    score_value = 1.0 if str(score_row["score_value"]).strip().lower() == "true" else 0.0
                    try:
                        categories = json.loads(score_row["score_category"] or "[]")
                        if categories:
                            detector = ", ".join(str(item) for item in categories)
                    except json.JSONDecodeError:
                        pass
            severity = _severity_for(passed, score_value if score_value is not None else (0.0 if passed else 1.0))

            cur.execute(
                "SELECT role, sequence, original_value, converted_value FROM PromptMemoryEntries "
                "WHERE conversation_id = ? ORDER BY sequence",
                (attack["conversation_id"],),
            )
            turns = [row for row in cur.fetchall() if row["role"] in ("user", "assistant")]

            turn_index = 0
            index = 0
            while index < len(turns) - 1:
                user_turn, assistant_turn = turns[index], turns[index + 1]
                if user_turn["role"] != "user" or assistant_turn["role"] != "assistant":
                    index += 1
                    continue
                findings.append(
                    {
                        "probe": str(attack["objective"] or "")[:200],
                        "detector": detector,
                        "passed": passed,
                        "score": score_value,
                        "severity": severity,
                        "prompt": user_turn["converted_value"] or user_turn["original_value"] or "",
                        "response": assistant_turn["converted_value"] or assistant_turn["original_value"] or "",
                        "conversation_id": str(attack["conversation_id"]),
                        "turn": turn_index,
                    }
                )
                turn_index += 1
                index += 2
    finally:
        conn.close()
    return findings


def parse_deepteam_findings(report_paths: list[str]) -> list[dict[str, Any]]:
    """Extract one finding per RTTestCase from DeepTeam's saved risk_assessment JSON
    (engine/tool_scan.py::_build_deepteam_command sets system_config.output_folder so
    RiskAssessment.save() always writes one). Each test case is either single-turn
    (input/actual_output as flat strings, e.g. Base64/Roleplay) or multi-turn (a
    `turns` list of {role, content} pairs, e.g. CrescendoJailbreaking/
    LinearJailbreaking) - confirmed by reading deepteam's actual RTTestCase source.
    A test case's score is deepteam's own evaluation-model verdict: score > 0 means
    the target successfully resisted the attack (matching TestCasesList.to_df()'s own
    "Passed" convention in deepteam's source), score == 0 means the attack succeeded.
    """

    json_path = next((path for path in report_paths if path.endswith(".json")), None)
    if json_path is None or not Path(json_path).exists():
        return []

    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    findings: list[dict[str, Any]] = []
    for index, case in enumerate(data.get("test_cases") or []):
        vulnerability = str(case.get("vulnerability", ""))
        vulnerability_type = case.get("vulnerability_type")
        probe = f"{vulnerability}.{vulnerability_type}" if vulnerability_type else vulnerability
        case_error = case.get("error")
        detector = str(case.get("attack_method") or ("error" if case_error else "unknown"))

        raw_score = case.get("score")
        score = float(raw_score) if raw_score is not None else None
        passed = bool(score is not None and score > 0)
        severity = _deepteam_severity_for(passed)
        conversation_id = f"deepteam-{index}"

        turns = [turn for turn in (case.get("turns") or []) if turn.get("role") in ("user", "assistant")]
        if turns:
            turn_index = 0
            turn_position = 0
            while turn_position < len(turns) - 1:
                user_turn, assistant_turn = turns[turn_position], turns[turn_position + 1]
                if user_turn.get("role") != "user" or assistant_turn.get("role") != "assistant":
                    turn_position += 1
                    continue
                findings.append(
                    {
                        "probe": probe,
                        "detector": detector,
                        "passed": passed,
                        "score": score,
                        "severity": severity,
                        "prompt": str(user_turn.get("content", "")),
                        "response": str(assistant_turn.get("content", "")),
                        "conversation_id": conversation_id,
                        "turn": turn_index,
                    }
                )
                turn_index += 1
                turn_position += 2
        else:
            # A test case can fail before ever producing a prompt/response - e.g.
            # DeepTeam's attack simulator itself raising (case["error"] set, "input"
            # and "actual_output" both null). Leaving prompt/response as empty strings
            # rendered as two blank chat bubbles in the UI with no explanation, so
            # surface the error text there instead of silently dropping it.
            prompt = str(case.get("input") or "")
            response = str(case.get("actual_output") or "")
            if not prompt and not response and case_error:
                prompt = "(No prompt was generated for this attack.)"
                response = f"DeepTeam error: {case_error}"
            findings.append(
                {
                    "probe": probe,
                    "detector": detector,
                    "passed": passed,
                    "score": score,
                    "severity": severity,
                    "prompt": prompt,
                    "response": response,
                    "conversation_id": conversation_id,
                    "turn": 0,
                }
            )
    return findings
