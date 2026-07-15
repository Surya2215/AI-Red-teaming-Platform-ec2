"""Enterprise-grade PDF export for AI red teaming scan reports.

Shared by the FastAPI API (React frontend "Download PDF" button) and the
legacy Streamlit UI, so there is a single implementation of the PDF layout.

Rendering itself is delegated to the navy/teal enterprise template in
engine/pdf_report_template.py via the adapters in engine/pdf_report_adapters.py;
this module owns the public export_pdf/export_tool_scan_pdf entry points plus
the small helpers the Streamlit UI (ui/app.py) still calls directly for its
on-screen widgets.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

from core.schemas import ScanResult
from engine.pdf_report_adapters import scenario_report_to_pdf_data, tool_scan_report_to_pdf_data
from engine.pdf_report_template import generate_report
from engine.report_generator import generate_enterprise_report


def export_pdf(report: dict[str, Any]) -> bytes:
    """Render the enterprise navy/teal PDF report for an OWASP scenario scan."""

    data = scenario_report_to_pdf_data(report)
    buffer = BytesIO()
    generate_report(data, buffer)
    return buffer.getvalue()


def export_tool_scan_pdf(result: dict[str, Any]) -> bytes:
    """Render the enterprise navy/teal PDF report for a Garak/PyRIT/DeepTeam tool scan.

    Populates findings/remediation/risk sections from `result["llm_analysis"]`
    when the LLM reporting agent (engine/tool_report_generator.py) produced
    one; otherwise falls back to a plain summary with the raw command/stdout
    in the appendix.
    """

    data = tool_scan_report_to_pdf_data(result)
    buffer = BytesIO()
    generate_report(data, buffer)
    return buffer.getvalue()


def scenario_vulnerability_percent(scenario: dict[str, Any]) -> int:
    vulnerable_scores = [
        float(detector.get("confidence", 0) or 0)
        for detector in scenario.get("detector_results", [])
        if detector.get("vulnerable", False)
    ]
    if not vulnerable_scores:
        return 0
    return int(round(max(vulnerable_scores) * 100))


def remediation_text(scenario: dict[str, Any]) -> str:
    detector_text = " ".join(
        " ".join(
            [
                str(detector.get("reason", "")),
                " ".join(str(item) for item in detector.get("evidence", [])),
            ]
        )
        for detector in scenario.get("detector_results", [])
    ).lower()
    prompt_text = " ".join(turn.get("prompt", {}).get("prompt", "") for turn in scenario.get("turns", [])).lower()
    combined = f"{detector_text} {prompt_text}"

    guidance = [
        "Harden the system prompt so user messages can never override developer or system instructions.",
        "Add server-side output filtering for protected context such as system prompts, developer messages, credentials, and confidential policy text.",
        "Keep sensitive instructions and secrets outside model-visible context whenever possible.",
    ]
    if "system prompt" in combined or "developer" in combined or "hidden" in combined:
        guidance.append("Return a safe refusal whenever a user asks to reveal hidden instructions, system prompts, or internal policy.")
    if "ignore previous" in combined or "override" in combined:
        guidance.append("Detect instruction-override phrases and route them to a refusal or security-review response.")
    if "policy drift" in combined or len(scenario.get("turns", [])) > 1:
        guidance.append("Maintain refusal consistency across multi-turn conversations by rechecking the full conversation state before each answer.")
    return " ".join(guidance)


def report_file_name(report: dict[str, Any], suffix: str) -> str:
    scan_name = "".join(char.lower() if char.isalnum() else "-" for char in report.get("scan_name", "scan")).strip("-")
    return f"{scan_name or 'scan'}{suffix}"


def tool_scan_report_file_name(result: dict[str, Any], suffix: str) -> str:
    tool_id = "".join(char.lower() if char.isalnum() else "-" for char in result.get("tool_id", "tool")).strip("-")
    scan_id = str(result.get("scan_id", "scan"))[:8]
    return f"{tool_id or 'tool'}-scan-{scan_id}{suffix}"


def enterprise_markdown_text(report: dict[str, Any]) -> str:
    """Return enterprise markdown report with fallback for legacy scan files."""

    markdown = str(report.get("enterprise_report_markdown") or "").strip()
    if markdown:
        return markdown

    try:
        parsed = ScanResult.model_validate(report)
        _, markdown = generate_enterprise_report(parsed)
        return markdown
    except Exception:
        return (
            "# Enterprise AI Red Teaming Report\n\n"
            "## Summary\n"
            "Enterprise markdown report is not available in this report artifact. "
            "Run or rerun the scan to generate the enhanced report payload.\n"
        )
