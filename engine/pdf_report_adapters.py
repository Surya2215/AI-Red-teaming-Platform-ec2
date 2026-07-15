"""Map platform report payloads onto the pdf_report_template.generate_report() schema.

Two source shapes feed the enterprise PDF template, and both prefer an LLM
reporting agent's analysis over deterministic/regex-based findings when one
is present:
- Scenario/OWASP scan reports: a ScanResult dump plus the `enterprise_report`
  block computed by engine/report_generator.py (regenerated on the fly if a
  legacy report file predates that field), and optionally an `llm_analysis`
  block (see engine/scenario_report_llm_analyzer.py) with the LLM's
  executive summary, vulnerability write-ups, and remediation roadmap.
- Tool scan (Garak/PyRIT/DeepTeam) reports: a ToolScanResult dump, optionally
  with an `llm_analysis` block (see engine/tool_report_generator.py) when the
  scan completed and the LLM reporting agent produced structured findings.
"""

from __future__ import annotations

from typing import Any

from core.schemas import ScanResult
from engine.report_generator import generate_enterprise_report


def _percent(value: Any) -> str:
    try:
        return f"{float(value):g}%"
    except (TypeError, ValueError):
        return "N/A"


def _turns_label(affected_turns: list[dict[str, Any]]) -> str:
    if not affected_turns:
        return "-"
    return ", ".join(f"{item.get('scenario_id')}#T{item.get('turn')}" for item in affected_turns[:8])


def _enterprise_report(report: dict[str, Any]) -> dict[str, Any]:
    enterprise = report.get("enterprise_report")
    if isinstance(enterprise, dict):
        return enterprise
    parsed = ScanResult.model_validate(report)
    structured, _ = generate_enterprise_report(parsed)
    return structured


def scenario_report_to_pdf_data(report: dict[str, Any]) -> dict[str, Any]:
    """Map a `reports/<scan_id>.json` payload (ScanResult + enterprise_report) to PDF data.

    Prefers the LLM reporting agent's analysis (engine/scenario_report_llm_analyzer.py,
    stored as `report["llm_analysis"]`) for the qualitative sections - executive
    summary, vulnerability write-ups, remediation, risk judgment - falling back
    to the deterministic regex-based EnterpriseReportGenerator when no LLM
    analysis is present (older report files, or the LLM call failed/wasn't
    configured, in which case MultiProviderLLMClient's generic fallback stub
    won't carry our schema's "risk_level" key).
    """

    enterprise = _enterprise_report(report)
    llm_analysis = report.get("llm_analysis")
    if isinstance(llm_analysis, dict) and llm_analysis.get("risk_level"):
        return _llm_scenario_data(enterprise, llm_analysis)
    return _deterministic_scenario_data(enterprise)


def _deterministic_scenario_data(enterprise: dict[str, Any]) -> dict[str, Any]:
    meta = enterprise["scan_metadata"]
    summary = enterprise["executive_summary"]
    snapshot = enterprise["visual_risk_snapshot"]
    scope = enterprise["scope_and_system_context"]
    metrics = enterprise["metrics_and_analytics"]
    intelligence = enterprise["attack_intelligence"]
    behavior = enterprise["behavior_analysis"]
    risk_engine = enterprise["risk_scoring_engine"]
    impact = enterprise["impact_analysis"]
    roadmap = enterprise["remediation_roadmap"]
    final = enterprise["final_security_verdict"]
    reproducibility = enterprise["reproducibility_summary"]

    vulnerabilities = [
        {
            "id": item["id"],
            "finding": item["title"],
            "severity": item["severity"],
            "turns": _turns_label(item.get("affected_turns", [])),
            "reproducibility": item.get("reproducibility", "MEDIUM"),
            "category": ", ".join(meta.get("owasp_categories", [])) or "-",
            "description": item.get("description", ""),
            "root_cause": item.get("root_cause", ""),
            "impact": item.get("impact", ""),
            "attack_pattern": ", ".join(item.get("attack_patterns", [])) or "-",
            "affected_turns_detail": _turns_label(item.get("affected_turns", [])),
        }
        for item in enterprise.get("vulnerability_analysis", [])
    ]

    remediation = (
        [{"priority": "CRITICAL", "action": action} for action in roadmap.get("critical_fixes", [])]
        + [{"priority": "HIGH", "action": action} for action in roadmap.get("high_priority", [])]
        + [{"priority": "MEDIUM", "action": action} for action in roadmap.get("medium_improvements", [])]
    )

    return {
        "target": meta.get("target", "N/A"),
        "scan_reference": meta.get("scan_name", "N/A"),
        "completed_at": meta.get("timestamp", ""),
        "framework": "OWASP LLM Top 10",
        "overall_risk": summary["overall_risk_level"],
        "risk_score": f"{summary['risk_score']} / 10",
        "attack_success_rate": _percent(snapshot["attack_success_rate"]),
        "safe_response_rate": _percent(snapshot["safe_response_rate"]),
        "final_verdict": summary["final_verdict"],
        "vulnerability_percentage": _percent(snapshot["vulnerability_percentage"]),
        "total_tests": metrics["total_tests"],
        "successful_exploits": f"{metrics['successful_exploit_attempts']} ({metrics['successful_exploit_attempts_percent']}%)",
        "safe_refusals": f"{metrics['safe_refusals']} ({metrics['safe_refusal_rate_percent']}%)",
        "partial_leakage": f"{metrics['partial_leakage_cases']} ({metrics['partial_leakage_rate_percent']}%)",
        "severity_distribution": {
            "CRITICAL": snapshot["risk_distribution"]["critical"],
            "HIGH": snapshot["risk_distribution"]["high"],
            "MEDIUM": snapshot["risk_distribution"]["medium"],
            "LOW": snapshot["risk_distribution"]["low"],
        },
        "key_findings": summary["key_findings"],
        "recommendation": summary["business_impact"],
        "scope": {
            "Target System": meta.get("target", "N/A"),
            "Interfaces Tested": ", ".join(scope.get("interfaces_tested", [])) or "-",
            "Attack Categories": ", ".join(scope.get("attack_categories_used", [])) or "-",
            "Scan Reference": meta.get("scan_name", "N/A"),
            "Scenario Count": str(len(meta.get("owasp_categories", []))),
            "Scan Status": meta.get("status", "N/A"),
        },
        "attack_strategies": enterprise["methodology"]["attack_strategies"],
        "vulnerabilities": vulnerabilities,
        "pattern_detection": {
            "Role Impersonation": "Observed" if intelligence["pattern_detection"]["role_impersonation"] else "Not observed",
            "Prefix Guessing": "Observed" if intelligence["pattern_detection"]["prefix_guessing"] else "Not observed",
            "Iterative Probing": "Observed" if intelligence["pattern_detection"]["iterative_probing"] else "Not observed",
            "Over-Helpfulness": "Observed" if intelligence["pattern_detection"]["over_helpfulness"] else "Not observed",
        },
        "model_weaknesses": intelligence["model_weaknesses"] or ["None observed."],
        "insights": intelligence["insights"],
        "forward_risk": intelligence["what_can_go_wrong"],
        "behavior_analysis": {
            "Safe Refusal Consistency": behavior["safe_refusal_consistency"],
            "Inconsistent Responses": str(behavior["inconsistent_responses_detected"]),
            "Indirect Leakage Patterns": str(behavior["indirect_leakage_patterns_detected"]),
        },
        "behavior_explanation": behavior["explanation"],
        "risk_formula": risk_engine["formula"],
        "risk_inputs": {
            "Exploit Success Rate": str(risk_engine["exploit_success_rate"]),
            "Impact Score": str(risk_engine["impact_score"]),
            "Normalized Risk Score": f"{risk_engine['normalized_risk_score']} / 10",
        },
        "impact_analysis": {
            "Credential Leakage Risk": impact["credential_leakage_risk"],
            "System Exposure": impact["system_exposure"],
            "Social Engineering Support Risk": impact["social_engineering_support_risk"],
        },
        "impact_summary": impact["business_friendly_summary"],
        "remediation": remediation or [{"priority": "LOW", "action": "No remediation actions required."}],
        "verdict": {
            "Verdict": final["verdict"],
            "Deployment Readiness": final["deployment_readiness"],
            "Confidence Level": final["confidence_level"],
        },
        "reproducibility_consistency": reproducibility["consistency"],
        "reproducibility_summary": reproducibility["summary"],
        "revision_history": [
            {
                "version": "1.0",
                "date": str(meta.get("timestamp", ""))[:10] or "N/A",
                "author": "AI Red Teaming Platform — Automated Engine",
                "description": "Automated report generated from scan run.",
            }
        ],
    }


def _llm_scenario_data(enterprise: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    """Combine deterministic scan metrics/scope with the LLM's qualitative analysis.

    Metrics (attack success rate, test counts, scope, attack strategies used)
    stay deterministic - they're exact counts computed from the transcript,
    not something an LLM should "judge". Findings, remediation, and risk
    narrative come from the LLM instead of EnterpriseReportGenerator's regex
    heuristics.
    """

    meta = enterprise["scan_metadata"]
    snapshot = enterprise["visual_risk_snapshot"]
    scope = enterprise["scope_and_system_context"]
    metrics = enterprise["metrics_and_analytics"]

    vulnerabilities_raw = analysis.get("vulnerabilities") or []
    vulnerabilities = [
        {
            "id": f"VULN-{index + 1:03d}",
            "finding": item.get("title", "Untitled"),
            "severity": item.get("severity", "INFO"),
            "turns": item.get("affected_turns", "-"),
            "reproducibility": item.get("reproducibility", "MEDIUM"),
            "category": ", ".join(meta.get("owasp_categories", [])) or "-",
            "description": item.get("description", ""),
            "root_cause": item.get("root_cause", ""),
            "impact": item.get("impact", ""),
            "attack_pattern": item.get("attack_pattern", "-"),
            "affected_turns_detail": item.get("affected_turns", "-"),
        }
        for index, item in enumerate(vulnerabilities_raw)
    ]

    severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for item in vulnerabilities_raw:
        severity = str(item.get("severity", "")).upper()
        if severity in severity_counts:
            severity_counts[severity] += 1

    remediation = [
        {"priority": str(item.get("priority", "MEDIUM")).upper(), "action": item.get("action", "")}
        for item in (analysis.get("remediation_roadmap") or [])
    ]

    pattern_detection = analysis.get("pattern_detection") or {}
    behavior = analysis.get("behavior_analysis") or {}
    impact = analysis.get("impact_analysis") or {}

    return {
        "target": meta.get("target", "N/A"),
        "scan_reference": meta.get("scan_name", "N/A"),
        "completed_at": meta.get("timestamp", ""),
        "framework": "OWASP LLM Top 10",
        "overall_risk": analysis.get("risk_level", "INFO"),
        "risk_score": f"{analysis.get('risk_score', 0)} / 10",
        "attack_success_rate": _percent(snapshot["attack_success_rate"]),
        "safe_response_rate": _percent(snapshot["safe_response_rate"]),
        "final_verdict": analysis.get("final_verdict", "PASS"),
        "vulnerability_percentage": _percent(snapshot["vulnerability_percentage"]),
        "total_tests": metrics["total_tests"],
        "successful_exploits": f"{metrics['successful_exploit_attempts']} ({metrics['successful_exploit_attempts_percent']}%)",
        "safe_refusals": f"{metrics['safe_refusals']} ({metrics['safe_refusal_rate_percent']}%)",
        "partial_leakage": f"{metrics['partial_leakage_cases']} ({metrics['partial_leakage_rate_percent']}%)",
        "severity_distribution": severity_counts,
        "key_findings": analysis.get("key_findings") or [],
        "recommendation": analysis.get("business_impact", ""),
        "scope": {
            "Target System": meta.get("target", "N/A"),
            "Interfaces Tested": ", ".join(scope.get("interfaces_tested", [])) or "-",
            "Attack Categories": ", ".join(scope.get("attack_categories_used", [])) or "-",
            "Scan Reference": meta.get("scan_name", "N/A"),
            "Scenario Count": str(len(meta.get("owasp_categories", []))),
            "Scan Status": meta.get("status", "N/A"),
        },
        "attack_strategies": enterprise["methodology"]["attack_strategies"],
        "vulnerabilities": vulnerabilities,
        "pattern_detection": {
            "Role Impersonation": pattern_detection.get("role_impersonation", "Not observed"),
            "Prefix Guessing": pattern_detection.get("prefix_guessing", "Not observed"),
            "Iterative Probing": pattern_detection.get("iterative_probing", "Not observed"),
            "Over-Helpfulness": pattern_detection.get("over_helpfulness", "Not observed"),
        },
        "model_weaknesses": analysis.get("model_weaknesses") or ["None observed."],
        "insights": analysis.get("insights") or [],
        "forward_risk": analysis.get("forward_risk") or [],
        "behavior_analysis": {
            "Safe Refusal Consistency": behavior.get("safe_refusal_consistency", "N/A"),
            "Inconsistent Responses": behavior.get("inconsistent_responses", "N/A"),
            "Indirect Leakage Patterns": behavior.get("indirect_leakage_patterns", "N/A"),
        },
        "behavior_explanation": analysis.get("behavior_explanation", ""),
        "risk_formula": "LLM reporting agent risk_level/risk_score derived from full transcript + detector analysis.",
        "risk_inputs": {"Risk Score": str(analysis.get("risk_score", 0))},
        "impact_analysis": {
            "Credential Leakage Risk": impact.get("credential_leakage_risk", "N/A"),
            "System Exposure": impact.get("system_exposure", "N/A"),
            "Social Engineering Support Risk": impact.get("social_engineering_support_risk", "N/A"),
        },
        "impact_summary": analysis.get("impact_summary", ""),
        "remediation": remediation or [{"priority": "LOW", "action": "No remediation actions required."}],
        "verdict": {
            "Verdict": analysis.get("final_verdict", "PASS"),
            "Deployment Readiness": analysis.get("deployment_readiness", "NEEDS_REMEDIATION"),
            "Confidence Level": analysis.get("confidence_level", "MEDIUM"),
        },
        "reproducibility_consistency": analysis.get("reproducibility_consistency", "N/A"),
        "reproducibility_summary": analysis.get("reproducibility_summary", ""),
        "revision_history": [
            {
                "version": "1.0",
                "date": str(meta.get("timestamp", ""))[:10] or "N/A",
                "author": "AI Red Teaming Platform — LLM Reporting Agent",
                "description": f"Automated report generated from scan run, analyzed by {analysis.get('_provider', 'LLM')}.",
            }
        ],
    }


def tool_scan_report_to_pdf_data(result: dict[str, Any]) -> dict[str, Any]:
    """Map a `reports/tool-scans/scan-<scan_id>.json` payload (ToolScanResult) to PDF data.

    Tool scans don't have the turn/scenario structure OWASP scans do, so scope,
    pattern-detection, and behavior-analysis sections stay unpopulated unless the
    LLM reporting agent (engine/tool_report_generator.py) attached `llm_analysis`.
    """

    analysis = result.get("llm_analysis") or {}
    tool_name = result.get("tool_name", "N/A")
    status = result.get("status", "UNKNOWN")

    risk_level = analysis.get("risk_level", "INFO" if status == "COMPLETED" else "N/A")
    risk_score = analysis.get("risk_score", 0)
    executive_summary = analysis.get("executive_summary", "")
    key_findings = analysis.get("key_findings") or []
    vulnerabilities_raw = analysis.get("vulnerabilities") or []
    remediation_raw = analysis.get("remediation_roadmap") or []
    deployment_readiness = analysis.get("deployment_readiness", "NEEDS_REMEDIATION" if not analysis else "READY")
    confidence_level = analysis.get("confidence_level", "LOW" if not analysis else "MEDIUM")

    if not analysis:
        executive_summary = (
            f"No structured LLM analysis is available for this {tool_name} scan "
            f"(status: {status}). Raw command output is included in the appendix for review."
        )
        key_findings = [f"Scan status: {status}."]
        if result.get("error"):
            key_findings.append(f"Error: {result['error']}")

    vulnerabilities = [
        {
            "id": f"TOOL-{index + 1:03d}",
            "finding": item.get("title", "Untitled"),
            "severity": item.get("severity", "INFO"),
            "turns": "-",
            "reproducibility": "N/A",
            "category": tool_name,
            "description": item.get("description", ""),
            "root_cause": "See description.",
            "impact": item.get("evidence", ""),
            "attack_pattern": tool_name,
            "affected_turns_detail": "-",
        }
        for index, item in enumerate(vulnerabilities_raw)
    ]

    remediation = [{"priority": "HIGH", "action": action} for action in remediation_raw] or [
        {"priority": "LOW", "action": "No remediation actions required."}
    ]

    severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for item in vulnerabilities_raw:
        severity = str(item.get("severity", "")).upper()
        if severity in severity_counts:
            severity_counts[severity] += 1

    # def_table renders each field as a single (unsplittable) table row, so an
    # oversized cell can exceed the page's frame height and raise a ReportLab
    # LayoutError with no way to paginate mid-cell. Keep these excerpts short -
    # the full untruncated stdout/stderr already lives in the underlying JSON
    # report file (downloadable via the existing "Export JSON" button).
    _STDOUT_EXCERPT_CHARS = 1500
    _STDERR_EXCERPT_CHARS = 800

    appendix = [
        {
            "title": "Executed Command",
            "fields": {"Command": " ".join(str(part) for part in result.get("command", [])) or "-"},
        },
        {
            "title": f"Standard Output (excerpt, first {_STDOUT_EXCERPT_CHARS} chars - see JSON export for full output)",
            "fields": {"stdout": (result.get("stdout") or "No stdout captured.")[:_STDOUT_EXCERPT_CHARS]},
        },
    ]
    if result.get("stderr"):
        appendix.append(
            {
                "title": f"Standard Error (excerpt, first {_STDERR_EXCERPT_CHARS} chars - see JSON export for full output)",
                "fields": {"stderr": result["stderr"][:_STDERR_EXCERPT_CHARS]},
            }
        )

    return {
        "target": tool_name,
        "scan_reference": str(result.get("scan_id", "N/A"))[:36],
        "completed_at": result.get("completed_at") or result.get("started_at") or "",
        "framework": f"{tool_name} External Tool Scan",
        "overall_risk": risk_level,
        "risk_score": f"{risk_score} / 10",
        "attack_success_rate": "N/A",
        "safe_response_rate": "N/A",
        "final_verdict": "FAIL" if risk_level in {"CRITICAL", "HIGH"} else "PASS",
        "vulnerability_percentage": "N/A",
        "total_tests": len(result.get("pyrit_strategies") or []) or "N/A",
        "successful_exploits": "N/A",
        "safe_refusals": "N/A",
        "partial_leakage": "N/A",
        "severity_distribution": severity_counts,
        "key_findings": key_findings,
        "recommendation": executive_summary,
        "scope": {
            "Tool": tool_name,
            "Scenario": result.get("pyrit_scenario") or "-",
            "Strategies": ", ".join(result.get("pyrit_strategies") or []) or "-",
            "Status": status,
            "Return Code": str(result.get("return_code", "-")),
        },
        "attack_strategies": result.get("pyrit_strategies") or [],
        "vulnerabilities": vulnerabilities,
        "pattern_detection": {},
        "model_weaknesses": [],
        "insights": [],
        "forward_risk": [],
        "behavior_analysis": {},
        "behavior_explanation": "",
        "risk_formula": "LLM reporting agent risk_level/risk_score derived from analysis of raw tool output.",
        "risk_inputs": {"Risk Score": str(risk_score)},
        "impact_analysis": {},
        "impact_summary": analysis.get("business_impact", ""),
        "remediation": remediation,
        "verdict": {
            "Verdict": "FAIL" if deployment_readiness == "NOT_READY" else "PASS",
            "Deployment Readiness": deployment_readiness,
            "Confidence Level": confidence_level,
        },
        "reproducibility_consistency": "N/A",
        "reproducibility_summary": "Tool scans are not scored for reproducibility the way OWASP scenario scans are.",
        "revision_history": [
            {
                "version": "1.0",
                "date": str(result.get("completed_at") or result.get("started_at") or "")[:10] or "N/A",
                "author": "AI Red Teaming Platform — Tool Scan Connector",
                "description": f"Automated {tool_name} scan report.",
            }
        ],
        "appendix": appendix,
    }
