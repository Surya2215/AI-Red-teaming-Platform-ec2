"""Pydantic contracts used by targets, scenarios, detectors, and scans."""

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, HttpUrl, field_validator


class HttpMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"


class Severity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TargetConfig(BaseModel):
    """Config-driven target application contract."""

    name: str = Field(min_length=1, max_length=160)
    url: str
    method: HttpMethod = HttpMethod.POST
    headers: dict[str, str] = Field(default_factory=dict)
    request_template: dict[str, Any] = Field(default_factory=dict)
    auth: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float | None = Field(default=None, ge=0, le=300)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        if value.startswith("mock://"):
            return value
        HttpUrl(value)
        return value


class ScanSettings(BaseModel):
    """User-tunable scan execution settings."""

    max_turns: int = Field(default=6, ge=1, le=25)
    timeout_seconds: float = Field(default=30, ge=1, le=300)
    concurrency: int = Field(default=2, ge=1, le=20)
    temperature: float = Field(default=0.2, ge=0, le=2)
    retry_count: int = Field(default=2, ge=0, le=5)
    crescendo_profile: str = "authority_escalation_system_prompt"
    prompt_injection_attack_types: list[str] = Field(default_factory=list)
    prompt_injection_include_single_turn: bool = True
    sensitive_information_attack_types: list[str] = Field(default_factory=list)
    supply_chain_attack_types: list[str] = Field(default_factory=list)
    data_model_poisoning_attack_types: list[str] = Field(default_factory=list)
    improper_output_handling_attack_types: list[str] = Field(default_factory=list)
    excessive_agency_attack_types: list[str] = Field(default_factory=list)
    excessive_agency_multi_turn_chains: list[str] = Field(default_factory=list)
    system_prompt_leakage_attack_types: list[str] = Field(default_factory=list)
    system_prompt_leakage_multi_turn_chains: list[str] = Field(default_factory=list)
    vector_embedding_attack_types: list[str] = Field(default_factory=list)
    vector_embedding_multi_turn_chains: list[str] = Field(default_factory=list)
    misinformation_attack_types: list[str] = Field(default_factory=list)
    misinformation_multi_turn_chains: list[str] = Field(default_factory=list)
    unbounded_consumption_attack_types: list[str] = Field(default_factory=list)
    unbounded_consumption_multi_turn_chains: list[str] = Field(default_factory=list)


class AttackPrompt(BaseModel):
    """Single prompt produced by an attack scenario."""

    prompt: str
    category: str
    stage: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TargetResponse(BaseModel):
    """Normalized target response."""

    status_code: int
    body: str
    headers: dict[str, str] = Field(default_factory=dict)
    elapsed_ms: float = 0
    error: str | None = None


class AttackTurn(BaseModel):
    """One prompt/response pair in an attack chain."""

    turn: int
    prompt: AttackPrompt
    response: TargetResponse
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    judge_decision: "JudgeDecision | None" = None


class JudgeDecision(BaseModel):
    """Structured decision emitted by the adaptive judge agent."""

    next_action: str
    reasoning: str
    risk_score: float = Field(ge=0, le=1)
    continue_attack: bool
    suggested_prompt: str | None = None
    evidence: list[str] = Field(default_factory=list)


class DetectorResult(BaseModel):
    """Normalized detector output."""

    detector_id: str
    vulnerable: bool
    confidence: float = Field(ge=0, le=1)
    reason: str
    evidence: list[str] = Field(default_factory=list)
    severity: Severity = Severity.INFO
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScenarioResult(BaseModel):
    """Result from a single scenario execution."""

    scenario_id: str
    scenario_name: str
    owasp_category: str
    turns: list[AttackTurn]
    detector_results: list[DetectorResult] = Field(default_factory=list)
    started_at: datetime
    completed_at: datetime | None = None


class ScanStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ScanRequest(BaseModel):
    """End-to-end scan request."""

    scan_name: str
    target: TargetConfig
    owasp_category: str = "LLM01-Prompt Injection"
    scenario_ids: list[str] = Field(default_factory=list)
    settings: ScanSettings = Field(default_factory=ScanSettings)
    scan_id: str = Field(default_factory=lambda: str(uuid4()))


class ScanResult(BaseModel):
    """Persistable scan result contract."""

    scan_id: str
    scan_name: str
    target_name: str
    status: ScanStatus
    scenario_results: list[ScenarioResult] = Field(default_factory=list)
    started_at: datetime
    completed_at: datetime | None = None
    error: str | None = None
    llm_analysis: dict[str, Any] | None = None

    @property
    def vulnerable(self) -> bool:
        return any(result.vulnerable for scenario in self.scenario_results for result in scenario.detector_results)


class PluginMetadata(BaseModel):
    """Metadata exposed by scenario and detector plugins."""

    id: str
    name: str
    owasp_category: str
    description: str
    type: Literal["single_turn", "multi_turn", "detector"]
    version: str = "1.0.0"
