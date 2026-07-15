"""DeepTeam's vulnerability/attack catalog, as actually wired into the installed CLI's
YAML config loader (deepteam.cli.main.VULN_MAP / ATTACK_MAP, deepteam==1.0.7 - the
latest available version, confirmed via `pip index versions deepteam`).

The deepteam.vulnerabilities package itself exports more classes than this (e.g.
ChildProtection, Ethics, Fairness, CrossContextRetrieval, ToolMetadataPoisoning), but
`deepteam run config.yaml` raises "Unknown vulnerability: X" for any name not in
VULN_MAP - so this catalog is deliberately scoped to what a scan can actually execute,
not the library's full class list. Types per vulnerability were extracted by
instantiating each class with no args and reading its default `.types`.
"""

from __future__ import annotations

DEEPTEAM_VULNERABILITIES: dict[str, list[str]] = {
    "Bias": ["religion", "politics", "gender", "race"],
    "Toxicity": ["profanity", "insults", "threats", "mockery"],
    "Misinformation": ["factual_errors", "unsupported_claims", "expertize_misrepresentation"],
    "IllegalActivity": [
        "weapons", "illegal_drugs", "violent_crimes", "non_violent_crimes", "sex_crimes",
        "cybercrime", "child_exploitation",
    ],
    "PromptLeakage": ["secrets_and_credentials", "instructions", "guard_exposure", "permissions_and_roles"],
    "ExcessiveAgency": ["functionality", "permissions", "autonomy"],
    "Robustness": ["input_overreliance", "hijacking"],
    "IntellectualProperty": ["imitation", "copyright_violations", "trademark_infringement", "patent_disclosure"],
    "Competition": ["competitor_mention", "market_manipulation", "discreditation", "confidential_strategies"],
    "GraphicContent": ["sexual_content", "graphic_content", "pornographic_content"],
    "PersonalSafety": ["bullying", "self_harm", "unsafe_practices", "dangerous_challenges", "stalking"],
    "BFLA": ["privilege_escalation", "function_bypass", "authorization_bypass"],
    "BOLA": ["object_access_bypass", "cross_customer_access", "unauthorized_object_manipulation"],
    "RBAC": ["role_bypass", "privilege_escalation", "unauthorized_role_assumption"],
    "DebugAccess": ["debug_mode_bypass", "development_endpoint_access", "administrative_interface_exposure"],
    "ShellInjection": ["command_injection", "system_command_execution", "shell_escape_sequences"],
    "SQLInjection": ["blind_sql_injection", "union_based_injection", "error_based_injection"],
    "SSRF": ["internal_service_access", "cloud_metadata_access", "port_scanning"],
    "GoalTheft": ["escalating_probing", "cooperative_dialogue", "social_engineering"],
    "RecursiveHijacking": ["self_modifying_goals", "recursive_objective_chaining", "goal_propagation_attacks"],
}

DEEPTEAM_VULNERABILITY_CATEGORIES: dict[str, list[str]] = {
    "Data Privacy": ["PromptLeakage"],
    "Responsible AI": ["Bias", "Toxicity"],
    "Security": ["BFLA", "BOLA", "RBAC", "DebugAccess", "ShellInjection", "SQLInjection", "SSRF"],
    "Safety": ["IllegalActivity", "GraphicContent", "PersonalSafety"],
    "Business": ["Misinformation", "IntellectualProperty", "Competition"],
    "Agentic": ["GoalTheft", "RecursiveHijacking", "ExcessiveAgency", "Robustness"],
}

# Single-turn attacks take no special config; multi-turn ones support a handful of
# tuning knobs in the YAML (see _build_attack in deepteam's CLI) but sensible defaults
# (no extra kwargs) are enough to run them.
DEEPTEAM_SINGLE_TURN_ATTACKS: list[str] = [
    "Base64", "GrayBox", "Leetspeak", "MathProblem", "Multilingual",
    "PromptInjection", "PromptProbing", "Roleplay", "ROT13",
]
DEEPTEAM_MULTI_TURN_ATTACKS: list[str] = [
    "CrescendoJailbreaking", "LinearJailbreaking", "TreeJailbreaking",
    "SequentialJailbreak", "BadLikertJudge",
]
DEEPTEAM_ATTACKS: list[str] = DEEPTEAM_SINGLE_TURN_ATTACKS + DEEPTEAM_MULTI_TURN_ATTACKS

# Providers deepteam.cli.model_callback.load_model() supports for target.model. Only
# ollama/azure/bedrock accept credentials via the YAML spec dict - openai/anthropic
# ignore any api_key passed in the spec and read solely from the worker's ambient
# OPENAI_API_KEY/ANTHROPIC_API_KEY env vars (same ones the fixed simulator/evaluation
# models already require), so those two providers get no credential fields here.
DEEPTEAM_PROVIDERS: tuple[str, ...] = ("ollama", "azure", "bedrock", "openai", "anthropic")

DEEPTEAM_CREDENTIAL_FIELDS: dict[str, list[dict[str, object]]] = {
    "ollama": [
        {"key": "base_url", "label": "Ollama host", "placeholder": "http://localhost:11434", "secret": False, "required": False},
    ],
    "azure": [
        {"key": "endpoint", "label": "Azure endpoint", "placeholder": "https://your-resource.openai.azure.com/", "secret": False, "required": True},
        {"key": "deployment_name", "label": "Deployment name", "placeholder": "gpt-4o-prod", "secret": False, "required": True},
        {"key": "api_version", "label": "API version", "placeholder": "2024-06-01", "secret": False, "required": False},
        {"key": "api_key", "label": "Azure API key", "placeholder": "Azure OpenAI API key", "secret": True, "required": True},
    ],
    "bedrock": [
        {"key": "region_name", "label": "AWS region", "placeholder": "us-east-1", "secret": False, "required": True},
        {"key": "aws_access_key_id", "label": "AWS access key ID", "placeholder": "AKIA...", "secret": True, "required": True},
        {"key": "aws_secret_access_key", "label": "AWS secret access key", "placeholder": "", "secret": True, "required": True},
    ],
    "openai": [],
    "anthropic": [],
}
DEEPTEAM_CREDENTIAL_SCHEMA: dict[str, tuple[str, ...]] = {
    provider: tuple(field["key"] for field in fields) for provider, fields in DEEPTEAM_CREDENTIAL_FIELDS.items()
}
DEEPTEAM_CREDENTIAL_REQUIRED: dict[str, tuple[str, ...]] = {
    provider: tuple(field["key"] for field in fields if field["required"])
    for provider, fields in DEEPTEAM_CREDENTIAL_FIELDS.items()
}
DEEPTEAM_SECRET_CREDENTIAL_KEYS: frozenset[str] = frozenset(
    field["key"] for fields in DEEPTEAM_CREDENTIAL_FIELDS.values() for field in fields if field["secret"]
)


def default_attacks_per_vulnerability_type(attack_count: int) -> int:
    """Scale attacks-per-vulnerability-type down as more attacks are selected, so
    total test volume (vulnerabilities x attacks x attacks_per_vulnerability_type)
    doesn't explode when someone selects many attacks at once. User-adjustable in the
    UI - this only picks a sane starting point."""

    if attack_count <= 2:
        return 3
    if attack_count <= 5:
        return 2
    return 1
