"""Attack engine that executes dynamically loaded scenarios."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from core.logging import get_logger
from core.schemas import ScanRequest, ScanSettings, ScenarioResult
from engine.judge_agent import JudgeAgent
from engine.scenario_loader import PluginLoader
from engine.target_executor import TargetExecutor


logger = get_logger(__name__)


SINGLE_TURN_ATTACK_FIELDS = {
    "llm01.prompt_injection": "prompt_injection_attack_types",
    "llm02.sensitive_information_disclosure": "sensitive_information_attack_types",
    "llm03.supply_chain": "supply_chain_attack_types",
    "llm04.data_model_poisoning": "data_model_poisoning_attack_types",
    "llm05.improper_output_handling": "improper_output_handling_attack_types",
    "llm06.excessive_agency": "excessive_agency_attack_types",
    "llm07.system_prompt_leakage": "system_prompt_leakage_attack_types",
    "llm08.vector_embedding_weaknesses": "vector_embedding_attack_types",
    "llm09.misinformation": "misinformation_attack_types",
    "llm10.unbounded_consumption": "unbounded_consumption_attack_types",
}


MULTI_TURN_CHAIN_FIELDS = {
    "llm06.excessive_agency": "excessive_agency_multi_turn_chains",
    "llm07.system_prompt_leakage": "system_prompt_leakage_multi_turn_chains",
    "llm08.vector_embedding_weaknesses": "vector_embedding_multi_turn_chains",
    "llm09.misinformation": "misinformation_multi_turn_chains",
    "llm10.unbounded_consumption": "unbounded_consumption_multi_turn_chains",
}


class AttackEngine:
    """Load scenarios and execute them asynchronously."""

    def __init__(
        self,
        loader: PluginLoader | None = None,
        target_executor: TargetExecutor | None = None,
        judge_agent: JudgeAgent | None = None,
    ) -> None:
        self.loader = loader or PluginLoader()
        self.target_executor = target_executor
        self.judge_agent = judge_agent

    async def run(self, request: ScanRequest) -> AsyncIterator[ScenarioResult]:
        scenarios = self.loader.discover_scenarios(request.owasp_category)
        selected_ids = request.scenario_ids or list(scenarios)
        for scenario_id in selected_ids:
            scenario = scenarios.get(scenario_id)
            if scenario is None:
                logger.warning("Requested scenario not found.", extra={"scenario": scenario_id, "scan_id": request.scan_id})
                continue
            for settings in self._expanded_scenario_settings(scenario, request.settings):
                executor = self.target_executor or TargetExecutor(settings, scan_id=request.scan_id)
                judge = self.judge_agent or JudgeAgent(settings=settings)
                logger.info("Executing scenario.", extra={"scenario": scenario_id, "scan_id": request.scan_id})
                result = await scenario.run(
                    target=request.target,
                    settings=settings,
                    target_executor=executor,
                    judge_agent=judge,
                    scan_id=request.scan_id,
                )
                self._label_expanded_result(scenario, settings, result)
                yield result

    def _expanded_scenario_settings(self, scenario: Any, settings: ScanSettings) -> list[ScanSettings]:
        scenario_id = scenario.metadata.id

        if settings.max_turns > 1:
            chain_field = MULTI_TURN_CHAIN_FIELDS.get(scenario_id)
            chain_defs = self._definitions(scenario, "multi_turn_attack_definitions", "chain_id")
            if chain_field and chain_defs:
                selected_chains = list(getattr(settings, chain_field, []) or [item["id"] for item in chain_defs])
                return [settings.model_copy(update={chain_field: [chain_id]}) for chain_id in selected_chains]
            return [settings]

        attack_field = SINGLE_TURN_ATTACK_FIELDS.get(scenario_id)
        attack_defs = self._definitions(scenario, "attack_definitions", "attack_id")
        if attack_field and attack_defs:
            selected_attacks = list(getattr(settings, attack_field, []) or [item["id"] for item in attack_defs])
            return [settings.model_copy(update={attack_field: [attack_id]}) for attack_id in selected_attacks]
        return [settings]

    def _definitions(self, scenario: Any, method_name: str, id_key: str) -> list[dict[str, str]]:
        method = getattr(scenario, method_name, None)
        if not callable(method):
            return []
        definitions: list[dict[str, str]] = []
        for item in method():
            identifier = item.get(id_key)
            if identifier:
                definitions.append(
                    {
                        "id": str(identifier),
                        "label": str(item.get("attack_type") or item.get("name") or identifier),
                    }
                )
        return definitions

    def _label_expanded_result(self, scenario: Any, settings: ScanSettings, result: ScenarioResult) -> None:
        scenario_id = scenario.metadata.id
        attack_field = SINGLE_TURN_ATTACK_FIELDS.get(scenario_id)
        if attack_field and settings.max_turns <= 1:
            selected = list(getattr(settings, attack_field, []) or [])
            if len(selected) == 1:
                labels = {item["id"]: item["label"] for item in self._definitions(scenario, "attack_definitions", "attack_id")}
                label = labels.get(selected[0], selected[0].replace("_", " ").title())
                if label not in result.scenario_name:
                    result.scenario_name = f"{result.scenario_name}: {label}"

