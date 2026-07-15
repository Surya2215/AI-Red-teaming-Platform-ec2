"""Dynamic scenario and detector plugin discovery."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol

from core.config import ROOT_DIR
from core.exceptions import PluginLoadError
from core.logging import get_logger
from core.schemas import PluginMetadata


logger = get_logger(__name__)


class ScenarioPlugin(Protocol):
    metadata: PluginMetadata


class DetectorPlugin(Protocol):
    metadata: PluginMetadata


class PluginLoader:
    """Discover and instantiate file-based Python plugins."""

    def __init__(self, root_dir: Path | None = None) -> None:
        self.root_dir = root_dir or ROOT_DIR
        self.scenario_root = self.root_dir / "scenarios"
        self.checks_root = self.root_dir / "checks"

    def discover_scenarios(self, category: str | None = None) -> dict[str, object]:
        return self._discover(self.scenario_root, "get_scenario", category)

    def discover_detectors(self, category: str | None = None) -> dict[str, object]:
        return self._discover(self.checks_root, "get_detector", category)

    def _discover(
        self,
        root: Path,
        factory_name: str,
        category: str | None,
    ) -> dict[str, object]:

        plugins: dict[str, object] = {}

        search_roots = (
            self._category_search_roots(root, category)
            if category
            else [path for path in root.iterdir() if path.is_dir()]
        )

        for search_root in search_roots:

            if not search_root.exists():
                continue

            for file_path in sorted(search_root.glob("*.py")):

                if file_path.name.startswith("__"):
                    continue

                try:
                    module = self._load_module(file_path)

                    factory = getattr(module, factory_name, None)

                    if factory is None:
                        logger.warning(
                            "Plugin factory missing.",
                            extra={"plugin": str(file_path)},
                        )
                        continue

                    plugin = factory()

                    metadata = getattr(plugin, "metadata", None)

                    if metadata is None:
                        logger.warning(
                            "Plugin metadata missing.",
                            extra={"plugin": str(file_path)},
                        )
                        continue

                    plugins[metadata.id] = plugin

                except Exception as exc:
                    raise PluginLoadError(
                        f"Failed loading plugin {file_path}: {exc}"
                    ) from exc

        return plugins

    def _category_search_roots(self, root: Path, category: str) -> list[Path]:
        variants = [
            category,
            category.replace("-", "_"),
            category.replace(" ", "_"),
            category.replace("-", "_").replace(" ", "_"),
            category.replace("_", "-"),
        ]
        seen: set[Path] = set()
        roots: list[Path] = []
        for variant in variants:
            path = root / variant
            if path not in seen:
                roots.append(path)
                seen.add(path)
        return roots

    def _load_module(self, file_path: Path) -> ModuleType:
        """
        Safely load plugin modules dynamically.

        Fixes Python 3.12 dataclass + dynamic import issues.
        """

        # Safe module name
        module_name = (
            f"redteam_plugin_{file_path.stem}"
            .replace("-", "_")
            .replace(" ", "_")
        )

        spec = importlib.util.spec_from_file_location(
            module_name,
            file_path,
        )

        if spec is None or spec.loader is None:
            raise PluginLoadError(
                f"Cannot create import spec for {file_path}"
            )

        module = importlib.util.module_from_spec(spec)

        # IMPORTANT:
        # Register module BEFORE execution
        # Required for Python 3.12 dataclasses
        sys.modules[module_name] = module

        # Set package/loader metadata
        module.__loader__ = spec.loader
        module.__package__ = module_name.rpartition(".")[0]

        # Execute module
        spec.loader.exec_module(module)

        return module
