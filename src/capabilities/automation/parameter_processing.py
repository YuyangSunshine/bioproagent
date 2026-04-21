"""Unified automation facade for public release."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.capabilities.automation.parameter_builder import parameter_builder
from src.capabilities.automation.parameter_filler import parameter_builder as parameter_filler
from src.capabilities.registry.resource_registry import resource_registry


class ParameterProcessor:
    def __init__(self):
        self.builder = parameter_builder
        self.filler = parameter_filler
        self.registry = resource_registry

    def construct(self, simplified_flow: Dict[str, Any]) -> Dict[str, Any]:
        return self.builder.build_experiment_flow(simplified_flow)

    def fill(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        return self.filler.build_node(operation)

    def validate(self, node: Dict[str, Any]) -> Tuple[bool, List[str]]:
        ok, msg = self.registry.validate_node_resource(node)
        return ok, ([msg] if msg else [])


parameter_processor = ParameterProcessor()
