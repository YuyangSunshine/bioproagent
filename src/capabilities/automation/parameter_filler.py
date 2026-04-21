"""Public parameter filler shell."""

from __future__ import annotations

from typing import Any, Dict


class ParameterBuilder:
    def build_node(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(operation, dict):
            operation = {'parameters': {'description': str(operation)}}
        if 'parameters' not in operation or not isinstance(operation['parameters'], dict):
            operation['parameters'] = {}
        operation.setdefault('resourceId', -1)
        operation.setdefault('templateNodeId', 1)
        return operation


parameter_builder = ParameterBuilder()
