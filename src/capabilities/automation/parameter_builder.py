"""Public automation builder shell.

This module keeps the interface only. Private mapping logic from aligned protocol
-> vendor-specific machine code is intentionally removed.
"""

from __future__ import annotations

from typing import Any, Dict, List


class ParameterBuilder:
    def __init__(self, resource_registry=None):
        self.resource_registry = resource_registry

    def build_node(self, operation: Dict[str, Any], idx: int) -> Dict[str, Any]:
        resource_id = operation.get('resourceId', -1)
        if resource_id is None:
            resource_id = -1
        try:
            resource_id = int(resource_id)
        except Exception:
            resource_id = -1
        return {
            'templateNodeId': int(operation.get('templateNodeId', idx + 1)),
            'resourceId': resource_id,
            'parameters': operation.get('parameters', {}),
        }

    def build_experiment_flow(self, simplified_flow: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(simplified_flow, dict):
            return {'nodes': [], 'consumables': [], 'connections': []}

        nodes_in = simplified_flow.get('nodes', [])
        if not isinstance(nodes_in, list):
            nodes_in = []

        nodes: List[Dict[str, Any]] = []
        for idx, op in enumerate(nodes_in):
            if not isinstance(op, dict):
                op = {'parameters': {'description': str(op)}}
            nodes.append(self.build_node(op, idx))

        consumables = simplified_flow.get('consumables', [])
        if not isinstance(consumables, list):
            consumables = []

        connections = simplified_flow.get('connections', [])
        if not isinstance(connections, list):
            connections = []

        return {
            'nodes': nodes,
            'consumables': consumables,
            'connections': connections,
        }


parameter_builder = ParameterBuilder()
