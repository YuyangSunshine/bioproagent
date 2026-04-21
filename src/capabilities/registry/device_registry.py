"""Public device registry prompt helper (privacy-preserving)."""

from __future__ import annotations

from typing import Dict, List, Optional

from src.capabilities.registry.base import BaseRegistry
from src.capabilities.registry.resource_registry import resource_registry


class DeviceRegistry(BaseRegistry):
    def __init__(self):
        self._devices = [d for d in resource_registry.list_available() if d.get('type') == 'instrument']

    def get_available_devices(self) -> List[Dict]:
        return self._devices

    def get_device_by_id(self, rid: int) -> Optional[Dict]:
        for d in self._devices:
            if d.get('resourceId') == rid:
                return d
        return None

    # Base protocol compatibility
    def get_by_id(self, rid: int) -> Optional[Dict]:
        return self.get_device_by_id(rid)

    def list_available(self) -> List[Dict]:
        return self.get_available_devices()

    def is_valid_id(self, rid: int) -> bool:
        return self.get_device_by_id(rid) is not None

    def summary_for_prompt(self) -> str:
        return self.generate_device_table_for_prompt()

    def generate_device_table_for_prompt(self) -> str:
        lines = [
            '### Public Device Registry (Example Only)',
            'This release intentionally provides only minimal examples.',
            'Please configure your own device catalog and parameters.',
            '',
            '| resourceId | Device Name |',
            '|------------|-------------|',
        ]
        for d in self._devices:
            lines.append(f"| {d.get('resourceId')} | {d.get('name', 'Unknown')} |")
        return '\n'.join(lines)

    def generate_device_params_guide(self) -> str:
        return (
            '### Parameter Guide (Template)\\n'
            '- Define your own per-device parameter schema in private deployment.\\n'
            '- Keep parameters validated by your local safety policy.'
        )


device_registry = DeviceRegistry()


def get_device_prompt_content() -> str:
    return device_registry.generate_device_table_for_prompt() + '\n\n' + device_registry.generate_device_params_guide()
