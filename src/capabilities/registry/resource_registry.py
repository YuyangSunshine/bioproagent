"""Public resource registry shell.

Only lightweight example resources are shipped. Users should replace these files
with their own protected registry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from config import settings
from src.capabilities.registry.base import BaseRegistry


@dataclass
class Instrument:
    resource_id: int
    name: str
    description: str
    model: str
    keywords: List[str]


@dataclass
class Consumable:
    resource_id: int
    name: str
    type: str
    keywords: List[str]


class ResourceRegistry(BaseRegistry):
    def __init__(self):
        self.instruments: Dict[int, Instrument] = {}
        self.consumables: Dict[int, Consumable] = {}
        self.valid_instrument_ids: Set[int] = set()
        self.valid_consumable_ids: Set[int] = set()
        self._load_examples()

    def _load_examples(self) -> None:
        self._load_instruments_from_json(str(settings.INSTRUMENTS_EN_PATH))
        self._load_consumables_from_json(str(settings.CONSUMABLES_EN_PATH))

    def _load_instruments_from_json(self, path: str) -> None:
        try:
            rows = json.load(open(path, 'r', encoding='utf-8'))
        except Exception:
            rows = []
        for row in rows if isinstance(rows, list) else []:
            rid = int(row.get('resourceId', 0) or 0)
            if rid <= 0:
                continue
            name = str(row.get('deviceName', f'Instrument-{rid}')).strip()
            desc = str(row.get('description', '')).strip()
            model = str(row.get('model', '')).strip()
            self.instruments[rid] = Instrument(
                resource_id=rid,
                name=name,
                description=desc,
                model=model,
                keywords=[name.lower(), model.lower()],
            )
            self.valid_instrument_ids.add(rid)

    def _load_consumables_from_json(self, path: str) -> None:
        try:
            rows = json.load(open(path, 'r', encoding='utf-8'))
        except Exception:
            rows = []
        for row in rows if isinstance(rows, list) else []:
            rid = int(row.get('resourceId', 0) or 0)
            if rid <= 0:
                continue
            name = str(row.get('name', row.get('materialName', f'Consumable-{rid}'))).strip()
            ctype = str(row.get('type', '')).strip()
            self.consumables[rid] = Consumable(
                resource_id=rid,
                name=name,
                type=ctype,
                keywords=[name.lower(), ctype.lower()],
            )
            self.valid_consumable_ids.add(rid)

    def validate_node_resource(self, node: Dict[str, Any]) -> Tuple[bool, str]:
        rid = node.get('resourceId', -1)
        if rid in (-1, None):
            return True, ''
        try:
            rid = int(rid)
        except Exception:
            return False, 'Invalid resourceId format'
        if self.valid_instrument_ids and rid not in self.valid_instrument_ids:
            return False, f'Unknown instrument resourceId: {rid}'
        return True, ''

    def is_valid_instrument_id(self, rid: int) -> bool:
        return rid in self.valid_instrument_ids or rid == -1

    def is_valid_consumable_id(self, rid: int) -> bool:
        return rid in self.valid_consumable_ids

    def match_instrument_by_text(self, text: str) -> Optional[int]:
        text_l = (text or '').lower()
        for rid, inst in self.instruments.items():
            if inst.name.lower() in text_l or inst.model.lower() in text_l:
                return rid
        return next(iter(self.valid_instrument_ids), None)

    def match_consumable_by_text(self, text: str) -> Optional[int]:
        text_l = (text or '').lower()
        for rid, con in self.consumables.items():
            if con.name.lower() in text_l or con.type.lower() in text_l:
                return rid
        return next(iter(self.valid_consumable_ids), None)

    def get_instrument_brief(self) -> str:
        if not self.instruments:
            return 'No public instrument examples provided.'
        return '\n'.join(f'- {i.resource_id}: {i.name}' for i in self.instruments.values())

    def get_consumable_brief(self) -> str:
        if not self.consumables:
            return 'No public consumable examples provided.'
        return '\n'.join(f'- {c.resource_id}: {c.name}' for c in self.consumables.values())

    # Unified base protocol compatibility
    def get_by_id(self, rid: int) -> Optional[Dict[str, Any]]:
        if rid in self.instruments:
            i = self.instruments[rid]
            return {'resourceId': i.resource_id, 'name': i.name, 'type': 'instrument'}
        if rid in self.consumables:
            c = self.consumables[rid]
            return {'resourceId': c.resource_id, 'name': c.name, 'type': 'consumable'}
        return None

    def list_available(self) -> List[Dict[str, Any]]:
        return [
            {'resourceId': i.resource_id, 'name': i.name, 'type': 'instrument'}
            for i in self.instruments.values()
        ]

    def is_valid_id(self, rid: int) -> bool:
        return rid in self.valid_instrument_ids or rid in self.valid_consumable_ids

    def summary_for_prompt(self) -> str:
        return (
            'Public registry examples (truncated). Replace with your protected registry.\\n'
            + self.get_instrument_brief()
            + '\\n'
            + self.get_consumable_brief()
        )


resource_registry = ResourceRegistry()
