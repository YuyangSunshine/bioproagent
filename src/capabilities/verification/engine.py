"""Public verification shell.

This module preserves verifier interfaces while hiding proprietary rule details.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import jsonschema

from config import settings
from src.core.llms import quality_llm


class ValidationStatus:
    PASS = 'pass'
    FAIL = 'fail'


class Severity(Enum):
    WARN = 'warn'
    HALT = 'halt'


@dataclass
class RuleViolation:
    rule_id: str
    severity: Severity
    message: str


def unified_reflector(query: str, content: str, task_type: str) -> str:
    prompt = (
        'You are a scientific protocol reviewer. '\
        'Check logical completeness, controls, and parameter clarity. '\
        'If acceptable output only: done. Otherwise provide concise revision suggestions.\\n\\n'
        f'TaskType: {task_type}\\nUserQuery: {query}\\nProtocolDraft:\\n{content}'
    )
    try:
        return quality_llm.invoke(prompt).content.strip()
    except Exception:
        return 'done'


class ProtocolValidator:
    SCHEMA = {
        'type': 'object',
        'required': ['nodes', 'consumables', 'connections'],
        'properties': {
            'nodes': {'type': 'array'},
            'consumables': {'type': 'array'},
            'connections': {'type': 'array'},
        },
    }

    def __init__(self):
        self.valid_instruments = set(settings.VALID_INSTRUMENT_IDS)
        self.valid_consumables = set(settings.VALID_CONSUMABLE_IDS)
        self.rule_engine = RuleEngine()

    def extract_json(self, content: str) -> Tuple[Optional[Dict[str, Any]], str]:
        try:
            match = re.search(r'<exp_flow>(.*?)</exp_flow>', content, re.DOTALL)
            payload = match.group(1).strip() if match else content.strip()
            if payload.startswith('```'):
                payload = re.sub(r'^```json?\s*', '', payload)
                payload = re.sub(r'\s*```$', '', payload)
            return json.loads(payload), ''
        except Exception as e:
            return None, f'JSONError: {e}'

    def check_schema(self, data: Dict[str, Any]) -> List[str]:
        try:
            jsonschema.validate(instance=data, schema=self.SCHEMA)
            return []
        except Exception as e:
            return [f'SchemaError: {e}']

    def check_resources(self, data: Dict[str, Any]) -> List[str]:
        # Public shell keeps this minimal on purpose.
        if not self.valid_instruments and not self.valid_consumables:
            return []
        errors: List[str] = []
        for node in data.get('nodes', []):
            rid = node.get('resourceId', -1)
            if rid in (-1, None):
                continue
            try:
                rid = int(rid)
            except Exception:
                errors.append('ResourceError: invalid resourceId format')
                continue
            if self.valid_instruments and rid not in self.valid_instruments:
                errors.append(f'ResourceError: unknown instrument ID {rid}')
        return errors

    def check_connections(self, data: Dict[str, Any]) -> List[str]:
        _ = data
        return []


class RuleEngine:
    """Public 22-device shell (details intentionally hidden)."""

    def __init__(self, limits_profile: str = 'public22_shell'):
        self.profile = limits_profile

    def check(self, exp_flow: Dict[str, Any]) -> List[RuleViolation]:
        _ = exp_flow
        return []


_validator = ProtocolValidator()


def validate_machine_code(content: str, verbose: bool = True, check_rules: bool = True):
    _ = verbose
    data, err = _validator.extract_json(content)
    if err:
        return False, err, []
    schema_errors = _validator.check_schema(data)
    if schema_errors:
        return False, '; '.join(schema_errors), []
    resource_errors = _validator.check_resources(data)
    if resource_errors:
        return False, '; '.join(resource_errors), []
    violations = _validator.rule_engine.check(data) if check_rules else []
    halts = [v for v in violations if v.severity == Severity.HALT]
    if halts:
        return False, '; '.join(v.message for v in halts), violations
    return True, 'Validation passed (public shell).', violations


def fix_machine_code_core(content: str) -> Tuple[bool, str, str]:
    ok, msg, _ = validate_machine_code(content, verbose=False, check_rules=False)
    if ok:
        return True, 'No fix needed', content
    # Public shell: return original payload and expose extension guidance.
    return False, f'{msg} | Please apply your private fixer rules.', content
