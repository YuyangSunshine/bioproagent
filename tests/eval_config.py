import json
import os
from typing import Any, Dict


def load_eval_config(config_path: str = "config/evaluation_profiles.json") -> Dict[str, Any]:
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_subset_test_file(cfg: Dict[str, Any], subset: str, fallback: str) -> str:
    return cfg.get("subsets", {}).get(subset, fallback)


def get_common(cfg: Dict[str, Any], key: str, fallback: Any) -> Any:
    return cfg.get("common", {}).get(key, fallback)


def get_nested(cfg: Dict[str, Any], path: str, fallback: Any) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return fallback
        cur = cur[part]
    return cur
