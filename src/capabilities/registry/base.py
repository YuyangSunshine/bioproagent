"""
Shared registry protocol to keep ResourceRegistry and DeviceRegistry APIs aligned.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseRegistry(ABC):
    @abstractmethod
    def get_by_id(self, rid: int) -> Optional[Any]:
        raise NotImplementedError

    @abstractmethod
    def list_available(self) -> List[Any]:
        raise NotImplementedError

    @abstractmethod
    def is_valid_id(self, rid: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    def summary_for_prompt(self) -> str:
        raise NotImplementedError

    def refresh(self) -> None:
        """
        Optional hook for registries backed by external files.
        """
        return None
