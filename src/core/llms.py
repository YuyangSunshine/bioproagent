from dataclasses import dataclass
from typing import Optional

from langchain_openai import ChatOpenAI
from mem0 import Memory

from config import settings


@dataclass
class _ChatModelConfig:
    model: str
    temperature: float = 0.0
    max_tokens: Optional[int] = None


def _build_chat(cfg: _ChatModelConfig) -> ChatOpenAI:
    kwargs = {
        'model': cfg.model,
        'openai_api_key': settings.MODEL_API_KEY,
        'temperature': cfg.temperature,
    }
    if settings.MODEL_BASE_URL:
        kwargs['openai_api_base'] = settings.MODEL_BASE_URL
    if cfg.max_tokens is not None:
        kwargs['max_tokens'] = cfg.max_tokens
    return ChatOpenAI(**kwargs)


fast_llm = _build_chat(
    _ChatModelConfig(
        model=settings.FAST_MODEL_NAME,
        temperature=0.0,
        max_tokens=min(settings.MODEL_MAX_TOKENS, 2000),
    )
)

general_llm = _build_chat(
    _ChatModelConfig(
        model=settings.GENERAL_MODEL_NAME,
        temperature=settings.MODEL_TEMPERATURE,
        max_tokens=settings.MODEL_MAX_TOKENS,
    )
)

quality_llm = _build_chat(
    _ChatModelConfig(
        model=settings.QUALITY_MODEL_NAME,
        temperature=0.1,
        max_tokens=settings.MODEL_MAX_TOKENS,
    )
)

long_context_llm = _build_chat(
    _ChatModelConfig(
        model=settings.LONG_CONTEXT_MODEL_NAME,
        temperature=0.0,
        max_tokens=settings.MODEL_MAX_TOKENS,
    )
)


class _NoopMemory:
    def add(self, *args, **kwargs):
        return None

    def search(self, *args, **kwargs):
        return []


try:
    mem = Memory.from_config(
        {
            'llm': {
                'provider': 'openai',
                'config': {
                    'model': settings.MEMORY_MODEL_NAME,
                    'api_key': settings.MODEL_API_KEY,
                    'base_url': settings.MODEL_BASE_URL or None,
                },
            }
        }
    )
except Exception:
    mem = _NoopMemory()
