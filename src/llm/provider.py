"""LLM provider factory with model tiering (fast/main), tenacity retries, and Gemini -> Groq fallback."""

from __future__ import annotations

import asyncio
from typing import Any, Literal, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)

_OVERRIDE_LLM: BaseChatModel | None = None


def set_override_llm(llm: BaseChatModel | None) -> None:
    """Inject a mock/fake LLM for deterministic integration tests."""
    global _OVERRIDE_LLM
    _OVERRIDE_LLM = llm


def _is_retryable_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    retry_signals = ("429", "rate limit", "resource exhausted", "503", "502", "timeout", "temporarily")
    return any(sig in msg for sig in retry_signals) or isinstance(exc, (TimeoutError, asyncio.TimeoutError))


def _build_single_model(provider: str, tier: Literal["fast", "main"]) -> BaseChatModel | None:
    settings = get_settings()
    env = settings.env
    models_cfg = settings.llm.models
    temp = settings.llm.temperature

    if provider == "gemini":
        if not env.gemini_api_key or "your_" in env.gemini_api_key:
            return None
        from langchain_google_genai import ChatGoogleGenerativeAI

        model_name = models_cfg.gemini_fast if tier == "fast" else models_cfg.gemini_main
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=env.gemini_api_key,
            temperature=temp,
            max_retries=1,
            timeout=30.0,
        )

    if provider == "groq":
        if not env.groq_api_key or "your_" in env.groq_api_key:
            return None
        from langchain_groq import ChatGroq

        model_name = models_cfg.groq_fast if tier == "fast" else models_cfg.groq_main
        return ChatGroq(
            model=model_name,
            groq_api_key=env.groq_api_key,
            temperature=temp,
            max_retries=1,
            timeout=30.0,
        )

    if provider == "openai_compatible":
        if not env.openai_compat_base_url:
            return None
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=models_cfg.kaggle_openweight,
            base_url=env.openai_compat_base_url,
            api_key=env.openai_compat_api_key,
            temperature=temp,
            max_retries=1,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")


def get_ordered_models(tier: Literal["fast", "main"] = "main") -> list[tuple[str, BaseChatModel]]:
    """Return configured (provider_name, chat_model) list in fallback priority order."""
    if _OVERRIDE_LLM is not None:
        return [("override", _OVERRIDE_LLM)]

    settings = get_settings()
    order: list[str] = []
    primary = settings.llm.primary_provider
    if primary not in order:
        order.append(primary)
    for fb in settings.llm.fallback_order:
        if fb not in order:
            order.append(fb)

    models: list[tuple[str, BaseChatModel]] = []
    for prov in order:
        model = _build_single_model(prov, tier=tier)
        if model is not None:
            models.append((prov, model))

    if not models:
        raise RuntimeError(
            "No LLM provider is configured. Set GEMINI_API_KEY and/or GROQ_API_KEY in .env."
        )
    return models


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1.0, max=12.0),
    retry=retry_if_exception(_is_retryable_error),
    reraise=True,
)
async def _ainvoke_once(runnable: Any, messages: Any) -> Any:
    return await runnable.ainvoke(messages)


async def ainvoke_text(messages: Any, *, tier: Literal["fast", "main"] = "main") -> str:
    """Invoke LLM with retry + automatic provider fallback and return plain text."""
    last_exc: Exception | None = None
    for prov, model in get_ordered_models(tier=tier):
        try:
            resp = await _ainvoke_once(model, messages)
            content = getattr(resp, "content", resp)
            if isinstance(content, list):
                return "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                ).strip()
            return str(content).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"LLM provider '{prov}' failed ({exc}); trying next fallback.")
            last_exc = exc
    raise RuntimeError(f"All LLM providers failed. Last error: {last_exc}") from last_exc


async def ainvoke_structured(
    schema: type[T],
    messages: Any,
    *,
    tier: Literal["fast", "main"] = "fast",
) -> T:
    """Invoke LLM with structured Pydantic output, retry, and automatic provider fallback."""
    last_exc: Exception | None = None
    for prov, model in get_ordered_models(tier=tier):
        try:
            structured = model.with_structured_output(schema)
            result = await _ainvoke_once(structured, messages)
            if isinstance(result, schema):
                return result
            if isinstance(result, dict):
                return schema.model_validate(result)
            raise ValueError(f"Unexpected structured output type: {type(result)}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Structured output on provider '{prov}' failed ({exc}); trying next fallback."
            )
            last_exc = exc
    raise RuntimeError(f"All LLM providers failed structured output. Last error: {last_exc}") from last_exc


async def astream_text(messages: Any, *, tier: Literal["fast", "main"] = "main"):
    """Stream text deltas from primary LLM with automatic fallback to next provider."""
    last_exc: Exception | None = None
    for prov, model in get_ordered_models(tier=tier):
        yielded = False
        try:
            async for chunk in model.astream(messages):
                content = getattr(chunk, "content", "")
                if isinstance(content, list):
                    delta = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                else:
                    delta = str(content or "")
                if delta:
                    yielded = True
                    yield delta
            return
        except Exception as exc:
            if yielded:
                raise
            logger.warning(f"Streaming on provider '{prov}' failed ({exc}); trying next fallback.")
            last_exc = exc
    raise RuntimeError(f"All LLM providers failed streaming. Last error: {last_exc}") from last_exc
