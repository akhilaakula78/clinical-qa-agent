"""Single source of truth for LLM client. Mockable in tests."""
from __future__ import annotations

from functools import lru_cache
from typing import Union

from clinical_qa.config import settings


@lru_cache(maxsize=1)
def get_llm():
    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0,
            max_output_tokens=2048,
        )
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        model=settings.claude_model,
        api_key=settings.anthropic_api_key,
        temperature=0,
        max_tokens=2048,
    )
