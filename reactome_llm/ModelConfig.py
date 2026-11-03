"""LangChain chat models for the gene-first pipeline and the draft builder.

Which model each role uses is set in ONE place, curator_llm/model_settings.py (roles 'gene', 'draft', 'convert'),
and can be overridden with LLM_MODEL_<ROLE> / LLM_THINKING_<ROLE>.
"""
import os
import sys
from typing import Optional, Tuple
from langchain_anthropic import ChatAnthropic

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from curator_llm.model_settings import langchain_kwargs, spec
import token_profiler

# kept for callers that read them; both follow the 'gene' role
REACTOME_MODEL_NAME = spec('gene').model
REACTOME_MODEL_TEMPERATURE = spec('gene').temperature


def get_reactome_model_settings() -> Tuple[str, float]:
    s = spec('gene')
    return s.model, s.temperature


def create_chat_model(role: str, max_tokens: Optional[int] = None) -> ChatAnthropic:
    """ChatAnthropic for a model role. Token-usage profiling is opt-in (TOKEN_PROFILE env var); the callbacks
    are None when it is off, so the model is constructed exactly as before with no callback attached.

    Pass `max_tokens` for long replies: without it langchain-anthropic takes the limit from its model profile,
    and falls back to 4096 for a model it has no profile for (claude-sonnet-5-5 in langchain-anthropic 1.7)."""
    extra = {'max_tokens': max_tokens} if max_tokens else {}
    return ChatAnthropic(callbacks=token_profiler.langchain_callbacks(), **langchain_kwargs(role), **extra)


def create_reactome_chat_model() -> ChatAnthropic:
    """The gene-first pipeline's model (role 'gene')."""
    return create_chat_model('gene')
