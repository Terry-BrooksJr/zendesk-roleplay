"""Common utilities and shared components."""

from .circuit_breaker import CircuitBreaker, CircuitBreakerError
from .embedding_cache import cache_embedding, get_cached_embedding
from .rate_limiter import rate_limit_middleware
from .storage import (
    add_milestone,
    end_session,
    get_session,
    get_session_milestones,
    get_session_transcript,
    log_turn,
    new_session,
    update_session,
)
from .validation import ChatRequestModel, validate_json_input

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerError",
    "cache_embedding",
    "get_cached_embedding",
    "rate_limit_middleware",
    "add_milestone",
    "end_session",
    "get_session",
    "get_session_milestones",
    "get_session_transcript",
    "log_turn",
    "new_session",
    "update_session",
    "ChatRequestModel",
    "validate_json_input",
]