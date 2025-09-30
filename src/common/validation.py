"""
Input validation and sanitization for chatbot endpoints.

Provides comprehensive validation for user inputs, message formatting,
and security sanitization to prevent injection attacks.
"""

import html
import re
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field, ValidationError, validator


class MessageModel(BaseModel):
    """Validated message structure for chatbot interactions."""

    role: str = Field(..., pattern=r"^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=8192)

    @validator("content")
    def sanitize_content(cls, v):
        """Sanitize message content to prevent XSS and injection."""
        # HTML escape
        v = html.escape(v)

        # Remove potentially dangerous patterns
        dangerous_patterns = [
            r"<script[^>]*>.*?</script>",  # JavaScript
            r"javascript:",  # JS URLs
            r"on\w+\s*=",  # Event handlers
            r"<iframe[^>]*>.*?</iframe>",  # Iframes
        ]

        for pattern in dangerous_patterns:
            v = re.sub(pattern, "", v, flags=re.IGNORECASE | re.DOTALL)

        # Normalize whitespace
        v = " ".join(v.split())

        return v


class SessionRequestModel(BaseModel):
    """Validated session creation request."""

    candidate_hash: Optional[str] = Field(None, max_length=128)
    scenario_id: str = Field(..., min_length=1, max_length=64)

    @validator("candidate_hash")
    def validate_hash(cls, v):
        """Validate candidate hash format."""
        if v and not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Invalid hash format")
        return v

    @validator("scenario_id")
    def validate_scenario_id(cls, v):
        """Validate scenario ID format."""
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Invalid scenario ID format")
        return v


class ChatRequestModel(BaseModel):
    """Validated chat request."""

    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=4096)

    @validator("session_id")
    def validate_session_id(cls, v):
        """Validate session ID format."""
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Invalid session ID format")
        return v

    @validator("message")
    def sanitize_message(cls, v):
        """Sanitize user message."""
        # HTML escape
        v = html.escape(v)

        # Remove excessive whitespace
        v = " ".join(v.split())

        # Basic profanity filter (extend as needed)
        profanity_patterns = [
            r"\b(spam|test|hack|exploit)\b",
        ]

        for pattern in profanity_patterns:
            v = re.sub(pattern, "[filtered]", v, flags=re.IGNORECASE)

        return v


def validate_json_input(
    data: Dict[str, Any], model_class: BaseModel
) -> Optional[BaseModel]:
    """Validate JSON input against a Pydantic model.

    Args:
        data: Raw JSON data
        model_class: Pydantic model class to validate against

    Returns:
        Validated model instance or None if invalid
    """
    try:
        return model_class(**data)
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        return None


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations."""
    # Remove path traversal attempts
    filename = filename.replace("..", "").replace("/", "").replace("\\", "")

    # Keep only alphanumeric, dots, dashes, underscores
    filename = re.sub(r"[^a-zA-Z0-9._-]", "", filename)

    # Limit length
    filename = filename[:100]

    return filename


def validate_rate_limit_key(key: str) -> bool:
    """Validate rate limiting key format."""
    return bool(re.match(r"^[a-zA-Z0-9._:-]+$", key)) and len(key) <= 256


def is_safe_redirect_url(url: str, allowed_hosts: List[str]) -> bool:
    """Check if URL is safe for redirects."""
    if not url:
        return False

    # Must be relative or from allowed hosts
    if url.startswith("/"):
        return True

    for host in allowed_hosts:
        if url.startswith(f"https://{host}") or url.startswith(f"http://{host}"):
            return True

    return False


def extract_mentions_and_hashtags(text: str) -> Dict[str, List[str]]:
    """Extract mentions and hashtags from text safely."""
    mentions = re.findall(r"@([a-zA-Z0-9_]+)", text)
    hashtags = re.findall(r"#([a-zA-Z0-9_]+)", text)

    return {
        "mentions": mentions[:10],  # Limit to prevent abuse
        "hashtags": hashtags[:10],
    }
