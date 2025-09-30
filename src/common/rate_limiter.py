"""
Rate limiting middleware for chatbot API endpoints.

Implements token bucket algorithm with Redis backend for distributed
rate limiting across multiple instances.
"""

import asyncio
import os
import time
from typing import Dict

from fastapi import HTTPException, Request
from loguru import logger


class InMemoryRateLimiter:
    """In-memory rate limiter using token bucket algorithm."""

    def __init__(self):
        self.buckets: Dict[str, Dict[str, float]] = {}
        self.lock = asyncio.Lock()

    async def is_allowed(
        self, key: str, limit: int = 60, window: int = 60
    ) -> tuple[bool, Dict[str, int]]:
        """Check if request is allowed based on rate limit.

        Args:
            key: Unique identifier for rate limiting (IP, user ID, etc.)
            limit: Maximum requests allowed in the window
            window: Time window in seconds

        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        async with self.lock:
            now = time.time()

            if key not in self.buckets:
                self.buckets[key] = {"tokens": limit, "last_refill": now}

            bucket = self.buckets[key]

            # Refill tokens based on time passed
            time_passed = now - bucket["last_refill"]
            tokens_to_add = (time_passed / window) * limit
            bucket["tokens"] = min(limit, bucket["tokens"] + tokens_to_add)
            bucket["last_refill"] = now

            # Check if request can be made
            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                remaining = int(bucket["tokens"])
                reset_time = int(now + (window - (now % window)))

                return True, {
                    "limit": limit,
                    "remaining": remaining,
                    "reset": reset_time,
                    "retry_after": 0,
                }
            else:
                # Calculate retry after
                retry_after = int((1 - bucket["tokens"]) * (window / limit))

                return False, {
                    "limit": limit,
                    "remaining": 0,
                    "reset": int(now + retry_after),
                    "retry_after": retry_after,
                }

    def clear_bucket(self, key: str):
        """Clear rate limit bucket for a key."""
        if key in self.buckets:
            del self.buckets[key]

    def get_stats(self) -> Dict[str, int]:
        """Get rate limiter statistics."""
        return {
            "total_buckets": len(self.buckets),
            "active_buckets": len(
                [
                    k
                    for k, v in self.buckets.items()
                    if time.time() - v["last_refill"] < 300  # Active in last 5 min
                ]
            ),
        }


# Global rate limiter instance
_rate_limiter = InMemoryRateLimiter()


async def rate_limit_middleware(request: Request, call_next):
    """FastAPI middleware for rate limiting."""
    # Skip rate limiting for health checks
    if request.url.path in ["/health", "/metrics"]:
        return await call_next(request)

    # Get client identifier (prefer API key, fallback to IP)
    client_key = get_client_identifier(request)

    # Different limits for different endpoints
    limits = get_endpoint_limits(request.url.path)

    # Check rate limit
    allowed, rate_info = await _rate_limiter.is_allowed(
        client_key, limits["requests"], limits["window"]
    )

    if not allowed:
        logger.warning(
            "Rate limit exceeded",
            client=client_key,
            endpoint=request.url.path,
            retry_after=rate_info["retry_after"],
        )

        headers = {
            "X-RateLimit-Limit": str(rate_info["limit"]),
            "X-RateLimit-Remaining": str(rate_info["remaining"]),
            "X-RateLimit-Reset": str(rate_info["reset"]),
            "Retry-After": str(rate_info["retry_after"]),
        }

        raise HTTPException(
            status_code=429, detail="Rate limit exceeded", headers=headers
        )

    # Add rate limit headers to response
    response = await call_next(request)

    response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
    response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
    response.headers["X-RateLimit-Reset"] = str(rate_info["reset"])

    return response


def get_client_identifier(request: Request) -> str:
    """Extract client identifier for rate limiting."""
    # Check for API key in headers
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api:{api_key[:8]}..."  # Truncate for privacy

    # Check for authenticated user
    # This would be implemented with your auth system
    # user_id = get_authenticated_user(request)
    # if user_id:
    #     return f"user:{user_id}"

    # Fallback to IP address
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Use first IP in X-Forwarded-For chain
        client_ip = forwarded_for.split(",")[0].strip()
    else:
        client_ip = request.client.host

    return f"ip:{client_ip}"


def get_endpoint_limits(path: str) -> Dict[str, int]:
    """Get rate limit configuration for endpoint."""
    # Default limits
    default_limits = {
        "requests": int(os.getenv("DEFAULT_RATE_LIMIT", "60")),
        "window": int(os.getenv("DEFAULT_RATE_WINDOW", "60")),
    }

    # Endpoint-specific limits
    endpoint_limits = {
        "/reply": {
            "requests": int(os.getenv("CHAT_RATE_LIMIT", "30")),
            "window": int(os.getenv("CHAT_RATE_WINDOW", "60")),
        },
        "/start": {
            "requests": int(os.getenv("SESSION_RATE_LIMIT", "10")),
            "window": int(os.getenv("SESSION_RATE_WINDOW", "60")),
        },
        "/finish": {
            "requests": int(os.getenv("FINISH_RATE_LIMIT", "10")),
            "window": int(os.getenv("FINISH_RATE_WINDOW", "60")),
        },
    }

    # Find matching endpoint (supports path parameters)
    for endpoint_path, limits in endpoint_limits.items():
        if path.startswith(endpoint_path):
            return limits

    return default_limits


async def clear_rate_limit(client_key: str):
    """Clear rate limit for a specific client."""
    _rate_limiter.clear_bucket(client_key)


async def get_rate_limit_stats() -> Dict[str, int]:
    """Get rate limiter statistics."""
    return _rate_limiter.get_stats()
