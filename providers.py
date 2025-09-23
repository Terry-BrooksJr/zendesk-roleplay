import os
from typing import Dict, List

import httpx
from huggingface_hub import login as hf_login
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

# ENV CONFIG (with safe defaults)
MODEL_PROVIDER = os.getenv(
    "MODEL_PROVIDER", "openai"
).lower()  # openai|anthropic|ollama
MODEL_NAME = os.getenv("MODEL_NAME", "")  # per-provider default applied if blank
REQ_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "30"))
MAX_OUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "512"))


class LLMProvider:
    """Base class for large language model (LLM) providers.

    Encapsulates common configuration and interface for LLM chat providers.
    """

    def __init__(self, system_prompt: str, temperature: float, top_p: float):
        self.system_prompt = (system_prompt or "").strip()
        self.temperature = float(temperature)
        self.top_p = float(top_p)

    async def chat(self, messages: List[Dict[str, str]]) -> str:
        """Generate a chat response from the LLM given a list of messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.

        Returns:
            The generated response as a string.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError


# ---------- Utilities ----------
def _inject_system_prompt(
    messages: List[Dict[str, str]], system_prompt: str
) -> List[Dict[str, str]]:
    """Prepends or merged = msgs.copy()
            merges a merged[0]["content"] = system prompt into the message history.

        If a system prompt is provided, this function ensures it is included as the first message.
        If the first message is already a f"{system_prompt}\n\n{merged[0].get('content', '')}".strip()
            return merged
        return [{"role": "system", "content": system_prompt}] + msgs
    =======
    def system message, the new prompt is prepended to its content.

        Args:
            messages: The list of message dictionaries to modify.
            system_prompt: _inject_system_prompt(messages: List[Dict[str, str]], system_prompt: str) -> List[Dict[str, str]]:
    """
    if not system_prompt:
        return messages
    msgs = messages or []
    if msgs and msgs[0].get("role") == "system":
        # prepend/merge, don't overwrite entirely
        merged = msgs.copy()
        merged[0][
            "content"
        ] = f"{system_prompt}\n\n{merged[0].get('content', '')}".strip()
        return merged
    return [{"role": "system", "content": system_prompt}] + msgs


def _truncate_history(
    messages: List[Dict[str, str]], keep: int = 12
) -> List[Dict[str, str]]:
    """Truncates the message history to keep only the most recent messages.

    Retains the first system message if present, and the last `keep` non-system messages.

    Args:
        messages: List of message dictionaries, each with a 'role' and 'content'.
        keep: The number of non-system messages to retain.

    Returns:
        A truncated list of messages with at most one system message and up to `keep` non-system messages.
    """
    if not messages:
        return messages
    sys_msgs = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    return (sys_msgs[:1] + rest[-keep:]) if sys_msgs else rest[-keep:]


# ---------- OpenAI ----------
class OpenAIProvider(LLMProvider):
    """Provider for interacting with OpenAI's chat models.

    Handles message formatting, API requests, and response parsing for OpenAI's chat completions API.
    """

    def __init__(self, system_prompt: str, temperature: float, top_p: float):
        super().__init__(system_prompt, temperature, top_p)
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.model = MODEL_NAME or "gpt-4o-mini"

    @retry(
        wait=wait_exponential_jitter(1, 8),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send a chat request to OpenAI's API and return the generated response.

        Formats the message history and system prompt, sends the request, and parses the response content.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.

        Returns:
            The generated response as a string.

        Raises:
            Exception: If the API response schema is unexpected or the request fails.
        """
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": MAX_OUT_TOKENS,
            "messages": _truncate_history(
                _inject_system_prompt(messages, self.system_prompt)
            ),
        }
        async with httpx.AsyncClient(timeout=REQ_TIMEOUT) as client:
            r = await client.post(
                url, headers={"Authorization": f"Bearer {self.api_key}"}, json=payload
            )
        r.raise_for_status()
        data = r.json()
        try:
            content = data["choices"][0]["message"]["content"]
            return content or ""
        except Exception as e:
            logger.error(f"[OpenAI] Unexpected response schema: {data} | {e}")
            raise


# ---------- Anthropic ----------
class AnthropicProvider(LLMProvider):
    def __init__(self, system_prompt: str, temperature: float, top_p: float):
        super().__init__(system_prompt, temperature, top_p)
        self.api_key = os.environ["ANTHROPIC_API_KEY"]
        self.model = MODEL_NAME or "claude-3-5-sonnet-latest"

    @retry(
        wait=wait_exponential_jitter(1, 8),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def chat(self, messages: List[Dict[str, str]]) -> str:
        # Anthropic uses separate system + messages
        url = "https://api.anthropic.com/v1/messages"
        user_msgs = [
            {"role": m["role"], "content": m["content"]}
            for m in _truncate_history(messages)
        ]
        payload = {
            "model": self.model,
            "system": self.system_prompt or None,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": MAX_OUT_TOKENS,
            "messages": user_msgs,
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        async with httpx.AsyncClient(timeout=REQ_TIMEOUT) as client:
            r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        try:
            blocks = data.get("content", [])
            text_parts = [b.get("text", "") for b in blocks if isinstance(b, dict)]
            return "\n".join([t for t in text_parts if t]).strip()
        except Exception as e:
            logger.error(f"[Anthropic] Unexpected response schema: {data} | {e}")
            raise


# ---------- Ollama (local) ----------
class OllamaProvider(LLMProvider):
    def __init__(self, system_prompt: str, temperature: float, top_p: float):
        super().__init__(system_prompt, temperature, top_p)
        self.base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.model = MODEL_NAME or "llama3.2"

    @retry(
        wait=wait_exponential_jitter(1, 8),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def chat(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self.base}/api/chat"
        payload = {
            "model": self.model,
            "options": {"temperature": self.temperature, "top_p": self.top_p},
            "messages": _truncate_history(
                _inject_system_prompt(messages, self.system_prompt)
            ),
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=REQ_TIMEOUT) as client:
            r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        try:
            return data["message"]["content"] or ""
        except Exception as e:
            logger.error(f"[Ollama] Unexpected response schema: {data} | {e}")
            raise


# ---------- Factory ----------
def make_provider(system_prompt: str, temperature: float, top_p: float):
    """Creates and returns an LLM provider instance based on environment configuration.

    Selects the appropriate provider (Anthropic, Ollama, or OpenAI) using environment variables and returns an instance configured with the given parameters.

    Args:
        system_prompt: The system prompt string to use for the provider.
        temperature: The temperature setting for the provider.
        top_p: The top_p setting for the provider.

    Returns:
        An instance of AnthropicProvider, OllamaProvider, or OpenAIProvider.
    """
    hf_login(
        token=os.getenv("HUGGINGFACE_API_KEY", ""),
        add_to_git_credential=True,
        skip_if_logged_in=True,
    )

    provider = os.getenv("MODEL_PROVIDER", MODEL_PROVIDER).lower()
    logger.info(f"Provider selected: {provider}")
    if provider == "anthropic":
        return AnthropicProvider(system_prompt, temperature, top_p)
    if provider == "ollama":
        return OllamaProvider(system_prompt, temperature, top_p)
    return OpenAIProvider(system_prompt, temperature, top_p)
