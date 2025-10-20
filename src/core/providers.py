import os
from typing import Dict, List
import yaml
import httpx
from claude_agent_sdk import (AssistantMessage, ClaudeSDKClient, ResultMessage,
                              TextBlock)
from loguru import logger
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential_jitter)

from ..common.circuit_breaker import CircuitBreaker, CircuitBreakerError
from ..data.anthropic_model_prompt import ChatClient

MODEL_PROVIDER = os.environ["MODEL_PROVIDER"].lower()  
MODEL_NAME = os.environ["MODEL_NAME"] # per-provider default applied if blank
REQ_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "30"))
MAX_OUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "512"))

def get_scenario():
    with open(os.environ["SCENARIO_FILE"], mode="r", encoding="utf-8") as f:
        scenario = yaml.safe_load(f)
    return scenario


class LLMProvider:
    """Base class for large language model (LLM) providers.

    Encapsulates common configuration and interface for LLM chat providers.
    """

    def __init__(self, system_prompt: str, temperature: float, top_p: float):
        self.system_prompt = (system_prompt or "").strip()
        self.temperature = (
            float(get_scenario()['temperature'])
            if 'temperature' in get_scenario()
            else temperature
        )
        self.top_p = float(get_scenario()['top_p']) if 'top_p' in get_scenario() else top_p

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
    messages: List[Dict[str, str]], keep: int = 50
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
        self.model = MODEL_NAME
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30.0,
            expected_exception=httpx.HTTPError
        )

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
        try:
            return await self.circuit_breaker.call(self._make_request, messages)
        except CircuitBreakerError:
            logger.error("OpenAI circuit breaker is open, falling back to default response")
            return "I'm experiencing technical difficulties. Please try again later."

    async def _make_request(self, messages: List[Dict[str, str]]) -> str:
        """Internal method to make the actual API request."""
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
        self.model = os.environ.get("MODEL_NAME") if os.environ.get("MODEL_NAME") else "claude-sonnet-4-5-20250929"
        self.chat_client = ChatClient(
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=MAX_OUT_TOKENS,
            tools=[
        {
            "name": "web_search",
            "type": "web_search_20250305"
        }
    ],thinking = {
        "type": "enabled",
        "budget_tokens": 16000
    },betas=["web-search-2025-03-05"])
        

    @retry(
        wait=wait_exponential_jitter(1, 8),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )


    async def chat(self, messages: List[Dict[str, str]]) -> str:
        # # Clear previous context and update with current conversation
        self.chat_client.clear_context_messages()

        # Add conversation messages to context
        user_msgs = [
            {"role": m["role"], "content": m["content"]}
            for m in _truncate_history(messages)
        ]

        for msg in user_msgs:
            self.chat_client.update_context_messages(msg)

        response = self.chat_client.create_message()
        try:
            blocks = response.content if hasattr(response, 'content') else response.get("content", [])
            if isinstance(blocks, str):
                return blocks.strip()

            text_parts = []
            for b in blocks:
                if hasattr(b, 'text'):
                    text_parts.append(b.text)
                elif isinstance(b, dict) and b.get("type") == "text":
                    text_parts.append(b.get("text", ""))

            return "\n".join([t for t in text_parts if t]).strip()
        except Exception as e:
            logger.error(f"[Anthropic] Unexpected response schema: {response} | {e}")
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


    provider = os.getenv("MODEL_PROVIDER", MODEL_PROVIDER).lower()
    logger.info(f"Provider selected: {provider}")
    if provider == "anthropic":
        return AnthropicProvider(system_prompt, temperature, top_p)
    if provider == "ollama":
        return OllamaProvider(system_prompt, temperature, top_p)
    return OpenAIProvider(system_prompt, temperature, top_p)
