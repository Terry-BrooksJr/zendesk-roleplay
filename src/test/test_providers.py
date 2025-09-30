import sys
from typing import Dict, List

import pytest

from ..core.providers import (
    LLMProvider,
    _inject_system_prompt,
    _truncate_history,
    make_provider,
)


class Dummy:
    # Minimal class to attach the chat method for testing
    async def chat(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "messages",
    [
        ([{"role": "user", "content": "Hello"}],),
        ([{"role": "system", "content": "System prompt"}],),
        ([{"role": "user", "content": ""}],),
        (
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
        ),
        ([],),
        (None,),
    ],
    ids=[
        "single_user_message",
        "single_system_message",
        "empty_content",
        "multiple_messages",
        "empty_list",
        "none_messages",
    ],
)
async def test_chat_always_raises_notimplementederror(messages):
    # Arrange
    dummy = Dummy()

    # Act & Assert
    with pytest.raises(NotImplementedError):
        await dummy.chat(messages)


@pytest.mark.parametrize(
    "system_prompt, temperature, top_import pytest p, expected_prompt,expected_temp, expected_top_p",
    [
        # happy path: normal values
        (
            "You are a helpful assistant.",
            0.7,
            0.9,
            "You are a helpful assistant.",
            0.7,
            0.9,
        ),
        # edge: system_prompt is None
        (None, 0.5, 0.8, "", 0.5, 0.8),
        # edge: system_prompt is empty string
        ("", 0.2, 0.3, "", 0.2, 0.3),
        # edge: system_prompt with whitespace
        ("  Hello!  ", 1.0, 1.0, "Hello!", 1.0, 1.0),
        # edge: temperature and top_p as int
        ("Prompt", 1, 1, "Prompt", 1.0, 1.0),
        # edge: temperature and top_p as string numbers
        ("Prompt", "0.1", "0.2", "Prompt", 0.1, 0.2),
        # edge: negative values
        ("Prompt", -1, -2, "Prompt", -1.0, -2.0),
    ],
    ids=[
        "normal_values",
        "none_prompt",
        "empty_prompt",
        "whitespace_prompt",
        "int_temp_top_p",
        "str_temp_top_p",
        "negative_temp_top_p",
    ],
)
def test_llmprovider_init(
    system_prompt, temperature, top_p, expected_prompt, expected_temp, expected_top_p
):
    # Act
    provider = LLMProvider(system_prompt, temperature, top_p)

    # Assert
    assert provider.system_prompt == expected_prompt
    assert provider.temperature == expected_temp
    assert provider.top_p == expected_top_p


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "messages",
    [
        # happy path: single message
        ([{"role": "user", "content": "Hello"}],),
        # edge: empty list
        ([],),
        # edge: list with empty dict
        ([{}],),
        # edge: list with missing keys
        ([{"role": "user"}],),
        ([{"content": "Hi"}],),
        # edge: list with multiple messages
        (
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
        ),
        # edge: None
        (None,),
    ],
    ids=[
        "single_message",
        "empty_list",
        "empty_dict",
        "missing_content",
        "missing_role",
        "multiple_messages",
        "none_messages",
    ],
)
async def test_llmprovider_chat_raises(messages):
    # Arrange
    provider = LLMProvider("Prompt", 0.5, 0.5)

    # Act & Assert
    with pytest.raises(NotImplementedError):
        await provider.chat(messages)


@pytest.mark.parametrize(
    "messages, system_prompt, expected",
    [
        # happy path: no system message, non-empty prompt
        (
            [{"role": "user", "content": "Hi"}],
            "System prompt",
            [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "Hi"},
            ],
        ),
        # happy path: already has system message, merges prompt
        (
            [
                {"role": "system", "content": "Existing"},
                {"role": "user", "content": "Hi"},
            ],
            "System prompt",
            [
                {"role": "system", "content": "System prompt\n\nExisting"},
                {"role": "user", "content": "Hi"},
            ],
        ),
        # edge: empty messages, non-empty prompt
        (
            [],
            "System prompt",
            [{"role": "system", "content": "System prompt"}],
        ),
        # edge: None messages, non-empty prompt
        (
            None,
            "System prompt",
            [{"role": "system", "content": "System prompt"}],
        ),
        # edge: system_prompt is empty, messages present
        (
            [{"role": "user", "content": "Hi"}],
            "",
            [{"role": "user", "content": "Hi"}],
        ),
        # edge: system_prompt is empty, messages is None
        (
            None,
            "",
            None,
        ),
        # edge: system_prompt is empty, messages is empty list
        (
            [],
            "",
            [],
        ),
        # edge: system message with empty content
        (
            [{"role": "system", "content": ""}, {"role": "user", "content": "Hi"}],
            "System prompt",
            [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "Hi"},
            ],
        ),
        # edge: system message with None content
        (
            [{"role": "system", "content": None}, {"role": "user", "content": "Hi"}],
            "System prompt",
            [
                {"role": "system", "content": "System prompt\n\n"},
                {"role": "user", "content": "Hi"},
            ],
        ),
        # edge: first message is not system, but later is
        (
            [{"role": "user", "content": "Hi"}, {"role": "system", "content": "Late"}],
            "System prompt",
            [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "Hi"},
                {"role": "system", "content": "Late"},
            ],
        ),
    ],
    ids=[
        "prepend_system_prompt",
        "merge_with_existing_system",
        "empty_messages",
        "none_messages",
        "empty_prompt_with_messages",
        "empty_prompt_none_messages",
        "empty_prompt_empty_messages",
        "system_message_empty_content",
        "system_message_none_content",
        "system_message_not_first",
    ],
)
def test_inject_system_prompt(messages, system_prompt, expected):
    # Act
    result = _inject_system_prompt(messages, system_prompt)

    # Assert
    assert result == expected


@pytest.mark.parametrize(
    "messages, keep, expected",
    [
        # happy path: system message + >keep user messages
        (
            [{"role": "system", "content": "sys"}]
            + [{"role": "user", "content": f"msg{i}"} for i in range(15)],
            10,
            [{"role": "system", "content": "sys"}]
            + [{"role": "user", "content": f"msg{i}"} for i in range(5, 15)],
        ),
        # happy path: system message + <keep user messages
        (
            [{"role": "system", "content": "sys"}]
            + [{"role": "user", "content": f"msg{i}"} for i in range(3)],
            10,
            [{"role": "system", "content": "sys"}]
            + [{"role": "user", "content": f"msg{i}"} for i in range(3)],
        ),
        # edge: no system message, >keep user messages
        (
            [{"role": "user", "content": f"msg{i}"} for i in range(15)],
            5,
            [{"role": "user", "content": f"msg{i}"} for i in range(10, 15)],
        ),
        # edge: no system message, <keep user messages
        (
            [{"role": "user", "content": f"msg{i}"} for i in range(3)],
            10,
            [{"role": "user", "content": f"msg{i}"} for i in range(3)],
        ),
        # edge: only system message
        (
            [{"role": "system", "content": "sys"}],
            5,
            [{"role": "system", "content": "sys"}],
        ),
        # edge: empty messages
        (
            [],
            5,
            [],
        ),
        # edge: None messages
        (
            None,
            5,
            None,
        ),
        # edge: keep=0, with system message
        (
            [{"role": "system", "content": "sys"}]
            + [{"role": "user", "content": f"msg{i}"} for i in range(3)],
            0,
            [{"role": "system", "content": "sys"}],
        ),
        # edge: keep=0, no system message
        (
            [{"role": "user", "content": "msg0"}],
            0,
            [],
        ),
        # edge: multiple system messages, only first kept
        (
            [
                {"role": "system", "content": "sys1"},
                {"role": "system", "content": "sys2"},
            ]
            + [{"role": "user", "content": "msg0"}],
            1,
            [
                {"role": "system", "content": "sys1"},
                {"role": "user", "content": "msg0"},
            ],
        ),
        # edge: messages with missing 'role' key
        (
            [{"role": "system", "content": "sys"}, {"content": "msg0"}],
            1,
            [{"role": "system", "content": "sys"}, {"content": "msg0"}],
        ),
        # edge: messages with non-string role
        (
            [{"role": None, "content": "msg0"}, {"role": "system", "content": "sys"}],
            1,
            [{"role": "system", "content": "sys"}, {"role": None, "content": "msg0"}],
        ),
    ],
    ids=[
        "system_and_many_users",
        "system_and_few_users",
        "no_system_many_users",
        "no_system_few_users",
        "only_system",
        "empty_messages",
        "none_messages",
        "keep_zero_with_system",
        "keep_zero_no_system",
        "multiple_systems_only_first_kept",
        "missing_role_key",
        "non_string_role",
    ],
)
def test_truncate_history(messages, keep, expected):
    # Act
    result = _truncate_history(messages, keep)

    # Assert
    assert result == expected


# Dummy provider classes for patching
class DummyAnthropic:
    def __init__(self, system_prompt, temperature, top_p):
        self.args = (system_prompt, temperature, top_p)


class DummyOllama:
    def __init__(self, system_prompt, temperature, top_p):
        self.args = (system_prompt, temperature, top_p)


class DummyOpenAI:
    def __init__(self, system_prompt, temperature, top_p):
        self.args = (system_prompt, temperature, top_p)


@pytest.fixture(autouse=True)
def patch_providers(monkeypatch):

    # Patch the provider classes in the module under test
    module = sys.modules["/Users/terry-brooks./Github/zendesk-roleplay.providers"]
    monkeypatch.setattr(module, "AnthropicProvider", DummyAnthropic)
    monkeypatch.setattr(module, "OllamaProvider", DummyOllama)
    monkeypatch.setattr(module, "OpenAIProvider", DummyOpenAI)
    # Patch MODEL_PROVIDER default
    monkeypatch.setattr(module, "MODEL_PROVIDER", "openai")
    yield


@pytest.mark.parametrize(
    "env_provider, expected_class, test_id",
    [
        ("anthropic", DummyAnthropic, "anthropic_env"),
        ("ollama", DummyOllama, "ollama_env"),
        ("openai", DummyOpenAI, "openai_env"),
        ("", DummyOpenAI, "empty_env_defaults_to_openai"),
        (None, DummyOpenAI, "none_env_defaults_to_openai"),
        ("UNKNOWN", DummyOpenAI, "unknown_env_defaults_to_openai"),
    ],
    ids=[
        "anthropic_env",
        "ollama_env",
        "openai_env",
        "empty_env_defaults_to_openai",
        "none_env_defaults_to_openai",
        "unknown_env_defaults_to_openai",
    ],
)
def test_make_provider(monkeypatch, env_provider, expected_class, test_id):
    # Arrange
    if env_provider is not None:
        monkeypatch.setenv("MODEL_PROVIDER", env_provider)
    else:
        monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    system_prompt = "sys"
    temperature = 0.5
    top_p = 0.9

    # Act
    provider = make_provider(system_prompt, temperature, top_p)

    # Assert
    assert isinstance(provider, expected_class)
    assert provider.args == (system_prompt, temperature, top_p)
