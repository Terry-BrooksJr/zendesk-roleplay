import os, httpx
from typing import List, Dict

class LLMProvider:
    def __init__(self, system_prompt:str, temperature:float, top_p:float):
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
    async def chat(self, messages:List[Dict[str,str]])->str:
        raise NotImplementedError

class OpenAIProvider(LLMProvider):
    async def chat(self, messages):
        # minimal, uses Chat Completions
        import os
        import httpx
        api_key = os.environ["OPENAI_API_KEY"]
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
              "https://api.openai.com/v1/chat/completions",
              headers={"Authorization": f"Bearer {api_key}"},
              json={
                "model": "gpt-4o-mini",
                "temperature": self.temperature,
                "top_p": self.top_p,
                "messages": [{"role":"system","content": self.system_prompt}] + messages
              })
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

class AnthropicProvider(LLMProvider):
    async def chat(self, messages):
        import os
        import httpx
        api_key = os.environ["ANTHROPIC_API_KEY"]
        user_msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
              "https://api.anthropic.com/v1/messages",
              headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
              },
              json={
                "model":"claude-3-5-sonnet-latest",
                "system": self.system_prompt,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "messages": user_msgs
              })
        r.raise_for_status()
        return r.json()["content"][0]["text"]

class OllamaProvider(LLMProvider):
    async def chat(self, messages):
        base = os.environ.get("OLLAMA_BASE_URL","http://localhost:11434")
        model = "llama3.2"
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(f"{base}/api/chat",
              json={"model": model,
                    "options": {"temperature": self.temperature, "top_p": self.top_p},
                    "messages": messages})
        r.raise_for_status()
        return r.json()["message"]["content"]

def make_provider(system_prompt:str, temperature:float, top_p:float):
    p = os.getenv("MODEL_PROVIDER","openai").lower()
    if p == "anthropic":
        return AnthropicProvider(system_prompt, temperature, top_p)
    if p == "ollama":
        return OllamaProvider(system_prompt, temperature, top_p)
    return OpenAIProvider(system_prompt, temperature, top_p)