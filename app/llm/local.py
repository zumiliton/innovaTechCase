import requests

from app.llm.base import LLMProvider


class LocalLLMProvider(LLMProvider):

    def __init__(
        self,
        base_url: str = "http://llama:8080",
        max_tokens: int = 512,
        temperature: float = 0.2,
    ):
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "chat_template_kwargs": {
                "enable_thinking": False,
                },
            },
            timeout=300,
        )

        response.raise_for_status()

        data = response.json()

        # print("\n=== RAW LLM RESPONSE ===")
        # print(data)

        return data["choices"][0]["message"]["content"].strip()