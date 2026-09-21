import os

from openai import OpenAI

from app.llm.base import LLMProvider


class APILLMProvider(LLMProvider):

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ):

        self.api_key = (
            api_key
            or os.getenv("LLM_API_KEY")
        )

        self.base_url = (
            base_url
            or os.getenv("LLM_BASE_URL")
        )

        self.model = (
            model
            or os.getenv("LLM_MODEL")
        )

        self.max_tokens = max_tokens
        self.temperature = temperature

        if not self.api_key:
            raise ValueError(
                "LLM_API_KEY is not configured."
            )

        if not self.base_url:
            raise ValueError(
                "LLM_BASE_URL is not configured."
            )

        if not self.model:
            raise ValueError(
                "LLM_MODEL is not configured."
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def generate(self, prompt):

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=1024,
        )

        choice = response.choices[0]

        print("========== LLM RESPONSE ==========")
        print(response)
        print("==================================")

        if choice.message.content:
            return choice.message.content.strip()

        if choice.message.reasoning:
            raise RuntimeError(
                "The model exhausted the completion limit while reasoning "
                "and did not generate a final answer. "
                f"finish_reason={choice.finish_reason}"
            )

        raise RuntimeError(
            f"LLM returned no content. "
            f"finish_reason={choice.finish_reason}"
        )


