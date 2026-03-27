import requests
from config.settings import settings


class OpenRouterClient:

    def __init__(self, model=None, max_tokens=None, api_key=None):
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        self.model = model or settings.OPENROUTER_MODEL
        self.max_tokens = max_tokens or settings.MAX_TOKENS

        if not self.api_key:
            raise ValueError("❌ OPENROUTER_API_KEY is missing in .env")

    def generate(self, prompt):

        try:
            res = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": 0.3
                },
                timeout=30
            )

            if res.status_code != 200:
                raise Exception(f"OpenRouter API Error: {res.text}")

            data = res.json()

            if "choices" not in data:
                raise Exception(f"Invalid response: {data}")

            return data["choices"][0]["message"]["content"]

        except requests.exceptions.Timeout:
            return "⚠️ Request timed out. Please try again."

        except Exception as e:
            if settings.ENABLE_LOGS:
                print("❌ LLM Error:", e)
            return "⚠️ AI service error. Please try again later."
