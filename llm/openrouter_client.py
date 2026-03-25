import requests
import os
from config.settings import OPENROUTER_MODEL, MAX_TOKENS


class OpenRouterClient:

    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")

    def generate(self, prompt):

        res = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": MAX_TOKENS,
                "temperature": 0.3
            }
        )

        data = res.json()

        if "choices" not in data:
            raise Exception(data)

        return data["choices"][0]["message"]["content"]
