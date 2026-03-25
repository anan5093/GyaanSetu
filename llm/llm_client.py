import requests
import os


class LLMClient:

    def __init__(self):

        self.api_key = os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError("❗ OPENROUTER_API_KEY not set")

        self.model = "nvidia/nemotron-3-super-120b-a12b:freeze-2024-08-01"

    def generate(self, prompt):

        response = requests.post(
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
                "temperature": 0.3
                "max_tokens": 300
            }
        )

        try:
            data = response.json()

            if "choices" not in data:
                print("\n🚨 OpenRouter Raw Response:\n", data)
                raise Exception("LLM response missing 'choices'")

            return data["choices"][0]["message"]["content"]

        except Exception as e:
            print("\n❌ LLM Parsing Error:", str(e))
            raise
