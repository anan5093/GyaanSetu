from config.settings import settings
from llm.mock_llm import MockLLM
from llm.openrouter_client import OpenRouterClient


class LLMFactory:

    @staticmethod
    def create():

        # 🧪 MOCK MODE (SAFE DEFAULT)
        if settings.USE_MOCK_LLM:
            if settings.ENABLE_LOGS:
                print("⚙️ LLM Mode: MOCK")
            return MockLLM()

        # 🚨 SAFETY: If API key missing → fallback
        if not settings.OPENROUTER_API_KEY:
            print("⚠️ No OpenRouter API key found. Falling back to MOCK.")
            return MockLLM()

        # 🌐 REAL LLM MODE
        try:
            if settings.ENABLE_LOGS:
                print("⚙️ LLM Mode: OPENROUTER")

            return OpenRouterClient(
                model=settings.OPENROUTER_MODEL,
                max_tokens=settings.MAX_TOKENS,
                api_key=settings.OPENROUTER_API_KEY
            )

        except Exception as e:
            # 🔥 FAILSAFE: never break system
            print(f"⚠️ LLM init failed: {e}")
            print("🔁 Falling back to MOCK LLM")

            return MockLLM()
