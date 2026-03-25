from config.settings import USE_MOCK_LLM
from llm.mock_llm import MockLLM
from llm.openrouter_client import OpenRouterClient


class LLMFactory:

    @staticmethod
    def create():

        if USE_MOCK_LLM:
            return MockLLM()

        return OpenRouterClient()