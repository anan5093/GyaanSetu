class MockLLM:

    def generate(self, prompt):

        print("\n🧪 MOCK LLM ACTIVE\n")

        return "This is a mock answer. RAG pipeline working."