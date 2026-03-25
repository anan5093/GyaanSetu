class RAGService:

    def __init__(self, vector_store, embedder, llm_client, top_k=3):

        self.vector_store = vector_store
        self.embedder = embedder
        self.llm_client = llm_client
        self.top_k = top_k

    def build_prompt(self, context, question):

        return f"""
You are NCERT Tutor.

Use ONLY the context to answer.

Context:
{context}

Question:
{question}

Answer:
"""

    def ask_with_debug(self, question):

        # 1 embed query
        qvec = self.embedder.embed([question])[0]

        # 2 retrieve chunks
        chunks = self.vector_store.search(qvec, k=self.top_k)

        # 3 build context
        context = "\n\n".join([c["text"] for c in chunks])

        # 4 build prompt
        prompt = self.build_prompt(context, question)

        # 5 call llm
        answer = self.llm_client.generate(prompt)

        return answer, chunks
