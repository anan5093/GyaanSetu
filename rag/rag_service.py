class RAGService:

    def __init__(
        self,
        vector_store,
        embedder,
        llm_client,
        prompt_builder,
        top_k=3
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
        self.top_k = top_k

    # ===============================
    # Retrieval Layer
    # ===============================
    def retrieve(self, question):

        qvec = self.embedder.embed([question])[0]

        chunks = self.vector_store.search(
            qvec,
            k=self.top_k
        )

        return chunks

    # ===============================
    # Answer Generation
    # ===============================
    def ask(self, question):

        chunks = self.retrieve(question)

        if not chunks:
            return "Answer not found in NCERT content."

        prompt = self.prompt_builder.build(
            chunks,
            question
        )

        answer = self.llm_client.generate(prompt)

        return answer

    # ===============================
    # Debug Mode
    # ===============================
    def ask_with_debug(self, question):

        chunks = self.retrieve(question)

        if not chunks:
            return "Answer not found in NCERT content.", []

        prompt = self.prompt_builder.build(
            chunks,
            question
        )

        answer = self.llm_client.generate(prompt)

        return answer, chunks