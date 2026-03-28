class RAGService:
    def __init__(
        self,
        vector_store,
        llm_client,
        prompt_builder,
        top_k=3,
        score_threshold=0.35
    ):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
        self.top_k = top_k
        self.score_threshold = score_threshold

    # ===============================
    # 🔍 Retrieval Layer
    # ===============================
    def retrieve(self, question):

        # ✅ Direct search (no embedder)
        results = self.vector_store.search(
            question,
            k=self.top_k
        ) or []

        # ✅ Threshold Filtering
        filtered = [
            c for c in results
            if isinstance(c, dict) and c.get("score", 0) >= self.score_threshold
        ]

        return filtered

    # ===============================
    # 🧠 Context Builder
    # ===============================
    def build_context(self, chunks):
        unique_texts = []
        seen = set()

        for c in chunks:
            text = ""
            if isinstance(c, dict):
                text = c.get("text", "")
            elif isinstance(c, str):
                text = c

            if text and text not in seen:
                unique_texts.append(text)
                seen.add(text)

        return "\n\n".join(unique_texts)

    # ===============================
    # 🤖 Core RAG Pipeline
    # ===============================
    def process_query(self, question):
        chunks = self.retrieve(question)

        if not chunks:
            return {
                "answer": "Answer not found in NCERT content.",
                "chunks": []
            }

        context = self.build_context(chunks)

        if not context.strip():
            return {
                "answer": "Answer not found in NCERT content.",
                "chunks": chunks
            }

        prompt = self.prompt_builder.build(
            context,
            question
        )

        answer = self.llm_client.generate(prompt)

        return {
            "answer": answer,
            "chunks": chunks
        }

    # ===============================
    # 🧾 Legacy Method
    # ===============================
    def ask(self, question):
        result = self.process_query(question)
        return result["answer"]

    # ===============================
    # 🧪 Debug Mode
    # ===============================
    def ask_with_debug(self, question):
        result = self.process_query(question)
        return result["answer"], result["chunks"]
