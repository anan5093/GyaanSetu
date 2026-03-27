import numpy as np

class RetrievalEvaluator:

    def __init__(self, rag_service):
        self.rag = rag_service

    def evaluate(self, test_data):

        results = []

        for item in test_data:
            q = item["question"]
            expected_topic = item["expected_topic"]

            query_vec = self.rag.embedder.embed([q])[0]
            chunks = self.rag.vector_store.search(query_vec, k=3)

            retrieved_topics = [c["topic"] for c in chunks]

            hit = expected_topic in retrieved_topics

            results.append({
                "question": q,
                "expected_topic": expected_topic,
                "retrieved_topics": retrieved_topics,
                "hit@3": int(hit)
            })

        accuracy = np.mean([r["hit@3"] for r in results])

        return {
            "accuracy": float(accuracy),
            "details": results
        }
