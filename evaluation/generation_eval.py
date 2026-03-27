from sentence_transformers import SentenceTransformer
import numpy as np


class GenerationEvaluator:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def semantic_similarity(self, a, b):
        emb = self.model.encode([a, b])
        return float(np.dot(emb[0], emb[1]))

    def evaluate(self, rag_service, test_data):

        results = []

        for item in test_data:
            q = item.get("question", "")
            gt = item.get("expected_answer", "")

            try:
                # ✅ Correct way to extract answer
                result = rag_service.process_query(q)
                pred = result.get("answer", "")

            except Exception as e:
                print(f"❌ Error in generation eval for question: {q}")
                print("Error:", str(e))
                pred = "ERROR"

            # ✅ Avoid crash if empty
            if not pred or not gt:
                score = 0.0
            else:
                score = self.semantic_similarity(pred, gt)

            results.append({
                "question": q,
                "score": float(score),
                "prediction": pred,
                "ground_truth": gt
            })

        # ✅ Safe average
        if results:
            avg_score = np.mean([r["score"] for r in results])
        else:
            avg_score = 0.0

        return {
            "avg_similarity": float(avg_score),
            "details": results
        }
