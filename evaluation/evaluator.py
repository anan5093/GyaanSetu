import json

from evaluation.retrieval_eval import RetrievalEvaluator
from evaluation.generation_eval import GenerationEvaluator
from evaluation.latency_eval import LatencyEvaluator
from evaluation.embedding_eval import EmbeddingEvaluator


class Evaluator:

    def __init__(self, rag_service):
        self.rag = rag_service

    def load_data(self):
        with open("evaluation/test_data.json") as f:
            return json.load(f)

    def run_all(self):

        data = self.load_data()

        retrieval = RetrievalEvaluator(self.rag).evaluate(data)
        generation = GenerationEvaluator().evaluate(self.rag, data)
        latency = LatencyEvaluator().evaluate(self.rag, data)

        embedding = EmbeddingEvaluator(self.rag.embedder).evaluate(
            ["force", "velocity", "acceleration", "mass"]
        )

        return {
            "retrieval": retrieval,
            "generation": generation,
            "latency": latency,
            "embedding": embedding
        }
