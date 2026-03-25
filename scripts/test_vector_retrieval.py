import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class TutorRetrieverTest:
    def __init__(
        self,
        class_id=10,
        subject="science",
        base_dir="data",
        top_k=5
    ):
        self.subject = subject.lower()
        self.top_k = top_k

        self.index_path = os.path.join(
            base_dir,
            "vector_store",
            f"class{class_id}",
            f"{self.subject}_faiss.index"
        )

        self.meta_path = os.path.join(
            base_dir,
            "vector_store",
            f"class{class_id}",
            f"{self.subject}_meta.json"
        )

        print("📦 Loading FAISS index...")
        self.index = faiss.read_index(self.index_path)

        print("📖 Loading metadata...")
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print("✅ Retriever ready.")

    def ask(self, question):
        print("\n🧠 QUESTION:", question)

        qvec = self.model.encode(
            [question],
            normalize_embeddings=True
        )

        scores, ids = self.index.search(
            np.array(qvec).astype('float32'),
            self.top_k
        )

        print("\n🔎 TOP RETRIEVED CHUNKS:\n")

        for rank, idx in enumerate(ids[0]):
            if idx == -1: continue
            chunk = self.chunks[idx]

            print("=" * 60)
            print(f"Rank: {rank+1}")
            print(f"Score: {scores[0][rank]:.4f}")
            print(f"Topic: {chunk['topic']}")
            print(f"Type: {chunk['type']}")
            print("\nTEXT:\n")
            print(chunk["text"][:500])
            print("=" * 60)

if __name__ == "__main__":
    retriever = TutorRetrieverTest()

    retriever.ask(
        "What is chemical reaction?"
    )

    retriever.ask(
        "Explain rusting of iron"
    )
