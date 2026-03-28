import faiss
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer  # ✅ added


class FAISSLoader:
    def __init__(self, class_id, subject, base_dir="data"):

        self.index_path = os.path.join(
            base_dir,
            "vector_store",
            f"class{class_id}",
            f"{subject}_faiss.index"
        )

        self.meta_path = os.path.join(
            base_dir,
            "vector_store",
            f"class{class_id}",
            f"{subject}_meta.json"
        )

        print("📦 Loading FAISS index...")
        self.index = faiss.read_index(self.index_path)

        print("📦 Loading metadata...")
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        # ✅ NEW: lightweight embedder
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def search(self, query, k=5):

        # ✅ NEW: handle both string + vector
        if isinstance(query, str):
            qvec = self.embedder.encode([query])[0]
        else:
            qvec = query

        qvec = np.array([qvec]).astype("float32")

        scores, ids = self.index.search(qvec, k)

        results = []

        for score, idx in zip(scores[0], ids[0]):

            if idx == -1:
                continue

            chunk = self.meta[idx].copy()
            chunk["score"] = float(score)

            results.append(chunk)

        return results
