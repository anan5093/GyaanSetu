import faiss
import json
import os
import numpy as np


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

    def search(self, qvec, k=5):

        qvec = np.array([qvec]).astype("float32")

        scores, ids = self.index.search(qvec, k)

        results = []

        for idx in ids[0]:
            if idx == -1:
                continue
            results.append(self.meta[idx])

        return results
