import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class TutorVectorIndexBuilder:
    def __init__(
        self,
        class_id: int,
        subject: str,
        base_dir: str = "data",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.class_id = class_id
        self.subject = subject.lower()

        self.index_dir = os.path.join(
            base_dir,
            "vector_store",
            f"class{class_id}"
        )

        self.chunk_path = os.path.join(
            base_dir,
            "processed_chunks",
            f"class{class_id}",
            f"ncert_{self.subject}_chunks.json"
        )

        self.index_file = os.path.join(self.index_dir, f"{self.subject}_faiss.index")
        self.meta_file = os.path.join(self.index_dir, f"{self.subject}_meta.json")
        self.model = SentenceTransformer(model_name)

    def load_chunks(self):
        print(f"📖 Loading tutor chunks → {self.chunk_path}")
        with open(self.chunk_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"✅ Total chunks loaded: {len(chunks)}")
        return chunks

    def embed_chunks(self, chunks):
        texts = [c["text"] for c in chunks]
        print("🧠 Generating embeddings...")
        vectors = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        print("✅ Embeddings shape:", vectors.shape)
        return vectors

    def build_index(self, vectors):
        dim = vectors.shape[1]
        print("⚡ Building FAISS index...")
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        print("✅ Total vectors indexed:", index.ntotal)
        return index

    def save_index(self, index, chunks):
        os.makedirs(self.index_dir, exist_ok=True)
        print("💾 Saving FAISS index...")
        faiss.write_index(index, self.index_file)
        print("💾 Saving metadata mapping...")
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print("✅ Index + metadata saved.")

    def run(self):
        chunks = self.load_chunks()
        vectors = self.embed_chunks(chunks)
        index = self.build_index(vectors)
        self.save_index(index, chunks)

if __name__ == "__main__":
    builder = TutorVectorIndexBuilder(
        class_id=10,
        subject="science"
    )
    builder.run()