import json
import os
import re


class DatasetCleaner:

    def __init__(
        self,
        class_id: int,
        subject: str,
        base_data_dir: str = "data",
        chunk_size: int = 320,
        overlap: int = 60
    ):

        self.class_id = class_id
        self.subject = subject.lower()

        self.input_path = os.path.join(
            base_data_dir,
            "raw_dataset",
            f"class{class_id}",
            f"ncert_{self.subject}{class_id}.json"
        )

        self.output_path = os.path.join(
            base_data_dir,
            "cleaned_dataset",
            f"class{class_id}",
            "cleaned_rows.json"
        )

        self.chunk_size = chunk_size
        self.overlap = overlap

    # ==============================
    # Sentence Aware Chunking
    # ==============================

    def chunk_text(self, text):

        if "Question:" in text:
            return [text.strip()]

        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = ""

        for sent in sentences:

            sent = sent.strip()
            if not sent:
                continue

            if len(sent) > self.chunk_size:

                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                start = 0
                while start < len(sent):
                    part = sent[start:start + self.chunk_size]
                    chunks.append(part.strip())
                    start += self.chunk_size - self.overlap

                continue

            if len(current_chunk) + len(sent) + 1 <= self.chunk_size:

                current_chunk = (
                    current_chunk + " " + sent
                    if current_chunk else sent
                )

            else:

                chunks.append(current_chunk.strip())

                overlap_text = (
                    current_chunk[-self.overlap:]
                    if len(current_chunk) > self.overlap
                    else current_chunk
                )

                current_chunk = overlap_text + " " + sent

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    # ==============================
    # Build Clean Dataset ⭐ IMPROVED
    # ==============================

    def build_clean_dataset(self):

        print(f"🧹 Cleaning dataset → Class {self.class_id} {self.subject}")

        with open(self.input_path, "r", encoding="utf-8") as f:
            rows = json.load(f)

        print("🔎 TOTAL ROWS:", len(rows))
        print("🔎 SAMPLE ROW:", rows[0])

        cleaned_rows = []

        for row in rows:

            # ⭐ Build unified text field from HF schema
            explanation = row.get("Explanation", "").strip()
            question = row.get("Question", "").strip()
            answer = row.get("Answer", "").strip()
            topic = row.get("Topic", "").strip()

            merged_text = (
                f"Topic: {topic}. "
                f"Explanation: {explanation} "
                f"Question: {question} "
                f"Answer: {answer}"
            ).strip()

            text_chunks = self.chunk_text(merged_text)

            for chunk in text_chunks:

                cleaned_rows.append({
                    "text": chunk,
                    "topic": topic,
                    "difficulty": row.get("Difficulty"),
                    "question_type": row.get("QuestionType"),
                    "complexity": row.get("QuestionComplexity")
                })

        print(f"✅ Cleaned rows created: {len(cleaned_rows)}")

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_rows, f, ensure_ascii=False, indent=2)

        print(f"💾 Saved at {self.output_path}")

        return cleaned_rows
