import json
import os


class DatasetChunker:

    def __init__(
        self,
        input_path="data/cleaned_dataset/cleaned_rows.json",
        output_path="data/processed_chunks/chunks.json"
    ):
        self.input_path = input_path
        self.output_path = output_path

    def _format_concept_text(self, explanation_text, row):
        """
        Convert flat explanation into structured study-card format
        """

        explanation_clean = (
          explanation_text
            .replace(f"Topic: {row['topic']}.", "")
            .replace("Topic:", "")
            .replace("Explanation:", "")
            .strip()
         )

        formatted = (
            "=== CONCEPT ===\n"
            f"Topic: {row['topic']}\n\n"
            "Explanation:\n"
            f"{explanation_clean}"
        )

        return formatted

    def _format_qa_text(self, qa_text, row):
        """
        Convert flat QA text into structured exam-card format
        """

        qa_body = qa_text.replace("Question:", "").strip()

        if "Answer:" in qa_body:
            question_part, answer_part = qa_body.split("Answer:", 1)
        else:
            question_part, answer_part = qa_body, ""

        formatted = (
            "=== EXAM QUESTION ===\n"
            f"Topic: {row['topic']}\n"
            f"Difficulty: {row['difficulty']}\n"
            f"Type: {row['question_type']}\n\n"
            "Question:\n"
            f"{question_part.strip()}\n\n"
            "Answer:\n"
            f"{answer_part.strip()}"
        )

        return formatted

    def build_chunks(self):

        print("📖 Loading cleaned dataset...")

        with open(self.input_path, "r", encoding="utf-8") as f:
            rows = json.load(f)

        all_chunks = []
        chunk_id = 0

        for row in rows:

            full_text = row["text"]

            # 🔵 Split concept and QA
            if "Question:" in full_text:
                explanation_part = full_text.split("Question:")[0].strip()
                qa_part = "Question:" + full_text.split("Question:", 1)[1].strip()
            else:
                explanation_part = full_text
                qa_part = None

            # 🔵 Concept Chunk
            concept_text = self._format_concept_text(explanation_part, row)

            all_chunks.append({
                "chunk_id": chunk_id,
                "text": concept_text,
                "type": "concept",
                "topic": row["topic"],
                "difficulty": row["difficulty"],
                "question_type": row["question_type"],
                "complexity": row["complexity"]
            })

            chunk_id += 1

            # 🔵 QA Chunk
            if qa_part:
                qa_text = self._format_qa_text(qa_part, row)

                all_chunks.append({
                    "chunk_id": chunk_id,
                    "text": qa_text,
                    "type": "qa",
                    "topic": row["topic"],
                    "difficulty": row["difficulty"],
                    "question_type": row["question_type"],
                    "complexity": row["complexity"]
                })

                chunk_id += 1

        print(f"✅ Total chunks created: {len(all_chunks)}")

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        print(f"💾 Saved chunks at {self.output_path}")

        return all_chunks
