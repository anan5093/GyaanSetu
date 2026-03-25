import json
import os
import re


class DatasetChunker:

    def __init__(
        self,
        class_id: int,
        subject: str,
        base_data_dir: str = "data"
    ):

        self.class_id = class_id
        self.subject = subject.lower()

        self.input_path = os.path.join(
            base_data_dir,
            "cleaned_dataset",
            f"class{class_id}",
            "cleaned_rows.json"
        )

        self.output_path = os.path.join(
            base_data_dir,
            "processed_chunks",
            f"class{class_id}",
            f"ncert_{self.subject}_chunks.json"
        )

    # =====================================
    # Formatting Helpers
    # =====================================

    def _format_concept(self, topic, explanation):

        return (
            "=== CONCEPT ===\n"
            f"Topic: {topic}\n\n"
            "Explanation:\n"
            f"{explanation.strip()}"
        )

    def _format_qa(self, topic, difficulty, qtype, question, answer):

        return (
            "=== EXAM QUESTION ===\n"
            f"Topic: {topic}\n"
            f"Difficulty: {difficulty}\n"
            f"Type: {qtype}\n\n"
            "Question:\n"
            f"{question.strip()}\n\n"
            "Answer:\n"
            f"{answer.strip()}"
        )

    # =====================================
    # Chapter Inference (Lightweight)
    # =====================================

    def _infer_chapter(self, topic):

        if not topic:
            return "Unknown Chapter"

        return topic.strip()

    # =====================================
    # Main Builder
    # =====================================

    def build_chunks(self):

        print(f"📦 Building tutor chunks → Class {self.class_id} {self.subject}")

        with open(self.input_path, "r", encoding="utf-8") as f:
            rows = json.load(f)

        all_chunks = []
        chunk_id = 0

        for row in rows:

            text = row["text"]
            topic = row.get("topic")
            difficulty = row.get("difficulty")
            qtype = row.get("question_type")
            complexity = row.get("complexity")

            chapter = self._infer_chapter(topic)

            # ============================
            # Extract Explanation / QA
            # ============================

            explanation_match = re.search(
                r"Explanation:(.*?)Question:",
                text,
                re.DOTALL
            )

            question_match = re.search(
                r"Question:(.*?)Answer:",
                text,
                re.DOTALL
            )

            answer_match = re.search(
                r"Answer:(.*)",
                text,
                re.DOTALL
            )

            explanation = (
                explanation_match.group(1).strip()
                if explanation_match else ""
            )

            question = (
                question_match.group(1).strip()
                if question_match else ""
            )

            answer = (
                answer_match.group(1).strip()
                if answer_match else ""
            )

            # ============================
            # Concept Chunk
            # ============================

            concept_text = self._format_concept(topic, explanation)

            all_chunks.append({
                "chunk_id": chunk_id,
                "text": concept_text,
                "type": "concept",
                "topic": topic,
                "class": self.class_id,
                "subject": self.subject,
                "chapter": chapter,
                "difficulty": difficulty,
                "question_type": qtype,
                "complexity": complexity
            })

            chunk_id += 1

            # ============================
            # QA Chunk
            # ============================

            if question and answer:

                qa_text = self._format_qa(
                    topic,
                    difficulty,
                    qtype,
                    question,
                    answer
                )

                all_chunks.append({
                    "chunk_id": chunk_id,
                    "text": qa_text,
                    "type": "qa",
                    "topic": topic,
                    "class": self.class_id,
                    "subject": self.subject,
                    "chapter": chapter,
                    "difficulty": difficulty,
                    "question_type": qtype,
                    "complexity": complexity
                })

                chunk_id += 1

        print(f"✅ Total tutor chunks created: {len(all_chunks)}")

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        print(f"💾 Saved at {self.output_path}")

        return all_chunks
