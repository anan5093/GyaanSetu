class PromptBuilder:

    def build(self, chunks, question):

        context_blocks = []

        for c in chunks:
            topic = c.get("topic", "Unknown Topic")
            text = c.get("text", "")

            block = f"""
[Topic: {topic}]
{text}
"""
            context_blocks.append(block.strip())

        context = "\n\n".join(context_blocks)

        return f"""
You are an NCERT academic tutor.

Answer ONLY using the NCERT context below.

If answer is not present → say:
"Answer not found in NCERT content."

NCERT Context:
{context}

Student Question:
{question}

Grounded Answer:
"""