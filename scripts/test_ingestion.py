from ingestion.dataset_chunker import DatasetChunker

chunker = DatasetChunker(
    class_id=10,
    subject="science"
)

chunks = chunker.build_chunks()

print("\n===== TOTAL CHUNKS =====")
print(len(chunks))

print("\n===== FIRST CHUNK TYPE =====")
print(chunks[0]["type"])

print("\n===== FIRST CHUNK TEXT =====")
print(chunks[0]["text"])

print("\n============================")

print("\n===== SECOND CHUNK TYPE =====")
print(chunks[1]["type"])

print("\n===== SECOND CHUNK TEXT =====")
print(chunks[1]["text"])
