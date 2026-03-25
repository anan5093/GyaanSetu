from ingestion.dataset_chunker import DatasetChunker

chunker = DatasetChunker()

chunks = chunker.build_chunks()

print("\nTOTAL CHUNKS:", len(chunks))

print("\nFirst chunk type:", chunks[0]["type"])
print("Second chunk type:", chunks[1]["type"])

print("\n--- SAMPLE CHUNK TEXT ---\n")

print(chunks[0]["text"])
print("\n" + "=" * 80 + "\n")
print(chunks[1]["text"])
