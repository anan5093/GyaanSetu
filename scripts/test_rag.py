import os
import sys
from dotenv import load_dotenv


# ===============================
# 🔧 Setup Project Root Path
# ===============================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# ===============================
# 🔐 Load Environment Variables
# ===============================
load_dotenv()


# ===============================
# 📦 Imports (After Path Fix)
# ===============================
from rag.rag_service import RAGService
from vector_store.faiss_loader import FAISSLoader
from llm.llm_client import LLMClient
from ingestion.embedder import Embedder


# ===============================
# ⚙️ Config (Centralised)
# ===============================
CLASS_ID = 10
SUBJECT = "science"
TOP_K = 5
DEBUG_MODE = True


# ===============================
# 🚀 Bootstrap Tutor
# ===============================
def build_tutor():

    print("\n🚀 Initialising NCERT RAG Tutor...\n")

    embedder = Embedder()

    vector_store = FAISSLoader(
        class_id=CLASS_ID,
        subject=SUBJECT
    )

    llm_client = LLMClient()

    rag = RAGService(
        vector_store=vector_store,
        embedder=embedder,
        llm_client=llm_client,
        top_k=TOP_K
    )

    print("✅ Tutor Ready!\n")

    return rag


# ===============================
# 💬 Interactive CLI Loop
# ===============================
def chat_loop(rag):

    print("💡 Ask NCERT Questions (type 'exit' to quit)\n")

    while True:

        try:
            question = input("🧠 You: ").strip()

            if not question:
                continue

            if question.lower() in ["exit", "quit", "q"]:
                print("\n👋 Exiting Tutor...\n")
                break

            answer, chunks = rag.ask_with_debug(question)

            if DEBUG_MODE:
                print("\n📚 Retrieved Chunks:\n")
                for c in chunks:
                    print(f"— {c.get('topic')} | {c.get('type')}")

            print("\n🤖 Tutor Answer:\n")
            print(answer)

            print("\n" + "=" * 70 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!\n")
            break

        except Exception as e:
            print("\n❌ Error:", str(e))
            print("Continuing...\n")


# ===============================
# 🎯 Entry Point
# ===============================
def main():

    rag = build_tutor()
    chat_loop(rag)


if __name__ == "__main__":
    main()
