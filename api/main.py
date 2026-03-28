from dotenv import load_dotenv
load_dotenv()


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router, set_rag_instance

from vector_store.faiss_loader import FAISSLoader
from rag.rag_service import RAGService
from llm.llm_factory import LLMFactory
from llm.prompt_builder import PromptBuilder


app = FastAPI(
    title="GyaanSetu NCERT Tutor API",
    version="1.1"
)

# ✅ CORS (IMPORTANT for React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def startup_event():

    print("\n🚀 Starting NCERT Tutor Backend...")

    # ⭐ Vector Store
    store = FAISSLoader(
        class_id=10,
        subject="science"
    )

    # ⭐ LLM
    llm = LLMFactory.create()

    # ⭐ Prompt Builder
    prompt_builder = PromptBuilder()

    # ⭐ RAG Service (embedder removed)
    rag = RAGService(
        store,
        llm,
        prompt_builder
    )

    # ✅ Inject into routes
    set_rag_instance(rag)

    print("✅ Tutor Backend Ready with RAG + Metrics + Sources!\n")
