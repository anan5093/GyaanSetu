from fastapi import APIRouter, HTTPException
from api.schemas import ChatRequest
import time

router = APIRouter()

# ⭐ Global RAG instance
rag_instance = None


def set_rag_instance(rag):
    global rag_instance
    rag_instance = rag


@router.post("/chat")
def chat_endpoint(req: ChatRequest):

    if rag_instance is None:
        raise HTTPException(status_code=500, detail="RAG not initialized")

    try:
        start_time = time.time()

        # 🔍 RAG processing
        result = rag_instance.process_query(req.question)

        answer = result["answer"]
        chunks = result["chunks"]

        latency = time.time() - start_time

        # 📚 Build sources
        sources = [
            {
                "topic": c.get("topic"),
                "chapter": c.get("chapter"),
                "type": c.get("type"),
            }
            for c in chunks[:3]
        ]

        return {
            "answer": answer,
            "sources": sources,
            "metrics": {
                "latency": round(latency, 2),
                "chunks_used": len(chunks),
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
