from __future__ import annotations

import uuid

from fastapi import APIRouter

from app.core.config import settings
from app.models.schemas import AskQuestionRequest, AskQuestionResponse, Citation
from app.services.chat_memory_service import ChatMemoryService
from app.services.embedding_service import EmbeddingService
from app.services.ollama_service import OllamaService
from app.services.vector_store_service import VectorStoreService

from app.utils.errors import http_400, http_404, http_500

router = APIRouter(prefix="/qa", tags=["qa"])

embedding_service = EmbeddingService(settings.embedding_model_name)
vstore = VectorStoreService(settings.faiss_dir)
mem = ChatMemoryService(settings.memory_dir, settings.memory_max_turns)
ollama = OllamaService(
    base_url=settings.ollama_base_url,
    model=settings.ollama_model,
    temperature=settings.ollama_temperature,
    num_ctx=settings.ollama_num_ctx,
)


@router.post("/ask-question", response_model=AskQuestionResponse)
async def ask_question(req: AskQuestionRequest) -> AskQuestionResponse:
    if not req.question.strip():
        raise http_400("question cannot be empty")

    if not vstore.index_exists(req.document_id):
        raise http_404("Unknown document_id (index not found)")

    session_id = req.session_id or str(uuid.uuid4())

    try:
        query_embedding = embedding_service.embed_query(req.question)
        search_result, retrieved_chunks = vstore.search(
            document_id=req.document_id,
            query_embedding=query_embedding,
            top_k=settings.top_k,
        )

        history = mem.load(session_id)
        prompt = ollama.build_prompt(
            question=req.question,
            context_chunks=retrieved_chunks,
            conversation_history=history,
        )
        answer = await ollama.generate_answer(prompt=prompt)

        updated_history = mem.append_turn(
            session_id=session_id,
            question=req.question,
            answer=answer,
        )

        citations = [
            Citation(chunk_index=i, text_preview=(retrieved_chunks[j][:200] + "..." if len(retrieved_chunks[j]) > 200 else retrieved_chunks[j]))
            for j, i in enumerate(search_result.chunk_indices)
        ]

        return AskQuestionResponse(
            document_id=req.document_id,
            session_id=session_id,
            answer=answer,
            citations=citations,
        )
    except Exception as e:
        raise http_500(f"Failed to answer question: {e}")

