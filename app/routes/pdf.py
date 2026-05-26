from __future__ import annotations

import os
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import settings
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.pdf_service import PdfService
from app.services.vector_store_service import VectorStoreService

from app.utils.errors import http_400, http_500

router = APIRouter(tags=["pdf"])


pdf_service = PdfService()

chunking_service = ChunkingService()
embedding_service = EmbeddingService(settings.embedding_model_name)
vstore = VectorStoreService(settings.faiss_dir)


@router.post("/upload-pdf", response_model=dict)
async def upload_pdf(file: UploadFile = File(...)) -> dict:

    if not file.filename.lower().endswith(".pdf"):
        raise http_400("Only .pdf files are supported")

    os.makedirs(settings.uploads_dir, exist_ok=True)
    document_id = str(uuid.uuid4())
    saved_path = os.path.join(settings.uploads_dir, f"{document_id}_{file.filename}")

    try:
        contents = await file.read()
        if not contents:
            raise http_400("Empty file")
        with open(saved_path, "wb") as f:
            f.write(contents)

        text = pdf_service.extract_text(saved_path, parser="pymupdf")
        chunks = chunking_service.chunk_text(
            text,
            chunk_size_chars=settings.chunk_size_chars,
            chunk_overlap_chars=settings.chunk_overlap_chars,
        )
        if not chunks:
            raise http_400("No extractable text found in PDF")

        embeddings = embedding_service.embed_texts(chunks)
        vstore.create_or_update_index(document_id=document_id, embeddings=embeddings, chunks=chunks)

        return {
            "document_id": document_id,
            "filename": file.filename,
            "chunks_indexed": len(chunks),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise http_500(f"Failed to index PDF: {e}")

