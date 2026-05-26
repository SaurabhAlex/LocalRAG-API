from __future__ import annotations

from typing import List, Dict, Optional

import httpx


class OllamaService:
    def __init__(self, *, base_url: str, model: str, temperature: float, num_ctx: int):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx

    def build_prompt(
        self,
        *,
        question: str,
        context_chunks: List[str],
        conversation_history: List[Dict[str, str]],
    ) -> str:
        context = "\n\n".join(
            [f"[Context {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)]
        )

        history_text = "\n".join(
            [f"{m['role'].capitalize()}: {m['content']}" for m in conversation_history]
        )

        return (
            "You are a helpful assistant for answering questions based strictly on the provided context.\n"
            "If the answer is not contained in the context, say you don't know.\n\n"
            f"Conversation history:\n{history_text or '(none)'}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer (concise):"
        )

    async def generate_answer(self, *, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": self.temperature,
            "options": {"num_ctx": self.num_ctx},
        }

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        return (data.get("response") or "").strip()

