class ChunkingService:
    def chunk_text(self, text: str, *, chunk_size_chars: int, chunk_overlap_chars: int) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size_chars)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(text):
                break
            start = max(0, end - chunk_overlap_chars)
        return chunks

