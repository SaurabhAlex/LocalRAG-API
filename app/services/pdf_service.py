from __future__ import annotations

from typing import Literal


class PdfService:
    def extract_text(self, file_path: str, parser: Literal["pymupdf", "pdfplumber"] = "pymupdf") -> str:
        if parser == "pdfplumber":
            return self._extract_pdfplumber(file_path)
        return self._extract_pymupdf(file_path)

    def _extract_pymupdf(self, file_path: str) -> str:
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        parts: list[str] = []
        for page in doc:
            parts.append(page.get_text("text"))
        return "\n".join(parts)

    def _extract_pdfplumber(self, file_path: str) -> str:
        import pdfplumber

        parts: list[str] = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                parts.append(page.extract_text() or "")
        return "\n".join(parts)

