from __future__ import annotations

import json
import os
from typing import Dict, List


class ChatMemoryService:
    def __init__(self, memory_dir: str, max_turns: int):
        self.memory_dir = memory_dir
        self.max_turns = max_turns
        os.makedirs(self.memory_dir, exist_ok=True)

    def _path(self, session_id: str) -> str:
        return os.path.join(self.memory_dir, f"{session_id}.json")

    def load(self, session_id: str) -> List[Dict[str, str]]:
        path = self._path(session_id)
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def append_turn(self, *, session_id: str, question: str, answer: str) -> List[Dict[str, str]]:
        history = self.load(session_id)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})

        # Keep last N turns (each turn = user+assistant => 2 messages)
        max_messages = self.max_turns * 2
        history = history[-max_messages:]

        with open(self._path(session_id), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False)
        return history

