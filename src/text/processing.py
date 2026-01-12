from __future__ import annotations

from typing import Dict, List


def extract_text(payload: Dict[str, object]) -> str:
    utterances = payload.get("utterances") or []
    parts: List[str] = []
    for item in utterances:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if text:
            parts.append(str(text))
    return " ".join(parts).strip()
