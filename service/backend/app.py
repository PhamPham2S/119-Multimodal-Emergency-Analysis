from __future__ import annotations

import os
import random
import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.data_pipeline import build_records
from service.backend.infer import InferenceEngine, LabelOrders, load_label_orders

DATA_ROOT = REPO_ROOT / "data" / "Sample"
MODEL_DIR = REPO_ROOT / "models"
FRONTEND_DIR = REPO_ROOT / "service" / "frontend"

AUDIO_MODEL = os.getenv("AUDIO_MODEL", "facebook/wav2vec2-base-960h")
TEXT_MODEL = os.getenv("TEXT_MODEL", "beomi/KcELECTRA-base")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
MAX_TEXT_LEN = int(os.getenv("MAX_TEXT_LEN", "256"))

app = FastAPI(title="Audio Text Inference Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/samples", StaticFiles(directory=str(DATA_ROOT)), name="samples")

_records = build_records(DATA_ROOT)
if not _records:
    raise RuntimeError("No records found under data/Sample")
_record_by_rel = {
    record.audio_path.relative_to(DATA_ROOT).as_posix(): record for record in _records
}

_label_orders = load_label_orders(MODEL_DIR, DATA_ROOT)
_engine = InferenceEngine(
    audio_model=AUDIO_MODEL,
    text_model=TEXT_MODEL,
    model_dir=MODEL_DIR,
    label_orders=_label_orders,
    sample_rate=SAMPLE_RATE,
    max_text_len=MAX_TEXT_LEN,
)


class InferRequest(BaseModel):
    audio_path: str
    text: str


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/samples/random")
def random_sample() -> dict[str, object]:
    record = random.choice(_records)
    rel = record.audio_path.relative_to(DATA_ROOT).as_posix()
    return {
        "audio_url": f"/samples/{rel}",
        "audio_path": rel,
        "text": record.text,
        "gold": {"urgency": record.urgency, "sentiment": record.sentiment},
    }


def _resolve_audio_path(relative_path: str) -> Path:
    candidate = (DATA_ROOT / relative_path).resolve()
    if DATA_ROOT not in candidate.parents:
        raise HTTPException(status_code=400, detail="Invalid audio path")
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return candidate


def _format_result(result, gold: Optional[dict[str, str]] = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "urgency": {
            "label": result.urgency_label,
            "logits": result.urgency_logits,
            "probs": result.urgency_probs,
        },
        "sentiment": {
            "label": result.sentiment_label,
            "logits": result.sentiment_logits,
            "probs": result.sentiment_probs,
        },
    }
    if gold:
        payload["gold"] = gold
    return payload


@app.post("/api/infer")
def infer(request: InferRequest) -> dict[str, object]:
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    audio_path = _resolve_audio_path(request.audio_path)
    result = _engine.predict_from_path(audio_path, request.text)
    rel = audio_path.relative_to(DATA_ROOT).as_posix()
    record = _record_by_rel.get(rel)
    gold = {"urgency": record.urgency, "sentiment": record.sentiment} if record else None
    return _format_result(result, gold=gold)


@app.post("/api/infer-upload")
async def infer_upload(
    audio_file: UploadFile = File(...),
    text: str = Form(""),
) -> dict[str, object]:
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    audio_bytes = await audio_file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file is required")
    result = _engine.predict_from_bytes(audio_bytes, text)
    return _format_result(result)


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
