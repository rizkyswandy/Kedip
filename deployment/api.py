"""
FastAPI server that exposes the blink detector for the browser UI.

The endpoint accepts base64-encoded JPEG frames from the web UI, runs them
through the existing real-time detector, and returns blink metrics.
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from inference import RealtimeBlinkDetector
from models import BlinkTransformer, create_model


def _to_camel(string: str) -> str:
    parts = string.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)


class InferenceRequest(CamelModel):
    image: str
    session_id: str = Field(default="default", description="Logical session id for buffering")


class InferenceResponse(CamelModel):
    has_blink: bool
    confidence: float
    eye_state: str
    total_blinks: int
    fps: float
    face_detected: bool
    buffer_full: bool
    timestamp: float


class HealthResponse(CamelModel):
    status: str
    device: str
    model_path: str
    sequence_length: int
    threshold: float


def choose_device(preferred: str | None = None) -> str:
    """Pick the best available device."""
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    if preferred == "mps" and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_path: Path, device: str) -> BlinkTransformer:
    """Load a trained model checkpoint."""
    if not model_path.exists():
        raise RuntimeError(
            f"Model checkpoint not found at {model_path}. "
            "Update MODEL_PATH or place your trained checkpoint there."
        )

    checkpoint = torch.load(model_path, map_location=device)

    if "config" in checkpoint:
        model_config = checkpoint["config"]["model"]
    else:
        model_config = {
            "backbone": "mobilenetv3_small_100",
            "hidden_dim": 256,
            "num_heads": 8,
            "num_layers": 4,
            "dropout": 0.1,
            "use_cross_modal": True,
            "use_linear_attention": True,
        }

    model = create_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


DEVICE = choose_device(os.getenv("DEVICE"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", "checkpoints/best_model.pth"))
SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", "16"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

torch.set_grad_enabled(False)

model = load_model(MODEL_PATH, DEVICE)
detector = RealtimeBlinkDetector(
    model=model,
    device=DEVICE,
    sequence_length=SEQUENCE_LENGTH,
    threshold=THRESHOLD,
)
detector_lock = asyncio.Lock()


app = FastAPI(
    title="Blink Detector API",
    description="Local API used by the browser-based blink detection UI.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def decode_image(data_url: str) -> np.ndarray:
    """Decode a base64 data URL into an OpenCV frame."""
    try:
        encoded = data_url.split(",")[-1]
        image_bytes = base64.b64decode(encoded)
        frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail="Invalid image data") from exc

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    return frame


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Simple readiness probe."""
    return HealthResponse(
        status="ok",
        device=DEVICE,
        model_path=str(MODEL_PATH),
        sequence_length=SEQUENCE_LENGTH,
        threshold=THRESHOLD,
    )


@app.post("/api/reset")
async def reset_state() -> dict[str, str]:
    """Reset the detector state and counters."""
    async with detector_lock:
        detector.reset()
    return {"status": "reset"}


@app.post("/api/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest) -> InferenceResponse:
    """Run inference on a single frame."""
    frame = decode_image(request.image)

    async with detector_lock:
        result = detector.process_frame(frame)

    eye_state_value = result.get("eye_state", 0)
    eye_state = "CLOSED" if eye_state_value > 0.5 else "OPEN"

    return InferenceResponse(
        has_blink=bool(result.get("has_blink", False)),
        confidence=float(result.get("confidence", 0.0)),
        eye_state=eye_state,
        total_blinks=int(result.get("total_blinks", 0)),
        fps=float(result.get("fps", 0.0)),
        face_detected=bool(result.get("face_detected", False)),
        buffer_full=bool(result.get("buffer_full", True)),
        timestamp=time.time(),
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    detector.close()
