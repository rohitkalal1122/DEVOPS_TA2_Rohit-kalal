
import json
import os
import time
import uuid
from pathlib import Path

import joblib
import numpy as np
import redis
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "artifacts"

# Redis configuration:
# Local development uses localhost.
# Docker Compose will override REDIS_URL with redis://redis:6379/0.
REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://localhost:6379/0",
)

app = FastAPI(
    title="Shadow Deployment Engine",
    description=(
        "Real-Time Risk Monitoring with "
        "Asynchronous Shadow Model Evaluation"
    ),
    version="1.0.0",
)

models = {}

client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=2,
    socket_timeout=5,
    health_check_interval=30,
)


# Load only the production model into the API.
@app.on_event("startup")
def load_models():
    model_path = MODEL_DIR / "v1.joblib"

    if not model_path.exists():
        raise RuntimeError(
            f"Production model not found: {model_path}"
        )

    models["v1"] = joblib.load(model_path)


# Request validation
class RiskRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=12,
        max_length=12,
        description="Exactly 12 numerical model features",
    )


# Enqueue shadow work without changing the production result.
def enqueue_shadow_job(job: dict):
    try:
        client.lpush(
            "shadow:queue",
            json.dumps(job),
        )
    except redis.RedisError as exc:
        # Shadow failures must not replace V1's prediction.
        print(f"Shadow queue unavailable: {exc}")


# API home
@app.get("/")
def home():
    return {
        "message": "Shadow Deployment Engine is running",
        "production_model": "v1",
        "shadow_models": ["v2", "v3"],
    }


# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy" if "v1" in models else "unhealthy",
        "production_model_loaded": "v1" in models,
    }


# Production prediction endpoint
@app.post("/predict")
def predict(
    request: RiskRequest,
    background_tasks: BackgroundTasks,
):
    if "v1" not in models:
        raise HTTPException(
            status_code=503,
            detail="Production model V1 is unavailable",
        )

    # Validate the input before inference.
    features = np.asarray(
        request.features,
        dtype=float,
    )

    if not np.isfinite(features).all():
        raise HTTPException(
            status_code=422,
            detail="Features must contain only finite numbers",
        )

    features = features.reshape(1, -1)

    # Only V1 determines the response returned to the caller.
    start = time.perf_counter()

    try:
        prediction = int(
            models["v1"].predict(features)[0]
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Production prediction failed",
        ) from exc

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    # Create a separate job for shadow evaluation.
    job = {
        "request_id": str(uuid.uuid4()),
        "features": request.features,
        "v1_prediction": prediction,
    }

    # Queue after the response is sent.
    # V2 and V3 inference runs in the separate worker.
    background_tasks.add_task(
        enqueue_shadow_job,
        job,
    )

    return {
        "request_id": job["request_id"],
        "production_model": "v1",
        "prediction": prediction,
        "latency_ms": round(latency_ms, 3),
    }


# Shadow metrics endpoint
@app.get("/metrics")
def metrics():
    try:
        from src.metrics import get_metrics

        return get_metrics()

    except redis.RedisError as exc:
        raise HTTPException(
            status_code=503,
            detail="Metrics store unavailable",
        ) from exc
