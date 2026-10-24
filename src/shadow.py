
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MODEL_DIR = Path(__file__).resolve().parent.parent / "artifacts"


client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=10,
    health_check_interval=30,
)



models = {
    version: joblib.load(MODEL_DIR / f"{version}.joblib")
    for version in ("v2", "v3")
}


def run_worker():
    print("Shadow worker started. Waiting for jobs...")

    while True:
        try:
            item = client.blpop("shadow:queue", timeout=5)

            if item is None:
                continue

            _, raw_job = item
            job = json.loads(raw_job)
            features = np.asarray(
                job["features"], dtype=float
            ).reshape(1, -1)

            for version, model in models.items():
                start = time.perf_counter()
                prediction = int(model.predict(features)[0])
                latency = (time.perf_counter() - start) * 1000

                result = {
                    "request_id": job["request_id"],
                    "model": version,
                    "prediction": prediction,
                    "v1_prediction": job["v1_prediction"],
                    "disagreement": (
                        prediction != job["v1_prediction"]
                    ),
                    "latency_ms": round(latency, 3),
                }

                client.lpush("shadow:results", json.dumps(result))
                client.ltrim("shadow:results", 0, 9999)
                print(json.dumps(result))

        except Exception as exc:
            print(f"Worker error: {exc}")
            time.sleep(1)


if __name__ == "__main__":
    run_worker()
