
import json
import os

import redis

client = redis.Redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    decode_responses=True,
)


def get_metrics():
    raw = client.lrange("shadow:results", 0, 9999)
    results = [json.loads(item) for item in raw]

    summary = {}

    for version in ("v2", "v3"):
        rows = [r for r in results if r["model"] == version]
        count = len(rows)
        disagreements = sum(r["disagreement"] for r in rows)

        summary[version] = {
            "completed_predictions": count,
            "disagreements_with_v1": disagreements,
            "disagreement_rate": (
                round(disagreements / count, 4)
                if count else None
            ),
            "average_latency_ms": (
                round(
                    sum(r["latency_ms"] for r in rows) / count,
                    3,
                )
                if count else None
            ),
        }

    return {
        "recorded_shadow_results": len(results),
        "models": summary,
    }
