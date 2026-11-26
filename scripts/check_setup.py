"""Verify Phase 0 setup: .env keys and Qdrant Cloud connectivity."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client import QdrantClient

from src.core.config import get_settings


def main() -> int:
    settings = get_settings()
    env = settings.env

    print("=== Phase 0 Setup Check ===")
    print(f"Runtime env     : {settings.app.runtime_env}")
    print(f"Qdrant mode     : {settings.effective_qdrant_mode}")
    print(f"GEMINI_API_KEY  : {'SET' if env.gemini_api_key and 'your_' not in env.gemini_api_key else 'MISSING'}")
    print(f"GROQ_API_KEY    : {'SET' if env.groq_api_key and 'your_' not in env.groq_api_key else 'MISSING'}")

    if settings.effective_qdrant_mode == "cloud":
        if not env.qdrant_url or "xxxx" in env.qdrant_url or not env.qdrant_api_key or "your_" in env.qdrant_api_key:
            print("[ERROR] QDRANT_URL or QDRANT_API_KEY is not configured in .env")
            return 1
        client = QdrantClient(url=env.qdrant_url, api_key=env.qdrant_api_key, timeout=30.0)
    else:
        client = QdrantClient(path=env.qdrant_local_path)

    collections = client.get_collections().collections
    names = [c.name for c in collections]
    print(f"Qdrant status   : CONNECTED (collections={names})")
    print("=== PHASE 0 OK ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
