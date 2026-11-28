"""One-command Zero-Docker local launcher: MCP Server (:8001) + FastAPI (:8000) + Chainlit (:8501)."""

from __future__ import annotations

import subprocess
import sys
import time

SERVICES = [
    ("MCP Server", ["uv", "run", "python", "-m", "src.mcp_server.server"]),
    ("FastAPI", ["uv", "run", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8000"]),
    ("Chainlit", ["uv", "run", "chainlit", "run", "ui/app.py", "--host", "127.0.0.1", "--port", "8501"]),
]


def main() -> int:
    procs: list[tuple[str, subprocess.Popen[bytes]]] = []
    try:
        for name, cmd in SERVICES:
            print(f"[START] {name}: {' '.join(cmd)}", flush=True)
            procs.append((name, subprocess.Popen(cmd)))
            time.sleep(2.5)

        print(
            "\n[READY] Chainlit UI : http://127.0.0.1:8501\n"
            "[READY] FastAPI Docs: http://127.0.0.1:8000/docs\n"
            "[READY] Health Check: http://127.0.0.1:8000/health\n"
            "Press Ctrl+C to stop all services cleanly.\n",
            flush=True,
        )

        while True:
            for name, proc in procs:
                if proc.poll() is not None:
                    raise RuntimeError(f"Service '{name}' exited unexpectedly (code={proc.returncode}).")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[STOP] Shutting down all services...", flush=True)
    finally:
        for _, proc in reversed(procs):
            if proc.poll() is None:
                proc.terminate()
        for _, proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
