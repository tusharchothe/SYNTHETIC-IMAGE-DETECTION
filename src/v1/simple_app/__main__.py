"""Entry point for the simple no-auth local workflow.

Starts the FastAPI backend and the Gradio UI in a single process using two
threads so a single terminal command launches everything.

Usage
-----
    # from the project root with the venv active:
    python -m v1.simple_app

    # override ports / paths via env vars:
    SIMPLE_API_PORT=8200 SIMPLE_UI_PORT=7901 python -m v1.simple_app

    # API only (useful for debugging):
    python -m v1.simple_app --api-only

    # UI only (assumes API is already running elsewhere):
    python -m v1.simple_app --ui-only
"""

from __future__ import annotations

import argparse
import sys
import threading
import time


def _start_api(host: str, port: int) -> None:
    """Run the FastAPI app with uvicorn (blocking)."""
    import uvicorn

    from v1.simple_app.api import app

    uvicorn.run(app, host=host, port=port, log_level="info")


def _start_ui(host: str, port: int, api_base: str) -> None:
    """Launch the Gradio UI (blocking)."""
    from v1.simple_app.ui import launch

    launch(host=host, port=port, api_base=api_base)


def _wait_for_api(api_base: str, retries: int = 20, delay: float = 0.5) -> bool:
    """Poll /health until the API is up or we run out of retries."""
    import httpx

    url = f"{api_base.rstrip('/')}/health"
    for _ in range(retries):
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(delay)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m v1.simple_app",
        description="Simple no-auth AI image detector — starts API + UI together.",
    )
    parser.add_argument("--api-only", action="store_true", help="Start the FastAPI server only.")
    parser.add_argument("--ui-only",  action="store_true", help="Start the Gradio UI only.")
    parser.add_argument("--api-host", default=None, help="Override API host (default: SIMPLE_API_HOST or 0.0.0.0).")
    parser.add_argument("--api-port", type=int, default=None, help="Override API port (default: SIMPLE_API_PORT or 8100).")
    parser.add_argument("--ui-host",  default=None, help="Override UI host  (default: SIMPLE_UI_HOST  or 0.0.0.0).")
    parser.add_argument("--ui-port",  type=int, default=None, help="Override UI port  (default: SIMPLE_UI_PORT  or 7900).")
    args = parser.parse_args(argv)

    # Import config after env has been loaded by the package
    from v1.simple_app.config import (
        API_BASE_URL,
        API_HOST,
        API_PORT,
        UI_HOST,
        UI_PORT,
    )

    api_host = args.api_host or API_HOST
    api_port = args.api_port or API_PORT
    ui_host  = args.ui_host  or UI_HOST
    ui_port  = args.ui_port  or UI_PORT
    api_base = f"http://localhost:{api_port}"

    if args.api_only and args.ui_only:
        print("Error: --api-only and --ui-only are mutually exclusive.", file=sys.stderr)
        return 1

    if args.api_only:
        print(f"Starting API on http://{api_host}:{api_port}")
        _start_api(api_host, api_port)
        return 0

    if args.ui_only:
        print(f"Starting UI on http://{ui_host}:{ui_port}  (API expected at {api_base})")
        _start_ui(ui_host, ui_port, api_base)
        return 0

    # ── Default: start both in separate threads ------------------------------
    print(f"Starting API  →  http://localhost:{api_port}")
    print(f"Starting UI   →  http://localhost:{ui_port}")
    print(f"API docs      →  http://localhost:{api_port}/docs")
    print()

    api_thread = threading.Thread(
        target=_start_api,
        args=(api_host, api_port),
        daemon=True,
        name="simple-api",
    )
    api_thread.start()

    # Give the API a moment to initialise before the UI tries to connect.
    print("Waiting for API to be ready ...", end="", flush=True)
    ready = _wait_for_api(api_base)
    if ready:
        print(" OK")
    else:
        print(
            "\nWarning: API did not respond before the UI started. "
            "The UI will retry automatically when you interact with it."
        )

    # Start the UI in the main thread (Gradio needs the main thread on some
    # platforms for its internal event loop).
    _start_ui(ui_host, ui_port, api_base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
