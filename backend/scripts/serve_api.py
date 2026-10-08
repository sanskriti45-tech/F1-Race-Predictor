"""
Start the local F1 prediction API server.

    python scripts/serve_api.py --port 8787

Serves GET /api/health and GET /api/prediction/next-race by calling the
real backend pipeline (src/app/api.py) — no prediction logic lives here
or in the frontend; this just binds a port. Requires
scripts/build_dataset_and_train.py (and optionally run_backtest.py) to
have been run first for /api/prediction/next-race to return populated
data; otherwise it honestly reports what's missing (see api.py).
"""
from __future__ import annotations

import argparse
import sys
from http.server import HTTPServer

from src.app.api import ApiHandler
from src.config import get_logger

logger = get_logger(__name__)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Serve the F1 prediction API")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--host", default="localhost")
    args = parser.parse_args(argv)

    server = HTTPServer((args.host, args.port), ApiHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"F1 Prediction API running at {url}")
    print(f"  GET {url}/api/health")
    print(f"  GET {url}/api/prediction/next-race")
    print("Press Ctrl+C to stop.")
    logger.info("Serving on %s", url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
