import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .elo_history import load_elo_snapshots

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_HTML_PATH = PROJECT_ROOT / "dashboard" / "elo_dashboard.html"
DEFAULT_LOG_PATH = PROJECT_ROOT / "artifacts" / "elo_history.jsonl"


def make_handler(log_path: Path):

    class EloDashboardHandler(BaseHTTPRequestHandler):

        def do_GET(self):
            parsed_url = urlsplit(self.path)

            if parsed_url.path in ("/", "/index.html"):
                self._send_html(DASHBOARD_HTML_PATH.read_text(encoding="utf-8"))
                return

            if parsed_url.path == "/api/elo-history":
                query = parse_qs(parsed_url.query)
                after = int(query.get("after", ["-1"])[0])
                snapshots = load_elo_snapshots(log_path)
                payload = {
                    "snapshots": snapshots[after + 1 :],
                    "total_snapshots": len(snapshots),
                }
                self._send_json(payload)
                return

            if parsed_url.path == "/health":
                self._send_json({"status": "ok"})
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def log_message(self, format, *args):
            return

        def _send_html(self, html: str):
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return EloDashboardHandler


def main():
    parser = argparse.ArgumentParser(description="Serve the local Elo dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    args = parser.parse_args()

    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(args.log_path.resolve()),
    )
    url = f"http://{args.host}:{args.port}"
    print(f"Serving Elo dashboard at {url}")
    print(f"Reading Elo snapshots from {args.log_path.resolve()}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
