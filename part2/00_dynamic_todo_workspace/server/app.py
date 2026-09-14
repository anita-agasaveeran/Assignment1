"""HTTP layer: JSON API, SSE stream, and static hosting for the SPA.

Run:  python3 run.py [--port 8000] [--host 127.0.0.1]
"""

import argparse
import json
import mimetypes
import os
import queue
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

if __package__ in (None, ""):                       # allow `python3 server/app.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from server import api, db, events
else:
    from . import api, db, events

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
MAX_BODY = 2 * 1024 * 1024
SSE_HEARTBEAT = 20          # seconds

mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("image/svg+xml", ".svg")
mimetypes.add_type("application/manifest+json", ".webmanifest")


class Handler(BaseHTTPRequestHandler):
    server_version = "TodoStack/1.0"
    protocol_version = "HTTP/1.1"

    # -- plumbing ---------------------------------------------------------- #
    def log_message(self, fmt, *args):
        if os.environ.get("TODO_QUIET"):
            return
        sys.stderr.write("  %s  %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def _send(self, status, body=b"", ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, separators=(",", ":")).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _error(self, status, message, field=None):
        self._send(status, {"error": {"status": status, "message": message, "field": field}})

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_BODY:
            raise api.ApiError(413, "Request body too large.")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            raise api.ApiError(400, "Request body must be valid JSON.")
        if not isinstance(data, dict):
            raise api.ApiError(400, "Request body must be a JSON object.")
        return data

    # -- verbs ------------------------------------------------------------- #
    def do_GET(self):     self._dispatch("GET")
    def do_HEAD(self):    self._dispatch("HEAD")
    def do_POST(self):    self._dispatch("POST")
    def do_PATCH(self):   self._dispatch("PATCH")
    def do_PUT(self):     self._dispatch("PUT")
    def do_DELETE(self):  self._dispatch("DELETE")

    def do_OPTIONS(self):
        self._send(204, b"", "text/plain", {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET,POST,PATCH,PUT,DELETE,OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type,X-Client-Id",
            "Access-Control-Max-Age": "600"})

    def _dispatch(self, method):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/api/events":
                return self._stream_events()
            if path.startswith("/api/"):
                body = self._read_json() if method in ("POST", "PATCH", "PUT") else {}
                status, payload = api.route(
                    method, path, body, parse_qs(parsed.query),
                    self.headers.get("X-Client-Id"))
                return self._send(status, payload,
                                  extra={"Access-Control-Allow-Origin": "*",
                                         "Cache-Control": "no-store"})
            return self._serve_static(path)
        except api.ApiError as exc:
            self._error(exc.status, exc.message, exc.field)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:                      # never leak a traceback to the client
            sys.stderr.write(f"!! {method} {path}: {exc!r}\n")
            self._error(500, "Something went wrong on the server.")

    # -- SSE --------------------------------------------------------------- #
    def _stream_events(self):
        q = events.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            self.wfile.write(b": connected\n\n")
            self.wfile.flush()
            while not getattr(self.server, "_shutting_down", False):
                try:
                    msg = q.get(timeout=SSE_HEARTBEAT)
                    self.wfile.write(f"data: {msg}\n\n".encode())
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            events.unsubscribe(q)
        self.close_connection = True

    # -- static ------------------------------------------------------------ #
    def _serve_static(self, path):
        rel = path.lstrip("/") or "index.html"
        target = os.path.normpath(os.path.join(WEB_DIR, rel))
        if not target.startswith(WEB_DIR):                 # path traversal
            return self._error(403, "Forbidden.")
        if os.path.isdir(target):
            target = os.path.join(target, "index.html")
        if not os.path.isfile(target):
            if "." in os.path.basename(rel):               # a real missing asset
                return self._error(404, "Not found.")
            target = os.path.join(WEB_DIR, "index.html")   # SPA deep link
        try:
            with open(target, "rb") as fh:
                data = fh.read()
        except OSError:
            return self._error(404, "Not found.")
        stat = os.stat(target)
        etag = f'W/"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
        if self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.end_headers()
            return
        ctype = mimetypes.guess_type(target)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "text/javascript"):
            ctype += "; charset=utf-8"
        self._send(200, data, ctype, {"ETag": etag, "Cache-Control": "no-cache"})


def _pick_port(host, port):
    for candidate in range(port, port + 20):
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, candidate))
                return candidate
            except OSError:
                continue
    raise SystemExit(f"No free port in {port}-{port + 19}.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the todo app (API + web UI).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-seed", action="store_true", help="start with an empty database")
    ap.add_argument("--reset", action="store_true", help="wipe the database first")
    args = ap.parse_args(argv)

    db.init(seed=False)
    if args.reset:
        db.reset()
    db.init(seed=not args.no_seed)
    db.purge_deleted(30)

    port = _pick_port(args.host, args.port)
    httpd = ThreadingHTTPServer((args.host, port), Handler)
    httpd.daemon_threads = True
    httpd._shutting_down = False

    url = f"http://{args.host}:{port}"
    print(f"\n  ✓ Todo app running at \033[1m{url}\033[0m")
    print(f"    API      {url}/api/state")
    print(f"    Live     {url}/api/events  (server-sent events)")
    print(f"    Database {db.DB_PATH}")
    print("    Ctrl-C to stop\n")
    try:
        httpd.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print("\n  Shutting down…")
    finally:
        httpd._shutting_down = True
        httpd.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    main()
