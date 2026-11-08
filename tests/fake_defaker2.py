"""Fake DefakerOne2 server for the tests (no model): POST /score -> respond(body), GET /health -> ready."""
from __future__ import annotations

import contextlib
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ANSWER = {"p_fake": 0.93, "p_real": 0.07, "device": "giả lập", "mode": "giả lập", "seconds": 0.01, "answer": "Fake",
          "top_logprobs": [["Fake", -0.07]], "model_input_size_wh": [512, 512]}


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False


@contextlib.contextmanager
def fake_server(respond=None, health=None):
    """respond(body: bytes) -> (status, dict).  Yields (score_url, list of received bodies)."""
    received = []
    respond = respond or (lambda body: (200, ANSWER))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, payload):
            data = json.dumps(payload).encode("utf-8")
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except OSError:
                pass

        def do_GET(self):
            self._send(200, health or {"ready": True, "device": "giả lập", "mode": "giả lập",
                                       "memory": {"rss_mib": 12.0}})

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            received.append(body)
            self._send(*respond(body))

    httpd = _Server(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/score", received
    finally:
        httpd.shutdown()
        httpd.server_close()


def dead_url() -> str:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}/score"
