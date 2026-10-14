"""Local web server: token, host check, upload limits, analysis, downloads, no stale results."""
import io
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from conftest import encode
from ifd.webapp.server import LocalHTTPServer, State, free_port, make_handler

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def server(analyzer):
    port = free_port("127.0.0.1")
    state = State(ROOT / "config.json", analyzer=analyzer)   # reuse the session's models (RAM)
    httpd = LocalHTTPServer(("127.0.0.1", port), make_handler(state, {f"127.0.0.1:{port}", f"localhost:{port}"}))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}", state
    httpd.shutdown()
    httpd.server_close()


def req(url, data=None, headers=None, method=None):
    r = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(r, timeout=300) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def test_requires_token_and_valid_host(server):
    base, state = server
    assert req(base + "/api/status")[0] == 403
    assert req(base + "/api/status", headers={"X-IFD-Token": "wrong"})[0] == 403
    code, body, _ = req(base + "/api/status", headers={"X-IFD-Token": state.token})
    assert code == 200 and json.loads(body)["ready"] is True
    assert req(base + "/", headers={"Host": "attacker.example:80"})[0] == 403
    code, body, hdr = req(base + "/")
    assert code == 200 and b"<title>" in body and "Content-Security-Policy" in hdr


def test_analyze_download_and_limits(server):
    base, state = server
    tok = {"X-IFD-Token": state.token}
    img = encode(np.random.default_rng(3).integers(0, 255, (80, 120, 3), dtype=np.uint8))
    code, body, _ = req(base + "/api/analyze", data=img, headers={**tok, "X-File-Name": urllib.parse.quote("ảnh thử.png")}, method="POST")
    assert code == 200, body
    data = json.loads(body)
    rep = data["report"]
    assert rep["source_name"] == "ảnh thử.png"
    assert rep["image"]["display_size_wh"] == [120, 80]
    rid = data["result_id"]
    code, z, hdr = req(f"{base}/api/result/{rid}/results.zip?t={state.token}&download=1")
    assert code == 200 and "attachment" in hdr.get("Content-Disposition", "")
    names = zipfile.ZipFile(io.BytesIO(z)).namelist()
    assert "result.json" in names and "overlay.png" in names
    inner = json.loads(zipfile.ZipFile(io.BytesIO(z)).read("result.json"))
    assert inner["image"]["sha256"] == rep["image"]["sha256"]
    assert req(f"{base}/api/result/{rid}/../../config.json?t={state.token}")[0] == 404
    assert req(f"{base}/api/result/{rid}/result.json")[0] == 403  # token also needed for downloads
    # invalid upload and empty body
    code, body, _ = req(base + "/api/analyze", data=b"garbage", headers=tok, method="POST")
    assert code == 422 and "hỏng" in json.loads(body)["error"]
    code, body, _ = req(base + "/api/analyze", data=b"", headers=tok, method="POST")
    assert code == 400


def test_results_are_isolated_per_upload(server):
    base, state = server
    tok = {"X-IFD-Token": state.token}
    a = encode(np.full((64, 64, 3), 10, np.uint8))
    b = encode(np.full((48, 96, 3), 200, np.uint8))
    ra = json.loads(req(base + "/api/analyze", data=a, headers={**tok, "X-File-Name": "a.png"}, method="POST")[1])
    rb = json.loads(req(base + "/api/analyze", data=b, headers={**tok, "X-File-Name": "b.png"}, method="POST")[1])
    assert ra["result_id"] != rb["result_id"]
    ja = json.loads(req(f"{base}/api/result/{ra['result_id']}/result.json?t={state.token}")[1])
    jb = json.loads(req(f"{base}/api/result/{rb['result_id']}/result.json?t={state.token}")[1])
    assert ja["source_name"] == "a.png" and jb["source_name"] == "b.png"
    assert ja["image"]["sha256"] != jb["image"]["sha256"]


def test_status_lists_detectors_and_tasks(server):
    base, state = server
    st = json.loads(req(base + "/api/status", headers={"X-IFD-Token": state.token})[1])
    tasks = {d["key"]: d["task"] for d in st["detectors"]}
    assert tasks == {"mesorch": "local_manipulation", "commfor_aigen": "ai_generated", "face_clip": "face_deepfake"}
    assert st["tasks_not_checked"] == [] and st["models"]["mesorch"]["sha256"]


def test_unexpected_error_is_logged_with_traceback(server, monkeypatch, caplog):
    """F07: an unexpected exception must give HTTP 500 with a Vietnamese message AND a logged traceback."""
    base, state = server

    def boom(*a, **k):
        raise RuntimeError("injected failure for test")

    monkeypatch.setattr(state.analyzer, "analyze", boom)
    with caplog.at_level("ERROR", logger="ifd.webapp"):
        code, body, _ = req(base + "/api/analyze", data=encode(np.full((40, 40, 3), 5, np.uint8)),
                            headers={"X-IFD-Token": state.token, "X-File-Name": "x.png"}, method="POST")
    assert code == 500 and "RuntimeError" in json.loads(body)["error"]
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "Traceback" in logged and "injected failure for test" in logged


def test_busy_port_fails_fast_with_exit_code_2(capsys):
    import socket
    from ifd.webapp.server import main
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    try:
        assert main(["--port", str(s.getsockname()[1]), "--no-browser"]) == 2
    finally:
        s.close()
    assert "Không mở được cổng" in capsys.readouterr().err
