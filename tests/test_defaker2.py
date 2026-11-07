"""DefakerOne2 add-on (experimental, HTTP): works / server down / timeout / bad answers, and it never changes the
overall decision, the task coverage or the decision-config hash.  A fake local server replaces the real model."""
import contextlib
import io
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import encode, light_config_data
from ifd.analysis import Analyzer, decision_config_hash, load_evaluated_configs
from ifd.config import Config, load_config
from ifd.detectors.base import TASKS
from ifd.detectors.defaker2 import DefakerTwoHTTPDetector
from ifd.errors import ForgeryError

ROOT = Path(__file__).resolve().parents[1]
OK_ANSWER = {"p_fake": 0.93, "p_real": 0.07, "device": "cuda:0 (giả lập)", "mode": "giả lập", "seconds": 0.01,
             "answer": "Fake", "top_logprobs": [["Fake", -0.07], ["Real", -2.7]], "model_input_size_wh": [512, 512]}


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False        # do not wait for a deliberately slow handler when the test ends


@contextlib.contextmanager
def fake_server(respond, delay: float = 0.0, health=None):
    """respond(body) -> (status, dict | bytes).  Yields (score_url, list of received requests)."""
    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, payload):
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except OSError:
                pass                  # the client already gave up (timeout test)

        def do_GET(self):
            self._send(200, health or {"ready": True, "device": "cuda:0 (giả lập)", "mode": "giả lập"})

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            received.append({"path": self.path, "ctype": self.headers.get("Content-Type"), "body": body})
            if delay:
                time.sleep(delay)
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
    s.close()                         # nothing listens on this port any more
    return f"http://127.0.0.1:{port}/score"


def settings(url, timeout=10):
    return {"enabled": True, "type": "defaker2_http", "url": url, "timeout_seconds": timeout}


def config_with_defaker2(url, timeout=10) -> dict:
    data = light_config_data()        # pscc + commfor only (small models)
    data["detectors"]["defaker2"] = settings(url, timeout)
    return data


def test_working_server_gives_experimental_p_fake(rgb_image):
    with fake_server(lambda body: (200, OK_ANSWER)) as (url, got):
        r = DefakerTwoHTTPDetector(settings(url)).predict(rgb_image)
    assert r.score == pytest.approx(0.93) and r.decision == "experimental"
    assert r.in_summary is False and r.calibrated is False and r.heatmap is None and r.thresholds == {}
    assert r.details["p_real"] == pytest.approx(0.07) and r.details["answer"] == "Fake"
    assert r.details["validation"]["status"] == "experimental" and "THỬ NGHIỆM" in r.score_type
    assert not r.warnings and r.timing["server_inference"] == pytest.approx(0.01)
    js = r.to_json()
    assert js["in_summary"] is False and js["decision_vi"].startswith("Thử nghiệm") and js["has_heatmap"] is False
    # the decoded image is sent losslessly as PNG to /score
    assert len(got) == 1 and got[0]["path"] == "/score" and got[0]["ctype"] == "image/png"
    sent = np.asarray(Image.open(io.BytesIO(got[0]["body"])))
    assert np.array_equal(sent, rgb_image)


def test_server_down_gives_unavailable(rgb_image):
    r = DefakerTwoHTTPDetector(settings(dead_url())).predict(rgb_image)
    assert r.decision == "unavailable" and r.score is None and r.in_summary is False
    assert "CHAY_DEFAKER2.bat" in r.warnings[0]
    assert r.to_json()["decision_vi"] == "Chưa khả dụng"


def test_timeout_gives_unavailable_quickly(rgb_image):
    with fake_server(lambda body: (200, OK_ANSWER), delay=4.0) as (url, _):
        t0 = time.perf_counter()
        r = DefakerTwoHTTPDetector(settings(url, timeout=1)).predict(rgb_image)
        dt = time.perf_counter() - t0
    assert r.decision == "unavailable" and r.score is None
    assert "1 giây" in r.warnings[0]
    assert dt < 3.5, f"timeout not honoured ({dt:.1f}s)"


@pytest.mark.parametrize("reply, needle", [
    ((503, {"error": "Đang nạp mô hình"}), "chưa sẵn sàng"),
    ((500, {"error": "RuntimeError: NaN"}), "HTTP 500"),
    ((200, b"{not json"), "không hợp lệ"),
    ((200, {"p_fake": "nan", "p_real": 0.5}), "NaN/inf"),
    ((200, {"p_fake": 1.7, "p_real": -0.7}), "ngoài [0, 1]"),
    ((200, {"device": "x"}), "không hợp lệ"),
])
def test_bad_answers_give_unavailable(rgb_image, reply, needle):
    with fake_server(lambda body: reply) as (url, _):
        r = DefakerTwoHTTPDetector(settings(url)).predict(rgb_image)
    assert r.decision == "unavailable" and r.score is None, r
    assert needle in r.warnings[0]


def test_health_reports_server_state():
    with fake_server(lambda body: (200, OK_ANSWER)) as (url, _):
        assert DefakerTwoHTTPDetector(settings(url)).health()["state"] == "ready"
    with fake_server(lambda body: (200, OK_ANSWER), health={"ready": False, "loading": True}) as (url, _):
        assert DefakerTwoHTTPDetector(settings(url)).health()["state"] == "loading"
    assert DefakerTwoHTTPDetector(settings(dead_url())).health(timeout=0.5)["state"] == "down"


@pytest.mark.parametrize("bad", [
    {"url": "http://example.com:8765/score"},          # images must stay on this machine
    {"url": "http://192.168.1.5:8765/score"},
    {"url": "https://127.0.0.1:8765/score"},
    {"url": "http://127.0.0.1/score"},                 # no port
    {"url": ""},
    {"url": "http://127.0.0.1:8765/score", "timeout_seconds": 0},
    {"url": "http://127.0.0.1:8765/score", "timeout_seconds": float("nan")},
])
def test_config_rejects_non_local_or_bad_settings(bad):
    data = config_with_defaker2("http://127.0.0.1:8765/score")
    data["detectors"]["defaker2"].update(bad)
    with pytest.raises(ForgeryError):
        Config(data, ROOT)


def test_demo_config_keeps_the_evaluated_decision_profile():
    std = load_config(ROOT / "config.json")
    demo = load_config(ROOT / "research" / "configs" / "config_demo_defaker2.json")
    assert demo.detector("defaker2")["type"] == "defaker2_http" and demo.detector("defaker2")["enabled"]
    for key in ("mesorch", "commfor_aigen", "face_clip", "pscc_net"):           # copied, thresholds untouched
        assert demo.detector(key)["thresholds"] == std.detector(key)["thresholds"]
        assert demo.path(demo.detector(key)["model"]).resolve() == std.path(std.detector(key)["model"]).resolve()
    assert decision_config_hash(demo) == decision_config_hash(std)
    profiles = load_evaluated_configs(ROOT)
    if profiles:                                                                 # shipped validation summary
        assert decision_config_hash(demo) in profiles and profiles[decision_config_hash(demo)].startswith("chuẩn")
    assert demo.notice_vi and "không được tính" in demo.notice_vi.lower()


@pytest.mark.parametrize("server_up", [True, False])
def test_overall_decision_ignores_defaker2(tmp_path, server_up):
    img = encode(np.random.default_rng(5).integers(0, 255, (96, 128, 3), dtype=np.uint8))
    base = Config(light_config_data(), ROOT)
    plain = Analyzer(base).analyze(img, name="x.png")
    ctx = fake_server(lambda body: (200, OK_ANSWER)) if server_up else contextlib.nullcontext((dead_url(), []))
    with ctx as (url, _):
        cfg = Config(config_with_defaker2(url), ROOT)
        assert decision_config_hash(cfg) == decision_config_hash(base)
        rep = Analyzer(cfg).analyze(img, name="x.png")
    d2 = next(r for r in rep.results if r.key == "defaker2")
    assert d2.decision == ("experimental" if server_up else "unavailable")
    assert rep.summary == plain.summary                       # decision, tasks and warnings unchanged
    assert rep.metadata["decision_config_sha256"] == plain.metadata["decision_config_sha256"]
    assert [r.score for r in rep.results if r.key != "defaker2"] == [r.score for r in plain.results]
    from ifd.analysis import report_files
    data = json.loads(report_files(rep, cfg.overlay_alpha, images=False)["result.json"])
    card = next(d for d in data["detectors"] if d["detector"] == "defaker2")
    assert card["in_summary"] is False and card["task"] == "whole_image_authenticity"
    assert data["models"]["defaker2"]["sha256"] is None and "Chưa rõ" in data["models"]["defaker2"]["license_weights"]
    assert all("in_summary" not in d for d in data["detectors"] if d["detector"] != "defaker2")


def test_web_status_and_analyze_show_fourth_card(tmp_path):
    from ifd.webapp.server import LocalHTTPServer, State, free_port, make_handler
    img = encode(np.random.default_rng(6).integers(0, 255, (80, 120, 3), dtype=np.uint8))
    with fake_server(lambda body: (200, OK_ANSWER)) as (url, _):
        data = config_with_defaker2(url)
        data["display"]["notice_vi"] = "Bản demo có DefakerOne2 (thử nghiệm)."
        cfg_path = tmp_path / "config_demo.json"
        cfg_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        state = State(cfg_path)
        state.load()
        assert state.analyzer is not None, state.error
        port = free_port("127.0.0.1")
        httpd = LocalHTTPServer(("127.0.0.1", port), make_handler(state, {f"127.0.0.1:{port}"}))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            tok = {"X-IFD-Token": state.token}
            with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/api/status", headers=tok)) as r:
                st = json.loads(r.read())
            item = next(d for d in st["detectors"] if d["key"] == "defaker2")
            assert item["experimental"] is True and item["validation_status"] == "experimental"
            assert item["server"]["state"] == "ready" and st["notice_vi"].startswith("Bản demo")
            assert "Thật/giả toàn ảnh" in item["task_vi"]
            assert st["tasks_not_checked"] == [TASKS["face_deepfake"]]     # the add-on does not "cover" a task
            req = urllib.request.Request(f"http://127.0.0.1:{port}/api/analyze", data=img, method="POST",
                                         headers={**tok, "X-File-Name": "a.png"})
            with urllib.request.urlopen(req, timeout=120) as r:
                rep = json.loads(r.read())["report"]
        finally:
            httpd.shutdown()
            httpd.server_close()
    cards = {d["detector"]: d for d in rep["detectors"]}
    assert list(cards)[-1] == "defaker2" and cards["defaker2"]["score"] == pytest.approx(0.93)
    assert cards["defaker2"]["in_summary"] is False
    assert rep["summary"]["overall_decision"] in ("no_clear_sign", "needs_review", "suspicious")


def _kit_dataset(base: Path) -> Path:
    data = base / "du_lieu"
    (data / "that" / "dien_thoai").mkdir(parents=True)
    (data / "sua" / "ghep_anh").mkdir(parents=True)
    tdata = ROOT / "tests" / "data"
    for name in ("authentic1.png", "authentic3.png"):
        (data / "that" / "dien_thoai" / name).write_bytes((tdata / name).read_bytes())
    rng = np.random.default_rng(7)
    (data / "that" / "dien_thoai" / "r.jpg").write_bytes(encode(rng.integers(0, 255, (80, 120, 3), dtype=np.uint8), "JPEG", quality=90))
    for name in ("splice1.png", "copymove2.png"):
        (data / "sua" / "ghep_anh" / name).write_bytes((tdata / name).read_bytes())
    return data


def _run_kit(data: Path, cfg_data: dict, out: Path, tmp: Path, name: str) -> dict:
    from ifd.company_eval import main as kit_main
    cfg_path = tmp / f"{name}.json"
    cfg_path.write_text(json.dumps(cfg_data), encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        rc = kit_main([str(data), "--output", str(out), "--config", str(cfg_path)])
    assert rc == 0
    work = next((out / "kiem_dinh").iterdir())
    rep = max((p for p in work.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime_ns)
    m = json.loads((rep / "metrics.json").read_text(encoding="utf-8"))
    m["_csv"] = (rep / "du_doan.csv").read_text(encoding="utf-8-sig")
    m["_html"] = (rep / "bao_cao_kiem_dinh.html").read_text(encoding="utf-8")
    m["_cache"] = [json.loads(l) for l in (work / "cache_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    return m


def test_validation_kit_reports_defaker2_separately(tmp_path):
    """The kit scores DefakerOne2 and reports it on its own (no thresholds, answer point 0.5), while decisions, the
    application-level block and the reference thresholds use only the evaluated detectors."""
    data = _kit_dataset(tmp_path)
    plain = _run_kit(data, light_config_data(), tmp_path / "out_plain", tmp_path, "plain")
    fakes = [np.asarray(Image.open(ROOT / "tests" / "data" / n).convert("RGB")) for n in ("splice1.png", "copymove2.png")]

    def respond(body):        # high p_fake for the two edited images (recognised by their pixels)
        px = np.asarray(Image.open(io.BytesIO(body)).convert("RGB"))
        fake = any(px.shape == f.shape and np.array_equal(px, f) for f in fakes)
        return 200, {**OK_ANSWER, "p_fake": 0.9 if fake else 0.1, "p_real": 0.1 if fake else 0.9}

    out = tmp_path / "out_demo"
    m1 = _run_kit(data, config_with_defaker2(dead_url(), timeout=3), out, tmp_path, "demo")   # 1) server down
    assert m1["experimental"]["defaker2"]["n_scored"] == 0 and m1["experimental"]["defaker2"]["n_unavailable"] == 5
    assert all(c.get("experimental_unavailable") for c in m1["_cache"])
    assert "không chấm được" in m1["_html"]
    with fake_server(respond) as (url, got):                                                # 2) server up
        m2 = _run_kit(data, config_with_defaker2(url), out, tmp_path, "demo2")
    exp = m2["experimental"]["defaker2"]
    assert exp["n_scored"] == 5 and exp["n_unavailable"] == 0 and len(got) == 5
    assert exp["auroc"] == pytest.approx(1.0) and exp["at_answer_point"]["tp"] == 2 and exp["at_answer_point"]["fp"] == 0
    assert exp["by_type"]["sua/ghep_anh"]["ge_answer_point"] == pytest.approx(1.0)
    for m in (m1, m2):
        assert set(m["thresholds"]) == set(plain["thresholds"]) == {"pscc_net", "commfor_aigen"}
        assert set(m["detectors"]) == set(plain["detectors"]) and set(m["reference_thresholds"]) == set(plain["reference_thresholds"])
        assert m["product"] == plain["product"]                                  # application-level result unchanged
        assert "THỬ NGHIỆM" in m["_html"] and "defaker2_score" in m["_csv"].splitlines()[0]
        assert "defaker2_decision" not in m["_csv"].splitlines()[0]
    assert "THỬ NGHIỆM" not in plain["_html"] and plain["experimental"] == {}


def test_validation_kit_rescoring_after_server_outage(tmp_path):
    """Images that the add-on could not score are scored again on the next run (same configuration)."""
    from ifd.company_eval import analyze_all, collect_folder
    data = _kit_dataset(tmp_path)
    items, _ = collect_folder(data)
    cache, key = tmp_path / "cache.jsonl", "same-settings"     # the key stands for "unchanged configuration"
    with fake_server(lambda body: (503, {"error": "Đang nạp mô hình"})) as (url, _):
        first = analyze_all(items, Config(config_with_defaker2(url), ROOT), cache, key, progress=lambda s: None)
    assert all(r["experimental_unavailable"] for r in first) and all(r["scores"]["defaker2"] is None for r in first)
    with fake_server(lambda body: (200, OK_ANSWER)) as (url, got):
        cfg = Config(config_with_defaker2(url), ROOT)
        second = analyze_all(items, cfg, cache, key, progress=lambda s: None)
    assert len(got) == len(items) and all(r["scores"]["defaker2"] == pytest.approx(0.93) for r in second)
    assert not any(r.get("experimental_unavailable") for r in second)
    third = analyze_all(items, cfg, cache, key, progress=lambda s: None)        # server gone, complete lines reused
    assert [r["scores"] for r in third] == [r["scores"] for r in second]


def _server_module():
    """addons/defaker2/defaker2_server.py imports only the standard library at module level (torch is loaded later)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("defaker2_server", ROOT / "addons" / "defaker2" / "defaker2_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _StubEngine:
    def __init__(self, ready):
        self.ready, self.loading, self.error, self.requests = ready, not ready, None, 0

    def status(self):
        return {"ready": self.ready, "loading": self.loading, "error": None}

    def score(self, body):
        self.requests += 1
        return {"p_fake": 0.25, "p_real": 0.75, "device": "stub", "mode": "stub", "seconds": 0.01, "answer": "Real",
                "image_size_wh": [8, 8]}


def test_server_http_layer_without_model():
    mod = _server_module()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(url, data=None, headers=None, method=None):
        try:
            with opener.open(urllib.request.Request(url, data=data, headers=headers or {}, method=method), timeout=10) as r:
                return r.status, r.headers.get("Content-Type", ""), r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type", ""), e.read()

    for ready in (False, True):
        engine = _StubEngine(ready)
        httpd = mod.LocalServer(("127.0.0.1", 0), mod.make_handler(engine, set(), 2**20))
        port = httpd.server_address[1]
        httpd.RequestHandlerClass = mod.make_handler(engine, {f"127.0.0.1:{port}"}, 2**20)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{port}"
        try:
            for path in ("/", "/score"):                       # opened in a browser: explanation page, not an error
                code, ctype, body = call(base + path)
                text = body.decode("utf-8")
                assert code == 200 and ctype.startswith("text/html") and "không phải giao diện" in text
                assert r"RUN_APP.bat --config research\configs\config_demo_defaker2.json" in text
                assert ("sẵn sàng" if ready else "đang nạp") in text
            assert json.loads(call(base + "/health")[2])["ready"] is ready
            assert call(base + "/khac")[0] == 404
            assert call(base + "/health", headers={"Host": "evil.example"})[0] == 403
            png = encode(np.zeros((8, 8, 3), np.uint8))
            assert call(base + "/score", data=b"abc", headers={"Content-Type": "text/plain"}, method="POST")[0] == 415
            assert call(base + "/score", data=b"not an image", headers={"Content-Type": "image/png"}, method="POST")[0] == 415
            code, _, body = call(base + "/score", data=png, headers={"Content-Type": "image/png"}, method="POST")
            if ready:
                assert code == 200 and json.loads(body)["p_fake"] == pytest.approx(0.25)
            else:
                assert code == 503 and "Đang nạp" in json.loads(body)["error"]
        finally:
            httpd.shutdown()
            httpd.server_close()


# ---------------------------------------------------------------- DefakerOne2 as the MAIN model (role "primary")
AUTHOR_TH = {"low": 0.5, "high": 0.5, "status": "theo tác giả"}


def primary_only_config(url, timeout=10) -> dict:
    data = light_config_data()
    for det in data["detectors"].values():
        det["enabled"] = False
    data["detectors"]["defaker2"] = {**settings(url, timeout), "role": "primary", "thresholds": dict(AUTHOR_TH)}
    return data


@pytest.mark.parametrize("p_fake, expected", [(0.93, "suspicious"), (0.5, "suspicious"), (0.1, "no_clear_sign")])
def test_primary_model_decides_with_author_rule(p_fake, expected):
    img = encode(np.random.default_rng(8).integers(0, 255, (64, 96, 3), dtype=np.uint8))
    with fake_server(lambda body: (200, {**OK_ANSWER, "p_fake": p_fake, "p_real": 1 - p_fake})) as (url, _):
        rep = Analyzer(Config(primary_only_config(url), ROOT)).analyze(img, name="x.png")
    r = rep.results[0]
    assert r.primary and r.in_summary and r.decision == expected and r.thresholds["high"] == 0.5
    s = rep.summary
    assert s["overall_decision"] == expected and s["decided_by"] == "defaker2" and s["primary_detector"] == "defaker2"
    assert s["tasks_not_checked"] == [] and "mô hình chính" in s["basis_vi"]
    js = r.to_json()
    assert js["primary"] is True and "in_summary" not in js


def test_primary_model_down_gives_no_conclusion():
    img = encode(np.full((64, 64, 3), 90, np.uint8))
    rep = Analyzer(Config(primary_only_config(dead_url()), ROOT)).analyze(img, name="x.png")
    assert rep.results[0].decision == "unavailable"
    assert rep.summary["overall_decision"] == "unavailable" and rep.summary["overall_decision_vi"] == "Chưa khả dụng"
    assert any("CHAY_DEFAKER2.bat" in w for w in rep.summary["warnings"])
    json.loads(__import__("ifd.analysis", fromlist=["report_files"]).report_files(rep, 0.5, images=False)["result.json"])


def test_primary_with_other_detectors_falls_back_when_down():
    img = encode(np.random.default_rng(9).integers(0, 255, (80, 80, 3), dtype=np.uint8))
    plain = Analyzer(Config(light_config_data(), ROOT)).analyze(img, name="x.png")
    data = light_config_data()
    data["detectors"]["defaker2"] = {**settings(dead_url()), "role": "primary", "thresholds": dict(AUTHOR_TH)}
    rep = Analyzer(Config(data, ROOT)).analyze(img, name="x.png")
    assert rep.summary["overall_decision"] == plain.summary["overall_decision"]
    assert rep.summary["decided_by"] == "fallback" and "TẠM" in rep.summary["warnings"][0]
    with fake_server(lambda body: (200, {**OK_ANSWER, "p_fake": 0.01, "p_real": 0.99})) as (url, _):
        data["detectors"]["defaker2"]["url"] = url
        rep2 = Analyzer(Config(data, ROOT)).analyze(img, name="x.png")
    assert rep2.summary["overall_decision"] == "no_clear_sign" and rep2.summary["decided_by"] == "defaker2"


def test_main_config_file_uses_only_defaker2_with_author_threshold():
    cfg = load_config(ROOT / "research" / "configs" / "config_defaker2_chinh.json")
    enabled = {k: d for k, d in cfg.data["detectors"].items() if d.get("enabled")}
    assert list(enabled) == ["defaker2"] and enabled["defaker2"]["role"] == "primary"
    assert enabled["defaker2"]["thresholds"]["low"] == enabled["defaker2"]["thresholds"]["high"] == 0.5
    assert decision_config_hash(cfg) != decision_config_hash(load_config(ROOT / "config.json"))
    assert "MÔ HÌNH CHÍNH" in cfg.notice_vi
    bat = (ROOT / "RUN_APP_DEFAKER2.bat").read_text(encoding="ascii")
    assert r"research\configs\config_defaker2_chinh.json" in bat


@pytest.mark.parametrize("mutate", [
    lambda d: d["detectors"]["defaker2"].pop("thresholds"),                      # primary needs thresholds
    lambda d: d["detectors"]["defaker2"].update(role="boss"),
    lambda d: d["detectors"].update(d2={**d["detectors"]["defaker2"]}),          # two primary models
])
def test_primary_config_validation(mutate):
    data = primary_only_config("http://127.0.0.1:8765/score")
    mutate(data)
    with pytest.raises(ForgeryError):
        Config(data, ROOT)


def test_validation_kit_with_primary_model(tmp_path):
    data = _kit_dataset(tmp_path)
    fakes = [np.asarray(Image.open(ROOT / "tests" / "data" / n).convert("RGB")) for n in ("splice1.png", "copymove2.png")]

    def respond(body):
        px = np.asarray(Image.open(io.BytesIO(body)).convert("RGB"))
        fake = any(px.shape == f.shape and np.array_equal(px, f) for f in fakes)
        return 200, {**OK_ANSWER, "p_fake": 0.9 if fake else 0.1, "p_real": 0.1 if fake else 0.9}

    with fake_server(respond) as (url, _):
        m = _run_kit(data, primary_only_config(url), tmp_path / "out", tmp_path, "main")
    assert set(m["thresholds"]) == {"defaker2"} and m["experimental"] == {}
    p = m["product"]["nghi_van_la_duong"]
    assert (p["tp"], p["fp"], p["tn"], p["fn"]) == (2, 0, 3, 0)
    assert any("mô hình chính defaker2" in w for w in m["warnings"])
