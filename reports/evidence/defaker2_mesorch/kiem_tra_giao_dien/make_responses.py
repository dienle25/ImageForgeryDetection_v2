"""Capture the real /api/status and /api/analyze JSON of the web app for harness.js: config DefakerOne2 + Mesorch,
FAKE DefakerOne2 server (tests/fake_defaker2.py, no model), real Mesorch ONNX, two generated images (no photo).

    runtime\\python.exe -X utf8 make_responses.py OUT_DIR
    node harness.js ..\\..\\..\\..\\ifd\\webapp\\static\\app.js OUT_DIR\\status.json OUT_DIR\\analyze_mesorch.json ..\\..\\..\\..\\ifd\\webapp\\static\\index.html
"""
import io
import json
import sys
import threading
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]          # repo root (reports/evidence/defaker2_mesorch/kiem_tra_giao_dien/)
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from fake_defaker2 import ANSWER, fake_server  # noqa: E402
from ifd.webapp.server import LocalHTTPServer, State, free_port, make_handler  # noqa: E402


def png(h, w, seed):
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


p_box = {"p": 0.93}
with fake_server(lambda body: (200, {**ANSWER, "p_fake": p_box["p"], "p_real": 1 - p_box["p"]})) as (url, _):
    data = json.loads((ROOT / "research/configs/config_defaker2_mesorch.json").read_text(encoding="utf-8"))
    data["detectors"]["defaker2"]["url"] = url
    data["detectors"]["mesorch"]["model"] = str(ROOT / "models" / "mesorch.onnx")
    cfgp = OUT / "cfg.json"
    cfgp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    st = State(cfgp)
    st.load()
    assert st.analyzer, st.error
    port = free_port("127.0.0.1")
    httpd = LocalHTTPServer(("127.0.0.1", port), make_handler(st, {f"127.0.0.1:{port}"}))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    tok = {"X-IFD-Token": st.token}

    def get(p):
        return json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}{p}", headers=tok)).read())

    def post(img, name):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/analyze", data=img, method="POST",
                                     headers={**tok, "X-File-Name": name})
        return json.loads(urllib.request.urlopen(req).read())

    (OUT / "status.json").write_text(json.dumps(get("/api/status"), ensure_ascii=False), encoding="utf-8")
    (OUT / "analyze_mesorch.json").write_text(json.dumps(post(png(300, 400, 1), "Screenshot 2026-10-18 181055.png"),
                                                         ensure_ascii=False), encoding="utf-8")
    p_box["p"] = 0.1
    (OUT / "analyze_mesorch_2.json").write_text(json.dumps(post(png(900, 1200, 2), "anh_thuong.png"), ensure_ascii=False),
                                                encoding="utf-8")
    httpd.shutdown()
for f in ("analyze_mesorch.json", "analyze_mesorch_2.json"):
    r = json.loads((OUT / f).read_text(encoding="utf-8"))["report"]
    print(f, r["summary"]["overall_decision"], [(d["detector"], d["decision"], round(d["score"], 4)) for d in r["detectors"]],
          "khong_thong_nhat" if r["summary"].get("disagreement") else "-", "do_tin_cay_thap" if r["input_quality"]["low_confidence"] else "-")
