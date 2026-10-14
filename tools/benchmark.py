"""CPU benchmark of the shipped pipeline (no torch): model load, first inference, warm p50/p95, end-to-end, peak RSS.

    python tools/benchmark.py [--runs 10] [--threads 0] [--out reports/benchmark/<host>.json]

Inputs are synthetic photo-like JPEGs at 0.25 / 2 / 12 MP plus one PNG with a face-free scene; cost of these models
does not depend on image content except the face classifier (runs once per detected face, max 3).
"""
from __future__ import annotations

import argparse
import io
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def photo_like(h, w, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([(xx / w) * 200 + 20, (yy / h) * 180 + 30, ((xx + yy) / (w + h)) * 160 + 40], -1)
    base += rng.normal(0, 6, size=(h, w, 3)).astype(np.float32)
    buf = io.BytesIO()
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def http_flow(an, config_path: Path, cases: dict, runs: int) -> dict:
    """What a user waits for in the browser: upload (POST /api/analyze) until the JSON answer, then the ZIP download.
    Local server on 127.0.0.1 with the SAME loaded analyzer (no second model set)."""
    import threading
    import urllib.request
    from ifd.webapp.server import LocalHTTPServer, State, free_port, make_handler
    state = State(config_path, analyzer=an)
    port = free_port("127.0.0.1")
    httpd = LocalHTTPServer(("127.0.0.1", port), make_handler(state, {f"127.0.0.1:{port}"}))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base, tok = f"http://127.0.0.1:{port}", {"X-IFD-Token": state.token}
    res = {}
    try:
        for name, data in cases.items():
            post, total = [], []
            for i in range(runs + 1):
                t0 = time.perf_counter()
                req = urllib.request.Request(base + "/api/analyze", data=data, method="POST",
                                             headers={**tok, "X-File-Name": name + ".jpg", "Content-Type": "application/octet-stream"})
                with urllib.request.urlopen(req, timeout=600) as r:
                    body = json.loads(r.read())
                t1 = time.perf_counter()
                with urllib.request.urlopen(urllib.request.Request(f"{base}/api/result/{body['result_id']}/results.zip?download=1",
                                                                   headers=tok), timeout=600) as r:
                    zbytes = len(r.read())
                t2 = time.perf_counter()
                if i == 0:
                    continue                      # warm-up
                post.append(t1 - t0)
                total.append(t2 - t0)
            res[name] = {"runs": runs, "upload_to_result_p50_s": round(float(np.percentile(post, 50)), 3),
                         "upload_to_result_p95_s": round(float(np.percentile(post, 95)), 3),
                         "incl_zip_download_p50_s": round(float(np.percentile(total, 50)), 3),
                         "incl_zip_download_p95_s": round(float(np.percentile(total, 95)), 3), "zip_bytes": zbytes}
    finally:
        httpd.shutdown()
        httpd.server_close()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--config", type=Path, default=ROOT / "config.json")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--face-image", type=Path, default=None,
                    help="optional JPEG/PNG with one face, to time the face path (not shipped: data images stay out of the package)")
    ap.add_argument("--http-runs", type=int, default=5, help="runs of the web flow (POST /api/analyze + GET results.zip)")
    a = ap.parse_args()
    from ifd.analysis import Analyzer, report_files
    from ifd.config import load_config
    from ifd.sysinfo import current_rss_mb, describe, peak_rss_mb
    cfg = load_config(a.config)
    if a.threads is not None:
        cfg.data.setdefault("runtime", {})["threads"] = a.threads
    t0 = time.perf_counter()
    an = Analyzer(cfg)
    load_s = time.perf_counter() - t0
    rss_load = current_rss_mb()
    cases = {"0.25MP_600x400": photo_like(400, 600, 1), "2MP_1920x1080": photo_like(1080, 1920, 2),
             "12MP_4000x3000": photo_like(3000, 4000, 3)}
    if a.face_image:
        cases["face_image_" + a.face_image.suffix.lstrip(".").lower()] = a.face_image.read_bytes()
    out = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "environment": describe(),
           "threads_config": cfg.threads, "model_load_seconds_incl_sha256": round(load_s, 2),
           "rss_after_load_mb": rss_load, "cases": {}}
    for name, data in cases.items():
        first = None
        analyze_t, e2e_t, per_det = [], [], {}
        for i in range(a.runs + 1):
            t1 = time.perf_counter()
            rep = an.analyze(data, name=name + ".jpg")
            t2 = time.perf_counter()
            files = report_files(rep, cfg.overlay_alpha)      # what the UI/CLI produce for download
            t3 = time.perf_counter()
            if i == 0:
                first = {"analyze_s": round(t2 - t1, 3), "end_to_end_s": round(t3 - t1, 3)}
                continue
            analyze_t.append(t2 - t1)
            e2e_t.append(t3 - t1)
            for r in rep.results:
                per_det.setdefault(r.key, []).append(next(iter(r.timing.values()), 0.0))
        faces = next((r.details.get("faces_found") for r in rep.results if r.task == "face_deepfake"), None)
        out["cases"][name] = {
            "first_run": first, "runs": a.runs, "faces_found": faces,
            "analyze_p50_s": round(float(np.percentile(analyze_t, 50)), 3), "analyze_p95_s": round(float(np.percentile(analyze_t, 95)), 3),
            "end_to_end_p50_s": round(float(np.percentile(e2e_t, 50)), 3), "end_to_end_p95_s": round(float(np.percentile(e2e_t, 95)), 3),
            "per_detector_p50_s": {k: round(float(np.percentile(v, 50)), 3) for k, v in per_det.items()},
            "output_bytes": sum(len(v) for v in files.values())}
        print(name, out["cases"][name], flush=True)
    if a.http_runs:
        out["http_end_to_end"] = http_flow(an, a.config, {k: cases[k] for k in ("2MP_1920x1080", "12MP_4000x3000")}, a.http_runs)
        print("http", out["http_end_to_end"], flush=True)
    out["rss_end_mb"] = current_rss_mb()
    out["peak_rss_mb"] = peak_rss_mb()
    dest = a.out or (ROOT / "reports" / "benchmark" / f"{platform.node() or 'host'}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("written", dest)


if __name__ == "__main__":
    main()
