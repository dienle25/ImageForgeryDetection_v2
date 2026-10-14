"""Acceptance self-test that runs the REAL app code on this machine (Windows or Linux) — no mocks.

    python -m ifd.selftest [--out reports/selftest] [--quick]

Writes <out>/<UTC>_<host>/selftest.json and selftest.txt with PASS / FAIL / SKIP per check, environment,
model hashes, cross-platform numeric parity (stored reference outputs) and CPU timings.
Exit code 0 = no FAIL, 1 = at least one FAIL.
"""
from __future__ import annotations

import argparse
import io
import json
import platform
import shutil
import socket
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTDATA = ROOT / "tests" / "data"


class Recorder:
    def __init__(self):
        self.checks = []

    def run(self, name: str, fn, *args, **kw):
        t0 = time.perf_counter()
        try:
            detail = fn(*args, **kw)
            status = "SKIP" if isinstance(detail, dict) and detail.get("_skip") else "PASS"
        except AssertionError as exc:
            status, detail = "FAIL", {"assertion": str(exc)}
        except Exception as exc:
            status, detail = "FAIL", {"exception": f"{type(exc).__name__}: {exc}",
                                      "traceback": traceback.format_exc()[-2000:]}
        rec = {"check": name, "status": status, "seconds": round(time.perf_counter() - t0, 3), "detail": detail}
        self.checks.append(rec)
        print(f"[{status:4s}] {name} ({rec['seconds']:.1f}s)", flush=True)
        if status == "FAIL":
            print("       " + json.dumps(detail, ensure_ascii=False)[:600], flush=True)
        return detail if status == "PASS" else None


# ----------------------------------------------------------------------------------------- image fixtures
def _jpeg_bytes(rgb, quality=92, exif_orientation=None, mode="RGB"):
    from PIL import Image
    im = Image.fromarray(rgb)
    if mode == "CMYK":
        im = im.convert("CMYK")
    buf = io.BytesIO()
    kw = {"quality": quality}
    if exif_orientation:
        ex = Image.Exif()
        ex[274] = exif_orientation
        kw["exif"] = ex.tobytes()
    im.save(buf, format="JPEG", **kw)
    return buf.getvalue()


def _png_bytes(arr, mode=None):
    from PIL import Image
    im = Image.fromarray(arr)            # mode follows the array (RGBA / L / I;16); the 'mode' argument is deprecated
    if mode:
        assert im.mode == mode, (im.mode, mode)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _photo_like(h, w, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([(xx / w) * 200 + 20, (yy / h) * 180 + 30, ((xx + yy) / (w + h)) * 160 + 40], -1)
    base += rng.normal(0, 6, size=(h, w, 3)).astype(np.float32)
    return np.clip(base, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------------------------- checks
def check_imports():
    import cv2
    import numpy
    import onnxruntime
    import PIL
    return {"numpy": numpy.__version__, "pillow": PIL.__version__, "onnxruntime": onnxruntime.__version__,
            "opencv": cv2.__version__, "ort_providers": onnxruntime.get_available_providers()}


def check_models():
    from .model_parts import assemble_all
    from .model_store import verify_model
    assembled = assemble_all(ROOT / "models", log=lambda *_: None)
    manifest = json.loads((ROOT / "models" / "manifest.json").read_text(encoding="utf-8"))
    out = {}
    for name in manifest["files"]:
        t0 = time.perf_counter()
        info = verify_model(ROOT / "models" / name, ROOT / "models" / "manifest.json")
        out[name] = {"sha256": info["sha256"], "bytes": info["size_bytes"], "verify_seconds": round(time.perf_counter() - t0, 2)}
    return {"assembled_from_parts": assembled, "models": out}


def check_probes():
    """Run each ONNX model on a deterministic input and compare with the outputs recorded on the build machine
    (tests/data/model_probes.json; ONNX Runtime CPU, each model itself checked against PyTorch at export)."""
    import numpy as np
    import onnxruntime as ort
    ref_path = TESTDATA / "model_probes.json"
    if not ref_path.exists():
        return {"_skip": True, "reason": "tests/data/model_probes.json not present"}
    refs = json.loads(ref_path.read_text(encoding="utf-8"))
    out = {}
    for name, ref in refs["models"].items():
        rng = np.random.default_rng(ref["seed"])
        x = rng.normal(size=tuple(ref["input_shape"])).astype(np.float32)
        sess = ort.InferenceSession(str(ROOT / "models" / name), providers=["CPUExecutionProvider"])
        y = sess.run(None, {ref["input_name"]: x})[0].astype(np.float64)
        flat = y.reshape(-1)
        idx = np.asarray(ref["sample_index"], dtype=np.int64)
        got = flat[idx]
        exp = np.asarray(ref["sample_values"], dtype=np.float64)
        diff = float(np.abs(got - exp).max())
        mean_diff = abs(float(flat.mean()) - ref["mean"])
        out[name] = {"max_abs_diff_sampled": diff, "mean_diff": mean_diff, "tolerance": ref["tolerance"]}
        assert diff <= ref["tolerance"] and mean_diff <= ref["tolerance"], f"{name}: output differs from reference"
        del sess
    return out


def check_analyzer_and_samples(state):
    from .analysis import Analyzer, report_files
    from .config import load_config
    from .sysinfo import current_rss_mb, peak_rss_mb
    t0 = time.perf_counter()
    cfg = load_config(ROOT / "config.json")
    an = Analyzer(cfg)
    load_s = time.perf_counter() - t0
    state["analyzer"], state["config"] = an, cfg
    ref_path = TESTDATA / "product_reference.json"
    refs = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else {"images": {}}
    out = {"model_load_seconds": round(load_s, 2), "rss_after_load_mb": current_rss_mb(),
           "peak_rss_so_far_mb": peak_rss_mb(), "images": {}}
    worst = 0.0
    for p in sorted(TESTDATA.glob("*.png")):
        t1 = time.perf_counter()
        rep = an.analyze(p)
        dt = time.perf_counter() - t1
        files = report_files(rep, cfg.overlay_alpha)
        h, w = rep.image.rgb.shape[:2]
        if rep.primary_heatmap is not None:
            assert rep.primary_heatmap.shape == (h, w), "heatmap not in display coordinates"
        assert "result.json" in files and json.loads(files["result.json"])["summary"]["overall_decision"]
        scores = {r.key: r.score for r in rep.results}
        rec = {"seconds": round(dt, 2), "scores": scores, "overall": rep.summary["overall_decision"]}
        exp = refs["images"].get(p.name)
        if exp:
            for k, v in exp["scores"].items():
                if v is None or scores.get(k) is None:
                    assert v == scores.get(k), f"{p.name}/{k}: expected {v}, got {scores.get(k)}"
                else:
                    worst = max(worst, abs(v - scores[k]))
            rec["reference_overall"] = exp["overall"]
        out["images"][p.name] = rec
    out["max_abs_score_diff_vs_linux_reference"] = worst
    tol = refs.get("tolerance", 2e-3)
    assert worst <= tol, f"scores differ from the reference run by {worst} (> {tol})"
    return out


def check_formats(state):
    import numpy as np
    from .errors import ForgeryError
    an, cfg = state["analyzer"], state["config"]
    base = _photo_like(300, 400, seed=1)
    out = {}
    # EXIF orientation 6 => displayed image is rotated (W,H swapped)
    rep = an.analyze(_jpeg_bytes(base, exif_orientation=6), name="exif6.jpg")
    assert list(rep.image.rgb.shape[:2]) == [400, 300], "EXIF orientation not applied"
    out["exif_orientation_6"] = rep.image.metadata["display_size_wh"]
    rgba = np.dstack([base, np.full(base.shape[:2], 128, np.uint8)])
    rep = an.analyze(_png_bytes(rgba, "RGBA"), name="alpha.png")
    assert any("alpha" in w for w in rep.image.metadata["warnings"]), "alpha note missing"
    out["png_alpha"] = "composited on white"
    rep = an.analyze(_png_bytes(base[..., 0], "L"), name="gray.png")
    out["gray_png"] = rep.summary["overall_decision"]
    rep = an.analyze(_png_bytes((base[..., 0].astype(np.uint16) * 257), "I;16"), name="gray16.png")
    out["png_16bit"] = rep.summary["overall_decision"]
    rep = an.analyze(_jpeg_bytes(base, mode="CMYK"), name="cmyk.jpg")
    out["jpeg_cmyk"] = rep.summary["overall_decision"]
    wide = _photo_like(700, 4200, seed=2)
    rep = an.analyze(_jpeg_bytes(wide, 90), name="wide.jpg")
    if rep.primary_heatmap is not None:
        assert rep.primary_heatmap.shape == (700, 4200)
    out["non_square_4200x700"] = rep.summary["overall_decision"]
    errors = {}
    for label, data in {"truncated_jpeg": _jpeg_bytes(base)[:2000], "text_as_jpg": b"not an image at all" * 10,
                        "empty": b"", "tiny_8x8": _png_bytes(base[:8, :8])}.items():
        try:
            an.analyze(data, name=label + ".jpg")
            raise AssertionError(f"{label} was accepted")
        except ForgeryError as exc:
            errors[label] = str(exc)[:120]
    from PIL import Image
    for fmt in ("BMP", "GIF", "WEBP", "TIFF"):
        buf = io.BytesIO()
        try:
            Image.fromarray(base).save(buf, format=fmt)
        except Exception:
            continue
        try:
            an.analyze(buf.getvalue(), name="x." + fmt.lower())
            raise AssertionError(f"{fmt} was accepted")
        except ForgeryError as exc:
            errors[fmt] = str(exc)[:120]
    try:
        an.analyze(Path(tempfile.gettempdir()) / "khong_ton_tai_ifd_selftest.jpg")
        raise AssertionError("missing path accepted")
    except ForgeryError as exc:
        errors["missing_path"] = str(exc)[:120]
    small = cfg.to_json()
    small["limits"]["max_file_mb"] = 0.1
    from .config import Config
    try:
        an.analyze(_png_bytes(_photo_like(600, 600, 3)), name="big.png", config=Config(small, cfg.base_dir))
        raise AssertionError("oversize accepted")
    except ForgeryError as exc:
        errors["oversize"] = str(exc)[:120]
    out["rejected_with_message"] = errors
    return out


def check_config_errors():
    from .config import Config, load_config
    from .errors import ForgeryError
    cfg = load_config(ROOT / "config.json")
    bad = {}
    cases = {
        "low_gt_high": lambda d: d["detectors"]["mesorch"]["thresholds"].update(low=0.9, high=0.1),
        "threshold_out_of_range": lambda d: d["detectors"]["mesorch"]["thresholds"].update(high=1.5),
        "unknown_type": lambda d: d["detectors"]["mesorch"].update(type="nope"),
        "nan_limit": lambda d: d["limits"].update(max_file_mb=float("nan")),
    }
    for name, mutate in cases.items():
        d = cfg.to_json()
        mutate(d)
        try:
            Config(d, cfg.base_dir)
            raise AssertionError(f"{name} accepted")
        except ForgeryError as exc:
            bad[name] = str(exc)[:120]
    tmp = Path(tempfile.mkdtemp(prefix="ifd_cfg_"))
    try:
        (tmp / "config.json").write_text("{ not json", encoding="utf-8")
        try:
            load_config(tmp / "config.json")
            raise AssertionError("broken JSON accepted")
        except ForgeryError as exc:
            bad["broken_json"] = str(exc)[:120]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return bad


def check_corrupt_model():
    from .errors import ForgeryError
    from .model_store import verify_model
    tmp = Path(tempfile.mkdtemp(prefix="ifd_model_"))
    try:
        name = "face_detection_yunet_2023mar.onnx"
        shutil.copyfile(ROOT / "models" / name, tmp / name)
        shutil.copyfile(ROOT / "models" / "manifest.json", tmp / "manifest.json")
        data = bytearray((tmp / name).read_bytes())
        data[len(data) // 2] ^= 0xFF
        (tmp / name).write_bytes(bytes(data))
        out = {}
        try:
            verify_model(tmp / name, tmp / "manifest.json")
            raise AssertionError("corrupt model accepted")
        except ForgeryError as exc:
            out["flipped_byte"] = str(exc)[:120]
        (tmp / name).write_bytes(bytes(data[:-10]))
        try:
            verify_model(tmp / name, tmp / "manifest.json")
            raise AssertionError("truncated model accepted")
        except ForgeryError as exc:
            out["truncated"] = str(exc)[:120]
        (tmp / name).unlink()
        try:
            verify_model(tmp / name, tmp / "manifest.json")
            raise AssertionError("missing model accepted")
        except ForgeryError as exc:
            out["missing"] = str(exc)[:120]
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _light_config(tmp: Path) -> Path:
    """Only the two small detectors (absolute model paths): tests CLI mechanics without loading a second copy of
    the ~2.5 GB model set inside this process (8 GB machines)."""
    data = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    for det in data["detectors"].values():
        det["enabled"] = det["type"] in ("pscc_onnx", "commfor_onnx")
        det["model"] = str((ROOT / det["model"]).resolve())
        if "face_model" in det:
            det["face_model"] = str((ROOT / det["face_model"]).resolve())
    p = tmp / "config_light.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _make_cli_inputs(tmp: Path) -> Path:
    src = tmp / "ảnh vào"            # non-ASCII + space in the path on purpose
    src.mkdir()
    (src / "a.jpg").write_bytes(_jpeg_bytes(_photo_like(240, 320, 5)))
    (src / "b.png").write_bytes(_png_bytes(_photo_like(200, 260, 6)))
    (src / "hong.jpg").write_bytes(b"\xff\xd8 broken")
    return src


def check_cli():
    """CLI mechanics (run folders, exit codes, CSV/JSON, no overwrite) in-process with the light config."""
    import contextlib
    from .cli import main as cli_main
    tmp = Path(tempfile.mkdtemp(prefix="ifd_cli_"))
    err = io.StringIO()
    try:
        cfg = _light_config(tmp)
        src = _make_cli_inputs(tmp)
        out_dir = tmp / "kết quả"
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc1 = cli_main([str(src), "--output", str(out_dir), "--config", str(cfg)])
        assert rc1 == 1, f"expected exit 1 (one bad file), got {rc1}; stderr: {err.getvalue()[-600:]}"
        latest = json.loads((out_dir / "latest_run.json").read_text(encoding="utf-8"))
        run1 = out_dir / latest["run_directory"]
        res = json.loads((run1 / "results.json").read_text(encoding="utf-8"))
        assert res["counts"] == {"ok": 2, "error": 1}, res["counts"]
        csv_text = (run1 / "results.csv").read_text(encoding="utf-8-sig")
        assert csv_text.count("\n") >= 4
        for d in [r["output_dir"] for r in res["results"] if r["status"] == "ok"]:
            assert (out_dir / d / "result.json").is_file()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc2 = cli_main([str(src / "a.jpg"), "--output", str(out_dir), "--config", str(cfg)])
        assert rc2 == 0, f"single good image should exit 0, got {rc2}; stderr: {err.getvalue()[-600:]}"
        latest2 = json.loads((out_dir / "latest_run.json").read_text(encoding="utf-8"))
        assert latest2["run_id"] != latest["run_id"] and run1.is_dir(), "previous run overwritten"
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc3 = cli_main([str(tmp / "khong_co"), "--output", str(out_dir), "--config", str(cfg)])
        assert rc3 == 2, f"missing input should exit 2, got {rc3}"
        return {"run1": latest["run_id"], "run2": latest2["run_id"], "exit_codes": [rc1, rc2, rc3],
                "csv_columns": csv_text.splitlines()[0]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_company_eval():
    """Validation kit for labelled company images (KIEM_DINH_CONG_TY.bat) in-process with the light config."""
    import contextlib
    from .company_eval import main as kit_main
    tmp = Path(tempfile.mkdtemp(prefix="ifd_kit_"))
    err, out = io.StringIO(), io.StringIO()
    try:
        cfg = _light_config(tmp)
        data = tmp / "dữ liệu công ty"
        (data / "Thật" / "máy ảnh").mkdir(parents=True)
        (data / "Sửa" / "ghép").mkdir(parents=True)
        (data / "Thật" / "máy ảnh" / "t1.jpg").write_bytes(_jpeg_bytes(_photo_like(240, 320, 7)))
        (data / "Thật" / "máy ảnh" / "t2.png").write_bytes(_png_bytes(_photo_like(200, 260, 8)))
        shutil.copy(TESTDATA / "authentic1.png", data / "Thật" / "máy ảnh" / "a1.png")
        shutil.copy(TESTDATA / "splice1.png", data / "Sửa" / "ghép" / "s1.png")
        shutil.copy(TESTDATA / "copymove2.png", data / "Sửa" / "ghép" / "c2.png")
        (data / "Sửa" / "ghép" / "hong.jpg").write_bytes(b"\xff\xd8 broken")
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            rc = kit_main([str(data), "--output", str(tmp / "kết quả"), "--config", str(cfg)])
        assert rc == 1, f"expected exit 1 (one unreadable file), got {rc}; stderr: {err.getvalue()[-600:]}"
        work = next((tmp / "kết quả" / "kiem_dinh").iterdir())
        rep = next(p for p in work.iterdir() if p.is_dir())
        m = json.loads((rep / "metrics.json").read_text(encoding="utf-8"))
        assert m["counts"] == {"found": 6, "ok": 5, "errors": 1, "real": 3, "fake": 2}, m["counts"]
        assert (rep / "bao_cao_kiem_dinh.html").is_file() and (rep / "du_doan.csv").is_file()
        return {"counts": m["counts"], "product_by_type": list(m["product"]["by_type"]),
                "report_files": sorted(p.name for p in rep.iterdir())}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_cli_full_subprocess():
    """The real folder-analysis path (what PHAN_TICH_THU_MUC.bat runs): a fresh process, default config, all models.
    Runs BEFORE this process loads its own models so that 8 GB machines are not asked to hold two copies."""
    import subprocess
    tmp = Path(tempfile.mkdtemp(prefix="ifd_clifull_"))
    try:
        src = _make_cli_inputs(tmp)
        out_dir = tmp / "kết quả"
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, "-X", "utf8", "-m", "ifd", str(src), "--output", str(out_dir)],
                           cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
        dt = time.perf_counter() - t0
        assert r.returncode == 1, f"expected exit 1, got {r.returncode}; stderr: {r.stderr[-800:]}"
        latest = json.loads((out_dir / "latest_run.json").read_text(encoding="utf-8"))
        res = json.loads((out_dir / latest["run_directory"] / "results.json").read_text(encoding="utf-8"))
        assert res["counts"] == {"ok": 2, "error": 1}, res["counts"]
        dets = sorted({d["detector"] for x in res["results"] if x["status"] == "ok" for d in x["detectors"]})
        assert dets == ["commfor_aigen", "face_clip", "mesorch"], dets
        return {"seconds_incl_model_load": round(dt, 1), "detectors": dets, "counts": res["counts"]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _http(url, data=None, headers=None, method=None):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def check_web(shared):
    from .webapp.server import LocalHTTPServer, State, free_port, make_handler, main as server_main
    state = State(ROOT / "config.json", analyzer=shared.get("analyzer"))
    state.load()
    assert state.analyzer is not None, state.error
    port = free_port("127.0.0.1")
    httpd = LocalHTTPServer(("127.0.0.1", port), make_handler(state, {f"127.0.0.1:{port}", f"localhost:{port}"}))
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{port}"
    tok = {"X-IFD-Token": state.token}
    out = {}
    try:
        code, _, body = _http(base + "/")
        assert code == 200 and b"<title>" in body
        code, _, body = _http(base + "/api/status", headers=tok)
        st = json.loads(body)
        assert code == 200 and st["ready"], body[:200]
        out["status_detectors"] = [d["key"] for d in st["detectors"]]
        assert _http(base + "/api/status")[0] == 403, "missing token accepted"
        assert _http(base + "/api/status", headers={**tok, "Host": "evil.example:80"})[0] == 403, "bad Host accepted"
        img = _jpeg_bytes(_photo_like(360, 480, 7))
        code, _, body = _http(base + "/api/analyze", data=img, method="POST",
                              headers={**tok, "X-File-Name": urllib.parse.quote("ảnh thử.jpg"),
                                       "Content-Type": "application/octet-stream"})
        assert code == 200, body[:300]
        resp = json.loads(body)
        rid = resp["result_id"]
        out["analyze_overall"] = resp["report"]["summary"]["overall_decision"]
        for name in resp["files"] + ["results.zip"]:
            c, h, b = _http(f"{base}/api/result/{rid}/{name}?download=1", headers=tok)
            assert c == 200 and len(b) > 0, name
            assert "attachment" in h.get("Content-Disposition", ""), name
        out["downloads_ok"] = resp["files"] + ["results.zip"]
        code, _, body = _http(base + "/api/analyze", data=b"\xff\xd8broken", method="POST", headers=tok)
        assert code == 422, code
        out["corrupt_upload"] = json.loads(body)["error"][:100]
        limit = int(state.config.limits["max_file_mb"] * 1024 * 1024)
        req = urllib.request.Request(base + "/api/analyze", data=b"0" * 16, method="POST",
                                     headers={**tok, "Content-Length": str(limit + 1)})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                code = r.status
        except urllib.error.HTTPError as e:
            code = e.code
        except (ConnectionError, urllib.error.URLError):
            code = 413  # server may close before the client finishes sending
        assert code == 413, code
        out["oversize_upload"] = code
    finally:
        httpd.shutdown()
        httpd.server_close()
    # port already in use -> clear message + exit code 2
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    busy = blocker.getsockname()[1]
    try:
        rc = server_main(["--port", str(busy), "--no-browser"])
        assert rc == 2, f"port in use should exit 2, got {rc}"
        out["port_in_use_exit_code"] = rc
    finally:
        blocker.close()
    return out


def check_timing(state):
    """Warm timings of the loaded analyzer: analysis only and end-to-end incl. the files the user downloads."""
    import numpy as np
    from .analysis import report_files
    from .sysinfo import current_rss_mb, peak_rss_mb
    an, cfg = state["analyzer"], state["config"]
    img12 = _jpeg_bytes(_photo_like(3000, 4000, 9), 92)
    img2 = _jpeg_bytes(_photo_like(1080, 1920, 10), 92)
    out = {}
    for label, data in (("2MP_1920x1080", img2), ("12MP_4000x3000", img12)):
        times, e2e = [], []
        for _ in range(5):
            t0 = time.perf_counter()
            rep = an.analyze(data, name=label + ".jpg")
            t1 = time.perf_counter()
            report_files(rep, cfg.overlay_alpha)
            t2 = time.perf_counter()
            times.append(t1 - t0)
            e2e.append(t2 - t0)
        per_det = {r.key: round(next(iter(r.timing.values()), 0.0), 2) for r in rep.results}
        out[label] = {"seconds": [round(x, 2) for x in times],
                      "analyze_p50_s": round(float(np.percentile(times, 50)), 2),
                      "analyze_p95_s": round(float(np.percentile(times, 95)), 2),
                      "end_to_end_p50_s": round(float(np.percentile(e2e, 50)), 2),
                      "end_to_end_p95_s": round(float(np.percentile(e2e, 95)), 2),
                      "per_detector_last": per_det}
    out["rss_now_mb"] = current_rss_mb()
    out["peak_rss_mb"] = peak_rss_mb()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Tự kiểm tra ứng dụng trên máy này (không mock).")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "selftest")
    ap.add_argument("--quick", action="store_true", help="bỏ qua đo thời gian ảnh 12MP")
    a = ap.parse_args(argv)
    from . import __version__
    from .sysinfo import describe
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = a.out / f"{stamp}_{platform.node() or 'host'}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Tự kiểm tra ifd {__version__} — kết quả ghi vào {out_dir}", flush=True)
    rec = Recorder()
    state = {}
    rec.run("imports", check_imports)
    rec.run("model_files_sha256", check_models)
    rec.run("model_numeric_parity_vs_build_machine", check_probes)
    rec.run("cli_folder_full_pipeline_subprocess", check_cli_full_subprocess)
    ok = rec.run("analyzer_load_and_reference_samples", check_analyzer_and_samples, state)
    if ok is not None:
        rec.run("input_formats_and_rejections", check_formats, state)
    rec.run("config_validation", check_config_errors)
    rec.run("corrupt_or_missing_model", check_corrupt_model)
    rec.run("cli_runs_and_artifacts", check_cli)
    rec.run("company_validation_kit", check_company_eval)
    rec.run("web_ui_api_downloads_security_port", check_web, state)
    if ok is not None and not a.quick:
        rec.run("cpu_timing", check_timing, state)
    counts = {s: sum(c["status"] == s for c in rec.checks) for s in ("PASS", "FAIL", "SKIP")}
    report = {"app_version": __version__, "created_utc": stamp, "environment": describe(),
              "counts": counts, "checks": rec.checks}
    (out_dir / "selftest.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str) + "\n",
                                           encoding="utf-8")
    lines = [f"ifd {__version__} self-test {stamp}", json.dumps(report["environment"], ensure_ascii=False), ""]
    lines += [f"{c['status']:4s}  {c['check']}  ({c['seconds']}s)" for c in rec.checks]
    lines += ["", f"PASS={counts['PASS']} FAIL={counts['FAIL']} SKIP={counts['SKIP']}"]
    (out_dir / "selftest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-1:]), flush=True)
    print(f"Báo cáo: {out_dir / 'selftest.txt'}", flush=True)
    return 0 if counts["FAIL"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
