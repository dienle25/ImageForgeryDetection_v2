"""Command line: one image or a folder; every run gets its own folder runs/<run_id>/."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .analysis import Analyzer, report_files
from .config import DEFAULT_CONFIG, load_config
from .errors import ForgeryError

EXTS = (".jpg", ".jpeg", ".png")
log = logging.getLogger("ifd.cli")


def safe_name(name: str) -> str:
    import re
    return re.sub(r"[^\w.-]+", "_", name, flags=re.UNICODE).strip("._")[:80] or "image"


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ifd",
                                     description="Phân tích ảnh JPEG/PNG trên CPU tại máy (chỉnh sửa cục bộ...).")
    parser.add_argument("input", type=Path, help="Ảnh JPEG/PNG hoặc thư mục ảnh")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Thư mục kết quả (mặc định: outputs)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--recursive", action="store_true", help="Đọc cả thư mục con")
    parser.add_argument("--threads", type=int, help="Số luồng CPU (0 = tự động)")
    parser.add_argument("--no-images", action="store_true",
                        help="Chỉ ghi điểm/kết luận (result.json, CSV), không ghi ảnh bản đồ nhiệt — nhanh hơn nhiều với ảnh lớn")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    try:
        config = load_config(args.config)
        if args.threads is not None:
            config.data.setdefault("runtime", {})["threads"] = args.threads
            config.validate()
        source, output = args.input.resolve(), args.output.resolve()
        if not source.exists():
            raise ForgeryError(f"Đường dẫn đầu vào không tồn tại: {args.input}")
        if source == output or (source.is_dir() and output in source.parents):
            raise ForgeryError("Thư mục kết quả không được là thư mục đầu vào hoặc thư mục cha của nó.")
        if source.is_dir():
            it = source.rglob("*") if args.recursive else source.iterdir()
            paths = sorted(p for p in it if p.is_file() and p.suffix.lower() in EXTS
                           and not p.resolve().is_relative_to(output))
        else:
            paths = [source]
        if not paths:
            raise ForgeryError("Không tìm thấy file .jpg/.jpeg/.png để xử lý.")
        analyzer = Analyzer(config)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        run_dir = output / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        results, rows = [], []
        det_keys = list(analyzer.detectors)
        for path in paths:
            rel = path.relative_to(source) if source.is_dir() else Path(path.name)
            ident = hashlib.sha256(rel.as_posix().encode("utf-8")).hexdigest()[:10]
            folder = run_dir / f"{safe_name(path.stem)}_{path.suffix[1:].lower()}_{ident}"
            row = {"input": rel.as_posix(), "status": "error", "overall_decision": "", "overall_decision_vi": ""}
            for k in det_keys:
                row[f"{k}_score"] = ""
                row[f"{k}_decision"] = ""
            row.update(faces_found="", output_dir="", error="")
            try:
                report = analyzer.analyze(path)
                files = report_files(report, config.overlay_alpha, images=not args.no_images)
                folder.mkdir(parents=True)
                for name, data in files.items():
                    _atomic_write(folder / name, data)
                data = json.loads(files["result.json"])
                row.update(status="ok", overall_decision=data["summary"]["overall_decision"],
                           overall_decision_vi=data["summary"]["overall_decision_vi"],
                           output_dir=folder.relative_to(output).as_posix())
                for d in data["detectors"]:
                    row[f"{d['detector']}_score"] = "" if d["score"] is None else d["score"]
                    row[f"{d['detector']}_decision"] = d["decision"]
                    if d["task"] == "face_deepfake":
                        row["faces_found"] = d["details"].get("faces_found", 0)
                results.append({"input": row["input"], "status": "ok", "output_dir": row["output_dir"],
                                "sha256": data["image"]["sha256"], "summary": data["summary"],
                                "detectors": [{k: d[k] for k in ("detector", "task", "score", "decision")} for d in data["detectors"]]})
            except ForgeryError as exc:
                row["error"] = str(exc)
                results.append({"input": row["input"], "status": "error", "error": str(exc), "output_dir": None})
            except Exception as exc:  # unexpected: keep going, but log the traceback
                log.exception("Lỗi không mong đợi khi xử lý %s", path)
                row["error"] = f"Lỗi không mong đợi: {type(exc).__name__}: {exc}"
                results.append({"input": row["input"], "status": "error", "error": row["error"], "output_dir": None})
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        counts = {"ok": sum(r["status"] == "ok" for r in results), "error": sum(r["status"] == "error" for r in results)}
        report = {"run_id": run_id, "run_directory": run_dir.relative_to(output).as_posix(),
                  "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "input": str(source), "config_file": str(Path(args.config).resolve()),
                  "models": analyzer.model_info, "model_load_seconds": analyzer.load_seconds,
                  "counts": counts, "results": results,
                  "note": "Mỗi lượt chạy có thư mục riêng; kết quả lượt trước được giữ nguyên, không bị ghi đè."}
        blob = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        _atomic_write(run_dir / "results.json", blob)
        import io
        sbuf = io.StringIO()
        writer = csv.DictWriter(sbuf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        csv_bytes = ("﻿" + sbuf.getvalue()).encode("utf-8")
        _atomic_write(run_dir / "results.csv", csv_bytes)
        _atomic_write(output / "latest_run.json", (json.dumps({"run_id": run_id, "run_directory": report["run_directory"],
                      "counts": counts}, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        print(f"Kết quả: {run_dir}", file=sys.stderr)
        return 1 if counts["error"] else 0
    except ForgeryError as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"Lỗi hệ thống tệp: {exc}", file=sys.stderr)
        return 2
