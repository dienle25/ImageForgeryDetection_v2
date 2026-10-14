"""BEFORE/AFTER on the 64 audit probes (F01-F04): v1 (PSCC-Net, torch, from reports/audit_probes.json of the
audited ZIP) vs v2 (shipped Analyzer: Mesorch + Community Forensics + face detector, ONNX).

DIAGNOSTIC ONLY - author-selected samples, scripted derivatives and procedural graphics; not an accuracy estimate.
Every input is regenerated deterministically and its SHA-256 must equal the audit record before it is scored.

  python research/before_after_probes.py --baseline AUDITED_PROJECT_DIR --pscc-samples PSCC_REPO/sample \
      --config config.json --out before_after_probes.json
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifd.analysis import Analyzer  # noqa: E402
from ifd.config import load_config  # noqa: E402

V1_LOW, V1_HIGH = 0.35, 0.65


def enc(image, fmt="PNG", **o):
    b = io.BytesIO()
    image.save(b, format=fmt, **o)
    return b.getvalue()


def manipulations(image, donor):
    yield "jpeg_q90", enc(image, "JPEG", quality=90)
    yield "jpeg_q50", enc(image, "JPEG", quality=50)
    yield "resize_x2", enc(image.resize((512, 512), Image.Resampling.BILINEAR))
    yield "brightness_x1.1", enc(ImageEnhance.Brightness(image).enhance(1.1))
    for size in (16, 64):
        box = (150, 140, 150 + size, 140 + size)
        for kind in ("copy", "splice", "blur_replace"):
            out = image.copy()
            if kind == "copy":
                patch = image.crop((20, 30, 20 + size, 30 + size))
            elif kind == "splice":
                patch = donor.crop((20, 30, 20 + size, 30 + size))
            else:
                patch = image.crop(box).filter(ImageFilter.GaussianBlur(8))
            out.paste(patch, box[:2])
            yield f"{kind}_{size}", enc(out)


def v1_label(score):
    return "suspicious" if score >= V1_HIGH else ("needs_review" if score >= V1_LOW else "no_clear_sign")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--pscc-samples", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    audit = json.loads((a.baseline / "reports" / "audit_probes.json").read_text())
    ref = {r["case"]: r for r in audit["records"]}
    inputs = {}
    samples = {n: (a.pscc_samples / n).read_bytes() for n in sorted(os.listdir(a.pscc_samples))}
    for n, raw in samples.items():
        inputs[n] = raw
    imgs = {n: Image.open(io.BytesIO(raw)).convert("RGB") for n, raw in samples.items()}
    scale_src = {}
    for src in ("authentic1.png", "authentic2.png", "authentic3.png"):
        for case, raw in manipulations(imgs[src], imgs["authentic4.png"]):
            inputs[f"{src}:{case}"] = raw
            if src == "authentic1.png" and case in ("copy_64", "splice_64"):
                scale_src[case] = Image.open(io.BytesIO(raw)).convert("RGB")
    scale_src["authentic"] = imgs["authentic1.png"]
    for name, image in scale_src.items():
        raw = enc(image.resize((768, 768), Image.Resampling.BILINEAR))
        for side in (128, 256, 512, 768):
            inputs[f"scale:{name}:{side}"] = raw     # v2 has no max_side knob: same bytes -> same result
    proc = a.baseline / "examples" / "inputs"
    for p in sorted(proc.iterdir()):
        inputs[f"procedural:{p.name}"] = p.read_bytes()
    an = Analyzer(load_config(a.config))
    rows, cache = [], {}
    for case, raw in inputs.items():
        r = ref.get(case)
        if r is None:
            continue
        sha = hashlib.sha256(raw).hexdigest()
        if sha != r["input_sha256"]:
            raise SystemExit(f"input bytes differ from audit record for {case}")
        if sha not in cache:
            rep = an.analyze(raw, name=case.split(":")[-1])
            dets = {x.key: {"score": x.score, "decision": x.decision} for x in rep.results}
            heat = rep.primary_heatmap
            cache[sha] = {"v2_overall": rep.summary["overall_decision"], "v2_detectors": dets,
                          "v2_fraction_map_ge_0_5": None if heat is None else float((heat >= 0.5).mean())}
        rows.append({"case": case, "group": case.split(":")[0], "v1_score": r["score"], "v1_decision": v1_label(r["score"]),
                     "v1_fraction_map_ge_0_5": r["fraction_map_ge_0_5"], **cache[sha]})
        print(case, round(r["score"], 4), rows[-1]["v2_detectors"].get("mesorch"), rows[-1]["v2_overall"], flush=True)
    out = {"purpose": "BEFORE (v1 PSCC-Net) vs AFTER (v2) on the audit probes - diagnostic only, not accuracy",
           "v1_thresholds": {"low": V1_LOW, "high": V1_HIGH}, "v2_config": str(a.config), "n": len(rows), "records": rows}
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
