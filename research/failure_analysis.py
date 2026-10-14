"""Where does the shipped product fail on the final test?  Breakdowns that the per-task metrics do not show.

    python research/failure_analysis.py --run test_v2.jsonl --manifest M ... --config config.json \
        --splits test test_korus test_cocoglide --out reports/test/failure_analysis.json

* false "Nghi vấn" on REAL images per detector, by source x resolution bucket x encoding;
* missed fakes (overall "Chưa thấy dấu hiệu rõ") by source and sub-type (operation / generator / DFF type);
* face detector: share of fakes where no face was found (so the face task could not fire);
* the most confident errors (image ids only, for manual inspection on the research machine).
Decisions are re-derived from stored scores with the given config (same rule as the app).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ORDER = {"no_clear_sign": 1, "needs_review": 2, "suspicious": 3}


def source(i: str) -> str:
    if i.startswith("mei_"):
        return "ctrl_mei"
    if i.startswith("korus_"):
        return "korus"
    if i.startswith("cocoglide"):
        return "cocoglide"
    if i.startswith("openfake_"):
        return "openfake"
    if i.startswith("dff_"):
        return "dff_" + i.split("_")[1]
    return "other"


def mp_bucket(m: dict) -> str:
    w, h = m.get("width"), m.get("height")
    if not w or not h:
        return "unknown"
    mp = w * h / 1e6
    return "<0.5MP" if mp < 0.5 else "0.5-2MP" if mp < 2 else "2-8MP" if mp < 8 else ">=8MP"


def subtype(m: dict, src: str) -> str:
    if src == "openfake":
        return str(m.get("generator") or m.get("real_source"))
    if src == "ctrl_mei":
        return f"{m.get('operation')}/{m.get('size_bucket')}"
    if src == "korus":
        return str(m.get("camera"))
    return str(m.get("operation") or src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, action="append", required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    th = {k: v["thresholds"] for k, v in cfg["detectors"].items() if v.get("enabled")}
    meta = {}
    for mp in a.manifest:
        for l in mp.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                meta.setdefault(r["image_id"], r)
    recs = []
    for l in a.run.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        m = meta.get(r["image_id"])
        if m is None or m.get("split") not in a.splits or m.get("exclude") or r.get("error"):
            continue
        dec = {}
        for k, t in th.items():
            d = r["detectors"].get(k)
            s = None if d is None else d["score"]
            dec[k] = ("not_applicable" if s is None else
                      "suspicious" if s >= t["high"] else "needs_review" if s >= t["low"] else "no_clear_sign")
        app = [v for v in dec.values() if v in ORDER]
        overall = max(app, key=lambda v: ORDER[v], default="no_clear_sign")
        recs.append({"id": r["image_id"], "m": m, "src": source(r["image_id"]), "label": int(m["label"]),
                     "scores": {k: (r["detectors"].get(k) or {}).get("score") for k in th}, "dec": dec, "overall": overall,
                     "faces": (r["detectors"].get("face_clip") or {}).get("details", {}).get("faces_found")})
    out = {"run": str(a.run), "config": str(a.config), "n": len(recs), "thresholds": th}
    # 1) false "Nghi vấn" on real images, per detector, by source / resolution / encoding
    fp = {}
    for k in th:
        tab = defaultdict(lambda: [0, 0])
        for r in recs:
            if r["label"] != 0:
                continue
            key = f"{r['src']} | {mp_bucket(r['m'])} | {r['m'].get('encoding', '?')}"
            tab[key][1] += 1
            tab[key][0] += r["dec"][k] == "suspicious"
        fp[k] = {kk: {"n": v[1], "suspicious": v[0], "rate": v[0] / v[1]} for kk, v in sorted(tab.items())}
    out["real_flagged_suspicious_by_detector_source_resolution_encoding"] = fp
    # 2) missed fakes by source / subtype
    miss = defaultdict(lambda: [0, 0])
    for r in recs:
        if r["label"] != 1:
            continue
        key = f"{r['src']} | {subtype(r['m'], r['src'])}"
        miss[key][1] += 1
        miss[key][0] += r["overall"] == "no_clear_sign"
    out["fakes_missed_overall_no_clear_sign_by_subtype"] = {k: {"n": v[1], "missed": v[0], "rate": v[0] / v[1]}
                                                           for k, v in sorted(miss.items())}
    # 3) face task: fakes where no face was found
    if "face_clip" in th:
        nf = defaultdict(lambda: [0, 0])
        for r in recs:
            if r["src"].startswith("dff_"):
                nf[r["src"]][1] += 1
                nf[r["src"]][0] += r["scores"]["face_clip"] is None
        out["face_not_found_by_dff_source"] = {k: {"n": v[1], "no_face": v[0], "rate": v[0] / v[1]} for k, v in sorted(nf.items())}
    # 4) most confident errors per detector (ids only)
    worst = {}
    for k in th:
        fps = sorted((r for r in recs if r["label"] == 0 and r["scores"][k] is not None),
                     key=lambda r: -r["scores"][k])[: a.top]
        worst[k] = {"highest_scored_real_images": [{"id": r["id"], "src": r["src"], "score": r["scores"][k]} for r in fps]}
    out["most_confident_false_positives"] = worst
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("written", a.out, "n", len(recs))


if __name__ == "__main__":
    main()
