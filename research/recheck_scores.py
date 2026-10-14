"""Post-freeze equivalence check: re-analyse a seeded random sample of images from a finished product run with
the CURRENT code + config and require identical scores/decisions (proves the shipped build reproduces the run).

  python research/recheck_scores.py --run test_v2.jsonl --manifest M ... --config config.json --n 100 --out recheck.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifd.analysis import Analyzer  # noqa: E402
from ifd.config import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, action="append", required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    paths = {}
    for m in a.manifest:
        for l in m.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                paths.setdefault(r["image_id"], m.parent / r["image"])
    recs = [json.loads(l) for l in a.run.read_text(encoding="utf-8").splitlines() if l.strip()]
    recs = [r for r in recs if not r.get("error") and r["image_id"] in paths]
    random.Random(a.seed).shuffle(recs)
    sample = recs[: a.n]
    an = Analyzer(load_config(a.config))
    worst, mismatches = 0.0, []
    for r in sample:
        rep = an.analyze(paths[r["image_id"]])
        now = {x.key: (x.score, x.decision) for x in rep.results}
        for k, d in r["detectors"].items():
            s_old, dec_old = d["score"], d["decision"]
            s_new, dec_new = now[k]
            if (s_old is None) != (s_new is None):
                mismatches.append({"image_id": r["image_id"], "detector": k, "old": s_old, "new": s_new})
                continue
            if s_old is not None:
                worst = max(worst, abs(s_old - s_new))
            if dec_old != dec_new:
                mismatches.append({"image_id": r["image_id"], "detector": k, "old_decision": dec_old, "new_decision": dec_new})
        if r["overall_decision"] != rep.summary["overall_decision"]:
            mismatches.append({"image_id": r["image_id"], "overall_old": r["overall_decision"], "overall_new": rep.summary["overall_decision"]})
    out = {"run": str(a.run), "n": len(sample), "max_abs_score_diff": worst, "tolerance": a.tol,
           "mismatches": mismatches, "ok": worst <= a.tol and not mismatches}
    a.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "mismatches"}, indent=1), "mismatches:", len(mismatches))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
