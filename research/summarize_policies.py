import json, sys, collections
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from ifd_core.metrics import auroc, average_precision, threshold_for_fpr, confusion, bootstrap_ci
man = {json.loads(l)["image_id"]: json.loads(l) for l in open(sys.argv[1])}
recs = [json.loads(l) for l in open(sys.argv[2])]
# dedupe identical decoded images (same source + same post-processing authentic pairs)
seen, rows = set(), []
for r in recs:
    h = man[r["image_id"]]["decoded_rgb_sha256"]
    if h in seen: continue
    seen.add(h); rows.append(r)
pols = [k for k in rows[0] if isinstance(rows[0][k], dict)]
y = np.array([r["label"] for r in rows]); g = np.array([man[r["image_id"]]["group_id"] for r in rows])
print(f"n={len(rows)} (pos={y.sum()}, neg={(1-y).sum()}, groups={len(set(g))})")
print(f"{'policy':22s} {'AUROC':>6s} {'95%CI':>13s} {'AP':>6s} {'TPR@5%':>7s} {'FPR@.5':>7s} {'TPR@.5':>7s} {'pixF1@.5':>8s} {'pixIoU':>7s} {'pixAUC':>6s} {'authFrac':>8s} {'sec':>5s}")
for p in pols:
    s = np.array([r[p]["score"] for r in rows])
    ci = bootstrap_ci(auroc, y, s, g, n=400)
    t5 = threshold_for_fpr(y, s, 0.05)
    c5 = confusion(y, s, t5); c50 = confusion(y, s, 0.5)
    pix = [r[p]["pixel"] for r in rows if r["label"] == 1 and "pixel" in r[p]]
    f1 = np.mean([q["f1"] or 0 for q in pix]); iou = np.mean([q["iou"] or 0 for q in pix])
    pauc = np.mean([q["auc"] for q in pix if q["auc"] is not None])
    afrac = np.mean([r[p]["pixel"]["frac_pred"] for r in rows if r["label"] == 0])
    sec = np.mean([r[p]["seconds"] for r in rows])
    print(f"{p:22s} {auroc(y,s):6.3f} [{ci[0]:.3f},{ci[1]:.3f}] {average_precision(y,s):6.3f} {c5['recall']:7.3f} {c50['fpr']:7.3f} {c50['recall']:7.3f} {f1:8.3f} {iou:7.3f} {pauc:6.3f} {afrac:8.3f} {sec:5.2f}")
if len(sys.argv) > 3:
    key = sys.argv[3]
    print("\nAUROC by", key)
    cats = collections.defaultdict(list)
    for r in rows:
        m = man[r["image_id"]]
        cats[m[key]].append(r)
    for cat, rs in sorted(cats.items()):
        # pair each category of positives with all negatives
        neg = [r for r in rows if r["label"] == 0]
        pos = [r for r in rs if r["label"] == 1]
        if not pos: continue
        line = f"{str(cat):18s} n+={len(pos):3d} "
        for p in pols:
            yy = np.r_[np.ones(len(pos)), np.zeros(len(neg))]; ss = np.r_[[r[p]['score'] for r in pos], [r[p]['score'] for r in neg]]
            line += f"{p[:12]}={auroc(yy, ss):.3f} "
        print(line)
