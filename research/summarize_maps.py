import json, sys, collections
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from ifd_core.metrics import auroc, average_precision, bootstrap_ci, threshold_for_fpr, confusion
man = {json.loads(l)["image_id"]: json.loads(l) for l in open(sys.argv[1])}
recs = [json.loads(l) for l in open(sys.argv[2])]
seen, rows = set(), []
for r in recs:
    if "error" in r: print("ERR", r["image_id"], r["error"][:100]); continue
    h = man[r["image_id"]]["decoded_rgb_sha256"]
    if h in seen: continue
    seen.add(h); rows.append(r)
y = np.array([r["label"] for r in rows]); g = np.array([man[r["image_id"]].get("group_id", r["image_id"]) for r in rows])
print(f"{sys.argv[2].split('/')[-1]} n={len(rows)} pos={y.sum()} groups={len(set(g))} mean_sec={np.mean([r['seconds'] for r in rows]):.2f}")
for stat in ["max", "p999", "p99", "top1pct_mean", "mean", "frac05"]:
    s = np.array([r["derived"][stat] for r in rows])
    ci = bootstrap_ci(auroc, y, s, g, n=300)
    t5 = threshold_for_fpr(y, s, 0.05); c5 = confusion(y, s, t5)
    print(f"  image score={stat:13s} AUROC={auroc(y,s):.3f} [{ci[0]:.3f},{ci[1]:.3f}] AP={average_precision(y,s):.3f} TPR@FPR5%={c5['recall']:.3f}")
pix = [r["pixel"] for r in rows if r["label"] == 1 and "pixel" in r and "f1" in r["pixel"]]
auth = [r["pixel"]["frac_pred"] for r in rows if r["label"] == 0 and "pixel" in r and "frac_pred" in r["pixel"]]
print(f"  pixel (edited, t=0.5): F1={np.mean([p['f1'] or 0 for p in pix]):.3f} IoU={np.mean([p['iou'] or 0 for p in pix]):.3f} pixAUC={np.mean([p['auc'] for p in pix if p['auc'] is not None]):.3f} | authentic activation={np.mean(auth):.3f}")
