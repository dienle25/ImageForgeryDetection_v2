# research/ — scripts that produced every number in the reports

Run from the project root. Data paths are always arguments; nothing points at a user's machine.
PyTorch reference adapters need `requirements-research.txt` and the upstream repositories
(set `IFD_UPSTREAM=/path/to/upstream` containing `Mesorch/`, `catnet/`, `spai/`).

| Script | Purpose |
|---|---|
| `make_controlled_set.py` | builds the lineage-tracked controlled set `ctrl_mei` (seed 20261012) |
| `build_manifests.py`, `prepare_korus.py` | manifests for CocoGlide/Korus; lossless Korus PNG copies + P02 split |
| `dedup_check.py` | bytes / decoded-pixel / near-duplicate (dHash, pigeonhole index) leakage audit |
| `eval_policies.py`, `summarize_policies.py` | PSCC-Net preprocessing policies (E06) |
| `eval_torch_adapter.py` + `adapters/mesorch_torch.py`, `adapters/catnet_torch.py` | candidate localisation models in PyTorch (E07, E08) |
| `eval_image_adapter.py` + `adapters/spai_torch.py` | SPAI (AI-generated candidate) on a stratified subset (E09) |
| `eval_faces.py` + `adapters/face_pipeline.py`, `adapters/yermandy_torch.py` | face-deepfake candidate in PyTorch (E10) |
| `eval_detectors.py` | product ONNX detectors one at a time |
| **`eval_product.py`** | **the shipped pipeline** (Analyzer + config.json) over manifests -> per-image JSONL (dev and final test) |
| `derive_thresholds.py` | operating thresholds from development scores only (FPR rule per negative source) |
| `summarize_scores.py`, `summarize_maps.py`, `final_report.py` | metrics with group-bootstrap CIs, per-group breakdowns, pixel metrics |
| `before_after_probes.py` | v1 vs v2 on the 64 audit probes (diagnostic) |

Metric definitions: `ifd_core/metrics.py` (AUROC with ties, AP, bootstrap over groups, FPR thresholds).
