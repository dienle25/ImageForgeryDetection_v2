"""Verify a delivery ZIP from a FRESH extraction (F09: never trust the build step).

    python tools/verify_package.py DIST/ImageForgeryDetection_v2.zip [--expect-sha256 HEX] [--selftest]

1. SHA-256 of the ZIP (and comparison with --expect-sha256 / <zip>.sha256 when present)
2. ZIP central directory + CRC of every member (zipfile.testzip)
3. extraction into a new empty temporary folder
4. every file listed in PACKAGE_MANIFEST.sha256 exists with the right SHA-256; no unlisted extra files
5. content audit: no credentials-like names, no venv, no caches, no images outside tests/data or runtime
6. optional: run `python -m ifd.selftest --quick` from the extracted copy (uses this interpreter)
Writes a JSON report next to the ZIP (<zip>.verify.json).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

FORBIDDEN = {".kaggle", "kaggle.json", ".env", "credentials", "id_rsa", ".netrc", ".git", ".venv", ".pytest_cache"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip", type=Path)
    ap.add_argument("--expect-sha256", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep the extracted copy")
    a = ap.parse_args()
    rep = {"zip": str(a.zip), "verified_on": platform.platform(), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "checks": {}}
    digest = sha256(a.zip)
    rep["sha256"] = digest
    expect = a.expect_sha256
    side = a.zip.with_name(a.zip.name + ".sha256")
    if not expect and side.exists():
        expect = side.read_text().split()[0]
    rep["checks"]["sha256_matches_expected"] = (expect is None) or (expect.lower() == digest)
    with zipfile.ZipFile(a.zip) as z:
        bad = z.testzip()
        rep["checks"]["crc_all_members"] = bad is None
        names = z.namelist()
        rep["members"] = len(names)
        tmp = Path(tempfile.mkdtemp(prefix="ifd_verify_"))
        z.extractall(tmp)
    roots = {n.split("/")[0] for n in names}
    rep["checks"]["single_top_folder"] = len(roots) == 1
    root = tmp / sorted(roots)[0]
    manifest = (root / "PACKAGE_MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
    listed, mism = {}, []
    for line in manifest:
        h, rel = line.split("  ", 1)
        listed[rel] = h
    for rel, h in listed.items():
        p = root / rel
        if not p.is_file() or sha256(p) != h:
            mism.append(rel)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} - {"PACKAGE_MANIFEST.sha256"}
    extra = sorted(actual - set(listed))
    rep["checks"]["all_manifest_files_match"] = not mism
    rep["checks"]["no_unlisted_files"] = not extra
    rep["mismatched"], rep["unlisted"] = mism[:50], extra[:50]
    leaks = []
    for rel in actual:
        parts = rel.split("/")
        if set(parts) & FORBIDDEN or ("__pycache__" in parts and not rel.startswith("runtime/")):
            leaks.append(rel)
        if Path(rel).suffix.lower() in IMAGE_EXT and not rel.startswith(("tests/data/", "runtime/", "reports/evidence/ui_e2e")):
            leaks.append(rel)
    rep["checks"]["no_forbidden_content"] = not leaks
    rep["forbidden"] = leaks[:50]
    if a.selftest:
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, "-X", "utf8", "-m", "ifd.selftest", "--quick", "--out", str(root / "reports" / "selftest")],
                           cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=3600)
        rep["selftest"] = {"returncode": r.returncode, "seconds": round(time.perf_counter() - t0, 1),
                           "stdout_tail": r.stdout[-3000:], "stderr_tail": r.stderr[-2000:]}
        rep["checks"]["selftest_from_fresh_extraction"] = r.returncode == 0
    rep["ok"] = all(rep["checks"].values())
    rep["extracted_to"] = str(root) if a.keep else None
    out = a.zip.with_name(a.zip.name + ".verify.json")
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k in ("sha256", "members", "checks", "ok")}, indent=1))
    if not a.keep:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
