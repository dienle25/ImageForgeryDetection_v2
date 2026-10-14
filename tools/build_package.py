"""Build the clean delivery package: folder -> PACKAGE_MANIFEST.sha256 -> ZIP -> (optional) transfer volumes.

    python tools/build_package.py --runtime RUNTIME_DIR --out DIST_DIR [--volume-mb 29]

* Copies only an explicit allow-list of project content (code, models, runtime, docs, reports, tests).
* Refuses to package: virtualenvs, caches, credentials-looking files, dataset images, run outputs.
* Writes PACKAGE_MANIFEST.sha256 (sha256 + relative path of every packaged file) inside the folder.
* ZIP (deflate) + its SHA-256 in <name>.zip.sha256; optional numbered volumes (<name>.zip.001 …) for transfer
  through size-limited channels, plus GHEP_ZIP.bat that re-joins them on Windows and checks the SHA-256.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "ImageForgeryDetection_v2"

INCLUDE_TOP = ["RUN_APP.bat", "SELF_TEST.bat", "PHAN_TICH_THU_MUC.bat", "KIEM_DINH_CONG_TY.bat", "run_app.sh", "README_VI.md", "LICENSE",
               "LICENSES_VI.md", "MODEL_CARD.md", "TASK_CONTRACT.md", "QUALITY_TARGETS.yaml", "STATUS.md", "RESUME.md",
               "CHANGELOG.md", "EXPERIMENT_LOG.jsonl", "config.json", "requirements.txt", "requirements-research.txt",
               "requirements-lock-win-cp313.txt", "pyproject.toml"]
INCLUDE_DIRS = ["ifd", "models", "tests", "tools", "docs", "reports", "third_party_licenses", "weights_source",
                "research", "ifd_core"]
EXCLUDE_GLOBS = ["*/__pycache__/*", "*.pyc", "*/.pytest_cache/*", "*.tmp", "*.assembling", "*/outputs/*",
                 "reports/selftest/*", "*.part[0-9][0-9][0-9]", "*/.venv/*", "*.log"]
FORBIDDEN_NAMES = {".kaggle", "kaggle.json", ".env", "credentials", "id_rsa", ".netrc", ".git"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
IMAGE_ALLOWED_PREFIX = ("tests/data/", "runtime/", "reports/evidence/ui_e2e")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def excluded(rel: str) -> bool:
    return any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch("/" + rel, g) for g in EXCLUDE_GLOBS)


def collect(runtime: Path | None) -> list[tuple[Path, str]]:
    items = []
    for name in INCLUDE_TOP:
        p = ROOT / name
        if p.is_file():
            items.append((p, name))
    for d in INCLUDE_DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file():
                rel = p.relative_to(ROOT).as_posix()
                if not excluded(rel):
                    items.append((p, rel))
    if runtime:
        for p in sorted(runtime.rglob("*")):
            if p.is_file():
                rel = "runtime/" + p.relative_to(runtime).as_posix()
                if not rel.endswith((".log", ".tmp")):
                    items.append((p, rel))     # runtime keeps its precompiled __pycache__ (unchecked-hash .pyc)
    return items


def audit(items) -> list[str]:
    problems = []
    for _, rel in items:
        parts = set(rel.split("/"))
        if parts & FORBIDDEN_NAMES:
            problems.append(f"forbidden name in {rel}")
        if Path(rel).suffix.lower() in IMAGE_EXT and not rel.startswith(IMAGE_ALLOWED_PREFIX):
            problems.append(f"image outside tests/data or runtime: {rel}")
    return problems


def write_volumes(zpath: Path, mb: int) -> list[Path]:
    size = mb * 1024 * 1024
    vols = []
    with zpath.open("rb") as f:
        i = 1
        while True:
            chunk = f.read(size)
            if not chunk:
                break
            v = zpath.with_name(f"{zpath.name}.{i:03d}")
            v.write_bytes(chunk)
            vols.append(v)
            i += 1
    return vols


def join_bat(zip_name: str, vols: list[Path], digest: str) -> str:
    plus = " + ".join(f'"{v.name}"' for v in vols)
    lines = [
        "@echo off",
        "setlocal EnableExtensions EnableDelayedExpansion",
        "cd /d \"%~dp0\"",
        "title Ghep file ZIP - Image Forgery Detection v2",
        f"echo Ghep {len(vols)} phan thanh {zip_name} ...",
        f"for %%F in ({' '.join(v.name for v in vols)}) do if not exist \"%%F\" (echo [LOI] Thieu phan %%F & goto :fail)",
        f"copy /b /y {plus} \"{zip_name}\" >nul",
        "if errorlevel 1 goto :fail",
        "echo Dang kiem tra SHA-256 - co the mat 10-30 giay ...",
        "set \"H=\"",
        f"for /f \"skip=1 delims=\" %%L in ('certutil -hashfile \"{zip_name}\" SHA256') do if not defined H set \"H=%%L\"",
        "set \"H=!H: =!\"",
        f"if /i not \"!H!\"==\"{digest}\" goto :badhash",
        f"echo OK: SHA-256 khop {digest}",
        f"echo Hay giai nen {zip_name} vao mot thu muc moi roi mo RUN_APP.bat ben trong.",
        "pause",
        "exit /b 0",
        ":badhash",
        "echo [LOI] SHA-256 khong khop: !H!",
        f"echo        mong doi:      {digest}",
        "echo File bi thieu hoac hong. Hay chep lai cac phan .zip.0xx roi chay lai.",
        "pause",
        "exit /b 1",
        ":fail",
        "echo [LOI] Khong ghep duoc file.",
        "pause",
        "exit /b 1",
    ]
    return "\r\n".join(lines) + "\r\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", type=Path, default=None, help="built Windows runtime folder (tools/build_windows_runtime.py)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--volume-mb", type=int, default=0, help="also write transfer volumes of this size (0 = no)")
    a = ap.parse_args()
    items = collect(a.runtime)
    problems = audit(items)
    if problems:
        raise SystemExit("refusing to package:\n" + "\n".join(problems[:50]))
    out = a.out
    folder = out / NAME
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    lines = []
    for src, rel in items:
        dst = folder / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        lines.append(f"{sha256(dst)}  {rel}")
    (folder / "PACKAGE_MANIFEST.sha256").write_text("\n".join(sorted(lines, key=lambda s: s[66:])) + "\n", encoding="utf-8")
    zpath = out / f"{NAME}.zip"
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                z.write(p, arcname=f"{NAME}/{p.relative_to(folder).as_posix()}")
    digest = sha256(zpath)
    (out / f"{NAME}.zip.sha256").write_text(f"{digest}  {zpath.name}\n", encoding="utf-8")
    info = {"zip": zpath.name, "bytes": zpath.stat().st_size, "sha256": digest, "files": len(lines)}
    if a.volume_mb:
        vols = write_volumes(zpath, a.volume_mb)
        (out / "GHEP_ZIP.bat").write_bytes(join_bat(zpath.name, vols, digest).encode("ascii"))
        info["volumes"] = [{"file": v.name, "bytes": v.stat().st_size, "sha256": sha256(v)} for v in vols]
    (out / "BUILD_INFO.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k != "volumes"}, indent=1), "volumes:", len(info.get("volumes", [])))


if __name__ == "__main__":
    main()
