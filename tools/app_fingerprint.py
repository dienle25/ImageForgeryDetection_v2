"""Fingerprint of everything that can change a score or a decision: app code, config, model manifest.

    python tools/app_fingerprint.py [--root DIR]

Printed digest = SHA-256 over sorted "relative-path sha256" lines of ifd/**/*.py, ifd/webapp/static/*,
config.json and models/manifest.json (the manifest pins every model file's SHA-256).  Recorded in
EXPERIMENT_LOG.jsonl when the configuration is locked (P04) and checked again on the delivered package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def fingerprint(root: Path) -> dict:
    files = sorted([p for p in (root / "ifd").rglob("*") if p.is_file() and "__pycache__" not in p.parts
                    and p.suffix in (".py", ".html", ".css", ".js")] + [root / "config.json", root / "models" / "manifest.json"])
    lines = []
    for p in files:
        lines.append(f"{p.relative_to(root).as_posix()} {hashlib.sha256(p.read_bytes()).hexdigest()}")
    digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    return {"app_fingerprint": digest, "files": len(lines),
            "config_sha256": hashlib.sha256((root / "config.json").read_bytes()).hexdigest(),
            "models_manifest_sha256": hashlib.sha256((root / "models" / "manifest.json").read_bytes()).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    print(json.dumps(fingerprint(ap.parse_args().root), indent=1))


if __name__ == "__main__":
    main()
