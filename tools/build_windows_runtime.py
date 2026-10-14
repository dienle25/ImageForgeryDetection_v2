"""Build the self-contained Windows runtime folder (no installer, no admin rights, no Internet at run time).

    python tools/build_windows_runtime.py --embed python-3.13.X-embed-amd64.zip --wheels wheels/runtime \
        [--msvc-wheel msvc_runtime-*.whl] --out PACKAGE/runtime

* Official CPython "embeddable package" for Windows x64 (python.org).
* Runtime wheels (numpy, pillow, onnxruntime, opencv-python-headless + their deps) unpacked into Lib/site-packages
  exactly as `pip install` would place them (wheels are zip archives; no build step is involved).
* `python3XX._pth` lists: the stdlib zip, the runtime dir, Lib/site-packages and `..` (the app root), then
  `import site`.  The launcher sets PYTHONNOUSERSITE=1 so nothing from the user's own Python leaks in.
* opencv_videoio_ffmpeg*.dll is dropped (video I/O is never used; avoids shipping FFmpeg); test suites and
  onnxruntime tools/transformers/quantization sub-packages are dropped (unused at run time, shorter paths).
* Optional: Microsoft VC++ runtime DLLs (msvcp140*.dll, vcruntime140*.dll, concrt140.dll) placed next to
  python.exe (app-local deployment of Microsoft redistributable files) so onnxruntime/OpenCV load on machines
  without the VC++ Redistributable installed.
Writes runtime/BUILD_INFO.json with the SHA-256 of every input.
"""
from __future__ import annotations

import argparse
import compileall
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", type=Path, required=True)
    ap.add_argument("--wheels", type=Path, required=True)
    ap.add_argument("--msvc-wheel", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = a.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    info = {"embeddable": {"file": a.embed.name, "sha256": sha256(a.embed), "source": "https://www.python.org/ftp/python/"},
            "wheels": [], "removed": [], "msvc_runtime": None}
    with zipfile.ZipFile(a.embed) as z:
        z.extractall(out)
    pth = sorted(out.glob("python3*._pth"))
    if len(pth) != 1:
        raise SystemExit("expected exactly one python3XX._pth in the embeddable package")
    stdlib_zip = sorted(out.glob("python3*.zip"))[0].name
    pth[0].write_text(f"{stdlib_zip}\n.\nLib\\site-packages\n..\nimport site\n", encoding="ascii")
    site = out / "Lib" / "site-packages"
    site.mkdir(parents=True)
    for whl in sorted(a.wheels.glob("*.whl")):
        with zipfile.ZipFile(whl) as z:
            z.extractall(site)
        info["wheels"].append({"file": whl.name, "sha256": sha256(whl)})
    for dll in site.glob("cv2/opencv_videoio_ffmpeg*.dll"):
        info["removed"].append(f"Lib/site-packages/cv2/{dll.name} (video I/O not used)")
        dll.unlink()
    # unused at run time; also keeps every path well below the Windows 260-character limit
    prune = [d for d in site.rglob("tests") if d.is_dir()]
    prune += [site / "onnxruntime" / n for n in ("tools", "transformers", "quantization") if (site / "onnxruntime" / n).is_dir()]
    for d in prune:
        info["removed"].append(f"{d.relative_to(out).as_posix()}/ (not used at run time)")
        shutil.rmtree(d)
    if a.msvc_wheel:
        with zipfile.ZipFile(a.msvc_wheel) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".dll")
                     and Path(n).name.lower().startswith(("msvcp140", "vcruntime140", "concrt140"))]
            copied = []
            for n in names:
                target = out / Path(n).name
                if target.exists():          # keep the embeddable package's own vcruntime140*.dll
                    continue
                target.write_bytes(z.read(n))
                copied.append(Path(n).name)
        info["msvc_runtime"] = {"wheel": a.msvc_wheel.name, "sha256": sha256(a.msvc_wheel), "copied": copied,
                                "note": "Microsoft Visual C++ redistributable files, app-local deployment"}
    # third-party bytecode compiled once with unchecked hashes (valid for any 3.13.x, never stale-checked)
    compileall.compile_dir(str(site), quiet=1, invalidation_mode=compileall.py_compile.PycInvalidationMode.UNCHECKED_HASH)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    info["total_bytes"] = total
    (out / "BUILD_INFO.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k != "wheels"}, indent=1))
    print("wheels:", [w["file"] for w in info["wheels"]], "size MB", round(total / 1e6, 1))


if __name__ == "__main__":
    main()
