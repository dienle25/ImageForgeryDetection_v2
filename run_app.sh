#!/usr/bin/env sh
# Linux/macOS launcher: uses ./.venv if present, else creates it from python3 (3.10-3.13).
set -e
APP="$(cd "$(dirname "$0")" && pwd)"
cd "$APP"
export PYTHONUTF8=1 PYTHONNOUSERSITE=1
if [ ! -x "$APP/.venv/bin/python" ]; then
  python3 -m venv "$APP/.venv"
  if [ -d "$APP/wheels/runtime" ]; then
    "$APP/.venv/bin/python" -m pip install --no-index --find-links "$APP/wheels/runtime" -r "$APP/requirements.txt"
  else
    "$APP/.venv/bin/python" -m pip install -r "$APP/requirements.txt"
  fi
fi
# glibc: hand large freed buffers back to the OS (model loading otherwise leaves ~2 GB of idle heap)
export MALLOC_MMAP_THRESHOLD_=65536
exec "$APP/.venv/bin/python" -X utf8 -m ifd.launcher "$@"
