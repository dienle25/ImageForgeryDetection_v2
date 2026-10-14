"""Shipped model parts: assembled only when complete and SHA-256-identical to the manifest."""
import hashlib
import json

import pytest

from ifd.errors import ForgeryError
from ifd.model_parts import assemble_all, split_file


def _setup(tmp_path, payload=b"0123456789" * 1000):
    m = tmp_path / "models"
    m.mkdir()
    (m / "x.onnx").write_bytes(payload)
    (m / "manifest.json").write_text(json.dumps({"files": {"x.onnx": {
        "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}}}))
    parts = split_file(m / "x.onnx", 3000)
    (m / "x.onnx").unlink()
    return m, parts, payload


def test_assemble_ok_and_parts_removed(tmp_path):
    m, parts, payload = _setup(tmp_path)
    assert len(parts) == 4
    assert assemble_all(m, log=lambda *_: None) == ["x.onnx"]
    assert (m / "x.onnx").read_bytes() == payload
    assert not list(m.glob("x.onnx.part*"))
    assert assemble_all(m, log=lambda *_: None) == []        # idempotent


def test_missing_part_rejected(tmp_path):
    m, parts, _ = _setup(tmp_path)
    parts[1].unlink()
    with pytest.raises(ForgeryError):
        assemble_all(m, log=lambda *_: None)
    assert not (m / "x.onnx").exists()


def test_corrupt_part_rejected_and_kept(tmp_path):
    m, parts, _ = _setup(tmp_path)
    data = bytearray(parts[2].read_bytes())
    data[5] ^= 1
    parts[2].write_bytes(bytes(data))
    with pytest.raises(ForgeryError, match="SHA-256"):
        assemble_all(m, log=lambda *_: None)
    assert not (m / "x.onnx").exists() and parts[2].exists()
