"""Read PyTorch zip-format checkpoints (state_dict of tensors) WITHOUT importing torch.

Only the tensor-rebuild functions needed for plain state_dicts are allowed; any other
global in the pickle raises an error (no arbitrary code execution).
"""
from __future__ import annotations

import collections
import pickle
import zipfile
from pathlib import Path

import numpy as np

_DTYPES = {
    "FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
    "LongStorage": np.int64, "IntStorage": np.int32, "ShortStorage": np.int16,
    "CharStorage": np.int8, "ByteStorage": np.uint8, "BoolStorage": np.bool_,
}


class _StorageType:
    def __init__(self, name):
        self.name = name
        self.dtype = _DTYPES[name]


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None):
    arr, = storage
    itemsize = arr.dtype.itemsize
    if len(size) == 0:
        return arr[storage_offset].copy()
    out = np.lib.stride_tricks.as_strided(arr[storage_offset:], shape=tuple(size),
                                          strides=tuple(s * itemsize for s in stride))
    return np.array(out, copy=True)


def _rebuild_parameter(data, requires_grad, backward_hooks):
    return data


class _Unpickler(pickle.Unpickler):
    def __init__(self, fileobj, archive: zipfile.ZipFile, prefix: str):
        super().__init__(fileobj)
        self.archive, self.prefix, self.cache = archive, prefix, {}

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "torch" and name in _DTYPES:
            return _StorageType(name)
        raise pickle.UnpicklingError(f"Refusing global {module}.{name}")

    def persistent_load(self, pid):
        kind, storage_type, key, location, numel = pid
        assert kind == "storage"
        if key not in self.cache:
            raw = self.archive.read(f"{self.prefix}/data/{key}")
            self.cache[key] = np.frombuffer(raw, dtype=storage_type.dtype)
        return (self.cache[key],)


def load_state_dict(path: str | Path) -> "collections.OrderedDict[str, np.ndarray]":
    with zipfile.ZipFile(path) as archive:
        pkl = [n for n in archive.namelist() if n.endswith("/data.pkl")]
        if len(pkl) != 1:
            raise ValueError(f"{path}: expected one data.pkl, found {pkl}")
        prefix = pkl[0][: -len("/data.pkl")]
        with archive.open(pkl[0]) as f:
            state = _Unpickler(f, archive, prefix).load()
    if not isinstance(state, dict):
        raise ValueError("checkpoint is not a state_dict")
    return collections.OrderedDict(((k[7:] if k.startswith("module.") else k), v) for k, v in state.items())


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        sd = load_state_dict(p)
        n = sum(int(np.prod(v.shape)) for v in sd.values())
        print(p, len(sd), "tensors", n, "elements")
        for k in list(sd)[:5]:
            print("  ", k, sd[k].shape, sd[k].dtype)
