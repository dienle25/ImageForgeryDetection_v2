"""Minimal ONNX (protobuf) writer with no dependency on the `onnx` package.

Implements only the message fields needed for static inference graphs.  Field numbers
follow onnx/onnx.proto (IR v8, opset 17).  Output is validated by loading it in ONNX Runtime.
"""
from __future__ import annotations

import struct

import numpy as np


def _varint(n: int) -> bytes:
    if n < 0:
        n += 1 << 64
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _key(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def f_int(field: int, v: int) -> bytes:
    return _key(field, 0) + _varint(int(v))


def f_bytes(field: int, v: bytes) -> bytes:
    return _key(field, 2) + _varint(len(v)) + v


def f_str(field: int, s: str) -> bytes:
    return f_bytes(field, s.encode("utf-8"))


def f_float(field: int, v: float) -> bytes:
    return _key(field, 5) + struct.pack("<f", float(v))


_DT = {np.dtype("float32"): 1, np.dtype("uint8"): 2, np.dtype("int8"): 3, np.dtype("int32"): 6,
       np.dtype("int64"): 7, np.dtype("bool"): 9, np.dtype("float16"): 10, np.dtype("float64"): 11}


def tensor_proto(name: str, arr: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(arr)
    body = b"".join(f_int(1, d) for d in arr.shape)
    body += f_int(2, _DT[arr.dtype]) + f_str(8, name) + f_bytes(9, arr.astype(arr.dtype.newbyteorder("<"), copy=False).tobytes())
    return body


def attribute(name: str, value) -> bytes:
    body = f_str(1, name)
    if isinstance(value, float):
        body += f_float(2, value) + f_int(20, 1)
    elif isinstance(value, (bool, int, np.integer)):
        body += f_int(3, int(value)) + f_int(20, 2)
    elif isinstance(value, str):
        body += f_bytes(4, value.encode()) + f_int(20, 3)
    elif isinstance(value, np.ndarray):
        body += f_bytes(5, tensor_proto("", value)) + f_int(20, 4)
    elif isinstance(value, (list, tuple)) and all(isinstance(v, (int, np.integer)) for v in value):
        body += b"".join(f_int(8, int(v)) for v in value) + f_int(20, 7)
    elif isinstance(value, (list, tuple)) and all(isinstance(v, float) for v in value):
        body += b"".join(f_float(7, v) for v in value) + f_int(20, 6)
    else:
        raise TypeError(f"unsupported attribute {name}={value!r}")
    return body


def node(op: str, inputs, outputs, name: str = "", **attrs) -> bytes:
    body = b"".join(f_str(1, i) for i in inputs) + b"".join(f_str(2, o) for o in outputs)
    body += f_str(3, name or outputs[0]) + f_str(4, op)
    body += b"".join(f_bytes(5, attribute(k, v)) for k, v in attrs.items())
    return body


def value_info(name: str, elem_type: int, dims) -> bytes:
    dim_msgs = b""
    for d in dims:
        dim = f_int(1, d) if isinstance(d, int) else f_str(2, str(d))
        dim_msgs += f_bytes(1, dim)
    tensor_type = f_int(1, elem_type) + f_bytes(2, dim_msgs)
    return f_str(1, name) + f_bytes(2, f_bytes(1, tensor_type))


class Graph:
    def __init__(self, name: str):
        self.name, self.nodes, self.inits, self.inputs, self.outputs = name, [], [], [], []
        self._n = 0

    def uid(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}_{self._n}"

    def init(self, arr: np.ndarray, prefix: str = "w") -> str:
        name = self.uid(prefix)
        self.inits.append(tensor_proto(name, arr))
        return name

    def op(self, op: str, inputs, n_out: int = 1, **attrs):
        outs = [self.uid(op.lower()) for _ in range(n_out)]
        self.nodes.append(node(op, inputs, outs, **attrs))
        return outs[0] if n_out == 1 else outs

    def serialize(self, producer: str = "ifd-onnx-writer", opset: int = 17, ir_version: int = 8, doc: str = "") -> bytes:
        graph = b"".join(f_bytes(1, n) for n in self.nodes) + f_str(2, self.name)
        graph += b"".join(f_bytes(5, t) for t in self.inits)
        graph += b"".join(f_bytes(11, v) for v in self.inputs) + b"".join(f_bytes(12, v) for v in self.outputs)
        opset_msg = f_str(1, "") + f_int(2, opset)
        model = f_int(1, ir_version) + f_str(2, producer) + f_str(3, "1.0")
        if doc:
            model += f_str(6, doc)
        model += f_bytes(7, graph) + f_bytes(8, opset_msg)
        return model
