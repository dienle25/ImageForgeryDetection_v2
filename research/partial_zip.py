"""Extract entries from a split zip when only some parts (incl. the last part with the central directory) are present."""
import io, json, struct, sys, zlib, hashlib
from pathlib import Path


class SparseZip(io.RawIOBase):
    def __init__(self, parts: dict, total: int):
        self.parts = parts          # {start_offset: bytes}
        self.total, self.pos = total, 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.total + off)
        return self.pos

    def readinto(self, b):
        n = len(b); start = self.pos; end = min(self.total, start + n)
        out = bytearray(end - start)
        for s, data in self.parts.items():
            e = s + len(data)
            a, z = max(start, s), min(end, e)
            if a < z:
                out[a - start:z - start] = data[a - s:z - s]
        b[:len(out)] = out
        self.pos = end
        return len(out)

    def covered(self, a, z):
        for s, data in self.parts.items():
            if s <= a and z <= s + len(data):
                return True
        return False


def open_partial(status_json: str, part_dir: str, have: list[int]):
    st = json.load(open(status_json))["result"]
    total = st["bytes"]
    parts, offset = {}, 0
    for i, p in enumerate(st["parts"]):
        if i in have:
            data = Path(part_dir, p["file"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == p["sha256"], p["file"]
            parts[offset] = data
        offset += p["bytes"]
    import zipfile
    sz = SparseZip(parts, total)
    zf = zipfile.ZipFile(io.BufferedReader(sz, buffer_size=1 << 16))
    return zf, sz


def extractable(zf, sz):
    ok = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        # local header (30 bytes + name + extra) + compressed data
        hdr_end = info.header_offset + 30 + len(info.filename.encode()) + 64 + info.compress_size + 1024
        if sz.covered(info.header_offset, min(hdr_end, sz.total)):
            ok.append(info)
    return ok
