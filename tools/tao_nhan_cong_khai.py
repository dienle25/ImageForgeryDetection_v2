"""Tạo file nhãn cho dữ liệu công khai (CocoGlide, Korus) và cho TẬP HIỆU CHỈNH riêng của DefakerOne2.

Chỉ giải nén ảnh, đổi TIFF -> PNG không mất dữ liệu và ghi file nhãn; KHÔNG chạy mô hình nào.

    python tools/tao_nhan_cong_khai.py [--bo cocoglide korus hieu_chinh] [--nguon THU_MUC_TAI_VE] [--dich THU_MUC_ANH]

Cách tách tập: dùng NGUYÊN cách tách dev/test đã khóa của repo (reports/manifests/*.jsonl; EXPERIMENT_LOG P02, P03),
không tách lại:
  * Korus: theo cặp cảnh (ảnh gốc + ảnh sửa cùng group_id), phân tầng theo máy ảnh, seed 20261012, 18 cặp mỗi máy
    -> dev_korus (72 cặp); còn lại -> test_korus (148 cặp).  Bước đo chỉ dùng test_korus; tập hiệu chỉnh chỉ dùng dev_korus.
  * CocoGlide: toàn bộ là test (test_cocoglide) -> chỉ dùng để đo, không bao giờ dùng để chọn ngưỡng.
  * DeepFakeFace (DFF) và OpenFake: chỉ lấy phần dev cho tập hiệu chỉnh; phần test của hai bộ này không dùng ở đâu.
  * ctrl_mei: ảnh nguồn không có trên máy (phải tải từ GitHub rồi tạo lại) -> không dùng.
  * OpenFake cần thư viện pyarrow (không có trong runtime của app): chạy script này bằng Python có pyarrow
    (ví dụ venv_research), nếu không OpenFake bị bỏ qua và ghi rõ trong tach_tap.json.
Kiểm tra: SHA-256 từng ảnh (Korus: SHA-256 file TIFF gốc và SHA-256 điểm ảnh) khớp manifest; không ảnh nào (theo mã ảnh,
nhóm, SHA-256 file hoặc SHA-256 điểm ảnh) vừa ở tập hiệu chỉnh vừa ở tập đo — nếu có thì dừng.

Kết quả trong --dich: ảnh, nhan_cocoglide.csv, nhan_korus_test.csv, nhan_hieu_chinh.csv và tach_tap.json (mô tả cách
tách, số ảnh, SHA-256 của file nhãn).  Không đưa thư mục ảnh này lên git.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "reports" / "manifests"
DEFAULT_SRC = Path(r"D:\ASUS\2\Image_Forgery_Detection_CPU\_cloud_bridge\downloads")
DEFAULT_DST = Path(r"D:\ASUS\2\Image_Forgery_Detection_CPU\du_lieu_cong_khai")
LABEL_FILES = {"cocoglide": "nhan_cocoglide.csv", "korus": "nhan_korus_test.csv", "hieu_chinh": "nhan_hieu_chinh.csv"}
SPLIT_FILE = "tach_tap.json"
TEST_SETS = ("cocoglide", "korus")
KORUS_ZIP_BYTES = 1697070982            # realistic-tampering-dataset.zip (tools/fetch_sources.py)
SPLIT_SEED = 20261012                   # seed of the locked Korus split (research/prepare_korus.py)
SPLIT_RULE_VI = {
    "cocoglide": "CocoGlide: toàn bộ 512 cặp là test (test_cocoglide), chỉ dùng để đo.",
    "korus": ("Korus: tách theo cặp cảnh (group_id), phân tầng theo máy ảnh, seed 20261012, 18 cặp/máy -> dev_korus; "
              "còn lại -> test_korus (research/prepare_korus.py, EXPERIMENT_LOG P02). Đo chỉ dùng test_korus."),
    "hieu_chinh": ("Tập hiệu chỉnh = phần dev đã khóa của repo: dev_korus + DFF dev (+ OpenFake dev nếu có pyarrow). "
                   "Không dùng data.zip, A–D, CocoGlide hay test_korus."),
}


class ConcatFile(io.RawIOBase):
    """Read-only view of several files as ONE file (a ZIP downloaded in parts), without writing a copy."""

    def __init__(self, files):
        self.files = [Path(f) for f in files]
        self.sizes = [f.stat().st_size for f in self.files]
        self.total, self.pos, self._fh = sum(self.sizes), 0, {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.total + off)
        return self.pos

    def readinto(self, b):
        want, got, start = len(b), 0, 0
        for f, size in zip(self.files, self.sizes):
            if got >= want:
                break
            if self.pos < start + size and self.pos + (want - got) > start:
                h = self._fh.get(f) or f.open("rb")
                self._fh[f] = h
                h.seek(self.pos - start)
                data = h.read(min(start + size - self.pos, want - got))
                b[got:got + len(data)] = data
                got += len(data)
                self.pos += len(data)
            start += size
        return got

    def close(self):
        for h in self._fh.values():
            h.close()
        self._fh.clear()
        super().close()


class SparseFile(io.RawIOBase):
    """A large file of which only some byte ranges were downloaded (OpenFake parquet row groups + footer).
    Reading outside the downloaded ranges raises OSError instead of returning wrong bytes."""

    def __init__(self, total: int, pieces):
        self.total, self.pos = int(total), 0
        self.pieces = sorted((int(s), Path(p), Path(p).stat().st_size) for s, p in pieces)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.total + off)
        return self.pos

    def readinto(self, b):
        want = min(len(b), self.total - self.pos)
        if want <= 0:
            return 0
        for start, path, size in self.pieces:
            if start <= self.pos and self.pos + want <= start + size:
                with path.open("rb") as h:
                    h.seek(self.pos - start)
                    data = h.read(want)
                b[:len(data)] = data
                self.pos += len(data)
                return len(data)
        raise OSError(f"vùng byte {self.pos}–{self.pos + want} của file chưa được tải về")


def open_zip(parts) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(ConcatFile(parts), buffer_size=1 << 20))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decoded_hash_of(rgb) -> str:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from ifd.image_io import decoded_pixel_hash      # same definition as the app and the manifests
    return decoded_pixel_hash(rgb)


def load_manifest(name: str, manifests: Path = MANIFESTS) -> list[dict]:
    rows = []
    for line in (manifests / name).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def safe_target(root: Path, rel: str) -> Path:
    """Destination inside root for a manifest path; refuses absolute paths and '..'."""
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"đường dẫn không hợp lệ trong manifest: {rel}")
    return root / p


def find_member(names: dict, rel: str):
    if rel in names:
        return names[rel]
    hits = [n for n in names if n.endswith("/" + rel)]
    return names[hits[0]] if len(hits) == 1 else None


def label_row(file: str, label: int, loai: str, nhom: str, source: dict) -> dict:
    return {"file": file, "label": int(label), "loai": loai, "nhom": nhom, "image_id": source["image_id"],
            "sha256": source.get("sha256"), "decoded": source.get("decoded_rgb_sha256"), "split": source.get("split")}


# ------------------------------------------------------------------------------------------------ datasets
def prep_cocoglide(src: Path, dst: Path, log, manifests: Path = MANIFESTS) -> list[dict]:
    rows = [r for r in load_manifest("cocoglide.jsonl", manifests) if not r.get("exclude")]
    zpath = src / "data" / "CocoGlide.zip"
    if not zpath.is_file():
        raise FileNotFoundError(f"Không thấy {zpath}")
    out, done = [], 0
    with zipfile.ZipFile(zpath) as zf:
        names = {i.filename: i for i in zf.infolist() if not i.is_dir()}
        for r in rows:
            target = safe_target(dst / "cocoglide", r["image"])
            if not (target.is_file() and sha256_file(target) == r["sha256"]):
                member = find_member(names, r["image"])
                if member is None:
                    raise FileNotFoundError(f"CocoGlide.zip thiếu {r['image']}")
                data = zf.read(member)
                if sha256_bytes(data) != r["sha256"]:
                    raise ValueError(f"SHA-256 khác manifest: {r['image']}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                done += 1
            out.append(label_row(f"cocoglide/{r['image']}", r["label"],
                                 "cocoglide_that" if int(r["label"]) == 0 else "cocoglide_glide", r["group_id"], r))
    log(f"CocoGlide: {len(out)} ảnh ({done} ảnh mới giải nén), SHA-256 khớp manifest.")
    return out


def korus_parts(src: Path) -> list[Path]:
    data = src / "data"
    head = sorted(data.glob("korus_realistic_tampering.zip.part*"))
    rest = sorted(data.glob("korus_rest_from_*.bin.part*"))
    if not head:
        raise FileNotFoundError(f"Không thấy korus_realistic_tampering.zip.part* trong {data}")
    if rest:
        offset = int(rest[0].name.split("_from_")[1].split(".")[0])
        if sum(p.stat().st_size for p in head) != offset:
            raise ValueError("Các phần korus_*.part không nối tiếp nhau (kích thước khác vị trí ghi trong tên file).")
    return head + rest


def prep_korus(src: Path, dst: Path, log, splits, manifests: Path = MANIFESTS) -> list[dict]:
    from PIL import Image, ImageOps
    import numpy as np
    rows = [r for r in load_manifest("korus_png.jsonl", manifests) if r.get("split") in splits and not r.get("exclude")]
    parts = korus_parts(src)
    total = sum(p.stat().st_size for p in parts)
    if total != KORUS_ZIP_BYTES:
        log(f"[CẢNH BÁO] Tổng kích thước các phần Korus {total} byte, khác {KORUS_ZIP_BYTES} byte của file gốc; "
            "vẫn thử đọc, từng ảnh vẫn được kiểm SHA-256.")
    out, done = [], 0
    zf = open_zip(parts)
    try:
        names = {i.filename: i for i in zf.infolist() if not i.is_dir()}
        for i, r in enumerate(rows, start=1):
            target = safe_target(dst / "korus", r["image_id"] + ".png")
            ok = False
            if target.is_file():
                with Image.open(target) as im:
                    ok = decoded_hash_of(np.asarray(im.convert("RGB"))) == r["decoded_rgb_sha256"]
            if not ok:
                member = find_member(names, r["original_file"])
                if member is None:
                    raise FileNotFoundError(f"ZIP Korus thiếu {r['original_file']}")
                tif = zf.read(member)
                if sha256_bytes(tif) != r["original_sha256"]:
                    raise ValueError(f"SHA-256 TIFF khác manifest: {r['original_file']}")
                with Image.open(io.BytesIO(tif)) as im:
                    rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
                if decoded_hash_of(rgb) != r["decoded_rgb_sha256"]:
                    raise ValueError(f"Điểm ảnh khác manifest: {r['original_file']}")
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_suffix(".tmp.png")
                Image.fromarray(rgb).save(tmp, format="PNG", compress_level=6)       # lossless, like prepare_korus.py
                with Image.open(tmp) as im2:
                    if decoded_hash_of(np.asarray(im2.convert("RGB"))) != r["decoded_rgb_sha256"]:
                        raise ValueError(f"PNG ghi ra khác điểm ảnh gốc: {target.name}")
                tmp.replace(target)
                done += 1
                if done % 25 == 0:
                    log(f"  Korus: đã đổi {done} ảnh TIFF -> PNG ({i}/{len(rows)})")
            out.append(label_row(f"korus/{target.name}", r["label"], "korus_that" if int(r["label"]) == 0 else "korus_sua",
                                 r["group_id"], r))
    finally:
        zf.close()
    log(f"Korus ({', '.join(splits)}): {len(out)} ảnh ({done} ảnh mới), điểm ảnh khớp manifest.")
    return out


def prep_dff_dev(src: Path, dst: Path, log, manifests: Path = MANIFESTS) -> list[dict]:
    rows, seen = [], set()
    for name in ("dff_faceswap.jsonl", "dff_extra.jsonl"):
        for r in load_manifest(name, manifests):
            if r.get("split") == "dev" and not r.get("exclude") and r["image_id"] not in seen:
                seen.add(r["image_id"])
                rows.append(r)
    out, done = [], 0
    for folder in sorted({r["image"].split("/")[0] for r in rows}):
        parts = sorted((src / "data" / "dff").glob(f"{folder}.zip.part*"))
        if not parts:
            raise FileNotFoundError(f"Không thấy {folder}.zip.part* trong {src / 'data' / 'dff'}")
        zf = open_zip(parts)
        try:
            names = {i.filename: i for i in zf.infolist() if not i.is_dir()}
            for r in (x for x in rows if x["image"].split("/")[0] == folder):
                target = safe_target(dst / "dff", r["image"])
                if not (target.is_file() and sha256_file(target) == r["sha256"]):
                    member = find_member(names, r["image"])
                    if member is None:
                        raise FileNotFoundError(f"{folder}.zip thiếu {r['image']}")
                    data = zf.read(member)
                    if sha256_bytes(data) != r["sha256"]:
                        raise ValueError(f"SHA-256 khác manifest: {r['image']}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                    done += 1
                loai = "dff_wiki" if int(r["label"]) == 0 else f"dff_{folder}"
                out.append(label_row(f"dff/{r['image']}", r["label"], loai, r["group_id"], r))
        finally:
            zf.close()
    log(f"DeepFakeFace dev: {len(out)} ảnh ({done} ảnh mới), SHA-256 khớp manifest.")
    return out


def prep_openfake_dev(src: Path, dst: Path, log, manifests: Path = MANIFESTS) -> list[dict] | None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        log("OpenFake: BỎ QUA — Python này không có pyarrow. Muốn thêm OpenFake dev vào tập hiệu chỉnh, chạy lại script "
            "bằng Python có pyarrow (ví dụ venv_research\\Scripts\\python.exe).")
        return None
    wanted = {}
    for r in load_manifest("openfake.jsonl", manifests):
        if r.get("split") == "dev" and not r.get("exclude"):
            wanted[r["image_id"][len("openfake_"):]] = r          # image_id = "openfake_" + row_id
    groups = json.loads((manifests / "openfake_row_groups.json").read_text(encoding="utf-8"))
    out, done, found = [], 0, set()
    try:
        for shard in sorted({g["shard"] for g in groups}):
            gs = [g for g in groups if g["shard"] == shard]
            pieces = []
            for g in gs:
                f = src / "data" / "openfake" / f"{shard}_rg{int(g['rg']):02d}_{int(g['start'])}.bin"
                if not f.is_file() or f.stat().st_size != int(g["end_excl"]) - int(g["start"]):
                    raise FileNotFoundError(f"thiếu hoặc sai kích thước {f.name}")
                pieces.append((g["start"], f))
            tail = src / "meta" / f"openfake_{shard}_tail.bin"
            if not tail.is_file():
                raise FileNotFoundError(f"thiếu {tail.name}")
            pieces.append((int(gs[0]["total"]) - tail.stat().st_size, tail))
            pf = pq.ParquetFile(SparseFile(int(gs[0]["total"]), pieces))
            for g in gs:
                tbl = pf.read_row_group(int(g["rg"]), columns=["image", "row_id"])
                for img, rid in zip(tbl.column("image").to_pylist(), tbl.column("row_id").to_pylist()):
                    r = wanted.get(rid)
                    if r is None:
                        continue
                    target = safe_target(dst / "openfake", r["image"])
                    if not (target.is_file() and sha256_file(target) == r["sha256"]):
                        data = img["bytes"] if isinstance(img, dict) else img
                        if sha256_bytes(data) != r["sha256"]:
                            raise ValueError(f"SHA-256 khác manifest: {r['image']}")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(data)
                        done += 1
                    found.add(rid)
    except (OSError, ValueError) as exc:
        log(f"OpenFake: BỎ QUA — không đọc được dữ liệu đã tải ({exc}).")
        return None
    missing = sorted(set(wanted) - found)
    if missing:
        log(f"OpenFake: BỎ QUA — {len(missing)} ảnh dev không có trong các phần đã tải (ví dụ {missing[0]}).")
        return None
    for rid in sorted(found):
        r = wanted[rid]
        loai = f"openfake_{r.get('real_source') or 'that'}" if int(r["label"]) == 0 else "openfake_ai"
        out.append(label_row(f"openfake/{r['image']}", r["label"], loai, r["group_id"], r))
    log(f"OpenFake dev: {len(out)} ảnh ({done} ảnh mới), SHA-256 khớp manifest.")
    return out


def check_sources(src: Path, log, manifests: Path = MANIFESTS) -> int:
    """--kiem-tra: every image listed by the manifests is present in the downloaded files (names and sizes only:
    ZIP directories, OpenFake row-group files and their row_id column).  Nothing is extracted or decoded."""
    problems = 0

    def report(name, rows, names, key):
        nonlocal problems
        missing = [r[key] for r in rows if find_member(names, r[key]) is None]
        problems += len(missing)
        log(f"{name}: {len(rows) - len(missing)}/{len(rows)} ảnh có trong file tải về"
            + (f" — THIẾU {len(missing)}, ví dụ {missing[0]}" if missing else ""))

    with zipfile.ZipFile(src / "data" / "CocoGlide.zip") as zf:
        report("CocoGlide (test)", load_manifest("cocoglide.jsonl", manifests),
               {i.filename: i for i in zf.infolist() if not i.is_dir()}, "image")
    parts = korus_parts(src)
    total = sum(p.stat().st_size for p in parts)
    log(f"Korus: {len(parts)} phần, tổng {total} byte (bản gốc {KORUS_ZIP_BYTES} byte){'' if total == KORUS_ZIP_BYTES else ' — KHÁC'}")
    problems += total != KORUS_ZIP_BYTES
    zf = open_zip(parts)
    try:
        names = {i.filename: i for i in zf.infolist() if not i.is_dir()}
        kor = load_manifest("korus_png.jsonl", manifests)
        for split in ("dev_korus", "test_korus"):
            report(f"Korus {split}", [r for r in kor if r.get("split") == split], names, "original_file")
    finally:
        zf.close()
    dff = {}
    for name in ("dff_faceswap.jsonl", "dff_extra.jsonl"):
        for r in load_manifest(name, manifests):
            if r.get("split") == "dev":
                dff[r["image_id"]] = r
    for folder in sorted({r["image"].split("/")[0] for r in dff.values()}):
        zf = open_zip(sorted((src / "data" / "dff").glob(f"{folder}.zip.part*")))
        try:
            report(f"DeepFakeFace dev/{folder}", [r for r in dff.values() if r["image"].split("/")[0] == folder],
                   {i.filename: i for i in zf.infolist() if not i.is_dir()}, "image")
        finally:
            zf.close()
    groups = json.loads((manifests / "openfake_row_groups.json").read_text(encoding="utf-8"))
    bad = [g for g in groups if not (src / "data" / "openfake" / f"{g['shard']}_rg{int(g['rg']):02d}_{int(g['start'])}.bin").is_file()
           or (src / "data" / "openfake" / f"{g['shard']}_rg{int(g['rg']):02d}_{int(g['start'])}.bin").stat().st_size
           != int(g["end_excl"]) - int(g["start"])]
    problems += len(bad)
    log(f"OpenFake: {len(groups) - len(bad)}/{len(groups)} phần row group đúng kích thước")
    if bad:
        log("OpenFake: không kiểm tên ảnh vì thiếu phần row group.")
        return problems
    try:
        import pyarrow.parquet as pq
    except ImportError:
        log("OpenFake: không kiểm được tên ảnh (Python này không có pyarrow).")
        return problems
    wanted = {r["image_id"][len("openfake_"):] for r in load_manifest("openfake.jsonl", manifests) if r.get("split") == "dev"}
    found = set()
    for shard in sorted({g["shard"] for g in groups}):
        gs = [g for g in groups if g["shard"] == shard]
        tail = src / "meta" / f"openfake_{shard}_tail.bin"
        pieces = [(g["start"], src / "data" / "openfake" / f"{shard}_rg{int(g['rg']):02d}_{int(g['start'])}.bin") for g in gs]
        pf = pq.ParquetFile(SparseFile(int(gs[0]["total"]), pieces + [(int(gs[0]["total"]) - tail.stat().st_size, tail)]))
        for g in gs:
            found |= set(pf.read_row_group(int(g["rg"]), columns=["row_id"]).column("row_id").to_pylist()) & wanted
    problems += len(wanted) - len(found)
    log(f"OpenFake dev: {len(found)}/{len(wanted)} ảnh có trong các row group đã tải (chỉ đọc cột row_id)")
    return problems


# ---------------------------------------------------------------------------------------------- label files
def write_labels(path: Path, rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["file", "label", "loai", "nhom"], lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: r[k] for k in ("file", "label", "loai", "nhom")})
    data = ("\ufeff" + buf.getvalue()).encode("utf-8")
    path.write_bytes(data)
    return sha256_bytes(data)


def manifest_keys(manifests: Path = MANIFESTS) -> dict:
    """image ids / groups / hashes of every TEST image of the measured sets (from the manifests, no image needed)."""
    keys = {"image_id": set(), "group": set(), "sha256": set(), "decoded": set()}
    test = [r for r in load_manifest("cocoglide.jsonl", manifests)] + \
           [r for r in load_manifest("korus_png.jsonl", manifests) if r.get("split") == "test_korus"]
    for r in test:
        keys["image_id"].add(r["image_id"])
        keys["group"].add(r["group_id"])
        for k, f in (("sha256", "sha256"), ("decoded", "decoded_rgb_sha256"), ("sha256", "original_sha256")):
            if r.get(f):
                keys[k].add(r[f])
    return keys


def overlap(calib: list[dict], manifests: Path = MANIFESTS) -> dict:
    keys = manifest_keys(manifests)
    return {"image_id": sum(r["image_id"] in keys["image_id"] for r in calib),
            "nhom": sum(r["nhom"] in keys["group"] for r in calib),
            "sha256": sum(bool(r.get("sha256")) and r["sha256"] in keys["sha256"] for r in calib),
            "sha256_diem_anh": sum(bool(r.get("decoded")) and r["decoded"] in keys["decoded"] for r in calib)}


def summary_of(rows: list[dict]) -> dict:
    return {"n": len(rows), "n_that": sum(r["label"] == 0 for r in rows), "n_sua": sum(r["label"] == 1 for r in rows),
            "theo_loai": dict(sorted(Counter(f"{'that' if r['label'] == 0 else 'sua'}/{r['loai']}" for r in rows).items())),
            "so_nhom": len({r["nhom"] for r in rows})}


def main(argv=None, log=print) -> int:
    ap = argparse.ArgumentParser(description="Tạo file nhãn CocoGlide/Korus và tập hiệu chỉnh DefakerOne2 (không chạy mô hình).")
    ap.add_argument("--bo", nargs="+", choices=["cocoglide", "korus", "hieu_chinh"],
                    default=["cocoglide", "korus", "hieu_chinh"])
    ap.add_argument("--nguon", type=Path, default=DEFAULT_SRC, help="thư mục _cloud_bridge\\downloads")
    ap.add_argument("--dich", type=Path, default=DEFAULT_DST, help="thư mục ghi ảnh và file nhãn (ngoài repo)")
    ap.add_argument("--manifests", type=Path, default=MANIFESTS, help=argparse.SUPPRESS)
    ap.add_argument("--kiem-tra", action="store_true",
                    help="chỉ kiểm tra file tải về có đủ ảnh của manifest (đọc tên và kích thước, không giải nén)")
    a = ap.parse_args(argv)
    src, dst, man = a.nguon.resolve(), a.dich.resolve(), a.manifests.resolve()
    if a.kiem_tra:
        try:
            n = check_sources(src, log, man)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            log(f"[LỖI] {type(exc).__name__}: {exc}")
            return 2
        log("Kiểm tra xong: đủ ảnh." if n == 0 else f"Kiểm tra xong: {n} vấn đề.")
        return 0 if n == 0 else 1
    if dst == ROOT or ROOT in dst.parents:
        log("[LỖI] --dich phải nằm ngoài repo (không đưa ảnh lên git).")
        return 2
    if not src.is_dir():
        log(f"[LỖI] Không thấy thư mục nguồn {src}")
        return 2
    dst.mkdir(parents=True, exist_ok=True)
    split_path = dst / SPLIT_FILE
    info = json.loads(split_path.read_text(encoding="utf-8")) if split_path.is_file() else {}
    info.update({"script": "tools/tao_nhan_cong_khai.py", "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "nguon": str(src), "manifests": {p.name: sha256_file(p) for p in sorted(man.glob("*.jsonl"))},
                 "seed_tach_korus": SPLIT_SEED, "quy_tac": SPLIT_RULE_VI})
    info.setdefault("tap", {})
    try:
        for bo in a.bo:
            log(f"== {bo}")
            skipped = []
            if bo == "cocoglide":
                rows = prep_cocoglide(src, dst, log, man)
            elif bo == "korus":
                rows = prep_korus(src, dst, log, ("test_korus",), man)
            else:
                rows = prep_korus(src, dst, log, ("dev_korus",), man) + prep_dff_dev(src, dst, log, man)
                of = prep_openfake_dev(src, dst, log, man)
                if of is None:
                    skipped.append("OpenFake dev (cần pyarrow hoặc thiếu dữ liệu; xem nhật ký)")
                else:
                    rows += of
                skipped.append("ctrl_mei dev (ảnh nguồn không có trên máy)")
                ov = overlap(rows, man)
                if any(ov.values()):
                    log(f"[LỖI] Tập hiệu chỉnh trùng với tập đo: {ov}. Dừng, không ghi file nhãn.")
                    return 2
                info["kiem_tra_trung_voi_tap_do"] = {**ov, "tap_do": "CocoGlide + test_korus (theo manifest)"}
                bad = [r for r in rows if not str(r.get("split", "")).startswith("dev")]
                if bad:
                    log(f"[LỖI] {len(bad)} ảnh không thuộc phần dev: {bad[0]['file']}. Dừng.")
                    return 2
            digest = write_labels(dst / LABEL_FILES[bo], rows)
            info["tap"][bo] = {"file_nhan": LABEL_FILES[bo], "sha256": digest, **summary_of(rows),
                               "tach": ("hieu_chinh" if bo == "hieu_chinh" else "test"), "bo_qua": skipped}
            log(f"   -> {dst / LABEL_FILES[bo]} ({len(rows)} dòng)")
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        log(f"[LỖI] {type(exc).__name__}: {exc}")
        return 2
    finally:
        split_path.write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    log(f"Xong. Mô tả cách tách: {split_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
