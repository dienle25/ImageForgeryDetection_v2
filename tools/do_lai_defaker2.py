"""Đo lại DefakerOne2 (mô hình chính) — kèm Mesorch tham khảo — trên dữ liệu độc lập.  NGƯỜI DÙNG tự chạy.

    tools\\DO_LAI_DEFAKER2.bat <bo> [--thu 10] [--mau N --seed S] [--chi-defaker2] [--config FILE]
        bo = A | B | C | D | cocoglide | korus | tat_ca

Cần server DefakerOne2 (addons\\defaker2\\CHAY_DEFAKER2.bat) đang chạy ở 127.0.0.1:8765; nếu chưa, script báo và thoát.
* A–D: 4 file nhãn trong C:\\Users\\ASUS\\Downloads\\data\\khac_phuc (theo HUONG_DAN.md).  Chưa có thư mục data thì script
  giải nén Downloads\\data.zip và chép khac_phuc (không ghi đè file nào); chưa có ảnh data\\doi_chung thì chạy
  tao_doi_chung.py trước.
* cocoglide / korus: ảnh và file nhãn do tools/tao_nhan_cong_khai.py tạo (tự chạy nếu chưa có).  korus = CHỈ phần test
  (phần dev_korus nằm trong tập hiệu chỉnh, không bao giờ được đo ở đây).
* Mỗi bộ: luật nền (luat_nen.py), kiểm định (python -m ifd.company_eval) với cấu hình DefakerOne2, rồi ghi
  reports\\defaker2\\<bo>_<ngày giờ>\\: tong_hop.md + tong_hop.json (chỉ số gộp: AUROC kèm khoảng tin cậy, tỷ lệ báo nhầm,
  tỷ lệ phát hiện, luật nền, thời gian mỗi ảnh, RAM đỉnh), luat_nen.txt, nhat_ky.log, và chi_tiet\\ (bộ công khai)
  hoặc rieng_tu\\ (A–D: có tên file, KHÔNG đưa lên git).
* Điểm từng ảnh lưu đệm trong outputs\\defaker2_kiem_dinh: chạy lại sau khi đổi ngưỡng không phải chấm lại.
* Không chọn lọc ảnh.  --thu N (chạy thử) và --mau N (mẫu ngẫu nhiên theo nhóm, có seed) luôn được ghi rõ trong tổng hợp.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(Path(__file__).resolve().parent)):      # the bundled runtime does not add these itself
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tao_nhan_cong_khai as tnck  # noqa: E402
from ifd import __version__  # noqa: E402
from ifd.config import load_config  # noqa: E402
from ifd.errors import ForgeryError  # noqa: E402
from ifd.input_quality import REASONS_VI as QUALITY_REASONS_VI  # noqa: E402
from ifd.sysinfo import process_memory_mb, system_memory_mb  # noqa: E402

DATA_DIR = Path(os.environ.get("IFD_DATA_DIR", r"C:\Users\ASUS\Downloads\data"))
KHAC_PHUC_GOC = Path(os.environ.get("IFD_KHAC_PHUC_GOC", r"D:\ASUS\2\bo_khac_phuc_data_zip\khac_phuc"))
PUBLIC_DIR = Path(os.environ.get("IFD_CONG_KHAI_DIR", str(tnck.DEFAULT_DST)))
CFG_MESORCH = ROOT / "research" / "configs" / "config_defaker2_mesorch.json"
CFG_CHINH = ROOT / "research" / "configs" / "config_defaker2_chinh.json"
CACHE_DIR = ROOT / "outputs" / "defaker2_kiem_dinh"
REPORTS = ROOT / "reports" / "defaker2"
RAM_LIMIT_GB = 6.5            # the user asked to be told when the peak goes above about 6.5 GB
SLOW_SECONDS = 60             # ... or when one image takes longer than 60 s
SETS = {
    "A": ("khac_phuc", "nhan_A_xoa_vat_the.csv", "A — xóa vật thể (ảnh thật / ảnh xóa vật thể)"),
    "B": ("khac_phuc", "nhan_B_ghep_anh.csv", "B — ghép ảnh / đối chứng cắt từ đúng ảnh gốc"),
    "C": ("khac_phuc", "nhan_C_ghep_mat.csv", "C — ghép mặt / đối chứng"),
    "D": ("khac_phuc", "nhan_D_anh_AI.csv", "D — ảnh AI / ảnh thật cùng cỡ (chỉ khớp cỡ và định dạng)"),
    "cocoglide": ("cong_khai", tnck.LABEL_FILES["cocoglide"], "CocoGlide (toàn bộ là test)"),
    "korus": ("cong_khai", tnck.LABEL_FILES["korus"], "Korus Realistic Tampering — chỉ phần test"),
}
ALL_SETS = ["A", "B", "C", "D", "cocoglide", "korus"]
# DefakerOne2 seconds per image on the GTX 1650 of docs/DEFAKER2_ADDON_VI.md section 6, by pixels fed to the model
_TIME_POINTS = [(262144, 4.0), (746496, 9.0), (1704960, 23.0), (2088960, 30.0), (2408448, 41.0)]


class PrepError(Exception):
    pass


class Log:
    """Prints and writes every line to nhat_ky.log (UTF-8) so the user can send the file back when something fails."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.f = path.open("a", encoding="utf-8")

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


# ------------------------------------------------------------------------------------------------- server
def server_health(url: str, timeout: float = 3.0) -> dict:
    health = url.rsplit("/", 1)[0] + "/health"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))       # 127.0.0.1 only, never a proxy
    try:
        with opener.open(health, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except (OSError, ValueError) as exc:
        return {"state": "down", "error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(data, dict):
        return {"state": "error", "error": "phản hồi /health không hợp lệ"}
    return {"state": "ready" if data.get("ready") else ("error" if data.get("error") else "loading"), **data}


def check_server(url: str, log) -> dict | None:
    h = server_health(url)
    if h["state"] == "ready":
        log(f"Server DefakerOne2 sẵn sàng: {h.get('device')} · {h.get('mode')}.")
        return h
    if h["state"] == "down":
        log(r"Hãy chạy addons\defaker2\CHAY_DEFAKER2.bat trước (chờ dòng 'DA NAP XONG'), rồi chạy lại lệnh này.")
        log(f"   (không kết nối được {url}: {h.get('error')})")
    elif h["state"] == "loading":
        log("Server DefakerOne2 đang nạp mô hình: chờ dòng 'DA NAP XONG' trong cửa sổ server rồi chạy lại.")
    else:
        log(f"Server DefakerOne2 báo lỗi: {h.get('error')}. Xem cửa sổ server.")
    return None


def listening_pid(port: int) -> int | None:
    """PID of the process listening on 127.0.0.1:<port> (the DefakerOne2 server), Windows only."""
    if os.name != "nt":
        return None
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, timeout=20,
                             errors="replace").stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].upper() == "TCP" and parts[1].endswith(f":{port}") \
                and (parts[2] in ("0.0.0.0:0", "[::]:0") or "LISTEN" in parts[3].upper()):
            try:
                return int(parts[-1])
            except ValueError:
                continue
    return None


class RamSampler(threading.Thread):
    """Samples the working set of the validation process and of the DefakerOne2 server, and the RAM used by the
    whole machine, every 0.5 s.  Peaks are therefore approximate (a peak shorter than 0.5 s can be missed)."""

    def __init__(self, pids: dict, interval: float = 0.5):
        super().__init__(daemon=True)
        self.pids, self.interval = dict(pids), interval
        self._halt = threading.Event()
        self.peak = {k: None for k in self.pids}
        self.peak_sum = None
        self.sys_used_peak = self.sys_avail_min = self.sys_total = None
        self.samples = 0
        self.last = {}

    def sample(self) -> None:
        total, seen = 0.0, False
        for k, pid in self.pids.items():
            if not pid:
                continue
            m = process_memory_mb(pid)
            if m and m.get("working_set") is not None:
                ws = m["working_set"]
                self.peak[k] = ws if self.peak[k] is None else max(self.peak[k], ws)
                self.last[k] = m
                total, seen = total + ws, True
        if seen:
            self.peak_sum = total if self.peak_sum is None else max(self.peak_sum, total)
        s = system_memory_mb()
        if s:
            used = s["total"] - s["available"]
            self.sys_total = s["total"]
            self.sys_used_peak = used if self.sys_used_peak is None else max(self.sys_used_peak, used)
            self.sys_avail_min = s["available"] if self.sys_avail_min is None else min(self.sys_avail_min, s["available"])
        self.samples += 1

    def run(self) -> None:
        while not self._halt.is_set():
            self.sample()
            self._halt.wait(self.interval)

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=10)


def gb(mib):
    return None if mib is None else round(mib * 2 ** 20 / 1e9, 2)


# ----------------------------------------------------------------------------------------------- data prep
def read_labels(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig")
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        r = {(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
        if r.get("file"):
            rows.append(r)
    return rows


def label01(value: str) -> int | None:
    v = str(value).strip().lower()
    return 1 if v in ("1", "1.0", "sua") else (0 if v in ("0", "0.0", "that") else None)


def safe_extract(zpath: Path, dest: Path, log) -> int:
    """Extract a ZIP into dest without ever leaving dest and without overwriting existing files."""
    n = 0
    dest = dest.resolve()
    with zipfile.ZipFile(zpath) as zf:
        for info in zf.infolist():
            target = (dest / info.filename).resolve()
            if target != dest and dest not in target.parents:
                log(f"   bỏ qua mục ZIP ra ngoài thư mục đích: {info.filename}")
                continue
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            n += 1
    return n


def run_logged(cmd: list, log, cwd: Path | None = None) -> int:
    log("   $ " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    p = subprocess.run([str(c) for c in cmd], cwd=str(cwd or ROOT), capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env)
    for line in (p.stdout + p.stderr).splitlines():
        log("   " + line)
    return p.returncode


def prepare_khac_phuc(data_dir: Path, label_name: str, log) -> tuple[Path, Path]:
    kp = data_dir / "khac_phuc"
    if not ((data_dir / "that").is_dir() and (data_dir / "sua").is_dir()):
        zp = data_dir.parent / "data.zip"
        if not zp.is_file():
            raise PrepError(f"Không thấy {data_dir}\\that, {data_dir}\\sua và cũng không thấy {zp}. Giải nén data.zip vào "
                            f"{data_dir} theo HUONG_DAN.md rồi chạy lại.")
        log(f"Chưa có thư mục ảnh {data_dir}: giải nén {zp} (chỉ tạo file mới, không ghi đè) ...")
        log(f"   đã giải nén {safe_extract(zp, data_dir, log)} file.")
    if not (kp / label_name).is_file():
        if not (KHAC_PHUC_GOC / label_name).is_file():
            raise PrepError(f"Không thấy {kp / label_name} (và không thấy bản gốc ở {KHAC_PHUC_GOC}). Chép thư mục "
                            f"khac_phuc vào {kp} theo HUONG_DAN.md.")
        log(f"Chép các file của {KHAC_PHUC_GOC} vào {kp} (không ghi đè file đã có) ...")
        kp.mkdir(parents=True, exist_ok=True)
        for p in KHAC_PHUC_GOC.iterdir():
            if p.is_file() and not (kp / p.name).exists():
                shutil.copy2(p, kp / p.name)
    label_csv = kp / label_name
    rows = read_labels(label_csv)
    need_dc = [r for r in rows if r["file"].replace("\\", "/").startswith("doi_chung/")]
    if need_dc and any(not (data_dir / r["file"]).is_file() for r in need_dc):
        script = kp / "tao_doi_chung.py"
        if not script.is_file():
            raise PrepError(f"Thiếu ảnh đối chứng trong {data_dir / 'doi_chung'} và không thấy {script}.")
        log("Chưa có đủ ảnh đối chứng data\\doi_chung: chạy tao_doi_chung.py ...")
        if run_logged([sys.executable, "-X", "utf8", script, data_dir], log) != 0:
            raise PrepError("tao_doi_chung.py báo lỗi (xem nhật ký).")
    missing = [r["file"] for r in rows if not (data_dir / r["file"]).is_file()]
    if missing:
        log(f"[CẢNH BÁO] {len(missing)} ảnh trong {label_name} không có trên đĩa (ví dụ {missing[0]}); "
            "công cụ kiểm định sẽ ghi lỗi riêng cho từng ảnh.")
    return data_dir, label_csv


def prepare_public(bo: str, public_dir: Path, log) -> tuple[Path, Path]:
    label = public_dir / tnck.LABEL_FILES[bo]
    if not label.is_file():
        log(f"Chưa có {label}: chạy tools/tao_nhan_cong_khai.py --bo {bo} (giải nén, không chạy mô hình) ...")
        py = os.environ.get("IFD_PY_PYARROW") if bo == "hieu_chinh" else None
        py = py if py and Path(py).is_file() else sys.executable
        rc = run_logged([py, "-X", "utf8", ROOT / "tools" / "tao_nhan_cong_khai.py", "--bo", bo, "--dich", public_dir], log)
        if rc != 0 or not label.is_file():
            raise PrepError(f"Không tạo được {label} (xem nhật ký).")
    return public_dir, label


def luat_nen_script(data_dir: Path = DATA_DIR) -> Path | None:
    for p in (Path(data_dir) / "khac_phuc" / "luat_nen.py", KHAC_PHUC_GOC / "luat_nen.py"):
        if p.is_file():
            return p
    return None


# --------------------------------------------------------------------------------------------- selection
def choose_rows(rows: list[dict], thu: int | None = None, mau: int | None = None, seed: int | None = None):
    if thu:
        pos = [r for r in rows if label01(r.get("label")) == 1]
        neg = [r for r in rows if label01(r.get("label")) == 0]
        k1 = min(len(pos), (thu + 1) // 2)
        k0 = min(len(neg), thu - k1)
        k1 = min(len(pos), thu - k0)                       # one class short: fill with the other
        keep = {id(r) for r in pos[:k1] + neg[:k0]}
        chosen = [r for r in rows if id(r) in keep]
        return chosen, f"CHẠY THỬ {len(chosen)} ảnh đầu tiên của file nhãn ({k0} ảnh thật + {k1} ảnh sửa đầu tiên)"
    if mau and mau < len(rows):
        order, groups = [], {}
        for r in rows:
            g = r.get("nhom") or r["file"]
            if g not in groups:
                groups[g] = []
                order.append(g)
            groups[g].append(r)
        rng = random.Random(seed)
        picked, n = set(), 0
        for g in rng.sample(order, len(order)):
            if n >= mau:
                break
            picked.add(g)
            n += len(groups[g])
        chosen = [r for r in rows if (r.get("nhom") or r["file"]) in picked]
        return chosen, (f"MẪU NGẪU NHIÊN theo nhóm: {len(chosen)}/{len(rows)} ảnh, {len(picked)}/{len(order)} nhóm, "
                        f"seed {seed} (random.Random, chọn cả nhóm)")
    return rows, f"toàn bộ file nhãn ({len(rows)} ảnh)"


def write_label_subset(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["file", "label", "loai", "nhom"], lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in ("file", "label", "loai", "nhom")})
    path.write_bytes(("\ufeff" + buf.getvalue()).encode("utf-8"))
    return path


def estimate_seconds(w: int, h: int, mesorch: bool) -> float:
    """Rough seconds per image: DefakerOne2 time by pixels fed to the model (GTX 1650 measurements) + app overhead."""
    px = min(max(w * h, _TIME_POINTS[0][0]), _TIME_POINTS[-1][0])
    t = _TIME_POINTS[-1][1]
    for (x0, y0), (x1, y1) in zip(_TIME_POINTS, _TIME_POINTS[1:]):
        if x0 <= px <= x1:
            t = y0 + (y1 - y0) * (px - x0) / (x1 - x0)
            break
    return t + 0.3 + 0.15 * w * h / 1e6 + (1.5 if mesorch else 0.0)


def estimate_minutes(data_dir: Path, rows: list[dict], mesorch: bool) -> float | None:
    from PIL import Image
    total, n = 0.0, 0
    for r in rows:
        try:
            with Image.open(data_dir / r["file"]) as im:          # header only: size, no decoding
                total += estimate_seconds(im.width, im.height, mesorch)
                n += 1
        except OSError:
            continue
    return total / 60 if n else None


# ---------------------------------------------------------------------------------------------- baseline
def _auroc(pos, neg):
    if not pos or not neg:
        return None
    s = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
    return s / (len(pos) * len(neg))


def baseline_rules(data_dir: Path, label_csv: Path, out_txt: Path, log, khac_phuc_data: Path = DATA_DIR) -> dict:
    """Run khac_phuc/luat_nen.py (the user's script) on this label file; same rules computed here if it is absent."""
    script = luat_nen_script(khac_phuc_data)
    if script is not None:
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        p = subprocess.run([sys.executable, "-X", "utf8", str(script), str(data_dir), str(label_csv)], capture_output=True,
                           text=True, encoding="utf-8", errors="replace", env=env)
        text = p.stdout + (("\n" + p.stderr) if p.stderr.strip() else "")
        out_txt.write_text(text, encoding="utf-8")

        def grab(pattern):
            m = re.search(pattern, text)
            if not m:
                return None
            try:
                v = float(m.group(1))
            except ValueError:
                return None
            return None if math.isnan(v) else v
        res = {"nguon": f"{script.name} (thư mục khac_phuc)", "ma_thoat": p.returncode,
               "auroc_anh_nho": grab(r"anh cang nho cang nghi sua'\s*:\s*(\S+)"),
               "auroc_la_png": grab(r"la PNG thi la sua'\s*:\s*(\S+)")}
        if p.returncode != 0:
            log(f"[CẢNH BÁO] luat_nen.py thoát với mã {p.returncode}; xem {out_txt.name}.")
        return res
    from PIL import Image
    small, png = {0: [], 1: []}, {0: [], 1: []}
    for r in read_labels(label_csv):
        y = label01(r.get("label"))
        if y is None:
            continue
        try:
            with Image.open(data_dir / r["file"]) as im:
                small[y].append(-(im.width * im.height))
                png[y].append(1 if im.format == "PNG" else 0)
        except OSError:
            continue
    res = {"nguon": "tính trong tools/do_lai_defaker2.py, cùng định nghĩa với luat_nen.py (không thấy luat_nen.py)",
           "auroc_anh_nho": _auroc(small[1], small[0]), "auroc_la_png": _auroc(png[1], png[0])}
    out_txt.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return res


# ---------------------------------------------------------------------------------------------- evaluation
def run_eval(data_dir: Path, label_csv: Path, cfg_path: Path, cache_dir: Path, server_pid, log):
    cmd = [sys.executable, "-X", "utf8", "-m", "ifd.company_eval", str(data_dir), "--labels", str(label_csv),
           "--config", str(cfg_path), "--output", str(cache_dir)]
    log("   $ " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    before = system_memory_mb()
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", env=env, bufsize=1)
    sampler = RamSampler({"app": proc.pid, "server": server_pid})
    sampler.start()
    report, slow = None, []
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            log("   " + line)
            m = re.search(r"\(([0-9]+(?:\.[0-9]+)?) s\)\s*$", line)
            if m and float(m.group(1)) > SLOW_SECONDS:
                slow.append(float(m.group(1)))
                log(f"   [CẢNH BÁO] ảnh vừa rồi mất {float(m.group(1)):.1f} s (> {SLOW_SECONDS} s).")
            m2 = re.match(r"Xong\. Báo cáo: (.+)$", line.strip())
            if m2:
                report = Path(m2.group(1).strip()).parent
        rc = proc.wait()
    finally:
        sampler.stop()
    app_seen = (sampler.peak.get("app") or 0) >= 50      # a venv python.exe is only a launcher: its own RAM is tiny
    ram = {"so_lan_lay_mau": sampler.samples, "chu_ky_giay": sampler.interval,
           "app_kiem_dinh_lay_mau_gb": gb(sampler.peak.get("app")) if app_seen else None,
           "server_defaker2_dinh_gb": gb(sampler.peak.get("server")),
           "app_cong_server_lay_mau_gb": gb(sampler.peak_sum) if app_seen else None,
           "server_dinh_tu_luc_khoi_dong_gb": gb((sampler.last.get("server") or {}).get("peak_working_set")),
           "he_thong_dang_dung_dinh_gb": gb(sampler.sys_used_peak), "he_thong_trong_it_nhat_gb": gb(sampler.sys_avail_min),
           "he_thong_tong_gb": gb(sampler.sys_total),
           "he_thong_dang_dung_truoc_khi_chay_gb": gb(before["total"] - before["available"]) if before else None,
           "ghi_chu": ("Working set lấy mẫu mỗi 0,5 s (đỉnh ngắn hơn có thể bị sót). 'app' = tiến trình kiểm định (nạp mô "
                       "hình ONNX như app), 'server' = server DefakerOne2. 'cận trên' = đỉnh app (tự đo) + đỉnh server, "
                       "hai đỉnh có thể không cùng lúc. GB = 10^9 byte.")}
    if server_pid is None:
        ram["ghi_chu"] += " Không tìm được tiến trình server (chỉ đo được trên Windows)."
    return rc, report, ram, slow


def _pick(d, *keys):
    for k in keys:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def _conf(c: dict | None) -> dict | None:
    if not c:
        return None
    return {"tp": c.get("tp"), "fp": c.get("fp"), "tn": c.get("tn"), "fn": c.get("fn"),
            "ty_le_bao_nham_fpr": c.get("fpr"), "ty_le_phat_hien_tpr": c.get("recall", c.get("tpr"))}


def summarize(bo: str, m: dict, extra: dict) -> dict:
    primary = m.get("primary_detector") or "defaker2"
    d = (m.get("detectors") or {}).get(primary) or {}
    out = {"bo": bo, "ten_vi": SETS[bo][2], **extra,
           "so_anh": m.get("counts"), "anh_trung_tinh_mot_lan": _pick(m, "dedup", "duplicates_counted_once"),
           "nguong": (m.get("thresholds") or {}).get(primary),
           primary: {"n_that": d.get("n_neg"), "n_sua": d.get("n_pos"), "auroc": d.get("auroc"),
                     "auroc_ci95": d.get("auroc_ci95"), "ap": d.get("ap"),
                     "tai_high_nghi_van": _conf(d.get("at_high")), "tai_low_khong_ket_luan_tro_len": _conf(d.get("at_low")),
                     "vung_khong_ket_luan": d.get("abstain_fraction")},
           "ket_luan_app": {"nghi_van_la_duong": _conf(_pick(m, "product", "nghi_van_la_duong")),
                            "tu_muc_giua_tro_len_la_duong": _conf(_pick(m, "product", "tu_can_kiem_tra_tro_len_la_duong")),
                            "theo_nhom": _pick(m, "product", "by_type")},
           "theo_canh_bao_do_tin_cay": {k: (v or {}).get("nghi_van_la_duong") for k, v in (m.get("by_input_quality") or {}).items()},
           "khong_thong_nhat": m.get("disagreement"), "thoi_gian_moi_anh_giay": m.get("timing_seconds_per_image"),
           "ram_dinh_tien_trinh_kiem_dinh_mib": m.get("process_peak_ram_mb"), "so_anh_cham_moi": m.get("n_scored_now"),
           "so_anh_loi": len(m.get("errors") or []), "canh_bao_kiem_dinh": m.get("warnings")}
    for k, b in (m.get("detectors") or {}).items():
        if k != primary:
            out[k] = {"vai_tro": "tham khảo (không quyết định kết luận)", "n_that": b.get("n_neg"), "n_sua": b.get("n_pos"),
                      "auroc": b.get("auroc"), "auroc_ci95": b.get("auroc_ci95"), "ap": b.get("ap"),
                      "tai_high": _conf(b.get("at_high")), "tai_low": _conf(b.get("at_low"))}
    return out


def pc(x):
    return "—" if x is None else f"{100 * x:.1f} %".replace(".", ",")


def g2(x):
    return "—" if x is None else f"{x:.2f}".replace(".", ",")


def when(stamp: str) -> str:
    try:
        return datetime.strptime(stamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return stamp


def f3(x):
    return "—" if x is None else f"{x:.3f}".replace(".", ",")


def ci_text(ci):
    return "" if not ci else f" [{f3(ci[0])}–{f3(ci[1])}]"


def render_md(s: dict) -> str:
    prim = "defaker2"
    d = s.get(prim) or {}
    th = s.get("nguong") or {}
    L = [f"# Đo lại DefakerOne2 — {s['ten_vi']}", "",
         f"- Thời điểm: {when(s['thoi_diem'])} · ứng dụng {s['phien_ban_app']} · commit `{s.get('git_commit') or '—'}`",
         f"- Cấu hình: `{s['cau_hinh']}` (SHA-256 {s['cau_hinh_sha256'][:12]}…)",
         f"- Ngưỡng DefakerOne2: low {th.get('low')} / high {th.get('high')} — {s.get('nguong_mo_ta') or ''}",
         f"- Server: {_pick(s, 'server', 'device') or '—'} · {_pick(s, 'server', 'mode') or '—'}",
         f"- Ảnh dùng: {s['chon_anh']}",
         f"- Số ảnh: {json.dumps(s.get('so_anh'), ensure_ascii=False)} · ảnh lỗi: {s.get('so_anh_loi')}", ""]
    if s.get("lan_do_truoc"):
        L += [f"- Các lần đo trước của bộ này (giữ nguyên, không chọn lần đẹp nhất): {', '.join(s['lan_do_truoc'])}", ""]
    L += ["## DefakerOne2 (mô hình chính)", "",
          f"- AUROC {f3(d.get('auroc'))}{ci_text(d.get('auroc_ci95'))} (khoảng tin cậy 95 %, bootstrap theo nhóm, 1000 lần) · "
          f"AP {f3(d.get('ap'))} · ảnh thật {d.get('n_that')} · ảnh sửa {d.get('n_sua')}", "",
          "| Mốc | TP | FP | TN | FN | Tỷ lệ báo nhầm (FPR) | Tỷ lệ phát hiện (TPR) |", "|---|---|---|---|---|---|---|"]
    for key, title in (("tai_high_nghi_van", "≥ high → “Nghi vấn”"), ("tai_low_khong_ket_luan_tro_len", "≥ low → “Không kết luận” trở lên")):
        c = d.get(key) or {}
        L.append(f"| {title} | {c.get('tp')} | {c.get('fp')} | {c.get('tn')} | {c.get('fn')} | "
                 f"{pc(c.get('ty_le_bao_nham_fpr'))} | {pc(c.get('ty_le_phat_hien_tpr'))} |")
    ln = s.get("luat_nen") or {}
    L += ["", "## Luật nền (không nhìn nội dung ảnh)", "",
          "| Cách | AUROC |", "|---|---|",
          f"| DefakerOne2 | {f3(d.get('auroc'))}{ci_text(d.get('auroc_ci95'))} |",
          f"| Luật “ảnh càng nhỏ càng nghi sửa” | {f3(ln.get('auroc_anh_nho'))} |",
          f"| Luật “là PNG thì là sửa” | {f3(ln.get('auroc_la_png'))} |", "",
          f"Nguồn luật nền: {ln.get('nguon', '—')}. DefakerOne2 không vượt rõ hai luật này thì kết quả chưa chứng minh được gì.", ""]
    for k, v in s.items():
        if isinstance(v, dict) and v.get("vai_tro"):
            L += [f"## {k} (tham khảo)", "",
                  f"- AUROC {f3(v.get('auroc'))}{ci_text(v.get('auroc_ci95'))} · AP {f3(v.get('ap'))} · "
                  f"tại high: FPR {pc(_pick(v, 'tai_high', 'ty_le_bao_nham_fpr'))}, TPR {pc(_pick(v, 'tai_high', 'ty_le_phat_hien_tpr'))}", ""]
    dis = s.get("khong_thong_nhat")
    if dis:
        L += ["## Hai mô hình không thống nhất", "",
              f"- {dis.get('n_khong_thong_nhat')}/{dis.get('n')} ảnh ({pc(dis.get('ty_le'))}); ảnh thật {pc(dis.get('ty_le_anh_that'))}, "
              f"ảnh sửa {pc(dis.get('ty_le_anh_sua'))}. {dis.get('rule_vi', '')}", ""]
    q = s.get("theo_canh_bao_do_tin_cay") or {}
    if q:
        L += ["## Tách theo cảnh báo “Độ tin cậy thấp” (kết luận “Nghi vấn” là dương)", "",
              "| Nhóm | n | ảnh thật | ảnh sửa | Tỷ lệ đúng | Báo nhầm (FPR) | Phát hiện (TPR) |", "|---|---|---|---|---|---|---|"]
        names = {"co_canh_bao": "Có cảnh báo", "khong_canh_bao": "Không cảnh báo", "khong_ro": "Không rõ"}
        names.update({f"ly_do:{c}": f"– lý do: {t}" for c, t in QUALITY_REASONS_VI.items()})
        for k, c in q.items():
            if c and (c.get("n") or not k.startswith("ly_do:")):
                L.append(f"| {names.get(k, k)} | {c.get('n')} | {c.get('n_that')} | {c.get('n_sua')} | {pc(c.get('ty_le_dung'))} | "
                         f"{pc(c.get('fpr'))} | {pc(c.get('tpr'))} |")
        L.append("")
    t = s.get("thoi_gian_moi_anh_giay") or {}
    ram = s.get("ram") or {}
    L += ["## Thời gian và RAM", "",
          f"- Mỗi ảnh (cả phân tích): trung bình {f3(t.get('mean'))} s · trung vị {f3(t.get('median'))} s · lâu nhất "
          f"{f3(t.get('max'))} s · {t.get('over_limit', '—')} ảnh quá {SLOW_SECONDS} s.",
          f"- Riêng DefakerOne2: trung vị {f3(_pick(t, 'by_detector', 'defaker2', 'median'))} s · lâu nhất "
          f"{f3(_pick(t, 'by_detector', 'defaker2', 'max'))} s.",
          f"- RAM đỉnh: app kiểm định {g2(ram.get('app_kiem_dinh_tu_do_gb'))} GB (tự đo) · server DefakerOne2 "
          f"{g2(ram.get('server_defaker2_dinh_gb'))} GB (lấy mẫu; từ lúc khởi động {g2(ram.get('server_dinh_tu_luc_khoi_dong_gb'))} GB)"
          f" · app + server {g2(ram.get('app_cong_server_lay_mau_gb'))} GB cùng lúc (lấy mẫu), cận trên "
          f"{g2(ram.get('app_cong_server_can_tren_gb'))} GB · cả máy đang dùng {g2(ram.get('he_thong_dang_dung_dinh_gb'))}/"
          f"{g2(ram.get('he_thong_tong_gb'))} GB (trước khi chạy {g2(ram.get('he_thong_dang_dung_truoc_khi_chay_gb'))} GB).",
          f"- Server tự báo (MiB): {json.dumps(_pick(s, 'server', 'memory') or {}, ensure_ascii=False)}", ""]
    if s.get("canh_bao"):
        L += ["## Cảnh báo", ""] + [f"- {w}" for w in s["canh_bao"]] + [""]
    L += ["## Giới hạn", "",
          "- Số liệu tác giả DefakerOne2 chưa được bình duyệt độc lập; ở đây chạy GPU fp16 + nén 8-bit.",
          "- Cỡ mẫu nhỏ: xem khoảng tin cậy; khác biệt nhỏ hơn độ rộng khoảng tin cậy không có ý nghĩa.",
          "- Bộ B–D dùng ảnh đối chứng do máy tạo (cắt/thu nhỏ từ ảnh gốc), không phải ảnh chụp màn hình thật; bộ D chỉ khớp "
          "cỡ và định dạng, không khớp nội dung.",
          "- Thời gian và RAM đo trên máy chạy lệnh, cùng các ứng dụng đang mở lúc đó.", ""]
    return "\n".join(L) + "\n"


def git_commit() -> str | None:
    """Short commit of the code that produced the numbers (+ a note when tracked files differ from it)."""
    try:
        p = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10)
        head = p.stdout.strip() or None
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return f"{head} + sửa chưa commit" if head and dirty else head
    except (OSError, subprocess.SubprocessError):
        return None


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_set(bo: str, a, cfg_path: Path, url: str, stamp: str, log, reports: Path) -> tuple[int, dict | None]:
    kind, label_name, title = SETS[bo]
    out_dir = reports / f"{bo}_{stamp}"
    detail = out_dir / ("rieng_tu" if kind == "khac_phuc" else "chi_tiet")
    detail.mkdir(parents=True, exist_ok=True)
    log("")
    log(f"===== {title} -> {out_dir}")
    warnings = []
    try:
        if kind == "khac_phuc":
            data_dir, label_csv = prepare_khac_phuc(a.data, label_name, log)
        else:
            data_dir, label_csv = prepare_public(bo, a.cong_khai, log)
    except (PrepError, OSError, zipfile.BadZipFile) as exc:
        log(f"[LỖI] {exc}")
        return 2, None
    rows = read_labels(label_csv)
    chosen, note = choose_rows(rows, a.thu, a.mau, a.seed)
    if len(chosen) != len(rows):
        label_csv = write_label_subset(chosen, detail / "nhan_dung_lan_nay.csv")
        warnings.append(f"Không phải toàn bộ bộ ảnh: {note}.")
    log(f"Ảnh dùng: {note}.")
    est = estimate_minutes(data_dir, chosen, mesorch=a.mesorch)
    if est is not None:
        log(f"Ước tính thời gian: khoảng {est:.0f} phút nếu chưa có điểm lưu đệm (theo số đo GTX 1650 ở "
            "docs/DEFAKER2_ADDON_VI.md mục 6; máy khác sẽ khác).")
    if not check_server(url, log):
        return 3, None
    log("Luật nền ...")
    ln = baseline_rules(data_dir, label_csv, out_dir / "luat_nen.txt", log, a.data)
    log(f"   luật 'ảnh nhỏ' AUROC {f3(ln.get('auroc_anh_nho'))} · luật 'là PNG' AUROC {f3(ln.get('auroc_la_png'))}")
    log("Kiểm định (ifd.company_eval) ...")
    port = urllib.parse.urlsplit(url).port
    rc, rep_dir, ram, slow = run_eval(data_dir, label_csv, cfg_path, a.luu_dem, listening_pid(port), log)
    if rep_dir is None or not (rep_dir / "metrics.json").is_file():
        log(f"[LỖI] Công cụ kiểm định không tạo được báo cáo (mã thoát {rc}). Gửi file {log.path} để được hỗ trợ.")
        return 2, None
    for name in ("metrics.json", "bao_cao_kiem_dinh.md", "bao_cao_kiem_dinh.html", "du_doan.csv"):
        shutil.copy2(rep_dir / name, detail / name)
    m = json.loads((rep_dir / "metrics.json").read_text(encoding="utf-8"))
    health = server_health(url)
    app_self = m.get("process_peak_ram_mb") if m.get("n_scored_now") else None   # measured inside company_eval
    ram["app_kiem_dinh_tu_do_gb"] = gb(app_self)
    srv = ram.get("server_defaker2_dinh_gb")
    ram["app_cong_server_can_tren_gb"] = round(ram["app_kiem_dinh_tu_do_gb"] + srv, 2) \
        if ram["app_kiem_dinh_tu_do_gb"] is not None and srv is not None else None
    if slow:
        warnings.append(f"{len(slow)} ảnh mất hơn {SLOW_SECONDS} s (lâu nhất {max(slow):.0f} s).")
    both = max([v for v in (ram.get("app_cong_server_lay_mau_gb"), ram.get("app_cong_server_can_tren_gb")) if v] or [0])
    if both > RAM_LIMIT_GB:
        warnings.append(f"RAM đỉnh app + server khoảng {both} GB, vượt {RAM_LIMIT_GB} GB.")
    if ram.get("he_thong_dang_dung_dinh_gb") and ram["he_thong_dang_dung_dinh_gb"] > RAM_LIMIT_GB:
        warnings.append(f"RAM cả máy đang dùng lúc đỉnh {ram['he_thong_dang_dung_dinh_gb']} GB, vượt {RAM_LIMIT_GB} GB "
                        "(gồm cả ứng dụng khác đang mở).")
    if rc == 1:
        warnings.append(f"Có {len(m.get('errors') or [])} ảnh lỗi (không tính vào chỉ số); danh sách trong "
                        f"{detail.name}/bao_cao_kiem_dinh.md.")
    from ifd.detectors.defaker2 import threshold_info
    cfg = load_config(cfg_path)
    prior = sorted(p.name for p in reports.glob(f"{bo}_*") if p.is_dir() and p != out_dir)
    extra = {"thoi_diem": stamp, "phien_ban_app": __version__, "git_commit": git_commit(),
             "cau_hinh": str(cfg_path.relative_to(ROOT)) if ROOT in cfg_path.resolve().parents else str(cfg_path),
             "cau_hinh_sha256": sha256_file(cfg_path), "nguong_mo_ta": threshold_info(cfg.detector("defaker2"))["text_vi"],
             "server": {k: health.get(k) for k in ("device", "mode", "pixels_policy", "memory", "model") if k in health},
             "chon_anh": note, "seed": a.seed if a.mau else None, "file_nhan": label_csv.name,
             "file_nhan_sha256": sha256_file(label_csv), "luat_nen": ln, "ram": ram, "canh_bao": warnings,
             "lan_do_truoc": prior, "ma_thoat_kiem_dinh": rc}
    s = summarize(bo, m, extra)
    (out_dir / "tong_hop.json").write_text(json.dumps(s, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (out_dir / "tong_hop.md").write_text(render_md(s), encoding="utf-8")
    log(f"Tổng hợp: {out_dir / 'tong_hop.md'}")
    for w in warnings:
        log(f"[CẢNH BÁO] {w}")
    return (1 if rc == 1 else 0), s


def render_all(results: dict, stamp: str) -> str:
    L = [f"# Đo lại DefakerOne2 — tất cả các bộ ({stamp})", "",
         "| Bộ | ảnh thật | ảnh sửa | AUROC [CI 95 %] | Báo nhầm tại high | Phát hiện tại high | Luật ảnh nhỏ | Luật PNG | "
         "Trung vị s/ảnh | Thư mục |", "|---|---|---|---|---|---|---|---|---|---|"]
    for bo, s in results.items():
        if not s:
            L.append(f"| {bo} | lỗi — xem nhật ký | | | | | | | | |")
            continue
        d = s.get("defaker2") or {}
        hi = d.get("tai_high_nghi_van") or {}
        L.append(f"| {bo} | {d.get('n_that')} | {d.get('n_sua')} | {f3(d.get('auroc'))}{ci_text(d.get('auroc_ci95'))} | "
                 f"{pc(hi.get('ty_le_bao_nham_fpr'))} | {pc(hi.get('ty_le_phat_hien_tpr'))} | "
                 f"{f3(_pick(s, 'luat_nen', 'auroc_anh_nho'))} | {f3(_pick(s, 'luat_nen', 'auroc_la_png'))} | "
                 f"{f3(_pick(s, 'thoi_gian_moi_anh_giay', 'median'))} | `{bo}_{stamp}` |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tools\\DO_LAI_DEFAKER2.bat",
                                 description="Đo lại DefakerOne2 (mô hình chính) trên dữ liệu độc lập. Cần server DefakerOne2.")
    ap.add_argument("bo", choices=ALL_SETS + ["tat_ca"], help="A, B, C, D, cocoglide, korus hoặc tat_ca")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--thu", type=int, metavar="N", help="chạy thử N ảnh đầu (nửa thật, nửa sửa) để kiểm tra đường chạy")
    g.add_argument("--mau", type=int, metavar="N", help="mẫu ngẫu nhiên khoảng N ảnh, chọn cả nhóm, có seed (bộ quá nặng)")
    ap.add_argument("--seed", type=int, default=20261108, help="seed cho --mau (mặc định 20261108)")
    ap.add_argument("--chi-defaker2", action="store_true",
                    help="dùng config_defaker2_chinh.json (chỉ DefakerOne2) thay cho bản DefakerOne2 + Mesorch")
    ap.add_argument("--config", type=Path, default=None, help="cấu hình khác (phải có DefakerOne2 role primary)")
    ap.add_argument("--data", type=Path, default=DATA_DIR, help="thư mục data của data.zip (có that, sua, khac_phuc)")
    ap.add_argument("--cong-khai", type=Path, default=PUBLIC_DIR, help="thư mục ảnh công khai của tao_nhan_cong_khai.py")
    ap.add_argument("--bao-cao", type=Path, default=REPORTS, help=argparse.SUPPRESS)
    ap.add_argument("--luu-dem", type=Path, default=CACHE_DIR, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if (a.thu is not None and a.thu <= 0) or (a.mau is not None and a.mau <= 0):
        ap.error("--thu / --mau phải là số dương")
    cfg_path = (a.config or (CFG_CHINH if a.chi_defaker2 else CFG_MESORCH)).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    a.bao_cao.mkdir(parents=True, exist_ok=True)
    log = Log(a.bao_cao / f"{a.bo}_{stamp}" / "nhat_ky.log")
    try:
        log(f"DO_LAI_DEFAKER2 {a.bo} · ứng dụng {__version__} · Python {sys.version.split()[0]} · {sys.executable}")
        try:
            cfg = load_config(cfg_path)
        except ForgeryError as exc:
            log(f"[LỖI] Cấu hình {cfg_path}: {exc}")
            return 2
        det = cfg.data["detectors"].get("defaker2") or {}
        if cfg_path == (ROOT / "config.json").resolve() or not (det.get("enabled") and det.get("role") == "primary"):
            log("[LỖI] Cấu hình phải bật DefakerOne2 làm mô hình chính (role primary); config.json đã khóa không dùng ở đây.")
            return 2
        a.mesorch = bool((cfg.data["detectors"].get("mesorch") or {}).get("enabled"))
        log(f"Cấu hình: {cfg_path} (Mesorch {'bật' if a.mesorch else 'tắt'})")
        if not check_server(det["url"], log):
            log(f"Nhật ký: {log.path}")
            return 3
        sets = ALL_SETS if a.bo == "tat_ca" else [a.bo]
        results, codes = {}, []
        for bo in sets:
            code, s = run_set(bo, a, cfg_path, det["url"], stamp, log, a.bao_cao)
            results[bo], codes = s, codes + [code]
            if code == 3:
                break
        if a.bo == "tat_ca":
            (a.bao_cao / f"tat_ca_{stamp}" / "tong_hop.md").write_text(render_all(results, stamp), encoding="utf-8")
            log(f"Bảng chung: {a.bao_cao / f'tat_ca_{stamp}' / 'tong_hop.md'}")
        log(f"Nhật ký: {log.path}")
        return max(codes) if codes else 0
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
