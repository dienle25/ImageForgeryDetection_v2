"""tools/tao_nhan_cong_khai.py, tools/do_lai_defaker2.py, tools/chon_nguong_defaker2.py on small generated data:
split ZIPs, tiny manifests and a fake DefakerOne2 server.  No real dataset and no model of DefakerOne2 is used."""
import csv
import hashlib
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fake_defaker2 import ANSWER, dead_url, fake_server

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import chon_nguong_defaker2 as cn  # noqa: E402
import do_lai_defaker2 as dl  # noqa: E402
import tao_nhan_cong_khai as tn  # noqa: E402
from ifd.config import load_config  # noqa: E402
from ifd.detectors.defaker2 import threshold_info  # noqa: E402
from ifd.image_io import decoded_pixel_hash  # noqa: E402

sha = lambda b: hashlib.sha256(b).hexdigest()


def img(h, w, seed, fmt):
    arr = np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt, **({"quality": 90} if fmt == "JPEG" else {}))
    return arr, buf.getvalue()


def zip_bytes(entries: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def make_downloads(tmp: Path):
    """Fake _cloud_bridge/downloads + manifests with the same layout as the real ones."""
    src, man = tmp / "downloads", tmp / "manifests"
    (src / "data" / "dff").mkdir(parents=True)
    (src / "meta").mkdir()
    man.mkdir()
    coco_rows, coco = [], {}
    for i in range(2):
        for lab, folder in ((0, "real"), (1, "fake")):
            arr, data = img(32, 32, 10 + 2 * i + lab, "PNG")
            name = f"{folder}/obj_{i}{'_up' if lab else ''}.png"
            coco[name] = data
            coco_rows.append({"image": name, "sha256": sha(data), "decoded_rgb_sha256": decoded_pixel_hash(arr),
                              "image_id": f"cocoglide_{folder}_{i}", "label": lab, "group_id": f"coco_{i}",
                              "split": "test_cocoglide"})
    (src / "data" / "CocoGlide.zip").write_bytes(zip_bytes(coco))
    korus_rows, korus = [], {}
    for i, split in enumerate(["dev_korus", "dev_korus", "test_korus"]):
        for lab, sub in ((0, "pristine"), (1, "tampered-realistic")):
            arr, tif = img(24, 40, 100 + 2 * i + lab, "TIFF")
            name = f"data-images/CamA/{sub}/P{i}.TIF"
            korus[name] = tif
            korus_rows.append({"image_id": f"korus_{'pristine' if lab == 0 else 'tampered'}_CamA_P{i}", "label": lab,
                               "group_id": f"korus_CamA_P{i}", "split": split, "original_file": name,
                               "original_sha256": sha(tif), "decoded_rgb_sha256": decoded_pixel_hash(arr)})
    zb = zip_bytes(korus)
    a, b = len(zb) // 3, 2 * len(zb) // 3
    (src / "data" / "korus_realistic_tampering.zip.part000").write_bytes(zb[:a])
    (src / "data" / "korus_realistic_tampering.zip.part001").write_bytes(zb[a:b])
    (src / "data" / f"korus_rest_from_{b}.bin.part000").write_bytes(zb[b:])
    dff_rows = []
    for folder, lab in (("wiki", 0), ("insight", 1)):
        entries = {}
        for i in range(3):
            _, data = img(40, 30, 200 + i + 10 * lab, "JPEG")
            name = f"{folder}/00/p{i}.jpg"
            entries[name] = data
            dff_rows.append({"image_id": f"dff_{folder}_00_p{i}", "image": name, "sha256": sha(data), "label": lab,
                             "group_id": f"dff_00_p{i}", "split": "dev"})
        zb = zip_bytes(entries)
        (src / "data" / "dff" / f"{folder}.zip.part000").write_bytes(zb[: len(zb) // 2])
        (src / "data" / "dff" / f"{folder}.zip.part001").write_bytes(zb[len(zb) // 2:])
    write_jsonl(man / "cocoglide.jsonl", coco_rows)
    write_jsonl(man / "korus_png.jsonl", korus_rows)
    write_jsonl(man / "dff_faceswap.jsonl", dff_rows)
    write_jsonl(man / "dff_extra.jsonl", [r for r in dff_rows if r["label"] == 0])      # same reals twice: deduplicated
    write_jsonl(man / "openfake.jsonl", [{"image_id": "openfake_openfake-x", "image": "images/openfake-x.png",
                                          "sha256": "0" * 64, "label": 0, "group_id": "of_x", "split": "dev",
                                          "real_source": "imagenet"}])
    (man / "openfake_row_groups.json").write_text(json.dumps([{"shard": "test0", "rg": 0, "start": 4, "end_excl": 10,
                                                               "total": 100, "rows": 1}]), encoding="utf-8")
    return src, man, korus_rows


def read_csv(p: Path):
    return list(csv.DictReader(io.StringIO(p.read_text(encoding="utf-8-sig"))))


def test_tao_nhan_extracts_verifies_and_keeps_the_locked_split(tmp_path):
    src, man, korus_rows = make_downloads(tmp_path)
    dst, logs = tmp_path / "cong_khai", []
    assert tn.main(["--nguon", str(src), "--dich", str(dst), "--manifests", str(man)], log=logs.append) == 0
    coco, ktest, calib = (read_csv(dst / tn.LABEL_FILES[k]) for k in ("cocoglide", "korus", "hieu_chinh"))
    assert len(coco) == 4 and {r["loai"] for r in coco} == {"cocoglide_that", "cocoglide_glide"}
    assert [r["file"] for r in ktest] == ["korus/korus_pristine_CamA_P2.png", "korus/korus_tampered_CamA_P2.png"]
    files = [r["file"] for r in calib]
    assert sum(f.startswith("korus/") for f in files) == 4 and sum(f.startswith("dff/") for f in files) == 6
    assert not set(files) & {r["file"] for r in ktest + coco}                     # calibration never in a test set
    assert {r["loai"] for r in calib} == {"korus_that", "korus_sua", "dff_wiki", "dff_insight"}
    for r in korus_rows:                                                          # lossless TIFF -> PNG
        with Image.open(dst / "korus" / f"{r['image_id']}.png") as im:
            assert decoded_pixel_hash(np.asarray(im.convert("RGB"))) == r["decoded_rgb_sha256"]
    info = json.loads((dst / tn.SPLIT_FILE).read_text(encoding="utf-8"))
    assert info["tap"]["hieu_chinh"]["sha256"] == sha((dst / tn.LABEL_FILES["hieu_chinh"]).read_bytes())
    assert info["tap"]["hieu_chinh"]["tach"] == "hieu_chinh" and info["tap"]["korus"]["tach"] == "test"
    assert set(info["kiem_tra_trung_voi_tap_do"].values()) <= {0, "CocoGlide + test_korus (theo manifest)"}
    assert any("OpenFake" in s for s in info["tap"]["hieu_chinh"]["bo_qua"]) and info["seed_tach_korus"] == 20261012
    logs.clear()
    assert tn.main(["--nguon", str(src), "--dich", str(dst), "--manifests", str(man)], log=logs.append) == 0
    assert any("(0 ảnh mới" in line for line in logs)                              # second run: nothing extracted again


def test_tao_nhan_stops_when_calibration_overlaps_a_test_set(tmp_path):
    src, man, korus_rows = make_downloads(tmp_path)
    coco = [json.loads(l) for l in (man / "cocoglide.jsonl").read_text(encoding="utf-8").splitlines()]
    coco[0]["decoded_rgb_sha256"] = korus_rows[0]["decoded_rgb_sha256"]           # same pixels in dev and test
    write_jsonl(man / "cocoglide.jsonl", coco)
    dst, logs = tmp_path / "cong_khai", []
    assert tn.main(["--bo", "hieu_chinh", "--nguon", str(src), "--dich", str(dst), "--manifests", str(man)],
                   log=logs.append) == 2
    assert not (dst / tn.LABEL_FILES["hieu_chinh"]).exists() and any("trùng với tập đo" in line for line in logs)
    assert tn.main(["--dich", str(ROOT / "anh_trong_repo"), "--nguon", str(src)], log=logs.append) == 2   # never in the repo


def test_tao_nhan_check_only_reads_names(tmp_path):
    src, man, _ = make_downloads(tmp_path)
    logs = []
    rc = tn.main(["--kiem-tra", "--nguon", str(src), "--dich", str(tmp_path / "khong_tao"), "--manifests", str(man)],
                 log=logs.append)
    text = "\n".join(logs)
    assert "CocoGlide (test): 4/4" in text and "Korus dev_korus: 4/4" in text and "Korus test_korus: 2/2" in text
    assert "DeepFakeFace dev/wiki: 3/3" in text and "OpenFake: 0/1" in text    # the fake row group file is absent
    assert rc == 1 and not (tmp_path / "khong_tao").exists()                    # nothing extracted or created


def test_sparse_and_concat_files(tmp_path):
    parts = []
    for i, chunk in enumerate((b"abc", b"defg", b"hi")):
        p = tmp_path / f"p{i}"
        p.write_bytes(chunk)
        parts.append(p)
    f = tn.ConcatFile(parts)
    f.seek(2)
    assert f.read(5) == b"cdefg" and f.read() == b"hi"
    sp = tn.SparseFile(20, [(5, parts[1])])
    sp.seek(5)
    assert sp.read(4) == b"defg"
    sp.seek(0)
    with pytest.raises(OSError):
        sp.read(3)


def test_do_lai_selection_and_estimate():
    rows = [{"file": f"t{i}", "label": "0", "nhom": f"g{i // 2}"} for i in range(6)] + \
           [{"file": f"s{i}", "label": "1", "nhom": f"h{i}"} for i in range(2)]
    chosen, note = dl.choose_rows(rows, thu=4)
    assert [r["file"] for r in chosen] == ["t0", "t1", "s0", "s1"] and "CHẠY THỬ" in note
    chosen, _ = dl.choose_rows(rows, thu=6)                                        # only 2 fakes: filled with reals
    assert sum(r["label"] == "1" for r in chosen) == 2 and len(chosen) == 6
    a, note = dl.choose_rows(rows, mau=4, seed=7)
    b, _ = dl.choose_rows(rows, mau=4, seed=7)
    assert a == b and "seed 7" in note and len(a) >= 4
    groups = {r["nhom"] for r in a}
    assert all(r in a for r in rows if r["nhom"] in groups)                       # whole groups only
    assert dl.choose_rows(rows)[0] == rows
    small, big = dl.estimate_seconds(256, 256, False), dl.estimate_seconds(1920, 1080, False)
    assert 4 <= small < big and dl.estimate_seconds(1920, 1080, True) > big


def make_khac_phuc(tmp: Path) -> Path:
    data = tmp / "data"
    (data / "that").mkdir(parents=True)
    (data / "sua" / "xoa_vat_the").mkdir(parents=True)
    (data / "khac_phuc").mkdir()
    lines = ["file,label,loai,nhom"]
    for i in range(3):
        (data / "that" / f"BIMAT_that_{i}.jpg").write_bytes(img(48, 64 + 8 * i, 300 + i, "JPEG")[1])
        lines.append(f"that/BIMAT_that_{i}.jpg,0,anh_chup,")
        (data / "sua" / "xoa_vat_the" / f"BIMAT_sua_{i}.jpg").write_bytes(img(48, 160 + 8 * i, 400 + i, "JPEG")[1])
        lines.append(f"sua/xoa_vat_the/BIMAT_sua_{i}.jpg,1,xoa_vat_the,")
    (data / "khac_phuc" / "nhan_A_xoa_vat_the.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (data / "khac_phuc" / "luat_nen.py").write_text(
        "print(\"nhan_A: 3 anh sua, 3 anh khong sua\")\n"
        "print(\"  AUROC luat 'anh cang nho cang nghi sua': 0.250\")\n"
        "print(\"  AUROC luat 'la PNG thi la sua':          0.500\")\n", encoding="utf-8")
    return data


def primary_cfg(tmp: Path, url: str) -> Path:
    data = json.loads((ROOT / "research" / "configs" / "config_defaker2_chinh.json").read_text(encoding="utf-8"))
    data["detectors"]["defaker2"]["url"] = url
    p = tmp / "cfg_chinh.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def test_do_lai_end_to_end_with_fake_server(tmp_path):
    data = make_khac_phuc(tmp_path)

    def respond(body):                     # sua images are wider: give them a high p_fake
        w = Image.open(io.BytesIO(body)).size[0]
        p = 0.9 if w >= 160 else 0.2
        return 200, {**ANSWER, "p_fake": p, "p_real": 1 - p}

    with fake_server(respond) as (url, received):
        rc = dl.main(["A", "--config", str(primary_cfg(tmp_path, url)), "--data", str(data),
                      "--bao-cao", str(tmp_path / "rep"), "--luu-dem", str(tmp_path / "cache")])
    assert rc == 0 and len(received) == 6
    out = next((tmp_path / "rep").glob("A_*"))
    s = json.loads((out / "tong_hop.json").read_text(encoding="utf-8"))
    d = s["defaker2"]
    assert d["auroc"] == pytest.approx(1.0) and d["n_that"] == 3 and d["n_sua"] == 3 and d["auroc_ci95"]
    assert d["tai_high_nghi_van"]["ty_le_bao_nham_fpr"] == 0 and d["tai_high_nghi_van"]["ty_le_phat_hien_tpr"] == 1
    assert s["luat_nen"]["auroc_anh_nho"] == pytest.approx(0.25) and s["luat_nen"]["auroc_la_png"] == pytest.approx(0.5)
    assert s["chon_anh"].startswith("toàn bộ") and s["thoi_gian_moi_anh_giay"]["n"] == 6
    assert s["ram"]["app_kiem_dinh_tu_do_gb"] is not None and s["ram"]["app_kiem_dinh_tu_do_gb"] > 0
    assert (out / "rieng_tu" / "du_doan.csv").is_file() and (out / "nhat_ky.log").is_file()
    md = (out / "tong_hop.md").read_text(encoding="utf-8")
    assert "Luật nền" in md and "Chưa hiệu chỉnh" in md
    for f in (out / "tong_hop.md", out / "tong_hop.json", out / "luat_nen.txt"):  # shareable files: no file names
        assert "BIMAT" not in f.read_text(encoding="utf-8")


def test_do_lai_stops_when_the_server_is_not_running(tmp_path, capsys):
    rc = dl.main(["B", "--config", str(primary_cfg(tmp_path, dead_url())), "--data", str(tmp_path / "khong_co"),
                  "--bao-cao", str(tmp_path / "rep"), "--luu-dem", str(tmp_path / "cache")])
    assert rc == 3 and r"Hãy chạy addons\defaker2\CHAY_DEFAKER2.bat trước" in capsys.readouterr().out
    assert dl.main(["A", "--config", str(ROOT / "config.json"), "--bao-cao", str(tmp_path / "rep")]) == 2


# --------------------------------------------------------------------------------------------- thresholds
def test_proposal_rule_per_real_source():
    rows = [{"file": f"a{i}", "label": 0, "loai": "nguon_a", "nhom": f"a{i}", "score": i / 100} for i in range(100)]
    rows += [{"file": f"b{i}", "label": 0, "loai": "nguon_b", "nhom": f"b{i}", "score": 0.6 + i / 100} for i in range(40)]
    rows += [{"file": f"s{i}", "label": 1, "loai": "sua", "nhom": f"s{i}", "score": 0.6 + i / 100} for i in range(30)]
    d = cn.propose(rows)
    assert d["nguon_quyet_dinh_high"] == "nguon_b" and d["high"] >= d["low"]
    for src, v in d["anh_that_theo_nguon"].items():                                 # <= 5 % / 20 % on EVERY source
        assert v["fpr_tai_high"]["ty_le"] <= 0.05 and v["fpr_tai_low"]["ty_le"] <= 0.20
    assert d["anh_that_theo_nguon"]["nguon_b"]["fpr_tai_high"]["so_anh_vuot"] == 2   # floor(0.05 * 40)
    assert d["high"] == cn.ceil9(d["high_chua_lam_tron"]) and d["phuong_an_gop_chung_tham_khao"]["high"] <= d["high"]
    assert any("nguon_b" in w and "< 50" in w for w in d["canh_bao"])
    assert d["gop"]["auroc"] is not None and d["anh_sua_theo_loai"]["sua"]["n"] == 30
    assert cn.ceil9(0.1234567891) == 0.123456790 and cn.ceil9(1.0000001) == 1.0
    lo, hi = cn.wilson(2, 40)
    assert lo < 0.05 < hi


def make_calibration(tmp: Path, n_per_source=60):
    pub = tmp / "cong_khai"
    pub.mkdir()
    rows, preds = [], []
    rng = np.random.default_rng(0)
    for src, prefix, lab in (("korus_that", "korus/k", 0), ("dff_wiki", "dff/wiki/w", 0), ("dff_insight", "dff/insight/i", 1)):
        for i in range(n_per_source):
            f = f"{prefix}{i}.png"
            rows.append({"file": f, "label": lab, "loai": src, "nhom": f"{src}_{i}"})
            score = float(rng.uniform(0.5, 1.0) if lab else rng.uniform(0.0, 0.9))
            preds.append({"file": f, "label": lab, "loai": src, "nhom": f"{src}_{i}", "status": "ok",
                          "defaker2_score": score})
    digest = tn.write_labels(pub / tn.LABEL_FILES["hieu_chinh"], rows)
    (pub / tn.SPLIT_FILE).write_text(json.dumps({"tap": {"hieu_chinh": {"sha256": digest, "tach": "hieu_chinh"}},
                                                 "quy_tac": {"hieu_chinh": "test"}}), encoding="utf-8")
    return pub, preds


def write_preds(path: Path, preds):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "loai", "nhom", "status", "defaker2_score"])
        w.writeheader()
        w.writerows(preds)
    return path


def test_chon_nguong_proposes_then_writes_only_with_ghi(tmp_path):
    pub, preds = make_calibration(tmp_path)
    du_doan = write_preds(tmp_path / "du_doan.csv", preds)
    rep = tmp_path / "rep"
    target = tmp_path / "research" / "configs" / "config_defaker2_mesorch.json"
    target.parent.mkdir(parents=True)
    shutil.copy(ROOT / "research" / "configs" / "config_defaker2_mesorch.json", target)
    before = target.read_bytes()
    common = ["--du-doan", str(du_doan), "--cong-khai", str(pub), "--data", str(tmp_path / "khong_co"), "--bao-cao", str(rep)]
    assert cn.main(common) == 0
    p = json.loads((rep / cn.PROPOSAL_NAME).read_text(encoding="utf-8"))
    assert p["da_ghi"] is None and 0 < p["de_xuat"]["low"] <= p["de_xuat"]["high"] <= 1
    assert (rep / "nguong_de_xuat.md").is_file() and target.read_bytes() == before        # nothing written without --ghi
    log = tmp_path / "EXPERIMENT_LOG.jsonl"
    log.write_text(json.dumps({"experiment_id": "D2T01_truoc_do"}) + "\n", encoding="utf-8")
    assert cn.main(common + ["--ghi", "--config-ghi", str(target), "--experiment-log", str(log)]) == 0
    p = json.loads((rep / cn.PROPOSAL_NAME).read_text(encoding="utf-8"))
    raw = target.read_bytes()
    assert b"\r\n" in raw                                                              # line endings kept
    th = load_config(target).detector("defaker2")["thresholds"]
    assert (th["low"], th["high"]) == (p["de_xuat"]["low"], p["de_xuat"]["high"])
    assert th["calibration"]["proposal_sha256"] == sha((rep / cn.PROPOSAL_NAME).read_bytes())
    assert th["calibration"]["experiment_id"].startswith("D2T02") and threshold_info({"thresholds": th})["calibrated"]
    entry = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert entry["experiment_id"].startswith("D2T02") and entry["ket_qua"]["high"] == th["high"]
    assert entry["config_json_da_khoa"] == "không đổi" and p["da_ghi"]["experiment_id"] == entry["experiment_id"]
    mes = load_config(target).detector("mesorch")["thresholds"]
    assert mes == load_config(ROOT / "config.json").detector("mesorch")["thresholds"]  # Mesorch untouched


def test_chon_nguong_refuses_test_data_and_the_locked_config(tmp_path):
    pub, preds = make_calibration(tmp_path)
    locked = (ROOT / "config.json").read_bytes()
    bad = preds + [{"file": "that/1792077546205_x.jpg", "label": 0, "loai": "anh_chup", "nhom": "", "status": "ok",
                    "defaker2_score": 0.97}]
    common = ["--cong-khai", str(pub), "--data", str(tmp_path / "khong_co"), "--bao-cao", str(tmp_path / "rep")]
    assert cn.main(["--du-doan", str(write_preds(tmp_path / "du_doan_ad.csv", bad)), "--ghi"] + common) == 2
    assert not (tmp_path / "rep" / cn.PROPOSAL_NAME).exists()
    good = write_preds(tmp_path / "du_doan.csv", preds)
    log = tmp_path / "log.jsonl"
    assert cn.main(["--du-doan", str(good), "--ghi", "--config-ghi", str(ROOT / "config.json"),
                    "--experiment-log", str(log)] + common) == 2
    assert (ROOT / "config.json").read_bytes() == locked and not log.exists()
    with pytest.raises(dl.PrepError):
        cn.check_target(ROOT / "config.json")
    (pub / tn.LABEL_FILES["hieu_chinh"]).write_text("file,label\nkorus/k0.png,0\n", encoding="utf-8")   # edited label file
    assert cn.main(["--du-doan", str(good)] + common) == 2
