"""Cảnh báo "Độ tin cậy thấp" (ifd/input_quality.py): one test per condition, images generated here (no real photo),
and the warning never changes a score or a decision."""
import contextlib
import csv
import io
import json
import unicodedata
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import light_config_data
from ifd.config import Config
from ifd.image_io import read_image
from ifd.input_quality import HEADLINE_VI, assess, looks_like_screenshot_name, mp_text

ROOT = Path(__file__).resolve().parents[1]
LIM = dict(max_file_mb=25, max_pixels=40_000_000, min_side=16)


def photo(h, w, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([xx / w * 200 + 20, yy / h * 180 + 30, (xx + yy) / (w + h) * 160 + 40], -1)
    return np.clip(base + rng.normal(0, 6, (h, w, 3)), 0, 255).astype(np.uint8)


def jpeg(h, w, exif=None):
    buf = io.BytesIO()
    kw = {"quality": 90}
    if exif:
        ex = Image.Exif()
        for k, v in exif.items():
            ex[k] = v
        kw["exif"] = ex.tobytes()
    Image.fromarray(photo(h, w)).save(buf, format="JPEG", **kw)
    return buf.getvalue()


def png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def rgba(h, w, transparent_pixel=False):
    a = np.concatenate([photo(h, w), np.full((h, w, 1), 255, np.uint8)], -1)
    if transparent_pixel:
        a[0, 0, 3] = 0
    return a


def quality(data: bytes, name: str) -> dict:
    img = read_image(data, name=name, **LIM)
    return assess(img.metadata, img.source_name)


def codes(q):
    return [r["code"] for r in q["reasons"]]


CAMERA = {271: "Canon", 272: "EOS 80D", 306: "2026:10:18 18:42:11"}


def test_small_image_below_half_megapixel():
    q = quality(jpeg(720, 480, CAMERA), "selfie.jpg")              # 0,3456 MP (like the reported false alarm)
    assert q["low_confidence"] is True and codes(q) == ["small_image"]
    assert q["reasons"][0]["text_vi"] == "0,34 MP, nhỏ hơn 0,5 MP"
    assert q["headline_vi"] == HEADLINE_VI == "Độ tin cậy thấp: ảnh nhỏ hoặc đã qua nén lại"
    assert q["text_vi"].startswith(HEADLINE_VI) and "0,34 MP" in q["text_vi"] and q["changes_decision"] is False


def test_png_alpha_without_transparent_pixel_is_screenshot_sign():
    q = quality(png(rgba(800, 1000)), "anh.png")                     # 0,8 MP, alpha channel fully opaque
    assert codes(q) == ["png_alpha_opaque"] and "trong suốt" in q["reasons"][0]["text_vi"]
    q2 = quality(png(rgba(800, 1000, transparent_pixel=True)), "anh.png")   # one transparent pixel: real alpha
    assert q2["low_confidence"] is False and q2["reasons"] == []


@pytest.mark.parametrize("name", ["Screenshot 2026-10-18 181055.png", "screenshot_20261018-184211.png",
                                  "Ảnh chụp màn hình 2026-10-18 184211.png",
                                  unicodedata.normalize("NFD", "Ảnh chụp màn hình 2026-10-18 184211.png"),
                                  "Ảnh màn hình 2026-10-18 lúc 18.42.11.png", "DC_Screenshot 2026-10-18 181055.png"])
def test_screenshot_file_name(name):
    q = quality(png(photo(800, 1000)), name)                        # RGB, no alpha, large: only the name triggers
    assert codes(q) == ["screenshot_name"] and "chụp màn hình" in q["reasons"][0]["text_vi"]


@pytest.mark.parametrize("name", ["IMG_2041.png", "man_hinh_tivi.png", "chup_anh_gia_dinh.png", "1792077546205_28.jpg",
                                  "thanh_man_hinh.png", "noscreenshot.png"])
def test_ordinary_names_are_not_screenshots(name):
    assert not looks_like_screenshot_name(name)


def test_jpeg_without_exif_below_2mp_is_messaging_app_sign():
    q = quality(jpeg(1000, 1200), "zalo.jpg")                        # 1,2 MP, no EXIF at all
    assert codes(q) == ["jpeg_no_exif_small"]
    assert "1,20 MP, nhỏ hơn 2 MP" in q["reasons"][0]["text_vi"] and "nhắn tin" in q["reasons"][0]["text_vi"]
    q2 = quality(jpeg(1000, 1200, {274: 1}), "zalo.jpg")             # orientation only = technical tag, still "no EXIF"
    assert codes(q2) == ["jpeg_no_exif_small"]


def test_no_exif_alone_is_not_a_warning():
    q = quality(jpeg(1200, 1800), "facebook.jpg")                    # 2,16 MP without EXIF: Zalo/Facebook HD
    assert q["low_confidence"] is False and q["reasons"] == []
    q2 = quality(jpeg(1000, 1200, CAMERA), "may_anh.jpg")            # 1,2 MP but with camera EXIF
    assert q2["low_confidence"] is False


def test_several_reasons_are_all_listed():
    q = quality(jpeg(500, 600), "Screenshot_1.jpg")                  # small + name + JPEG without EXIF
    assert codes(q) == ["small_image", "screenshot_name", "jpeg_no_exif_small"]


def test_megapixel_text_is_rounded_down():
    assert mp_text(0.4999) == "0,49" and mp_text(0.3456) == "0,34" and mp_text(0.065536) == "0,065"
    assert mp_text(1.9999) == "1,99"


def test_warning_never_changes_scores_or_decisions():
    """Same pixels, one name looks like a screenshot: identical scores and summary, only input_quality differs."""
    from ifd.analysis import Analyzer, report_files
    an = Analyzer(Config(light_config_data(), ROOT))                 # pscc + commfor (small models)
    data = png(photo(240, 320))                                      # 0,08 MP
    a = an.analyze(data, name="anh_thuong.png")
    b = an.analyze(data, name="Screenshot 2026-10-18 181055.png")
    assert [r.score for r in a.results] == [r.score for r in b.results]
    assert a.summary == b.summary
    js = json.loads(report_files(b, 0.5, images=False)["result.json"])
    assert js["input_quality"]["low_confidence"] is True
    assert codes(js["input_quality"]) == ["small_image", "screenshot_name"]
    assert codes(a.metadata["input_quality"]) == ["small_image"]
    assert js["image"]["alpha"] == {"present": False} and "exif_tags" in js["image"]


def _kit(tmp_path, cfg_data, labels: str, name="cfg"):
    from ifd.company_eval import main as kit_main
    cfg = tmp_path / f"{name}.json"
    cfg.write_text(json.dumps(cfg_data), encoding="utf-8")
    lab = tmp_path / f"{name}_nhan.csv"
    lab.write_text(labels, encoding="utf-8")
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        rc = kit_main([str(tmp_path / "d"), "--labels", str(lab), "--output", str(tmp_path / "out"), "--config", str(cfg)])
    work = next((tmp_path / "out" / "kiem_dinh").iterdir())
    rep = max((p for p in work.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime_ns)
    m = json.loads((rep / "metrics.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader(io.StringIO((rep / "du_doan.csv").read_text(encoding="utf-8-sig"))))
    return rc, m, rows, (rep / "bao_cao_kiem_dinh.md").read_text(encoding="utf-8"), out.getvalue(), work


def _kit_data(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "nho.jpg").write_bytes(jpeg(300, 400))                                   # warning: small + no EXIF
    (d / "lon.png").write_bytes(png(photo(800, 1000, seed=1)))                    # no warning
    (d / "Screenshot 2026-10-18 181055.png").write_bytes(png(photo(800, 1000, seed=2)))   # warning: name
    (d / "sua.png").write_bytes(png(photo(800, 1000, seed=3)))                    # no warning
    return "file,label,loai,nhom\nnho.jpg,0,that,g1\nlon.png,0,that,g2\nScreenshot 2026-10-18 181055.png,1,sua,g3\nsua.png,1,sua,g4\n"


def test_validation_kit_column_and_table_by_warning(tmp_path):
    labels = _kit_data(tmp_path)
    rc, m, rows, md, out, work = _kit(tmp_path, light_config_data(), labels)
    assert rc == 0
    q = {r["file"]: (r["input_quality"], r["input_quality_ly_do"]) for r in rows}
    assert q["nho.jpg"][0] == "canh_bao" and "nhỏ hơn 0,5 MP" in q["nho.jpg"][1]
    assert q["Screenshot 2026-10-18 181055.png"][0] == "canh_bao"
    assert q["lon.png"] == ("khong_canh_bao", "") and q["sua.png"][0] == "khong_canh_bao"
    assert all(r["seconds"] for r in rows)
    g = m["by_input_quality"]
    assert g["co_canh_bao"]["nghi_van_la_duong"]["n"] == 2 and g["khong_canh_bao"]["nghi_van_la_duong"]["n"] == 2
    c = g["co_canh_bao"]["nghi_van_la_duong"]
    assert c["dung"] + c["sai"] == c["n"] and c["n_that"] == 1 and c["n_sua"] == 1
    assert g["ly_do:screenshot_name"]["nghi_van_la_duong"]["n"] == 1
    assert "Tách theo cảnh báo “Độ tin cậy thấp”" in md and "Có cảnh báo" in md and "Không cảnh báo" in md
    assert m["timing_seconds_per_image"]["n"] == 4 and "Thời gian và bộ nhớ" in md
    # scores read from an OLDER cache (no input_quality field): the warning is recomputed without loading a model
    cache = work / "cache_predictions.jsonl"
    lines = [json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines()]
    for line in lines:
        line.pop("input_quality", None)
    cache.write_text("".join(json.dumps(l, ensure_ascii=False) + "\n" for l in lines), encoding="utf-8")
    rc2, m2, rows2, _, out2, _ = _kit(tmp_path, light_config_data(), labels)
    assert rc2 == 0 and "Đang nạp mô hình" not in out2 and m2["n_scored_now"] == 0
    assert {r["file"]: r["input_quality"] for r in rows2} == {k: v[0] for k, v in q.items()}
    assert m2["by_input_quality"]["co_canh_bao"] == m["by_input_quality"]["co_canh_bao"]
