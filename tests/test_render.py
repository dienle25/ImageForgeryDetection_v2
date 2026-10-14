import numpy as np

from conftest import encode
from ifd.analysis import Analyzer, report_files
from ifd.config import load_config
from ifd.render import colorize, overlay


def test_overlay_with_precomputed_colors_is_identical():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (37, 53, 3), dtype=np.uint8)
    hm = rng.random((37, 53)).astype(np.float32)
    assert np.array_equal(overlay(rgb, hm, 0.55), overlay(rgb, hm, 0.55, colored=colorize(hm)))


def test_report_files_same_bytes_and_arrays_for_previews(light_config):
    an = Analyzer(load_config(light_config))
    rng = np.random.default_rng(1)
    rep = an.analyze(encode(rng.integers(0, 255, (90, 140, 3), dtype=np.uint8)), name="x.png")
    arrays = {}
    a = report_files(rep, 0.55, arrays)
    b = report_files(rep, 0.55)
    assert {k: v for k, v in a.items() if k != "result.json"} == {k: v for k, v in b.items() if k != "result.json"}
    assert np.array_equal(arrays["heatmap"], colorize(rep.primary_heatmap))
    assert np.array_equal(arrays["overlay"], overlay(rep.image.rgb, rep.primary_heatmap, 0.55))


def test_colorize_matches_reference_formula_bit_for_bit():
    anchors = np.array([[20, 35, 120], [0, 170, 225], [255, 230, 55], [210, 25, 35]], dtype=np.float32)
    rng = np.random.default_rng(2)
    hm = rng.random((120, 90)).astype(np.float32)
    hm.flat[:6] = [0, 1, 1 / 3, 2 / 3, -0.5, 1.5]
    scaled = np.clip(hm, 0, 1) * 3
    index = np.minimum(scaled.astype(np.int32), 2)
    weight = (scaled - index)[..., None]
    reference = np.round(anchors[index] * (1 - weight) + anchors[index + 1] * weight).astype(np.uint8)
    assert np.array_equal(colorize(hm), reference)


def test_png_is_lossless():
    import io
    from PIL import Image
    from ifd.render import png_bytes
    rng = np.random.default_rng(3)
    for arr in (rng.integers(0, 255, (40, 60, 3), dtype=np.uint8), rng.integers(0, 65535, (40, 60), dtype=np.uint16)):
        back = np.asarray(Image.open(io.BytesIO(png_bytes(arr))))
        assert np.array_equal(back, arr)
