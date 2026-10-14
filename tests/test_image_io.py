import io

import numpy as np
import pytest
from PIL import Image

from conftest import encode
from ifd.errors import ForgeryError
from ifd.image_io import decoded_pixel_hash, read_image

LIM = dict(max_file_mb=5, max_pixels=4_000_000, min_side=16)


def test_png_and_jpeg_decode(rgb_image):
    for fmt in ("PNG", "JPEG"):
        img = read_image(encode(rgb_image, fmt), name=f"x.{fmt.lower()}", **LIM)
        assert img.rgb.shape == rgb_image.shape and img.rgb.dtype == np.uint8
        assert img.metadata["format"] == fmt
        assert len(img.sha256) == 64


def test_exif_orientation_applied_and_reported():
    arr = np.zeros((40, 80, 3), np.uint8)
    arr[:, :10] = 255  # white stripe on the left edge
    im = Image.fromarray(arr)
    exif = im.getexif()
    exif[274] = 6  # rotate 90 CW on display
    buf = io.BytesIO()
    im.save(buf, format="JPEG", exif=exif.tobytes(), quality=95)
    img = read_image(buf.getvalue(), name="rot.jpg", **LIM)
    assert img.rgb.shape[:2] == (80, 40), "display coordinates must follow EXIF orientation"
    assert img.metadata["exif_orientation"] == 6
    assert any("EXIF" in w for w in img.metadata["warnings"])
    assert img.rgb[:10, :, 0].mean() > 200, "white stripe moves to the top after rotation"


def test_alpha_composited_on_white():
    rgba = np.zeros((32, 32, 4), np.uint8)
    rgba[..., 3] = 0  # fully transparent
    img = read_image(encode(rgba, "PNG"), name="a.png", **LIM)
    assert (img.rgb == 255).all()
    assert any("alpha" in w for w in img.metadata["warnings"])


def test_grayscale_and_16bit_png():
    gray = (np.arange(64 * 64).reshape(64, 64) % 256).astype(np.uint8)
    img = read_image(encode(gray, "PNG"), name="g.png", **LIM)
    assert img.rgb.shape == (64, 64, 3)
    g16 = (np.arange(64 * 64).reshape(64, 64) * 16).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(g16).save(buf, format="PNG")
    img16 = read_image(buf.getvalue(), name="g16.png", **LIM)
    assert img16.rgb.dtype == np.uint8 and img16.rgb.shape == (64, 64, 3)


@pytest.mark.parametrize("data,msg", [
    (b"", "trống"),
    (b"not an image at all", "hỏng"),
    (b"GIF89a" + b"\x00" * 50, "hỏng"),
])
def test_rejects_invalid(data, msg):
    with pytest.raises(ForgeryError, match=msg):
        read_image(data, name="bad.png", **LIM)


def test_truncated_jpeg_rejected(rgb_image):
    raw = encode(rgb_image, "JPEG", quality=90)
    with pytest.raises(ForgeryError):
        read_image(raw[: len(raw) // 2], name="cut.jpg", **LIM)


def test_limits(rgb_image):
    with pytest.raises(ForgeryError, match="Cạnh nhỏ nhất"):
        read_image(encode(np.zeros((8, 100, 3), np.uint8)), name="thin.png", **LIM)
    with pytest.raises(ForgeryError, match="pixel"):
        read_image(encode(np.zeros((2100, 2100, 3), np.uint8)), name="big.png", **LIM)
    with pytest.raises(ForgeryError, match="MiB"):
        read_image(b"x" * (6 * 1024 * 1024), name="huge.png", **LIM)


def test_extension_does_not_matter_content_does(rgb_image, tmp_path):
    p = tmp_path / "photo.png"
    p.write_bytes(encode(rgb_image, "JPEG"))
    img = read_image(p, **LIM)
    assert img.metadata["format"] == "JPEG"


def test_decoded_hash_ignores_container_bytes(rgb_image):
    a = read_image(encode(rgb_image, "PNG", compress_level=0), name="a.png", **LIM)
    b = read_image(encode(rgb_image, "PNG", compress_level=9), name="b.png", **LIM)
    assert a.sha256 != b.sha256
    assert decoded_pixel_hash(a.rgb) == decoded_pixel_hash(b.rgb)
