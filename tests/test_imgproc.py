import io
import os
import sys

import cv2
import numpy as np
import pytest
from PIL import Image
from scipy.signal import convolve2d as sp_conv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imgproc as ip  # noqa: E402

ASSET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "sample_astronaut.png")
rng = np.random.default_rng(0)


def make_bytes(arr_or_img, fmt="PNG", **kw):
    img = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


# ---------- 載入 ----------
@pytest.mark.parametrize("fmt,name", [("PNG", "a.png"), ("JPEG", "a.jpg"), ("BMP", "a.bmp")])
def test_load_formats(fmt, name):
    arr = rng.integers(0, 256, (40, 60, 3), dtype=np.uint8)
    out, info = ip.load_image(make_bytes(arr, fmt), name)
    assert out.shape == (40, 60, 3) and out.dtype == np.uint8
    assert (info.width, info.height, info.channels) == (60, 40, 3)
    assert info.file_size_bytes > 0


def test_load_grayscale_and_rgba():
    g = rng.integers(0, 256, (10, 12), dtype=np.uint8)
    out, info = ip.load_image(make_bytes(g), "g.png")
    assert out.shape == (10, 12, 3) and info.channels == 1
    rgba = np.zeros((8, 8, 4), np.uint8)       # 全透明 → 白底
    out, info = ip.load_image(make_bytes(rgba), "t.png")
    assert info.channels == 4 and np.all(out == 255)


def test_load_errors():
    with pytest.raises(ValueError):
        ip.load_image(b"not an image", "x.png")
    with pytest.raises(ValueError):
        ip.load_image(b"", "x.png")
    with pytest.raises(ValueError):
        ip.load_image(b"abc", "x.gif")


def test_large_image_resized():
    arr = np.zeros((10, ip.MAX_SIDE + 500, 3), np.uint8)
    out, info = ip.load_image(make_bytes(arr), "big.png")
    assert max(out.shape[:2]) == ip.MAX_SIDE and info.resized_from is not None


# ---------- 色彩空間 / 直方圖 ----------
def test_yuv_known_values():
    px = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]]], np.uint8)
    yuv = ip.rgb_to_yuv(px)
    assert list(yuv["Y"][0]) == [76, 150, 29, 255, 0]       # 0.299/0.587/0.114
    # 白、黑為無色彩 → U、V 位於中點（≈127~128）
    assert abs(int(yuv["U"][0, 3]) - 128) <= 1 and abs(int(yuv["V"][0, 4]) - 128) <= 1
    # 純藍 U 最大、純紅 V 最大
    assert yuv["U"][0, 2] == 255 and yuv["V"][0, 0] == 255


def test_yuv_matches_opencv_luma():
    img = rng.integers(0, 256, (30, 30, 3), dtype=np.uint8)
    ref = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    assert np.max(np.abs(ip.to_gray(img).astype(int) - ref.astype(int))) <= 1


def test_histogram():
    ch = rng.integers(0, 256, (50, 50), dtype=np.uint8)
    h = ip.histogram(ch)
    assert h.shape == (256,) and h.sum() == ch.size
    assert np.array_equal(h, np.histogram(ch, bins=256, range=(0, 256))[0])


# ---------- 卷積 ----------
@pytest.mark.parametrize("k", [3, 5, 7])
def test_convolve_matches_scipy(k):
    ch = rng.integers(0, 256, (37, 29), dtype=np.uint8)
    ker = rng.normal(size=(k, k))
    mine = ip.convolve2d(ch, ker, pad="zero")
    ref = sp_conv(ch.astype(float), ker, mode="same", boundary="fill")
    assert np.allclose(mine, ref)


def test_convolve_reflect_matches_opencv():
    ch = rng.integers(0, 256, (31, 33), dtype=np.uint8)
    ker = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], float) / 16
    ref = cv2.filter2D(ch.astype(np.float64), -1, ker[::-1, ::-1], borderType=cv2.BORDER_REFLECT_101)
    assert np.allclose(ip.convolve2d(ch, ker, "reflect"), ref)


def test_identity_gives_infinite_psnr():
    img = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
    k = ip.parse_kernel(ip.PRESETS["恆等 (Identity) 3×3"])
    out = ip.apply_filter(img, k)
    assert np.array_equal(out, img) and ip.psnr(img, out) == float("inf")


def test_clipping():
    img = np.full((9, 9, 3), 200, np.uint8)
    k = np.zeros((3, 3)); k[1, 1] = 5          # 放大 5 倍 → 截斷於 255
    assert np.all(ip.apply_filter(img, k) == 255)
    k[1, 1] = -5
    assert np.all(ip.apply_filter(img, k) == 0)


def test_bias():
    img = np.full((5, 5, 3), 100, np.uint8)
    k = ip.parse_kernel("0 0 0\n0 1 0\n0 0 0")
    assert np.all(ip.apply_filter(img, k, bias=28) == 128)


# ---------- 卷積核解析 / 防呆 ----------
def test_parse_ok_formats():
    assert ip.parse_kernel("1,2,1\n2,4,2\n1,2,1").shape == (3, 3)
    assert ip.parse_kernel("1;2;1\n2;4;2\n1;2;1").shape == (3, 3)
    k = ip.parse_kernel("1/9 1/9 1/9\n1/9 1/9 1/9\n1/9 1/9 1/9")
    assert np.isclose(k.sum(), 1.0)
    assert ip.parse_kernel("5").shape == (1, 1)


@pytest.mark.parametrize("bad", [
    "", "   \n  ", "1 2 3\n4 5", "1 2\n3 4", "1 2 3\n4 5 6", "a b c\nd e f\ng h i",
    "0 0 0\n0 0 0\n0 0 0", "nan 1 1\n1 1 1\n1 1 1", "inf 1 1\n1 1 1\n1 1 1", "1/0 1 1\n1 1 1\n1 1 1",
])
def test_parse_rejects(bad):
    with pytest.raises(ip.KernelError):
        ip.parse_kernel(bad)


def test_parse_too_large():
    n = ip.MAX_KERNEL + 2
    with pytest.raises(ip.KernelError):
        ip.parse_kernel("\n".join(" ".join(["1"] * n) for _ in range(n)))


def test_normalize():
    k = ip.parse_kernel(ip.PRESETS["高斯平滑 3×3"])
    kn, w = ip.normalize_kernel(k, "sum")
    assert w is None and np.isclose(kn.sum(), 1)
    lap = ip.parse_kernel(ip.PRESETS["邊緣偵測 (Laplacian) 3×3"])
    kn, w = ip.normalize_kernel(lap, "sum")          # 總和 0 → 不可除
    assert w is not None and np.array_equal(kn, lap)
    kn, w = ip.normalize_kernel(lap, "abs")
    assert w is None and np.isclose(np.abs(kn).sum(), 1)


# ---------- PSNR ----------
def test_psnr_formula():
    a = np.zeros((10, 10, 3), np.uint8)
    b = a + 10                                       # MSE = 100
    assert np.isclose(ip.psnr(a, b), 10 * np.log10(255 ** 2 / 100))
    assert np.isclose(ip.psnr(a, b), 28.1308, atol=1e-3)


def test_psnr_matches_opencv_and_skimage():
    from skimage.metrics import peak_signal_noise_ratio as sk
    a = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    b = np.clip(a.astype(int) + rng.integers(-20, 21, a.shape), 0, 255).astype(np.uint8)
    assert np.isclose(ip.psnr(a, b), cv2.PSNR(a, b))
    assert np.isclose(ip.psnr(a, b), sk(a, b, data_range=255))


def test_psnr_shape_mismatch():
    with pytest.raises(ValueError):
        ip.psnr(np.zeros((4, 4, 3), np.uint8), np.zeros((5, 4, 3), np.uint8))


def test_blur_psnr_reasonable_and_ordering():
    img, _ = ip.load_image(open(ASSET, "rb").read(), "s.png")
    g3 = ip.normalize_kernel(ip.parse_kernel(ip.PRESETS["高斯平滑 3×3"]), "sum")[0]
    g5 = ip.normalize_kernel(ip.parse_kernel(ip.PRESETS["高斯平滑 5×5"]), "sum")[0]
    p3, p5 = ip.psnr(img, ip.apply_filter(img, g3)), ip.psnr(img, ip.apply_filter(img, g5))
    assert 20 < p5 < p3 < 60                         # 越大的平滑核，失真越大、PSNR 越低


# ---------- 人臉偵測 ----------
def test_face_detection_on_astronaut():
    img, _ = ip.load_image(open(ASSET, "rb").read(), "s.png")
    faces, model = ip.detect_faces(img, 1.1, 3, 40)
    assert len(faces) >= 1
    assert all(f.confidence is None or 0 <= f.confidence <= 1 for f in faces)
    out = ip.draw_faces(img, faces)
    assert out.shape == img.shape and not np.array_equal(out, img)


def test_face_detection_no_face():
    blank = np.full((200, 200, 3), 128, np.uint8)
    faces, _ = ip.detect_faces(blank)
    assert faces == []


def test_face_params_validation():
    with pytest.raises(ValueError):
        ip.detect_faces(np.zeros((50, 50, 3), np.uint8), scale_factor=1.0)


def test_weight_to_confidence_monotonic_and_bounded():
    vals = [ip.weight_to_confidence(w) for w in (-1, 0, 0.5, 1, 3, 10, 100)]
    assert vals[0] == 0 and vals[1] == 0
    assert all(a <= b for a, b in zip(vals, vals[1:])) and all(0 <= v < 1.0000001 for v in vals)


def test_nms_removes_nested_and_keeps_disjoint():
    F = ip.Face
    nested = [F(100, 100, 100, 100, None, None), F(120, 120, 60, 60, None, None)]
    assert len(ip._suppress_overlaps(nested)) == 1
    far = [F(0, 0, 50, 50, None, None), F(200, 200, 50, 50, None, None)]
    assert len(ip._suppress_overlaps(far)) == 2
    hi = [F(0, 0, 50, 50, 1.0, 0.2), F(5, 5, 50, 50, 5.0, 0.8)]
    out = ip._suppress_overlaps(hi)
    assert len(out) == 1 and out[0].confidence == 0.8
