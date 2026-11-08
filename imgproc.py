"""
imgproc.py — 影像處理核心演算法（與 UI 完全解耦，方便單元測試）

內容：
  1. 影像載入與中繼資訊
  2. RGB / YUV 色彩空間轉換
  3. 直方圖計算
  4. 自訂卷積核解析、正規化、2D 卷積（NumPy 手刻實作）
  5. MSE / PSNR
  6. 人臉偵測（OpenCV Haar Cascade，含信心度）
"""
from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

SUPPORTED_EXT = ("png", "jpg", "jpeg", "bmp")
MAX_SIDE = 2000          # 影像最長邊上限，避免卷積過慢
MAX_KERNEL = 15          # 卷積核最大邊長


# --------------------------------------------------------------------------- #
# 1. 影像載入
# --------------------------------------------------------------------------- #
@dataclass
class ImageInfo:
    filename: str
    fmt: str
    width: int
    height: int
    channels: int
    mode: str
    file_size_bytes: int
    resized_from: tuple | None = None   # 若因過大被縮小，記錄原尺寸

    @property
    def file_size_text(self) -> str:
        b = self.file_size_bytes
        return f"{b / 1024:.1f} KB" if b < 1024 * 1024 else f"{b / 1024 / 1024:.2f} MB"


def load_image(data: bytes, filename: str = "image") -> tuple[np.ndarray, ImageInfo]:
    """將位元組讀成 RGB uint8 ndarray (H, W, 3) 並回傳中繼資訊。

    - 依 EXIF 修正方向
    - 含透明度 (RGBA / LA / P) 者以白底合成
    - 灰階影像擴展為 3 通道
    - 最長邊 > MAX_SIDE 時等比例縮小
    """
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    if ext and ext not in SUPPORTED_EXT:
        raise ValueError(f"不支援的檔案格式 .{ext}，僅支援：{', '.join(SUPPORTED_EXT)}")
    if not data:
        raise ValueError("檔案內容為空。")

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"無法解析影像檔案（可能已損毀）：{e}") from e

    fmt = (img.format or ext or "?").upper()
    orig_mode = img.mode
    img = ImageOps.exif_transpose(img)

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba).convert("RGB")
    else:
        img = img.convert("RGB")

    resized_from = None
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / max(w, h)
        resized_from = (w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)

    arr = np.asarray(img, dtype=np.uint8)
    n_ch = {"L": 1, "1": 1, "P": 3, "RGB": 3, "RGBA": 4, "LA": 2, "CMYK": 4}.get(orig_mode, 3)
    info = ImageInfo(
        filename=filename, fmt=fmt, width=arr.shape[1], height=arr.shape[0],
        channels=n_ch, mode=orig_mode, file_size_bytes=len(data), resized_from=resized_from,
    )
    return arr, info


# --------------------------------------------------------------------------- #
# 2. 色彩空間
# --------------------------------------------------------------------------- #
def split_rgb(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """回傳 R、G、B 三個 uint8 灰階通道圖。"""
    return {"R": rgb[..., 0], "G": rgb[..., 1], "B": rgb[..., 2]}


# BT.601 類比 YUV：U = 0.492 (B - Y)，V = 0.877 (R - Y)
def rgb_to_yuv(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """RGB → YUV (BT.601)，並將 Y/U/V 線性映射到 0–255 供顯示與直方圖使用。

    Y = 0.299 R + 0.587 G + 0.114 B                (範圍 0 ~ 255)
    U = 0.492 (B − Y)                              (理論範圍 ≈ −111.2 ~ +111.2)
    V = 0.877 (R − Y)                              (理論範圍 ≈ −156.8 ~ +156.8)

    顯示用映射：U' = (U − U_min) / (U_max − U_min) × 255，V' 同理（含 128 附近的中性色）。
    """
    f = rgb.astype(np.float64)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = 0.492 * (b - y)
    v = 0.877 * (r - y)

    u_lim = 0.492 * 255.0 * (1 - 0.114)    # 111.18
    v_lim = 0.877 * 255.0 * (1 - 0.299)    # 156.77
    u_d = (u + u_lim) / (2 * u_lim) * 255.0
    v_d = (v + v_lim) / (2 * v_lim) * 255.0
    return {
        "Y": np.clip(np.rint(y), 0, 255).astype(np.uint8),
        "U": np.clip(np.rint(u_d), 0, 255).astype(np.uint8),
        "V": np.clip(np.rint(v_d), 0, 255).astype(np.uint8),
    }


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return rgb_to_yuv(rgb)["Y"]


# --------------------------------------------------------------------------- #
# 3. 直方圖
# --------------------------------------------------------------------------- #
def histogram(channel: np.ndarray) -> np.ndarray:
    """256 bins 的像素數量統計（橫軸 0–255）。"""
    return np.bincount(channel.ravel(), minlength=256).astype(np.int64)


# --------------------------------------------------------------------------- #
# 4. 卷積核
# --------------------------------------------------------------------------- #
PRESETS: dict[str, str] = {
    "恆等 (Identity) 3×3": "0 0 0\n0 1 0\n0 0 0",
    "邊緣銳化 (Sharpen) 3×3": "0 -1 0\n-1 5 -1\n0 -1 0",
    "方框平滑 (Box Blur) 3×3": "1 1 1\n1 1 1\n1 1 1",
    "高斯平滑 3×3": "1 2 1\n2 4 2\n1 2 1",
    "高斯平滑 5×5": "1 4 6 4 1\n4 16 24 16 4\n6 24 36 24 6\n4 16 24 16 4\n1 4 6 4 1",
    "浮雕 (Emboss) 3×3": "-2 -1 0\n-1 1 1\n0 1 2",
    "邊緣偵測 (Laplacian) 3×3": "0 1 0\n1 -4 1\n0 1 0",
    "Sobel 水平邊緣 3×3": "-1 0 1\n-2 0 2\n-1 0 1",
    "自訂（可自行編輯）": "0 0 0\n0 1 0\n0 0 0",
}


class KernelError(ValueError):
    """卷積核輸入不合法。"""


def parse_kernel(text: str) -> np.ndarray:
    """將文字解析成 2D float64 陣列。

    允許以換行分列、以空白 / 逗號 / 分號 / Tab 分隔數值，支援分數 (如 1/9)。
    驗證：非空、每列等長、方陣、奇數邊長、有限數值、不超過 MAX_KERNEL、非全零。
    """
    if text is None or not text.strip():
        raise KernelError("卷積核不可為空，請輸入例如：\n0 0 0\n0 1 0\n0 0 0")

    rows = [r for r in re.split(r"[\r\n]+", text.strip()) if r.strip()]
    parsed: list[list[float]] = []
    for i, row in enumerate(rows, 1):
        tokens = [t for t in re.split(r"[,\s;]+", row.strip()) if t]
        vals = []
        for t in tokens:
            try:
                if "/" in t:
                    a, b = t.split("/")
                    if float(b) == 0:
                        raise KernelError(f"第 {i} 列的分數「{t}」分母為 0。")
                    vals.append(float(a) / float(b))
                else:
                    vals.append(float(t))
            except KernelError:
                raise
            except ValueError:
                raise KernelError(f"第 {i} 列含有無法辨識的數值「{t}」。") from None
        parsed.append(vals)

    widths = {len(r) for r in parsed}
    if len(widths) != 1:
        raise KernelError(f"各列的元素個數不一致（{sorted(widths)}），卷積核必須是矩形陣列。")

    k = np.array(parsed, dtype=np.float64)
    h, w = k.shape
    if not np.all(np.isfinite(k)):
        raise KernelError("卷積核含有 NaN 或無限大 (inf)。")
    if h != w:
        raise KernelError(f"卷積核為 {h}×{w} 的非方陣；請輸入 N×N（N 為奇數）的方陣，例如 3×3 或 5×5。")
    if h % 2 == 0:
        raise KernelError(f"卷積核邊長 {h} 為偶數，無法決定中心點；請使用奇數邊長（3、5、7…）。")
    if h > MAX_KERNEL:
        raise KernelError(f"卷積核 {h}×{h} 過大，上限為 {MAX_KERNEL}×{MAX_KERNEL}。")
    if np.all(k == 0):
        raise KernelError("卷積核全為 0，濾波結果會是全黑影像；請至少設定一個非零權重。")
    return k


def normalize_kernel(k: np.ndarray, mode: str) -> tuple[np.ndarray, str | None]:
    """正規化卷積核，回傳 (新核, 警告訊息或 None)。

    mode:
      "none"  — 不處理
      "sum"   — 除以所有元素總和（保持整體亮度）。總和≈0 時（如邊緣偵測核）無法使用，略過並警告
      "abs"   — 除以所有元素絕對值總和（使 L1 範數為 1，永不除以 0，因為核非全零）
    """
    if mode == "none":
        return k, None
    if mode == "sum":
        s = k.sum()
        if abs(s) < 1e-9:
            return k, "卷積核元素總和為 0（如邊緣偵測 / 浮雕核），無法以總和正規化，已略過此步驟以避免除以零。"
        return k / s, None
    if mode == "abs":
        s = np.abs(k).sum()
        return k / s, None
    raise ValueError(f"未知的正規化模式：{mode}")


_PAD_MODES = {"reflect": "reflect", "zero": "constant", "edge": "edge"}


def convolve2d(channel: np.ndarray, kernel: np.ndarray, pad: str = "reflect") -> np.ndarray:
    """對單一通道做 2D 卷積（回傳 float64，未截斷）。

    數學定義（真正的卷積，核需翻轉 180°）：
        g(x, y) = Σ_{i=-a}^{a} Σ_{j=-b}^{b} k(i, j) · f(x − i, y − j)

    實作：以 np.pad 補邊後，對核的每個元素 (di, dj) 累加「平移後的影像 × 權重」，
    複雜度 O(K²·HW)，完全向量化，不依賴 cv2.filter2D / scipy。
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    mode = _PAD_MODES[pad]
    padded = np.pad(channel.astype(np.float64), ((ph, ph), (pw, pw)), mode=mode)
    flipped = kernel[::-1, ::-1]       # 卷積 = 翻轉後的相關 (correlation)
    H, W = channel.shape
    out = np.zeros((H, W), dtype=np.float64)
    for di in range(kh):
        for dj in range(kw):
            wgt = flipped[di, dj]
            if wgt != 0.0:
                out += wgt * padded[di:di + H, dj:dj + W]
    return out


def apply_filter(rgb: np.ndarray, kernel: np.ndarray, pad: str = "reflect", bias: float = 0.0) -> np.ndarray:
    """對 RGB 三通道分別卷積，加上 bias，截斷至 [0, 255] 並轉 uint8。"""
    chans = [convolve2d(rgb[..., c], kernel, pad) + bias for c in range(3)]
    out = np.stack(chans, axis=-1)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 5. 品質指標
# --------------------------------------------------------------------------- #
def mse(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"影像尺寸不一致：{a.shape} vs {b.shape}")
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(d * d))


def psnr(original: np.ndarray, processed: np.ndarray, max_val: float = 255.0) -> float:
    """PSNR (dB) = 10 · log10( MAX² / MSE )，MSE = 0 時回傳 +inf。"""
    m = mse(original, processed)
    if m == 0.0:
        return float("inf")
    return 10.0 * np.log10(max_val ** 2 / m)


# --------------------------------------------------------------------------- #
# 6. 人臉偵測
# --------------------------------------------------------------------------- #
@dataclass
class Face:
    x: int
    y: int
    w: int
    h: int
    raw_weight: float | None      # OpenCV levelWeights；後端不提供時為 None
    confidence: float | None      # 0–1 的歸一化分數；後端不提供時為 None


def weight_to_confidence(w: float) -> float:
    """將 cascade 的 levelWeights 映射為 0–1 的啟發式信心度：c = 1 − exp(−w / 3)。

    注意：Haar/LBP Cascade 並非機率模型，此值僅用於相對排序與顯示，不是校準後的機率。
    """
    return float(1.0 - np.exp(-max(w, 0.0) / 3.0))


def _cascade_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    try:
        candidates.append(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    except Exception:  # noqa: BLE001
        pass
    candidates.append(os.path.join(here, "assets", "haarcascade_frontalface_default.xml"))
    candidates.append(os.path.join(here, "assets", "lbpcascade_frontalface_opencv.xml"))
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("找不到人臉偵測 cascade 模型檔。")


_CASCADE = None


def _get_cascade():
    """回傳 (後端名稱, 分類器物件, 模型檔名)。

    優先使用 OpenCV 4.x 的 CascadeClassifier（可取得信心度）；
    若環境是 OpenCV 5（已移除 CascadeClassifier），退回 scikit-image 的 LBP Cascade。
    """
    global _CASCADE
    if _CASCADE is not None:
        return _CASCADE
    if hasattr(cv2, "CascadeClassifier"):
        path = _cascade_path()
        c = cv2.CascadeClassifier(path)
        if c.empty():
            raise RuntimeError(f"cascade 模型載入失敗：{path}")
        _CASCADE = ("opencv", c, os.path.basename(path))
    else:
        try:
            from skimage.feature import Cascade
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("此環境的 OpenCV 不含 CascadeClassifier，請安裝 opencv-python-headless<5。") from e
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "assets", "lbpcascade_frontalface_opencv.xml")
        _CASCADE = ("skimage", Cascade(path), os.path.basename(path))
    return _CASCADE


def _suppress_overlaps(faces: list[Face], iou_thr: float = 0.3, contain_thr: float = 0.8) -> list[Face]:
    """非極大值抑制 (NMS)：移除與較佳框高度重疊或被其包含的重複偵測。

    分數優先序：信心度（若有），否則框面積。
    重複條件：IoU > iou_thr，或交集 / 較小框面積 > contain_thr（巢狀框）。
    """
    def score(f: Face) -> float:
        return f.confidence if f.confidence is not None else float(f.w * f.h)

    kept: list[Face] = []
    for f in sorted(faces, key=score, reverse=True):
        dup = False
        for g in kept:
            iw = min(f.x + f.w, g.x + g.w) - max(f.x, g.x)
            ih = min(f.y + f.h, g.y + g.h) - max(f.y, g.y)
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            union = f.w * f.h + g.w * g.h - inter
            if inter / union > iou_thr or inter / min(f.w * f.h, g.w * g.h) > contain_thr:
                dup = True
                break
        if not dup:
            kept.append(f)
    return kept


def detect_faces(rgb: np.ndarray, scale_factor: float = 1.1, min_neighbors: int = 5,
                 min_size: int = 40) -> tuple[list[Face], str]:
    """偵測人臉，回傳 (人臉列表, 後端與模型描述)。"""
    if scale_factor <= 1.0:
        raise ValueError("scaleFactor 必須大於 1.0")
    backend, cascade, name = _get_cascade()
    gray = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    faces: list[Face] = []

    if backend == "opencv":
        rects, _levels, weights = cascade.detectMultiScale3(
            gray, scaleFactor=scale_factor, minNeighbors=int(min_neighbors),
            minSize=(int(min_size), int(min_size)), outputRejectLevels=True,
        )
        for (x, y, w, h), wt in zip(rects, np.asarray(weights).ravel()):
            faces.append(Face(int(x), int(y), int(w), int(h), float(wt), weight_to_confidence(float(wt))))
    else:
        max_side = max(gray.shape)
        dets = cascade.detect_multi_scale(
            gray, scale_factor=scale_factor, step_ratio=1,
            min_size=(int(min_size), int(min_size)), max_size=(max_side, max_side),
            min_neighbor_number=int(min_neighbors), intersection_score_threshold=0.5,
        )
        for d in dets:
            faces.append(Face(int(d["c"]), int(d["r"]), int(d["width"]), int(d["height"]), None, None))
    faces = _suppress_overlaps(faces)
    if backend == "opencv":
        faces.sort(key=lambda f: f.confidence, reverse=True)
    else:
        faces.sort(key=lambda f: f.w * f.h, reverse=True)
    return faces, f"{backend} / {name}"


def draw_faces(rgb: np.ndarray, faces: list[Face]) -> np.ndarray:
    """在影像上畫出邊界框與編號 / 信心度。"""
    out = np.ascontiguousarray(rgb.copy())
    thick = max(2, round(max(out.shape[:2]) / 300))
    scale = max(0.5, max(out.shape[:2]) / 1200)
    for i, f in enumerate(faces, 1):
        cv2.rectangle(out, (f.x, f.y), (f.x + f.w, f.y + f.h), (0, 220, 70), thick)
        label = f"#{i}" if f.confidence is None else f"#{i} {f.confidence * 100:.0f}%"
        ty = f.y - 6 if f.y > 24 else f.y + f.h + 20
        cv2.putText(out, label, (f.x, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 220, 70), max(1, thick - 1), cv2.LINE_AA)
    return out
