"""
AI 輔助之互動式影像處理與人臉分析工具（Streamlit）

執行：  streamlit run app.py
"""
from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from PIL import Image

import imgproc as ip

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(HERE, "assets", "sample_astronaut.png")

st.set_page_config(page_title="影像處理與人臉分析工具", page_icon="🖼️", layout="wide")

# 通道名稱 → 繪圖顏色
CH_COLOR = {"R": "#d62728", "G": "#2ca02c", "B": "#1f77b4", "Y": "#444444", "U": "#7a5bd1", "V": "#d98a1c"}


# --------------------------------------------------------------------------- #
# 快取包裝
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=8)
def cached_load(data: bytes, name: str):
    return ip.load_image(data, name)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_filter(rgb: np.ndarray, kernel_key: tuple, pad: str, bias: float) -> np.ndarray:
    return ip.apply_filter(rgb, np.array(kernel_key, dtype=np.float64), pad, bias)


def to_png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# 繪圖
# --------------------------------------------------------------------------- #
def hist_figure(channels: dict[str, np.ndarray], title: str, fill: bool = True, height: float = 2.6) -> Figure:
    """繪製直方圖：橫軸像素強度 0–255，縱軸像素數量。"""
    fig = Figure(figsize=(5.2, height), dpi=110)
    ax = fig.subplots()
    x = np.arange(256)
    for name, ch in channels.items():
        h = ip.histogram(ch)
        c = CH_COLOR.get(name, "#333333")
        if fill and len(channels) == 1:
            ax.bar(x, h, width=1.0, color=c, alpha=0.85)
        else:
            ax.plot(x, h, color=c, linewidth=1.2, label=name)
            ax.fill_between(x, h, color=c, alpha=0.12)
    ax.set_xlim(0, 255)
    ax.set_xlabel("Pixel intensity (0-255)", fontsize=8)
    ax.set_ylabel("Pixel count", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    ax.ticklabel_format(axis="y", style="plain")
    if len(channels) > 1:
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def tinted(ch: np.ndarray, name: str) -> np.ndarray:
    """將 R/G/B 單通道以對應顏色著色（其餘通道為 0）。"""
    out = np.zeros(ch.shape + (3,), np.uint8)
    out[..., "RGB".index(name)] = ch
    return out


# --------------------------------------------------------------------------- #
# 側邊欄：載入影像
# --------------------------------------------------------------------------- #
st.sidebar.header("影像來源")
uploaded = st.sidebar.file_uploader("上傳影像（PNG、JPG、JPEG、BMP）", type=list(ip.SUPPORTED_EXT))
use_sample = st.sidebar.button("使用範例影像")
if use_sample:
    st.session_state["use_sample"] = True
if uploaded is not None:
    st.session_state["use_sample"] = False

st.title("影像處理與人臉分析工具")

data, fname = None, None
if uploaded is not None:
    data, fname = uploaded.getvalue(), uploaded.name
elif st.session_state.get("use_sample"):
    with open(SAMPLE_PATH, "rb") as fh:
        data, fname = fh.read(), "sample_astronaut.png"

if data is None:
    st.info("請從左側上傳一張影像，或按「使用範例影像」開始。")
    st.stop()

try:
    rgb, info = cached_load(data, fname)
except ValueError as e:
    st.error(f"影像載入失敗：{e}")
    st.stop()

if info.resized_from:
    st.warning(
        f"影像過大（{info.resized_from[0]}×{info.resized_from[1]}），為確保運算速度已等比例縮小為 "
        f"{info.width}×{info.height} 後處理。"
    )

tab_info, tab_color, tab_filter, tab_face = st.tabs(
    ["影像資訊", "色彩空間與直方圖", "卷積濾波與 PSNR", "人臉偵測"]
)

# --------------------------------------------------------------------------- #
# 功能 1：影像載入與資訊
# --------------------------------------------------------------------------- #
with tab_info:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.image(rgb, caption=info.filename)
    with c2:
        st.subheader("中繼資訊")
        rows = {
            "檔名": info.filename,
            "格式": info.fmt,
            "解析度（寬 × 高）": f"{info.width} × {info.height} px",
            "總像素數": f"{info.width * info.height:,}",
            "原始色彩模式": info.mode,
            "原始通道數": str(info.channels),
            "處理時色彩維度": "3（RGB, 8-bit / 通道）",
            "檔案大小": info.file_size_text,
        }
        st.dataframe(pd.DataFrame({"項目": rows.keys(), "內容": rows.values()}), hide_index=True)
        st.caption("含透明度的影像會以白色背景合成；灰階影像會複製為三通道以統一處理。")

# --------------------------------------------------------------------------- #
# 功能 2：色彩空間與同步直方圖
# --------------------------------------------------------------------------- #
with tab_color:
    view = st.radio("檢視模式", ["原始影像 (Original)", "RGB 分量", "YUV 分量"], horizontal=True)

    if view.startswith("原始"):
        l, r = st.columns(2)
        with l:
            st.image(rgb, caption="Original")
        with r:
            st.pyplot(hist_figure(ip.split_rgb(rgb), "RGB histogram (overlay)"))
            st.pyplot(hist_figure({"Y": ip.to_gray(rgb)}, "Luminance (Y) histogram"))

    elif view == "RGB 分量":
        colorize = st.checkbox("以對應色彩顯示通道（取消則為灰階）", value=False)
        comps = ip.split_rgb(rgb)
        cols = st.columns(3)
        for col, (name, ch) in zip(cols, comps.items()):
            with col:
                st.image(tinted(ch, name) if colorize else ch, caption=f"{name} channel")
                st.pyplot(hist_figure({name: ch}, f"{name} histogram"))

    else:
        comps = ip.rgb_to_yuv(rgb)
        cols = st.columns(3)
        notes = {"Y": "Y（亮度）", "U": "U（藍色差，已映射至 0–255）", "V": "V（紅色差，已映射至 0–255）"}
        for col, (name, ch) in zip(cols, comps.items()):
            with col:
                st.image(ch, caption=notes[name])
                st.pyplot(hist_figure({name: ch}, f"{name} histogram"))
        with st.expander("YUV 轉換公式（BT.601）"):
            st.latex(r"Y = 0.299R + 0.587G + 0.114B")
            st.latex(r"U = 0.492\,(B - Y), \qquad V = 0.877\,(R - Y)")
            st.caption(
                "U、V 為帶正負號的色差量（U 約 ±111、V 約 ±157），顯示與直方圖時線性映射至 0–255，"
                "中性灰（無色彩）落在約 128。"
            )

# --------------------------------------------------------------------------- #
# 功能 3：自訂卷積核濾波與 PSNR
# --------------------------------------------------------------------------- #
def _on_preset_change():
    st.session_state["kernel_text"] = ip.PRESETS[st.session_state["preset"]]


with tab_filter:
    if "kernel_text" not in st.session_state:
        st.session_state["kernel_text"] = ip.PRESETS["邊緣銳化 (Sharpen) 3×3"]
        st.session_state["preset"] = "邊緣銳化 (Sharpen) 3×3"

    left, right = st.columns([2, 3])
    with left:
        st.subheader("卷積核設定")
        st.selectbox("預設範本", list(ip.PRESETS), key="preset", on_change=_on_preset_change)
        st.text_area(
            "卷積核矩陣（每列一行；數值以空白、逗號或分號分隔；可用分數如 1/9）",
            key="kernel_text", height=170,
        )
        norm_label = st.radio(
            "正規化 (Normalization)",
            ["不正規化", "除以元素總和", "除以元素絕對值總和"],
            help="除以總和可維持整體亮度（適用於平滑核）；總和為 0 的核（邊緣偵測、浮雕）會自動略過並提示。",
        )
        norm_mode = {"不正規化": "none", "除以元素總和": "sum", "除以元素絕對值總和": "abs"}[norm_label]
        pad_label = st.radio("邊界處理", ["鏡射 (reflect)", "補零 (zero)", "複製邊緣 (edge)"], horizontal=True)
        pad = {"鏡射 (reflect)": "reflect", "補零 (zero)": "zero", "複製邊緣 (edge)": "edge"}[pad_label]
        bias = st.slider("輸出偏移 (bias)", -255, 255, 0,
                         help="卷積結果加上此值再截斷至 0–255；浮雕效果常用 +128。")

    kernel, warn, err = None, None, None
    try:
        kernel = ip.parse_kernel(st.session_state["kernel_text"])
        kernel, warn = ip.normalize_kernel(kernel, norm_mode)
    except ip.KernelError as e:
        err = str(e)

    with left:
        if err:
            st.error(err)
        else:
            if warn:
                st.warning(warn)
            st.caption(f"實際使用的卷積核（{kernel.shape[0]}×{kernel.shape[1]}，元素總和 = {kernel.sum():.4g}）")
            st.dataframe(pd.DataFrame(kernel).round(4), hide_index=True)

    with right:
        if err:
            st.info("請修正左側的卷積核輸入後，這裡會顯示濾波結果。")
        else:
            key = tuple(map(tuple, kernel.tolist()))
            with st.spinner("計算中…"):
                filtered = cached_filter(rgb, key, pad, float(bias))
            p = ip.psnr(rgb, filtered)
            m = ip.mse(rgb, filtered)

            mc1, mc2 = st.columns(2)
            mc1.metric("PSNR", "∞ dB（兩圖完全相同）" if np.isinf(p) else f"{p:.2f} dB")
            mc2.metric("MSE", f"{m:.2f}")

            show = st.radio("比對方式", ["並排", "疊加（混合）", "差異圖（放大）"], horizontal=True)
            if show == "並排":
                a, b = st.columns(2)
                a.image(rgb, caption="濾波前")
                b.image(filtered, caption="濾波後")
            elif show == "疊加（混合）":
                alpha = st.slider("濾波後影像比例", 0.0, 1.0, 0.5, 0.05)
                blend = np.clip(rgb * (1 - alpha) + filtered * alpha, 0, 255).astype(np.uint8)
                st.image(blend, caption=f"疊加（濾波後 {alpha:.0%}）")
            else:
                gain = st.slider("差異放大倍率", 1, 20, 5)
                diff = np.clip(np.abs(rgb.astype(int) - filtered.astype(int)) * gain, 0, 255).astype(np.uint8)
                st.image(diff, caption=f"|原圖 − 濾波後| × {gain}")

            st.download_button("下載濾波後影像 (PNG)", to_png_bytes(filtered), "filtered.png", "image/png")

            with st.expander("PSNR 計算公式"):
                st.latex(r"\mathrm{MSE}=\frac{1}{H\,W\,C}\sum_{x=1}^{H}\sum_{y=1}^{W}\sum_{c=1}^{C}\big[I(x,y,c)-K(x,y,c)\big]^2")
                st.latex(r"\mathrm{PSNR}=10\log_{10}\!\left(\frac{\mathrm{MAX}_I^{2}}{\mathrm{MSE}}\right)\ \text{dB},\quad \mathrm{MAX}_I=255")
                st.caption("I 為原圖、K 為濾波後影像（皆為 8-bit RGB，C = 3）。MSE = 0 時 PSNR 為無限大。"
                           "PSNR 越高代表與原圖越接近；平滑、銳化、浮雕等刻意改變影像的濾波，PSNR 低是預期結果。")

# --------------------------------------------------------------------------- #
# 功能 4：人臉偵測
# --------------------------------------------------------------------------- #
with tab_face:
    st.subheader("人臉偵測（Cascade Classifier）")
    p1, p2, p3 = st.columns(3)
    scale = p1.slider("scaleFactor", 1.02, 1.5, 1.1, 0.01, help="每次縮放金字塔的比例；越小越精細但越慢。")
    neigh = p2.slider("minNeighbors", 1, 10, 5, help="越大誤報越少，但可能漏偵測。")
    msize = p3.slider("最小人臉尺寸 (px)", 20, 300, 40, 5)

    try:
        with st.spinner("偵測中…"):
            faces, model_name = ip.detect_faces(rgb, scale, neigh, msize)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        st.error(f"人臉偵測無法執行：{e}")
        faces = None

    if faces is not None:
        a, b = st.columns([3, 2])
        with a:
            st.image(ip.draw_faces(rgb, faces), caption="偵測結果")
        with b:
            st.metric("偵測到的人臉數量", len(faces))
            st.caption(f"模型：{model_name}")
            if not faces:
                st.info("未偵測到人臉。可嘗試調低 minNeighbors 或縮小「最小人臉尺寸」。")
            else:
                df = pd.DataFrame([
                    {
                        "編號": i, "x": f.x, "y": f.y, "寬": f.w, "高": f.h,
                        "信心度": "N/A" if f.confidence is None else f"{f.confidence * 100:.1f}%",
                        "原始權重": "N/A" if f.raw_weight is None else round(f.raw_weight, 3),
                    }
                    for i, f in enumerate(faces, 1)
                ])
                st.dataframe(df, hide_index=True)
            if faces and faces[0].confidence is None:
                st.caption("目前環境的偵測後端不提供信心度；使用 OpenCV 4.x 時會顯示。")
            else:
                st.caption("信心度 = 1 − exp(−w/3)，w 為 Cascade 最後一級累積權重；"
                           "為相對分數而非校準後的機率。")
