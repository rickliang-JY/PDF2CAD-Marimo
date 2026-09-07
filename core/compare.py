"""compare.py —— PDF 原页 vs DXF 渲染对比。

- render_pdf_page: pymupdf 渲染 PDF 页为 RGB 数组
- render_dxf:      ezdxf.addons.drawing + matplotlib(Agg) 渲染 DXF 为 RGB 数组
- compare_page:    并排对比 + RGB 叠差图（PDF 红 / DXF 绿 / 重合黄）+ 墨迹 IoU
"""
from __future__ import annotations

import gc
import io
from typing import Optional, Tuple

import cv2
import numpy as np
import pymupdf

import matplotlib
matplotlib.use("Agg")  # 硬性规则：一律 Agg 后端
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# 中文字体显式设置（DXF 中可能含中文 TEXT 实体）
matplotlib.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei"]
matplotlib.rcParams["axes.unicode_minus"] = False

import ezdxf
from ezdxf import recover as ezrecover  # ezdxf 1.4 需显式导入 recover 子模块
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

# 墨迹判定阈值：灰度低于该值视为墨迹（白底 255）
INK_THRESHOLD = 200


def render_pdf_page(pdf_path: str, page_num: int, dpi: int = 150) -> np.ndarray:
    """渲染 PDF 页为 RGB uint8 数组（处理 pix.n 为 3/4 的情况）。"""
    doc = pymupdf.open(pdf_path)
    try:
        pix = doc[page_num].get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif pix.n == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = img.copy()  # 已是 RGB，脱离 buffer 所有权
        return img
    finally:
        doc.close()
        gc.collect()


def _fig_to_rgb(fig: Figure) -> np.ndarray:
    """matplotlib Figure -> RGB uint8 数组（Agg buffer）。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi)
    buf.seek(0)
    bgr = cv2.imdecode(np.frombuffer(buf.read(), np.uint8),
                       cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def render_dxf(dxf_path: str, dpi: int = 150,
               size_hint: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """渲染 DXF 为 RGB uint8 数组。

    size_hint=(h, w) 时输出缩放到与该尺寸一致（cv2.resize），
    便于与 PDF 渲染图逐像素对齐做叠差。
    """
    # recover.readfile 对非严格规范的 DXF 更稳
    doc, auditor = ezrecover.readfile(dxf_path)
    msp = doc.modelspace()
    if size_hint is not None:
        h, w = size_hint
        figsize = (max(w, 1) / dpi, max(h, 1) / dpi)
    else:
        figsize = (12, 9)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])  # 满幅无边距，坐标不被留白压缩
    ax.set_axis_off()
    try:
        ctx = RenderContext(doc)
        backend = MatplotlibBackend(ax)
        Frontend(ctx, backend).draw_layout(msp, finalize=True)
        img = _fig_to_rgb(fig)
    finally:
        plt.close(fig)
        gc.collect()
    if size_hint is not None:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img


def _ink_mask(rgb: np.ndarray) -> np.ndarray:
    """RGB 图 -> 二值墨迹掩膜（True=墨迹）。"""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray < INK_THRESHOLD


def _png_bytes(rgb: np.ndarray) -> bytes:
    """RGB 数组 -> PNG bytes（cv2 编码需 BGR）。"""
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG 编码失败")
    return buf.tobytes()


def _error_overlay(h: int, w: int, msg: str) -> bytes:
    """渲染失败时的占位叠差图：灰底 + 错误说明文字。"""
    img = np.full((max(h, 100), max(w, 400), 3), 245, np.uint8)
    cv2.putText(img, "render failed:", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
    # 长错误信息按行截断绘制，避免溢出
    for i, seg in enumerate([msg[j:j + 60] for j in range(0, len(msg), 60)][:6]):
        cv2.putText(img, seg, (10, 60 + 22 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
    return _png_bytes(img)


def compare_page(pdf_path: str, dxf_path: str, page_num: int,
                 dpi: int = 150) -> dict:
    """对比一页：返回三张 PNG bytes + 墨迹 IoU + 两侧墨迹像素数。

    返回 {'pdf_png': bytes, 'dxf_png': bytes, 'overlay_png': bytes,
          'iou': float|None, 'ink_pdf': int, 'ink_dxf': int}
    - overlay：PDF 墨迹红色、DXF 墨迹绿色、重合黄色的 RGB 叠差图
    - 渲染失败时 iou=None 并在 overlay 写错误说明
    """
    pdf_rgb = render_pdf_page(pdf_path, page_num, dpi=dpi)
    pdf_png = _png_bytes(pdf_rgb)
    h, w = pdf_rgb.shape[:2]
    ink_pdf = _ink_mask(pdf_rgb)
    n_pdf = int(ink_pdf.sum())

    try:
        dxf_rgb = render_dxf(dxf_path, dpi=dpi, size_hint=(h, w))
        dxf_png = _png_bytes(dxf_rgb)
        ink_dxf = _ink_mask(dxf_rgb)
    except Exception as exc:
        # 渲染失败：iou=None，overlay 写错误说明，不抛异常
        return {
            "pdf_png": pdf_png,
            "dxf_png": _error_overlay(h, w, f"DXF: {exc}"),
            "overlay_png": _error_overlay(h, w, f"DXF render failed: {exc}"),
            "iou": None,
            "ink_pdf": n_pdf,
            "ink_dxf": 0,
        }
    n_dxf = int(ink_dxf.sum())

    inter = int((ink_pdf & ink_dxf).sum())
    union = int((ink_pdf | ink_dxf).sum())
    iou = (inter / union) if union > 0 else 1.0  # 双空白视为完全一致

    # 叠差图：白底；PDF 独有=红，DXF 独有=绿，重合=黄
    overlay = np.full((h, w, 3), 255, np.uint8)
    overlay[ink_pdf & ~ink_dxf] = (255, 0, 0)    # 红：PDF 有 DXF 无
    overlay[~ink_pdf & ink_dxf] = (0, 180, 0)    # 绿：DXF 有 PDF 无
    overlay[ink_pdf & ink_dxf] = (255, 220, 0)   # 黄：重合
    overlay_png = _png_bytes(overlay)

    del pdf_rgb, dxf_rgb, overlay
    gc.collect()
    return {
        "pdf_png": pdf_png,
        "dxf_png": dxf_png,
        "overlay_png": overlay_png,
        "iou": float(iou),
        "ink_pdf": n_pdf,
        "ink_dxf": n_dxf,
    }
