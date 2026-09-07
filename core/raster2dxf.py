#!/usr/bin/env python3
"""raster2dxf.py —— 栅格图像/扫描件 -> DXF（光栅矢量化）。

与 pdf2dxf.py 的矢量解析互补：本模块处理"只有像素"的输入
（扫描 PDF 页、照片、PNG/JPG），通过 二值化 -> 骨架化 -> 骨架追踪
重建线稿为 LWPOLYLINE，辅以 Hough 圆检测与 OCR 文本。

注意（与矢量管线的本质区别）：光栅矢量化是近似重建，
精度受扫描质量/分辨率影响，不保证原生矢量 PDF 转换的坐标级精度。

用法:
    python raster2dxf.py --input scan.png --output out.dxf [--dpi 150]
    python raster2dxf.py --input scan.pdf --output out_dir/ [--dpi 150] [--page -1]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import ezdxf
import numpy as np

logger = logging.getLogger("raster2dxf")

DXF_VERSION = "R2010"
LAYER_LINE = "SCAN_LINE"
LAYER_TEXT = "SCAN_TEXT"

# 8 邻域偏移
_NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# --------------------------------------------------------------------------- #
# 预处理
# --------------------------------------------------------------------------- #
def binarize(img_bgr: np.ndarray) -> np.ndarray:
    """灰度 -> 去噪 -> 自适应二值化。返回墨迹掩膜（墨迹=True）。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7,
                                    searchWindowSize=21)
    # 自适应阈值对扫描件的光照不均更鲁棒
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, blockSize=35, C=12)
    # 去孤立噪点：开运算（1px）
    kernel = np.ones((2, 2), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    return bw > 0


def deskew(img_bgr: np.ndarray, ink: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """按墨迹主方向纠偏（仅当倾角在 ±5° 内时，避免误纠旋转内容）。"""
    ys, xs = np.nonzero(ink)
    if len(xs) < 1000:
        return img_bgr, ink
    coords = np.column_stack([xs, ys]).astype(np.float32)
    angle = cv2.minAreaRect(coords)[-1]
    if angle > 45:
        angle -= 90
    if abs(angle) < 0.05 or abs(angle) > 5:
        return img_bgr, ink
    h, w = ink.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img_bgr = cv2.warpAffine(img_bgr, M, (w, h), borderValue=(255, 255, 255))
    ink = cv2.warpAffine(ink.astype(np.uint8), M, (w, h),
                         borderValue=0) > 0
    logger.info("纠偏 %.2f°", angle)
    return img_bgr, ink


# --------------------------------------------------------------------------- #
# 骨架追踪
# --------------------------------------------------------------------------- #
def _degrees(skel: np.ndarray) -> np.ndarray:
    """每个骨架像素的 8 邻域骨架像素数。"""
    k = np.ones((3, 3), np.uint8)
    k[1, 1] = 0
    deg = cv2.filter2D(skel.astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT)
    return np.where(skel, deg, 0)


def trace_skeleton(skel: np.ndarray, min_len_px: float = 4.0
                   ) -> List[List[Tuple[int, int]]]:
    """把骨架像素图追踪成折线列表（像素坐标 (x, y)）。

    策略：节点（端点 deg=1 / 交点 deg>=3）之间沿 deg=2 链行走；
    剩余纯环（全 deg=2）从任一未访问像素起走。按有向边去重。
    """
    skel = skel.astype(bool)
    deg = _degrees(skel)
    visited: set = set()          # 有向边 (p, q)，p/q 为 (y, x)
    paths: List[List[Tuple[int, int]]] = []

    def neighbors(p):
        y, x = p
        for dy, dx in _NB8:
            q = (y + dy, x + dx)
            if 0 <= q[0] < skel.shape[0] and 0 <= q[1] < skel.shape[1] and skel[q]:
                yield q

    def walk(start, first):
        path = [start, first]
        visited.add((start, first))
        prev, cur = start, first
        while deg[cur] == 2:      # 沿链前进
            nxt = None
            for q in neighbors(cur):
                if q != prev and (cur, q) not in visited:
                    nxt = q
                    break
            if nxt is None:
                break
            visited.add((cur, nxt))
            prev, cur = cur, nxt
            path.append(cur)
        return path

    ys, xs = np.nonzero(skel & (deg != 2))
    nodes = list(zip(ys.tolist(), xs.tolist()))
    for node in nodes:                       # 节点出发的链
        for q in neighbors(node):
            if (node, q) not in visited:
                paths.append(walk(node, q))
    ys, xs = np.nonzero(skel & (deg == 2))   # 纯环
    for p in zip(ys.tolist(), xs.tolist()):
        free = [q for q in neighbors(p) if (p, q) not in visited]
        if free:
            path = walk(p, free[0])
            if len(path) > 2:
                path.append(path[0])         # 闭环
            paths.append(path)

    # 像素坐标 (y,x) -> (x,y)，过滤碎片
    out = []
    for path in paths:
        pts = [(x, y) for y, x in path]
        if len(pts) >= 2:
            arr = np.asarray(pts, np.float32)
            seg = np.diff(arr, axis=0)
            if float(np.hypot(seg[:, 0], seg[:, 1]).sum()) >= min_len_px:
                out.append(pts)
    return out


def simplify(path: List[Tuple[int, int]], epsilon: float = 1.2
             ) -> List[Tuple[float, float]]:
    """Douglas-Peucker 折线简化。"""
    arr = np.asarray(path, np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(arr, epsilon, False)
    return [(float(p[0][0]), float(p[0][1])) for p in approx]


# --------------------------------------------------------------------------- #
# 主转换器
# --------------------------------------------------------------------------- #
class RasterToDXFConverter:
    """栅格图像 -> DXF。1 DXF 单位 = 1 pt（按 dpi 换算像素），Y 轴上指。"""

    def __init__(self, dpi: int = 150, do_ocr: bool = True,
                 do_circles: bool = True, min_len_px: float = 4.0):
        self.dpi = dpi
        self.do_ocr = do_ocr
        self.do_circles = do_circles
        self.min_len_px = min_len_px
        self.stats: Dict[str, int] = {}

    # 像素 -> DXF（pt, Y 翻转）
    def _map(self, x: float, y: float, h_px: int) -> Tuple[float, float]:
        s = 72.0 / self.dpi
        return x * s, (h_px - y) * s

    def convert_image(self, img_bgr: np.ndarray, dxf_path: str) -> str:
        t0 = time.perf_counter()
        ink = binarize(img_bgr)
        img_bgr, ink = deskew(img_bgr, ink)
        h_px, w_px = ink.shape

        from skimage.morphology import skeletonize
        skel = skeletonize(ink)
        logger.info("骨架像素: %d / 墨迹像素: %d", int(skel.sum()), int(ink.sum()))

        doc = ezdxf.new(DXF_VERSION, setup=True)
        doc.header["$INSUNITS"] = 0
        msp = doc.modelspace()
        doc.layers.add(LAYER_LINE, color=7)
        doc.layers.add(LAYER_TEXT, color=3)

        n_poly = 0
        for path in trace_skeleton(skel, self.min_len_px):
            pts = simplify(path)
            if len(pts) < 2:
                continue
            mapped = [self._map(x, y, h_px) for x, y in pts]
            closed = len(path) > 2 and path[0] == path[-1]
            try:
                msp.add_lwpolyline(mapped, close=closed,
                                   dxfattribs={"layer": LAYER_LINE})
                n_poly += 1
            except Exception as exc:
                logger.debug("折线写入失败: %s", exc)
        self.stats["LWPOLYLINE"] = n_poly

        if self.do_circles:
            self.stats["CIRCLE"] = self._detect_circles(skel, msp, h_px)
        if self.do_ocr:
            self.stats["TEXT"] = self._ocr_text(img_bgr, msp, h_px)

        os.makedirs(os.path.dirname(os.path.abspath(dxf_path)), exist_ok=True)
        doc.saveas(dxf_path)
        aud = doc.audit()
        self.stats["audit_errors"] = len(aud.errors)
        self.stats["elapsed_sec"] = round(time.perf_counter() - t0, 2)
        logger.info("已保存 %s: %s", dxf_path, self.stats)
        return dxf_path

    def _detect_circles(self, skel: np.ndarray, msp, h_px: int) -> int:
        """Hough 圆检测 + 圆周支撑率验证 + 去重。

        Hough 在密集线稿上极易误检，必须逐个验证：圆周采样点落在骨架上的
        比例 >= 0.6 才接受；近同心同半径的候选只保留支撑率最高的一个。
        """
        edge = (skel.astype(np.uint8)) * 255
        min_r = max(8, int(0.008 * min(edge.shape)))
        circles = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT, dp=1.5,
                                   minDist=min_r, param1=50, param2=24,
                                   minRadius=min_r,
                                   maxRadius=int(0.2 * min(edge.shape)))
        if circles is None:
            return 0
        H, W = skel.shape
        sk_d = skel.astype(np.uint8)
        # 骨架稍作膨胀用于支撑率判定（容忍 1px 偏差）
        sk_d = cv2.dilate(sk_d, np.ones((3, 3), np.uint8))

        def support(x: float, y: float, r: float) -> float:
            ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
            xs = np.clip(np.round(x + r * np.cos(ang)).astype(int), 0, W - 1)
            ys = np.clip(np.round(y + r * np.sin(ang)).astype(int), 0, H - 1)
            return float(sk_d[ys, xs].mean())

        cands = []
        for x, y, r in circles[0]:
            s = support(float(x), float(y), float(r))
            if s >= 0.6:
                cands.append((float(x), float(y), float(r), s))
        # 去重：中心距 < 0.5r 且半径差 < 30% 视为同一圆
        cands.sort(key=lambda c: -c[3])
        kept: List[Tuple[float, float, float, float]] = []
        for c in cands:
            dup = any(abs(c[0] - k[0]) < 0.5 * k[2]
                      and abs(c[1] - k[1]) < 0.5 * k[2]
                      and abs(c[2] - k[2]) < 0.3 * k[2] for k in kept)
            if not dup:
                kept.append(c)
        for x, y, r, _s in kept:
            cx, cy = self._map(x, y, h_px)
            msp.add_circle((cx, cy), r * 72.0 / self.dpi,
                           dxfattribs={"layer": LAYER_LINE})
        return len(kept)

    def _ocr_text(self, img_bgr: np.ndarray, msp, h_px: int) -> int:
        """tesseract OCR -> TEXT 实体（位置/字高为像素近似值）。"""
        try:
            import pytesseract
        except ImportError:
            logger.warning("pytesseract 不可用，跳过 OCR")
            return 0
        data = pytesseract.image_to_data(
            img_bgr, output_type=pytesseract.Output.DICT, config="--psm 11")
        n = 0
        for i, txt in enumerate(data["text"]):
            txt = txt.strip()
            try:
                conf = float(data["conf"][i])
            except (ValueError, TypeError):
                conf = -1
            if not txt or conf < 60:
                continue
            # 过滤把图形区域误识别为文字的巨型框（字高超过页高 3% 或无字母数字）
            w_i, h_i = data["width"][i], data["height"][i]
            if h_i > 0.03 * h_px or h_i < 4 or w_i < 4:
                continue
            if not any(ch.isalnum() for ch in txt):
                continue
            x, y = data["left"][i], data["top"][i] + data["height"][i]
            h_pt = data["height"][i] * 72.0 / self.dpi
            mx, my = self._map(float(x), float(y), h_px)
            try:
                msp.add_text(txt, height=max(h_pt, 1.0),
                             dxfattribs={"layer": LAYER_TEXT}
                             ).set_placement((mx, my))
                n += 1
            except Exception as exc:
                logger.debug("文本写入失败: %s", exc)
        return n

    # ------------------------------------------------------------------ #
    def convert_file(self, path: str, out: str, page: int = -1) -> List[str]:
        """入口：png/jpg 直接转；pdf 逐页渲染后转（用于扫描版 PDF）。"""
        p = Path(path)
        ext = p.suffix.lower()
        outs: List[str] = []
        if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            img = cv2.imread(str(p))
            if img is None:
                raise ValueError(f"无法读取图像: {p}")
            out_path = out if out.lower().endswith(".dxf") else str(
                Path(out) / (p.stem + ".dxf"))
            outs.append(self.convert_image(img, out_path))
        elif ext == ".pdf":
            import fitz
            doc = fitz.open(str(p))
            pages = range(doc.page_count) if page < 0 else [page]
            for pg in pages:
                pix = doc[pg].get_pixmap(dpi=self.dpi)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n)
                img = cv2.cvtColor(
                    img, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
                if out.lower().endswith(".dxf"):
                    out_path = out
                else:
                    name = (f"{p.stem}_p{pg + 1:03d}.dxf" if doc.page_count > 1
                            else f"{p.stem}.dxf")
                    out_path = str(Path(out) / name)
                outs.append(self.convert_image(img, out_path))
        else:
            raise ValueError(f"不支持的输入类型: {ext}")
        return outs


# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="栅格图像/扫描 PDF -> DXF（光栅矢量化）")
    ap.add_argument("--input", required=True, help="png/jpg/tif/bmp 或扫描版 pdf")
    ap.add_argument("--output", default="raster_output", help="输出 dxf 或目录")
    ap.add_argument("--dpi", type=int, default=150, help="扫描分辨率（坐标换算基准）")
    ap.add_argument("--page", type=int, default=-1, help="PDF 页码（0 起），-1=全部")
    ap.add_argument("--no-ocr", action="store_true", help="关闭 OCR 文本提取")
    ap.add_argument("--no-circles", action="store_true", help="关闭 Hough 圆检测")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    conv = RasterToDXFConverter(dpi=args.dpi, do_ocr=not args.no_ocr,
                                do_circles=not args.no_circles)
    outs = conv.convert_file(args.input, args.output, page=args.page)
    for o in outs:
        print(o)
    print("统计:", conv.stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
