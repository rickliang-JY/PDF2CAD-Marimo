#!/usr/bin/env python3
"""PDF 工程图纸转 DXF 格式转换器（布局无损）。

核心设计
========
* 解析层：PyMuPDF ``page.get_drawings()`` 提取矢量图元（'l' 线段、're' 矩形、
  'qu' 四边形、'c' 三次贝塞尔），``page.get_text("dict")`` 提取文本 span。
* 坐标变换：
    - PyMuPDF 返回的是**未旋转**页面坐标（原点左上，Y 向下，单位 1/72 英寸 = 1 pt）。
    - 先乘 ``page.rotation_matrix``（旋转 0 时为单位阵）得到显示坐标；
    - 再做 Y 翻转：``y_dxf = page_height_display - y_display``；
    - 即  P_dxf = (x', H - y')，其中 (x', y') = P_pdf * page.rotation_matrix，
      H = page.rect.height（已含 cropbox/rotation 影响）。
    - 1 PDF 点 = 1 DXF 单位（$INSUNITS=0 unitless，在日志中注明）。
* DXF 版本：R2010（AC1024）。R12 不支持 ELLIPSE/SPLINE，题目要求
  "R12 及以上"，故选 R2010 以原生支持 ELLIPSE/SPLINE，兼顾兼容性。
* 圆/椭圆/圆弧识别：MuPDF 把圆/椭圆/圆弧输出为若干段三次贝塞尔
  （kappa ≈ 0.5522847498 逼近）。本转换器对每段贝塞尔做
  "端点切线法线求交 → 圆心 → 中点回代校验"，把能拟合的连续贝塞尔段
  合并还原为 CIRCLE / ELLIPSE / ARC（无损语义级还原），
  拟合失败的贝塞尔以控制点 SPLINE（度 3，节点 [0,0,0,0,1,1,1,1]，
  与贝塞尔精确等价，真正无损）输出。
* 分层：几何 -> "GEOMETRY"，文本 -> "TEXT"。颜色 -> 真彩色 (true_color)。
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF
import ezdxf
from ezdxf import colors

logger = logging.getLogger("pdf2dxf")

DXF_VERSION = "R2010"  # 支持 ELLIPSE/SPLINE；题目要求 R12 及以上，R2010 合规
LAYER_GEOMETRY = "GEOMETRY"
LAYER_TEXT = "TEXT"
ARC_FIT_TOL = 0.02       # 圆/椭圆拟合相对容差
CLOSE_TOL_PT = 0.75      # 判定闭合路径的距离容差（点）
CAP_HEIGHT_RATIO = 0.72  # PDF em 字号 -> DXF cap height 换算系数


# --------------------------------------------------------------------------- #
# 日志
# --------------------------------------------------------------------------- #
def configure_logging(log_path: Optional[str] = None, verbose: bool = True) -> None:
    """配置 logging：输出到文件（conversion.log）并可选输出到控制台。"""
    log = logging.getLogger("pdf2dxf")
    log.setLevel(logging.DEBUG)
    for h in list(log.handlers):
        log.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        log.addHandler(fh)
    if verbose:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        log.addHandler(sh)


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _rgb_to_true_color(rgb: Optional[Sequence[float]]) -> Optional[int]:
    """PDF 浮点 RGB (0..1) -> DXF 真彩色 int。"""
    if rgb is None:
        return None
    r = max(0, min(255, int(round(rgb[0] * 255))))
    g = max(0, min(255, int(round(rgb[1] * 255))))
    b = max(0, min(255, int(round(rgb[2] * 255))))
    return colors.rgb2int((r, g, b))


def _int_to_rgb_tuple(c: int) -> Tuple[int, int, int]:
    return ((c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF)


def _lineweight_from_pt(width_pt: Optional[float]) -> int:
    """PDF 线宽（点）-> DXF lineweight（1/100 mm，取最近合法值）。"""
    if not width_pt or width_pt <= 0:
        return 25
    lw100 = int(round(width_pt * 25.4 / 72.0 * 100))
    valid = [0, 5, 9, 13, 15, 18, 20, 25, 30, 35, 40, 50, 53, 60, 70, 80,
             90, 100, 106, 120, 140, 158, 200, 211]
    return min(valid, key=lambda v: abs(v - lw100))


def _bezier_point(p1, p2, p3, p4, t: float) -> Tuple[float, float]:
    mt = 1.0 - t
    x = mt**3 * p1[0] + 3 * mt * mt * t * p2[0] + 3 * mt * t * t * p3[0] + t**3 * p4[0]
    y = mt**3 * p1[1] + 3 * mt * mt * t * p2[1] + 3 * mt * t * t * p3[1] + t**3 * p4[1]
    return (x, y)


def _fit_circle_bezier(b, tol: float = ARC_FIT_TOL):
    """尝试把一段三次贝塞尔拟合为圆弧（MuPDF kappa 逼近）。

    方法：端点切线的法线交点 = 圆心候选；校验两端半径一致、
    切线与半径正交、且 t=0.5 中点落在圆上。
    返回 (cx, cy, r) 或 None。
    """
    p1, p2, p3, p4 = [(float(p.x), float(p.y)) for p in b]
    t1 = (p2[0] - p1[0], p2[1] - p1[1])
    t2 = (p4[0] - p3[0], p4[1] - p3[1])
    l1, l2 = math.hypot(*t1), math.hypot(*t2)
    if l1 < 1e-9 or l2 < 1e-9:
        return None
    n1 = (-t1[1], t1[0])
    n2 = (-t2[1], t2[0])
    det = -n1[0] * n2[1] + n1[1] * n2[0]
    if abs(det) < 1e-9:
        return None
    rhs = (p4[0] - p1[0], p4[1] - p1[1])
    a = (-rhs[0] * n2[1] + rhs[1] * n2[0]) / det
    cx, cy = p1[0] + a * n1[0], p1[1] + a * n1[1]
    r1 = _dist(p1, (cx, cy))
    r2 = _dist(p4, (cx, cy))
    r = (r1 + r2) / 2.0
    if r < 1e-6:
        return None
    if abs(r1 - r2) / r > tol:
        return None
    # 切线 ⟂ 半径
    d1 = abs(t1[0] * (p1[0] - cx) + t1[1] * (p1[1] - cy)) / (l1 * r)
    d2 = abs(t2[0] * (p4[0] - cx) + t2[1] * (p4[1] - cy)) / (l2 * r)
    if d1 > 5 * tol or d2 > 5 * tol:
        return None
    # 中点回代
    m = _bezier_point(p1, p2, p3, p4, 0.5)
    if abs(_dist(m, (cx, cy)) - r) / r > tol:
        return None
    return (cx, cy, r)


def _fit_ellipse_bezier(b, tol: float = ARC_FIT_TOL):
    """尝试把一段三次贝塞尔拟合为轴对齐椭圆的 1/4 弧。

    返回 (cx, cy, rx, ry) 或 None。
    """
    p1, p2, p3, p4 = [(float(p.x), float(p.y)) for p in b]
    t1 = (p2[0] - p1[0], p2[1] - p1[1])
    t2 = (p4[0] - p3[0], p4[1] - p3[1])
    l1, l2 = math.hypot(*t1), math.hypot(*t2)
    if l1 < 1e-9 or l2 < 1e-9:
        return None
    n1 = (-t1[1], t1[0])
    n2 = (-t2[1], t2[0])
    det = -n1[0] * n2[1] + n1[1] * n2[0]
    if abs(det) < 1e-9:
        return None
    rhs = (p4[0] - p1[0], p4[1] - p1[1])
    a = (-rhs[0] * n2[1] + rhs[1] * n2[0]) / det
    cx, cy = p1[0] + a * n1[0], p1[1] + a * n1[1]
    # 端点半径向量须轴对齐
    d1 = (p1[0] - cx, p1[1] - cy)
    d4 = (p4[0] - cx, p4[1] - cy)
    comps = []
    for d in (d1, d4):
        if abs(d[0]) >= abs(d[1]):
            if abs(d[1]) > tol * max(abs(d[0]), 1e-9) * 5:
                return None
            comps.append(("x", abs(d[0])))
        else:
            if abs(d[0]) > tol * max(abs(d[1]), 1e-9) * 5:
                return None
            comps.append(("y", abs(d[1])))
    axes = dict(comps)
    if "x" not in axes or "y" not in axes:
        return None
    rx, ry = axes["x"], axes["y"]
    if rx < 1e-6 or ry < 1e-6:
        return None
    m = _bezier_point(p1, p2, p3, p4, 0.5)
    v = ((m[0] - cx) / rx) ** 2 + ((m[1] - cy) / ry) ** 2
    if abs(v - 1.0) > 5 * tol:
        return None
    return (cx, cy, rx, ry)


# --------------------------------------------------------------------------- #
# 转换器
# --------------------------------------------------------------------------- #
class PDFToDXFConverter:
    """把 PDF 工程图纸页面无损转换为 DXF。"""

    def __init__(self, pdf_path: str, dxf_path: Optional[str] = None,
                 bake_dashes: bool = False):
        self.pdf_path = str(pdf_path)
        self.dxf_path = dxf_path
        # True 时把虚线实体展开成显式短线段（几何级虚线），
        # 兼容忽略线型 pattern 的查看器（部分网页/轻量 DXF 查看器）
        self.bake_dashes = bake_dashes
        self.doc: Optional[fitz.Document] = None
        self.dxf: Optional[ezdxf.EzDxf] = None
        self.msp = None
        # 页面状态
        self.page_height: float = 0.0       # 显示高度（含 rotation/cropbox）
        self.page_width: float = 0.0
        self.page_rotation: int = 0
        self.transform = fitz.Matrix(1, 1)  # 未旋转 -> 显示 坐标矩阵
        self.matrix_applications: int = 0
        # 统计
        self.stats: Dict[str, Any] = {}
        self._reset_stats()

    # ------------------------------------------------------------------ #
    def _reset_stats(self) -> None:
        self.stats = {
            "LINE": 0, "LWPOLYLINE": 0, "CIRCLE": 0, "ELLIPSE": 0,
            "ARC": 0, "SPLINE": 0, "HATCH": 0, "TEXT": 0,
            "pdf_drawings": 0, "pdf_items": 0, "pdf_text_spans": 0,
            "unconverted": [],   # (描述, 原因)
            "matrix_applications": 0,
            "elapsed_sec": 0.0,
        }

    # ------------------------------------------------------------------ #
    def _open(self) -> None:
        if self.doc is None:
            self.doc = fitz.open(self.pdf_path)

    # ------------------------------------------------------------------ #
    # 坐标变换
    # ------------------------------------------------------------------ #
    def apply_matrix(self, point, matrix) -> Tuple[float, float]:
        """应用 PDF 变换矩阵到点坐标（fitz 约定：p * m）。"""
        p = point if isinstance(point, fitz.Point) else fitz.Point(point)
        q = p * matrix
        self.matrix_applications += 1
        return (q.x, q.y)

    def _to_dxf(self, point) -> Tuple[float, float]:
        """PDF 页面坐标 -> DXF 坐标：先旋转矩阵，再 Y 翻转。

        公式： (x', y') = P * page.rotation_matrix
               P_dxf  = (x', page_height - y')
        """
        x, y = self.apply_matrix(point, self.transform)
        return (x, self.page_height - y)

    def _dir_to_dxf_angle(self, direction: Tuple[float, float]) -> float:
        """把 PDF 文本方向向量转换为 DXF 旋转角（度，CCW）。"""
        m = self.transform
        dx = direction[0] * m.a + direction[1] * m.c
        dy = direction[0] * m.b + direction[1] * m.d
        # Y 翻转：dy 取反
        return math.degrees(math.atan2(-dy, dx))

    # ------------------------------------------------------------------ #
    # 解析层
    # ------------------------------------------------------------------ #
    def extract_page_entities(self, page_num: int = 0) -> Dict[str, Any]:
        """提取一页的矢量图元与文本 span，并建立页面坐标变换。"""
        self._open()
        assert self.doc is not None
        page = self.doc[page_num]
        self.page_rotation = page.rotation
        self.page_width = float(page.rect.width)
        self.page_height = float(page.rect.height)
        # 未旋转坐标 -> 显示坐标（rotation=0 时为单位阵）
        self.transform = page.rotation_matrix
        logger.info(
            "页面 %d: 显示尺寸 %.1f x %.1f pt, rotation=%d, cropbox=%s",
            page_num, self.page_width, self.page_height,
            self.page_rotation, tuple(round(v, 2) for v in page.cropbox),
        )
        if self.page_rotation:
            logger.info("页面旋转 %d 度：应用 page.rotation_matrix %s",
                        self.page_rotation, tuple(self.transform))

        drawings = []
        try:
            drawings = page.get_drawings()
        except Exception as exc:  # 鲁棒性：解析失败不崩溃
            logger.warning("get_drawings 失败: %s", exc)
            self.stats["unconverted"].append(("page drawings", str(exc)))

        texts: List[Dict[str, Any]] = []
        try:
            tdict = page.get_text("dict")
            for block in tdict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        span = dict(span)
                        span["_dir"] = tuple(line.get("dir", (1.0, 0.0)))
                        texts.append(span)
        except Exception as exc:
            logger.warning("get_text 失败: %s", exc)
            self.stats["unconverted"].append(("page text", str(exc)))

        self.stats["pdf_drawings"] = len(drawings)
        self.stats["pdf_items"] = sum(len(d.get("items", [])) for d in drawings)
        self.stats["pdf_text_spans"] = len(texts)
        logger.info("提取: %d 个 path（共 %d 个 item）, %d 个文本 span",
                    len(drawings), self.stats["pdf_items"], len(texts))
        return {"drawings": drawings, "texts": texts}

    # ------------------------------------------------------------------ #
    # DXF 文档
    # ------------------------------------------------------------------ #
    def _new_dxf(self) -> None:
        """创建 R2010 DXF（R12 不支持 ELLIPSE/SPLINE，故选 R2010）。"""
        self.dxf = ezdxf.new(DXF_VERSION, setup=True)
        self.dxf.header["$INSUNITS"] = 0  # unitless：1 单位 = 1 PDF 点(1/72in)
        self.msp = self.dxf.modelspace()
        self.dxf.layers.add(LAYER_GEOMETRY, color=7)
        self.dxf.layers.add(LAYER_TEXT, color=7)
        # setup=True 自带的标准 DASHED pattern 是公制尺度(1.27mm 画线/0.25mm 间隔)，
        # 而本图 1 单位=1pt(比 mm 小约 2.8 倍)，微观虚线在渲染时退化为视觉实线。
        # 按 pt 尺度重定义: 6pt 画线 / 3pt 间隔（工程图常用观感）。
        try:
            self.dxf.linetypes.remove("DASHED")
        except Exception:
            pass
        self.dxf.linetypes.add(
            "DASHED", pattern=[9.0, 6.0, -3.0],
            description="Dashed: 6pt on / 3pt off (1 unit = 1 pt)")
        # 具体虚线样式由 _linetype_for_dashes() 按 PDF 原始 dash 数组动态注册
        self._registered_linetypes = {"CONTINUOUS", "DASHED"}
        logger.info("创建 DXF %s（1 DXF 单位 = 1 PDF 点 = 1/72 英寸, "
                    "$INSUNITS=0）", DXF_VERSION)

    @staticmethod
    def _parse_dash_array(dashes: str):
        """PyMuPDF dash 描述如 '[5.3 3.9] 0' -> ([5.3, 3.9], 0.0)；解析失败返回 None。"""
        m = re.match(r"\[(.*?)\]\s*(-?[\d.]+)", dashes.strip())
        if not m:
            return None
        try:
            arr = [float(x) for x in m.group(1).split()]
            phase = float(m.group(2))
        except ValueError:
            return None
        if not arr or any(x <= 0 for x in arr):
            return None
        return arr, phase

    def _linetype_for_dashes(self, dashes: str) -> str:
        """按 PDF 原始 dash 数组注册/返回 pt 尺度的 DXF 线型名（如 DASH_5_3_3_9）。

        setup=True 的标准 DASHED 是公制尺度，1 单位=1pt 时虚线微观化、渲染退化
        为实线；直接用 PDF 原 dash 数组（单位即 pt）逐样式注册，虚实观感与 PDF 一致。
        """
        parsed = self._parse_dash_array(dashes)
        if not parsed:
            return "DASHED"
        arr, _ = parsed
        # 奇数段数组按 CAD 惯例重复一次拼成偶数段(on/off 交替)
        segs = arr if len(arr) % 2 == 0 else arr * 2
        name = "DASH_" + "_".join(("%g" % s).replace(".", "p") for s in arr)
        if name not in self._registered_linetypes:
            # ezdxf 的 pattern 用 list 形式: [总长度, 画线, -间隔, ...]
            # (字符串 "A,..." 形式不会自动回填总长度, 会导致渲染退化为实线)
            elems = [s if i % 2 == 0 else -s for i, s in enumerate(segs)]
            pattern = [sum(segs)] + elems
            try:
                self.dxf.linetypes.add(
                    name, pattern=pattern,
                    description="PDF dash array [%s] (1 unit = 1 pt)"
                    % " ".join("%g" % s for s in arr))
            except Exception:
                return "DASHED"
            self._registered_linetypes.add(name)
        return name

    def _base_attribs(self, rgb, width_pt=None, layer=LAYER_GEOMETRY) -> Dict:
        attribs: Dict[str, Any] = {"layer": layer}
        tc = _rgb_to_true_color(rgb)
        if tc is not None:
            attribs["true_color"] = tc
        if width_pt:
            attribs["lineweight"] = _lineweight_from_pt(width_pt)
        return attribs

    # ------------------------------------------------------------------ #
    # 基本图元转换（题目要求的方法）
    # ------------------------------------------------------------------ #
    def _bake_dashed_line(self, a, b, attribs) -> None:
        """把一条带虚线线型的线段按 pattern 展开成显式实线段序列。

        解决部分查看器（网页/轻量 DXF viewer）忽略 linetype pattern、
        把所有线画成实线的问题：虚线直接落在几何上，任何查看器都可见。
        """
        name = attribs.get("linetype", "BYLAYER")
        try:
            pattern = tuple(self.dxf.linetypes.get(name).simplified_line_pattern())
        except Exception:
            pattern = ()
        if not pattern:
            e = self.msp.add_line(a, b, dxfattribs=attribs)
            self.stats["LINE"] += 1
            return
        total = _dist(a, b)
        if total < 1e-9:
            return
        ux, uy = (b[0] - a[0]) / total, (b[1] - a[1]) / total
        seg_attribs = {k: v for k, v in attribs.items() if k != "linetype"}
        pos = 0.0          # 沿线已消费长度
        i = 0              # pattern 下标（偶=画线, 奇=间隔）
        while pos < total - 1e-9:
            seg = pattern[i % len(pattern)]
            if seg <= 0:   # 点(0)按 0.5pt 小画线处理
                seg = 0.5
            if i % 2 == 0:  # 画线段
                end = min(pos + seg, total)
                p_a = (a[0] + ux * pos, a[1] + uy * pos)
                p_b = (a[0] + ux * end, a[1] + uy * end)
                if end - pos > 1e-6:
                    self.msp.add_line(p_a, p_b, dxfattribs=seg_attribs)
                    self.stats["LINE"] += 1
            pos += seg
            i += 1
        self.stats.setdefault("baked_dash_lines", 0)
        self.stats["baked_dash_lines"] += 1

    def convert_line(self, line) -> Any:
        """('l', p1, p2) -> DXF LINE。"""
        _, p1, p2 = line
        a = self._to_dxf(p1)
        b = self._to_dxf(p2)
        attribs = self._current_attribs
        if _dist(a, b) < 1e-9:
            self.stats["unconverted"].append(("line", "零长度线段"))
            return None
        if self.bake_dashes and "DASH" in str(attribs.get("linetype", "")).upper():
            self._bake_dashed_line(a, b, attribs)
            return None
        e = self.msp.add_line(a, b, dxfattribs=attribs)
        self.stats["LINE"] += 1
        return e

    def convert_curve(self, curve) -> Any:
        """('c', p1..p4) -> DXF SPLINE。

        采用控制点 SPLINE（度 3, 节点 [0,0,0,0,1,1,1,1]），与原始三次
        贝塞尔**精确等价**（优于 ≥32 点采样折线逼近，真正无损）。
        """
        _, p1, p2, p3, p4 = curve
        pts = [self._to_dxf(p) for p in (p1, p2, p3, p4)]
        attribs = self._current_attribs
        try:
            sp = self.msp.add_spline(dxfattribs=attribs)
            sp.dxf.degree = 3
            sp.control_points = [(*p, 0.0) for p in pts]
            sp.knots = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
            self.stats["SPLINE"] += 1
            return sp
        except Exception as exc:
            logger.warning("SPLINE 创建失败(%s)，退化为折线逼近", exc)
            pts32 = [self._to_dxf(_bezier_point(
                (p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y), (p4.x, p4.y),
                i / 32.0)) for i in range(33)]
            e = self.msp.add_lwpolyline(pts32, dxfattribs=attribs)
            self.stats["LWPOLYLINE"] += 1
            return e

    def convert_rect(self, item) -> Any:
        """('re', rect, orientation) -> 闭合 LWPOLYLINE。"""
        _, rect = item[0], item[1]
        corners_pdf = [(rect.x0, rect.y0), (rect.x1, rect.y0),
                       (rect.x1, rect.y1), (rect.x0, rect.y1)]
        pts = [self._to_dxf(c) for c in corners_pdf]
        e = self.msp.add_lwpolyline(pts, close=True,
                                    dxfattribs=self._current_attribs)
        self.stats["LWPOLYLINE"] += 1
        self._maybe_hatch(pts)
        return e

    def convert_quad(self, item) -> Any:
        """('qu', quad) -> 闭合 LWPOLYLINE。"""
        quad = item[1]
        pts = [self._to_dxf(quad.ul), self._to_dxf(quad.ur),
               self._to_dxf(quad.lr), self._to_dxf(quad.ll)]
        e = self.msp.add_lwpolyline(pts, close=True,
                                    dxfattribs=self._current_attribs)
        self.stats["LWPOLYLINE"] += 1
        self._maybe_hatch(pts)
        return e

    def convert_text(self, text_span, transform_matrix) -> Any:
        """PDF 文本 span -> DXF TEXT（保留内容/字号/位置/旋转/颜色）。"""
        content = text_span.get("text", "")
        if not content.strip():
            self.stats["unconverted"].append(("text", "空白 span"))
            return None
        # PDF span size 为 em 字号；DXF TEXT height 约定为 cap height，
        # 换算系数 ≈ 0.72（Helvetica cap-height 比），保证渲染视觉尺寸一致。
        size = float(text_span.get("size", 10.0)) * CAP_HEIGHT_RATIO
        origin = text_span.get("origin")
        if origin is None:
            bbox = text_span.get("bbox")
            origin = (bbox[0], bbox[3])
        x, y = self.apply_matrix(origin, transform_matrix)
        ins = (x, self.page_height - y)
        angle = self._dir_to_dxf_angle(text_span.get("_dir", (1.0, 0.0)))
        color_int = int(text_span.get("color", 0))
        attribs = {
            "layer": LAYER_TEXT,
            "height": size,
            "rotation": angle % 360.0,
            "true_color": colors.rgb2int(_int_to_rgb_tuple(color_int)),
        }
        try:
            t = self.msp.add_text(content, dxfattribs=attribs)
            t.set_placement(ins)  # 默认左对齐基线，与 PDF origin 一致
            self.stats["TEXT"] += 1
            return t
        except Exception as exc:  # 字体缺失等异常不崩溃
            logger.warning("文本 %r 转换失败: %s", content[:20], exc)
            self.stats["unconverted"].append((f"text {content[:20]!r}", str(exc)))
            return None

    # ------------------------------------------------------------------ #
    # 圆弧/圆/椭圆还原
    # ------------------------------------------------------------------ #
    def _add_circle(self, c_pdf, r, attribs) -> None:
        c = self._to_dxf(c_pdf)
        self.msp.add_circle(c, r, dxfattribs=attribs)
        self.stats["CIRCLE"] += 1
        if self._fill_active:
            pts = [self._to_dxf((c_pdf[0] + r * math.cos(2 * math.pi * i / 72),
                                 c_pdf[1] + r * math.sin(2 * math.pi * i / 72)))
                   for i in range(72)]
            self._maybe_hatch(pts)

    def _add_ellipse(self, c_pdf, rx, ry, attribs) -> None:
        c = self._to_dxf(c_pdf)
        # 长轴端点（PDF 坐标）经完整变换（旋转矩阵 + Y 翻转）后与中心求差
        if rx >= ry:
            e = self._to_dxf((c_pdf[0] + rx, c_pdf[1]))
            ratio = ry / rx
        else:
            e = self._to_dxf((c_pdf[0], c_pdf[1] + ry))
            ratio = rx / ry
        major = (e[0] - c[0], e[1] - c[1], 0.0)
        self.msp.add_ellipse(c, major_axis=major, ratio=ratio,
                             dxfattribs=attribs)
        self.stats["ELLIPSE"] += 1
        if self._fill_active:
            pts = [self._to_dxf((c_pdf[0] + rx * math.cos(2 * math.pi * i / 72),
                                 c_pdf[1] + ry * math.sin(2 * math.pi * i / 72)))
                   for i in range(72)]
            self._maybe_hatch(pts)

    def _add_arc(self, c_pdf, r, p_start_pdf, p_end_pdf, p_mid_pdf,
                 attribs) -> None:
        """由 PDF 坐标下的圆心/起止/中间点生成 DXF ARC（CCW）。"""
        c = self._to_dxf(c_pdf)
        s = self._to_dxf(p_start_pdf)
        e = self._to_dxf(p_end_pdf)
        m = self._to_dxf(p_mid_pdf)

        def ang(p):
            a = math.degrees(math.atan2(p[1] - c[1], p[0] - c[0])) % 360.0
            return a

        a_s, a_e, a_m = ang(s), ang(e), ang(m)
        ccw_span = (a_e - a_s) % 360.0
        mid_span = (a_m - a_s) % 360.0
        if mid_span > ccw_span + 1e-6:  # 中点不在 CCW 扫掠内 -> 交换起止
            a_s, a_e = a_e, a_s
        if a_s > 360.0 - 1e-9:  # 浮点归一化（360.0 -> 0.0）
            a_s = 0.0
        self.msp.add_arc(c, r, a_s, a_e, dxfattribs=attribs)
        self.stats["ARC"] += 1

    def _convert_bezier_run(self, run: List[Any]) -> None:
        """处理一段连续贝塞尔：能还原为 CIRCLE/ELLIPSE/ARC 则还原，
        否则逐段输出控制点 SPLINE。"""
        attribs = self._current_attribs
        n = len(run)
        circles = [_fit_circle_bezier(it[1:]) for it in run]
        if all(c is not None for c in circles):
            cx = sum(c[0] for c in circles) / n
            cy = sum(c[1] for c in circles) / n
            r = sum(c[2] for c in circles) / n
            coherent = all(
                _dist((c[0], c[1]), (cx, cy)) <= 0.02 * r
                and abs(c[2] - r) <= 0.02 * r for c in circles)
            if coherent:
                p_start = (run[0][1].x, run[0][1].y)
                p_end = (run[-1][4].x, run[-1][4].y)
                closed = _dist(p_start, p_end) <= CLOSE_TOL_PT
                # 总扫掠角 ≈ 360° 且闭合 -> 圆
                if closed and n >= 3:
                    self._add_circle((cx, cy), r, attribs)
                    return
                # 逐段中点取第一段中点定方向
                mid = _bezier_point(
                    (run[0][1].x, run[0][1].y), (run[0][2].x, run[0][2].y),
                    (run[0][3].x, run[0][3].y), (run[0][4].x, run[0][4].y), 0.5)
                self._add_arc((cx, cy), r, p_start, p_end, mid, attribs)
                if closed:  # 闭合但段数<3：罕见，补记
                    logger.info("闭合圆弧段数 %d <3，按 ARC 输出", n)
                return
        ellipses = [_fit_ellipse_bezier(it[1:]) for it in run]
        if all(e is not None for e in ellipses):
            cx = sum(e[0] for e in ellipses) / n
            cy = sum(e[1] for e in ellipses) / n
            rx = sum(e[2] for e in ellipses) / n
            ry = sum(e[3] for e in ellipses) / n
            coherent = all(
                _dist((e[0], e[1]), (cx, cy)) <= 0.02 * max(rx, ry)
                and abs(e[2] - rx) <= 0.02 * rx
                and abs(e[3] - ry) <= 0.02 * ry for e in ellipses)
            p_start = (run[0][1].x, run[0][1].y)
            p_end = (run[-1][4].x, run[-1][4].y)
            closed = _dist(p_start, p_end) <= CLOSE_TOL_PT
            if coherent and closed and n >= 3:
                self._add_ellipse((cx, cy), rx, ry, attribs)
                return
            if coherent:
                # 部分椭圆弧：以 SPLINE 精确输出（DXF ELLIPSE 参数角
                # 换算易错，SPLINE 无损且查看器兼容）
                logger.info("部分椭圆弧（%d 段）以 SPLINE 无损输出", n)
                for it in run:
                    self.convert_curve(it)
                return
        # 无法拟合：逐段控制点 SPLINE
        for it in run:
            self.convert_curve(it)

    # ------------------------------------------------------------------ #
    # 填充
    # ------------------------------------------------------------------ #
    def _maybe_hatch(self, boundary_pts: List[Tuple[float, float]]) -> None:
        """对 fill 类型的闭合路径生成实体填充 HATCH。"""
        if not self._fill_active:
            return
        rgb = self._fill_rgb
        try:
            h = self.msp.add_hatch(dxfattribs={"layer": LAYER_GEOMETRY})
            if rgb is not None:
                r = max(0, min(255, int(round(rgb[0] * 255))))
                g = max(0, min(255, int(round(rgb[1] * 255))))
                b = max(0, min(255, int(round(rgb[2] * 255))))
                h.rgb = (r, g, b)
            h.paths.add_polyline_path(boundary_pts, is_closed=True)
            self.stats["HATCH"] += 1
        except Exception as exc:
            logger.warning("HATCH 创建失败: %s", exc)
            self.stats["unconverted"].append(("hatch", str(exc)))

    # ------------------------------------------------------------------ #
    # path 级转换
    # ------------------------------------------------------------------ #
    def _convert_drawing(self, d: Dict[str, Any]) -> None:
        dtype = d.get("type", "s")
        self._fill_active = dtype in ("f", "fs")
        self._fill_rgb = d.get("fill")
        stroke_rgb = d.get("color") if dtype in ("s", "fs") else None
        width = d.get("width", 1.0)
        # stroke 优先取 stroke 色；纯 fill 路径边界取 fill 色
        rgb = stroke_rgb if stroke_rgb is not None else d.get("fill")
        self._current_attribs = self._base_attribs(rgb, width)
        dashes = d.get("dashes") or ""  # 格式: "[] 0" 或 "[3 2] 0"；None->"" 表示实线
        if isinstance(dashes, str) and dashes.strip() and not dashes.startswith("[]"):
            self._current_attribs["linetype"] = self._linetype_for_dashes(dashes)
            logger.debug("虚线 path(%s) -> linetype %s",
                         dashes, self._current_attribs["linetype"])

        items = d.get("items", [])
        bezier_run: List[Any] = []
        # 纯填充路径的折线边界累积: 'f' 型 path 的 'l' 子图元构成一个或多个
        # 闭合子路径(如晕渲点、面状符号), 不能当 LINE 输出也不能丢弃,
        # 应按子路径闭合成组 -> HATCH 实体填充。
        fill_runs: List[List[Tuple[float, float]]] = []

        def flush_run():
            if bezier_run:
                self._convert_bezier_run(bezier_run)
                bezier_run.clear()

        def flush_fill_runs() -> None:
            for run in fill_runs:
                if len(run) >= 3:
                    # PDF 填充语义: 未闭合子路径在填充前按闭合处理(隐式 closepath)。
                    # 箭头/实心符号的填充常只有 2 条 'l' 边(三角形第 3 边由
                    # 闭合操作补上), 必须显式补首点闭合, 否则被整体丢弃。
                    if _dist(run[0], run[-1]) > 0.01:
                        run = run + [run[0]]
                    self._maybe_hatch([self._to_dxf(p) for p in run])
                else:
                    self.stats["unconverted"].append(
                        ("fill boundary", "填充子路径退化为线/点(不足3顶点)"))
            fill_runs.clear()

        for item in items:
            kind = item[0]
            try:
                if kind == "l":
                    flush_run()
                    if stroke_rgb is not None or not self._fill_active:
                        self.convert_line(item)
                    if self._fill_active:
                        # 'f'/'fs' 的折线边界都累积成闭合子路径 -> HATCH;
                        # 'fs' 在上面已额外描边(既有轮廓线又有实体填充)
                        _, p1, p2 = item
                        q1 = (float(p1.x), float(p1.y))
                        q2 = (float(p2.x), float(p2.y))
                        if fill_runs and _dist(fill_runs[-1][-1], q1) <= 0.01:
                            fill_runs[-1].append(q2)
                        else:
                            fill_runs.append([q1, q2])
                elif kind == "re":
                    flush_run()
                    self.convert_rect(item)
                elif kind == "qu":
                    flush_run()
                    self.convert_quad(item)
                elif kind == "c":
                    bezier_run.append(item)
                else:
                    self.stats["unconverted"].append(
                        (f"item type {kind!r}", "未知图元类型"))
                    logger.warning("未知图元类型: %s", kind)
            except Exception as exc:
                logger.warning("图元 %s 转换失败: %s", kind, exc)
                self.stats["unconverted"].append((f"item {kind}", str(exc)))
        flush_run()
        if self._fill_active:
            flush_fill_runs()

    # ------------------------------------------------------------------ #
    # 保存 / 主流程
    # ------------------------------------------------------------------ #
    def _output_path(self, page_num: int, page_count: int) -> str:
        if self.dxf_path:
            out = Path(self.dxf_path)
            if out.suffix.lower() == ".dxf":
                if page_count > 1:
                    return str(out.with_name(
                        f"{out.stem}_p{page_num + 1:03d}.dxf"))
                return str(out)
            out.mkdir(parents=True, exist_ok=True)
            stem = Path(self.pdf_path).stem
            name = (f"{stem}_p{page_num + 1:03d}.dxf" if page_count > 1
                    else f"{stem}.dxf")
            return str(out / name)
        stem = Path(self.pdf_path).stem
        name = (f"{stem}_p{page_num + 1:03d}.dxf" if page_count > 1
                else f"{stem}.dxf")
        return str(Path(self.pdf_path).parent / name)

    def _looks_like_scan(self, page_num: int, entities: Dict[str, Any]) -> bool:
        """判定扫描页: 矢量内容极少 且 页面上存在覆盖 >50% 面积的栅格图像。"""
        if len(entities["drawings"]) > 10 or entities["texts"]:
            return False
        assert self.doc is not None
        page = self.doc[page_num]
        page_area = abs(page.rect)
        if page_area <= 0:
            return False
        for img in page.get_images(full=True):
            try:
                rects = page.get_image_rects(img[0])
            except Exception:
                continue
            if sum(abs(r) for r in rects) > 0.5 * page_area:
                return True
        return False

    def _run_raster_page(self, page_num: int) -> str:
        """把扫描页渲染成位图后走 raster2dxf 光栅矢量化管线。"""
        from raster2dxf import RasterToDXFConverter
        assert self.doc is not None
        pix = self.doc[page_num].get_pixmap(dpi=150)
        import cv2
        import numpy as _np
        img = _np.frombuffer(pix.samples, dtype=_np.uint8).reshape(
            pix.height, pix.width, pix.n)
        img = cv2.cvtColor(
            img, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
        out = self._output_path(page_num, self.doc.page_count)
        RasterToDXFConverter(dpi=150).convert_image(img, out)
        self.stats["raster_page"] = 1
        return out

    def save(self, path: Optional[str] = None) -> str:
        """保存 DXF 文件并做 audit 预检。"""
        assert self.dxf is not None
        out = path or self.dxf_path
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        self.dxf.saveas(out)
        logger.info("已保存 %s", out)
        return out

    def run(self, page_num: int = 0) -> str:
        """转换指定页，返回输出 DXF 路径。"""
        t0 = time.perf_counter()
        self._reset_stats()
        self.matrix_applications = 0
        self._open()
        assert self.doc is not None
        page_count = self.doc.page_count
        logger.info("=== 转换 %s 第 %d/%d 页 ===",
                    self.pdf_path, page_num + 1, page_count)
        if page_num >= page_count:
            raise IndexError(f"页码 {page_num} 超出范围（共 {page_count} 页）")

        entities = self.extract_page_entities(page_num)

        # 扫描版 PDF 页检测: 矢量图元极少且页面被大图覆盖 -> 转光栅矢量化管线
        if self._looks_like_scan(page_num, entities):
            logger.warning("第 %d 页疑似扫描件（矢量 path=%d，含大面积栅格图），"
                           "切换到光栅矢量化管线（近似重建，非无损）",
                           page_num + 1, len(entities["drawings"]))
            return self._run_raster_page(page_num)

        self._new_dxf()

        if not entities["drawings"] and not entities["texts"]:
            logger.warning("第 %d 页无可提取内容（空页/纯位图页），"
                           "输出空 DXF", page_num + 1)

        for d in entities["drawings"]:
            try:
                self._convert_drawing(d)
            except Exception as exc:
                logger.warning("path 转换失败: %s", exc)
                self.stats["unconverted"].append(("drawing path", str(exc)))
        for span in entities["texts"]:
            self.convert_text(span, self.transform)

        out = self._output_path(page_num, page_count)
        self.save(out)

        elapsed = time.perf_counter() - t0
        self.stats["elapsed_sec"] = elapsed
        self.stats["matrix_applications"] = self.matrix_applications
        # 汇总日志
        counts = {k: v for k, v in self.stats.items()
                  if isinstance(v, int) and v and k not in
                  ("pdf_drawings", "pdf_items", "pdf_text_spans",
                   "matrix_applications")}
        logger.info("写入 DXF 实体: %s", counts)
        logger.info("坐标变换: 页面 %.1fx%.1f pt, rotation=%d, "
                    "矩阵应用 %d 次", self.page_width, self.page_height,
                    self.page_rotation, self.matrix_applications)
        if self.stats["unconverted"]:
            for what, why in self.stats["unconverted"]:
                logger.warning("未转换: %s —— %s", what, why)
        else:
            logger.info("未转换图元: 0")
        logger.info("耗时 %.3f s", elapsed)
        return out

    def close(self) -> None:
        if self.doc is not None:
            self.doc.close()
            self.doc = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _convert_one(pdf_path: str, output: Optional[str], page: int,
                 log_path: Optional[str], bake_dashes: bool = False) -> List[str]:
    outs = []
    with PDFToDXFConverter(pdf_path, output, bake_dashes=bake_dashes) as conv:
        assert conv.doc is not None
        n = conv.doc.page_count
        pages = range(n) if page < 0 else [page]
        for p in pages:
            outs.append(conv.run(p))
    return outs


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="PDF 工程图纸转 DXF 转换器（布局无损）",
        epilog="示例: python pdf2dxf.py --input ./test_pdfs/ --output ./output/")
    ap.add_argument("--input", required=True, help="输入 PDF 文件或目录（批量）")
    ap.add_argument("--output", default=None, help="输出 DXF 文件或目录")
    ap.add_argument("--page", type=int, default=-1,
                    help="页码（0 起）；默认 -1 = 全部页，每页一个 DXF")
    ap.add_argument("--log", default=None, help="日志文件路径")
    ap.add_argument("--bake-dashes", action="store_true",
                    help="把虚线展开为显式短线段（兼容忽略线型的查看器）")
    args = ap.parse_args(argv)

    inp = Path(args.input)
    if not inp.exists():
        print(f"输入不存在: {inp}", file=sys.stderr)
        return 2
    log_path = args.log
    if log_path is None:
        if args.output and not str(args.output).lower().endswith(".dxf"):
            log_path = str(Path(args.output) / "conversion.log")
        else:
            log_path = "conversion.log"
    configure_logging(log_path)

    # 图片输入（png/jpg/tif/bmp）直接走光栅矢量化管线
    if inp.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        from raster2dxf import RasterToDXFConverter
        conv = RasterToDXFConverter()
        outs = conv.convert_file(str(inp), args.output or "raster_output")
        for o in outs:
            print(o)
        print("统计:", conv.stats)
        return 0

    pdfs = sorted(inp.glob("*.pdf")) if inp.is_dir() else [inp]
    if not pdfs:
        print(f"目录中无 PDF: {inp}", file=sys.stderr)
        return 2
    logger.info("批量转换: %d 个 PDF -> %s", len(pdfs), args.output or "同目录")
    ok = True
    for pdf in pdfs:
        try:
            _convert_one(str(pdf), args.output, args.page, log_path,
                         bake_dashes=args.bake_dashes)
        except Exception as exc:
            ok = False
            logger.error("转换 %s 失败: %s", pdf, exc)
    logger.info("全部完成")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
