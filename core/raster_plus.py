"""raster_plus.py —— CV+ 增强视觉管线（在经典骨架管线基础上的三项升级）。

升级（全部纯 OpenCV/numpy/可选依赖，无 GPU）：
1. 文字掩膜分离：tesseract image_to_data 取文字框（含 90°/180°/270° 三个旋转
   方向，覆盖旋转文字），把框内墨迹（外扩 3px）从 ink 中抠除后再骨架化，
   解决"文字粘连图元"导致的乱线；被抠文字照常 OCR 写入 TEXT 实体。
2. LSD 线段检测 + 间隙容忍合并：cv2.createLineSegmentDetector 检测线段，
   对近共线（角度差 <3°、法向距离 <2.5px）且端点间隙 < gap_px 的线段迭代
   合并为一条——把虚线/点划线的断裂段重建为单图元；合并段数 ≥3 且
   总间隙/总长 > 15% 的组写入 DASHED 图层（DASHED 线型），其余写 LINE。
3. 圆/弧拟合升级：对骨架追踪出的较长折线做最小二乘圆拟合，残差与弧覆盖角
   （30°~330°）达标时改写为 ARC 实体；失败保留原折线，绝不减少实体。

另有可选 vtracer 通道（use_vtracer=True，lazy import，缺库时中文 warning 跳过）：
把 ink 位图矢量化为 SVG path，解析 M/L/C/Q/Z（贝塞尔分段采样为折线）写入
VTRACER 图层，适合曲线密集区域。

复用 raster2dxf.py 既有组件（binarize/deskew/trace_skeleton/simplify/_map/
OCR 思路/Hough 圆检测），不重造轮子。
"""
from __future__ import annotations

import gc
import logging
import math
import os
import re
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import cv2
import ezdxf
import numpy as np

# 让 core/ 下的既有模块能以顶层模块方式 import（与 backends.py 同一约定）
import sys as _sys
_CORE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CORE_DIR not in _sys.path:
    _sys.path.insert(0, _CORE_DIR)

try:
    from core.raster2dxf import (  # noqa: E402 包方式导入
        DXF_VERSION, LAYER_LINE, LAYER_TEXT, RasterToDXFConverter,
        binarize, deskew, simplify, trace_skeleton)
except ImportError:  # 直接以 core/ 为顶层目录运行时
    from raster2dxf import (  # noqa: E402 同目录导入
        DXF_VERSION, LAYER_LINE, LAYER_TEXT, RasterToDXFConverter,
        binarize, deskew, simplify, trace_skeleton)

logger = logging.getLogger("raster_plus")

LAYER_DASHED = "DASHED"      # 重建出的虚线/点划线
LAYER_VTRACER = "VTRACER"    # vtracer 可选通道输出

# LSD 合并阈值
_MERGE_ANGLE_TOL_DEG = 3.0   # 近共线：角度差上限
_MERGE_DIST_PX = 2.5         # 近共线：法向距离上限
_DASH_GAP_RATIO = 0.15       # 虚线判定：总间隙/总长 阈值
_DASH_MIN_SEGS = 3           # 虚线判定：合并原始段数下限
_TEXT_EXPAND_PX = 3          # 文字框外扩像素（抠除墨迹用）
_MAX_LSD_SEGS = 6000         # LSD 段数上限（超出时提高 min_seg_len 过滤，保内存）

# 圆拟合阈值
_ARC_MIN_POINTS = 15         # 参与拟合的骨架路径最少点数
_ARC_MIN_LEN_PX = 30.0       # 参与拟合的路径最短弧长
_ARC_MIN_RADIUS_PX = 6.0     # 半径下限（避免把文字残片/噪点拟合成弧）
_ARC_MIN_COVER_DEG = 30.0    # 弧覆盖角下限（不足则不像弧）
_ARC_MAX_COVER_DEG = 330.0   # 弧覆盖角上限（整圆交给 Hough 验证路径）


# --------------------------------------------------------------------------- #
# 文字探测（多方向 OCR：原始 + 90°/180°/270°，覆盖旋转文字）
# --------------------------------------------------------------------------- #
def _rotation_passes(img_bgr: np.ndarray):
    """产出 (旋转后图像, DXF 文字旋转角, 框逆映射函数)。

    逆映射把旋转图上的框 (l, t, w, h) 映回原图像素坐标：
    返回 (x, y, w, h, ins_x, ins_y)，其中 (ins_x, ins_y) 是文字基点
    （baseline 起点，供 DXF TEXT 定位）。
    """
    h, w = img_bgr.shape[:2]

    def inv_0(l, t, bw, bh):
        return l, t, bw, bh, l, t + bh

    def inv_90(l, t, bw, bh):  # 原图逆时针 90° 的文字（竖排，自下而上）
        # cv2.ROTATE_90_CLOCKWISE 后检测：rot(X,Y) = orig(Y, H-1-X)
        return t, h - 1 - l - bw, bh, bw, t + bh, h - 1 - l

    def inv_180(l, t, bw, bh):
        return w - 1 - l - bw, h - 1 - t - bh, bw, bh, w - 1 - l, h - 1 - t - bh

    def inv_270(l, t, bw, bh):  # 原图顺时针 90° 的文字（竖排，自上而下）
        # cv2.ROTATE_90_COUNTERCLOCKWISE 后检测：rot(X,Y) = orig(W-1-Y, X)
        return w - 1 - t - bh, l, bh, bw, w - 1 - t - bh, l

    yield img_bgr, 0.0, inv_0
    yield cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE), 90.0, inv_90
    yield cv2.rotate(img_bgr, cv2.ROTATE_180), 180.0, inv_180
    yield cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE), 270.0, inv_270


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """两框 (x,y,w,h) 的交并比（用于跨方向去重）。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def detect_text_items(img_bgr: np.ndarray, h_px: int
                      ) -> Tuple[List[dict], Optional[str]]:
    """多方向 tesseract 文字探测。

    返回 (items, error)；items 元素:
        {'box': (x,y,w,h) 原图像素框, 'rotation': DXF 角度,
         'insert': (x,y) 文字基点（像素）, 'text_h_px': 字高, 'text': str}
    error 非 None 表示 tesseract 完全不可用（调用方据此跳过掩膜并告警）。
    """
    try:
        import pytesseract
    except Exception as exc:
        return [], f"pytesseract 导入失败（{exc}）"

    items: List[dict] = []
    n_pass_fail = 0
    for img_rot, rot_deg, inv in _rotation_passes(img_bgr):
        try:
            data = pytesseract.image_to_data(
                img_rot, output_type=pytesseract.Output.DICT, config="--psm 11")
        except Exception as exc:
            n_pass_fail += 1
            logger.warning("tesseract 探测失败（旋转 %.0f°）: %s", rot_deg, exc)
            continue
        for i, txt in enumerate(data["text"]):
            txt = str(txt).strip()
            try:
                conf = float(data["conf"][i])
            except (ValueError, TypeError):
                conf = -1
            if not txt or conf < 60:
                continue
            w_i, h_i = int(data["width"][i]), int(data["height"][i])
            # 与 _ocr_text 相同的过滤：巨型框/碎片/无字母数字
            if h_i > 0.03 * h_px or h_i < 4 or w_i < 4:
                continue
            if not any(ch.isalnum() for ch in txt):
                continue
            x, y, bw, bh, ix, iy = inv(int(data["left"][i]), int(data["top"][i]),
                                       w_i, h_i)
            box = (int(x), int(y), int(bw), int(bh))
            # 跨方向去重：与已保留框 IoU>0.3 视为同一文字
            if any(_iou(box, it["box"]) > 0.3 for it in items):
                continue
            items.append({"box": box, "rotation": rot_deg,
                          "insert": (float(ix), float(iy)),
                          "text_h_px": float(h_i), "text": txt})
    if n_pass_fail == 4:
        return [], "tesseract 四个方向探测均失败"
    return items, None


# --------------------------------------------------------------------------- #
# LSD 线段检测 + 间隙容忍合并
# --------------------------------------------------------------------------- #
class _Seg:
    """参与合并的线段：端点 p/q（像素），count=原始段数，gap_sum=累计间隙。"""
    __slots__ = ("p", "q", "angle", "length", "count", "gap_sum")

    def __init__(self, p, q, count=1, gap_sum=0.0):
        self.p = np.asarray(p, np.float64)
        self.q = np.asarray(q, np.float64)
        d = self.q - self.p
        self.length = float(np.hypot(d[0], d[1]))
        self.angle = math.degrees(math.atan2(d[1], d[0])) % 180.0
        self.count = count
        self.gap_sum = gap_sum


def merge_segments(segs: List[_Seg], gap_px: float,
                   angle_tol: float = _MERGE_ANGLE_TOL_DEG,
                   dist_tol: float = _MERGE_DIST_PX) -> List[_Seg]:
    """迭代合并近共线且端点间隙 < gap_px 的线段（重建虚线/点划线）。

    策略：按角度分桶（3°/桶，含 0/180 环绕），桶内+相邻桶两两检查；
    满足条件的用并查集连通成组，整组重投影为一条线段；重复至收敛。
    """
    segs = list(segs)
    for _pass in range(20):
        n = len(segs)
        if n < 2:
            break
        parent = list(range(n))
        gap_edge: Dict[Tuple[int, int], float] = {}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b, gap):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
                gap_edge[(min(a, b), max(a, b))] = gap

        # 角度分桶：桶 i 只与桶 i、i+1（及环绕）比较，避免 O(n²) 全量
        nbins = 60
        buckets: List[List[int]] = [[] for _ in range(nbins)]
        for idx, s in enumerate(segs):
            buckets[int(s.angle // 3.0) % nbins].append(idx)

        merged_any = False
        for b in range(nbins):
            cand = buckets[b] + buckets[(b + 1) % nbins]
            m = len(cand)
            if m < 2:
                continue
            P = np.array([segs[idx].p for idx in cand])
            Q = np.array([segs[idx].q for idx in cand])
            ANG = np.array([segs[idx].angle for idx in cand])
            MID = (P + Q) / 2.0
            for ii in range(m - 1):
                si = segs[cand[ii]]
                if si.length < 1e-9:
                    continue
                ui = (si.q - si.p) / si.length
                ni = np.array([-ui[1], ui[0]])
                # 角度差（mod 180°）
                dang = np.abs(ANG[ii + 1:] - ANG[ii]) % 180.0
                dang = np.minimum(dang, 180.0 - dang)
                ok = dang < angle_tol
                if not ok.any():
                    continue
                # 法向距离：j 中点到 i 直线的距离
                off = np.abs((MID[ii + 1:] - MID[ii]) @ ni)
                ok &= off < dist_tol
                if not ok.any():
                    continue
                # 端点间隙：投影区间 [lo,hi] 的间距（重叠为 0）
                lo = np.minimum(P @ ui, Q @ ui)
                hi = np.maximum(P @ ui, Q @ ui)
                gap = np.maximum(0.0, np.maximum(lo[ii], lo[ii + 1:])
                                 - np.minimum(hi[ii], hi[ii + 1:]))
                ok &= gap < gap_px
                for k in np.nonzero(ok)[0]:
                    union(cand[ii], cand[ii + 1 + int(k)], float(gap[k]))
                    merged_any = True
        if not merged_any:
            break
        # 连通组重投影为单线段
        groups: Dict[int, List[int]] = {}
        for idx in range(n):
            groups.setdefault(find(idx), []).append(idx)
        new_segs: List[_Seg] = []
        for root, members in groups.items():
            if len(members) == 1:
                new_segs.append(segs[members[0]])
                continue
            pts, lens = [], []
            cnt, gap_sum = 0, 0.0
            for mi in members:
                s = segs[mi]
                pts += [s.p, s.q]
                lens += [s.length, s.length]
                cnt += s.count
                gap_sum += s.gap_sum
            # 组内合并边带来的间隙（只累计同组内的 union 边）
            for (a, b), g in gap_edge.items():
                if find(a) == root:
                    gap_sum += g
            pts = np.asarray(pts)
            lens = np.asarray(lens)
            # 双角法求加权平均方向（角度 mod 180°）
            ang = np.array([math.radians(2.0 * segs[mi].angle)
                            for mi in members for _ in (0, 1)])
            ca = float((lens * np.cos(ang)).sum())
            sa = float((lens * np.sin(ang)).sum())
            mean_ang = math.degrees(math.atan2(sa, ca)) / 2.0
            u = np.array([math.cos(mean_ang), math.sin(mean_ang)])
            nv = np.array([-u[1], u[0]])
            t = pts @ u
            c = float((pts @ nv * lens).sum() / lens.sum())
            lo, hi = float(t.min()), float(t.max())
            p = lo * u + c * nv
            q = hi * u + c * nv
            new_segs.append(_Seg(p, q, count=cnt, gap_sum=gap_sum))
        segs = new_segs
    return segs


def detect_lines_lsd(ink: np.ndarray, min_seg_len_px: float, gap_px: float
                     ) -> Tuple[List[_Seg], List[_Seg]]:
    """LSD 检测 + 合并。返回 (实线段组, 虚线段组)。"""
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines, *_ = lsd.detect((ink.astype(np.uint8)) * 255)
    if lines is None:
        return [], []
    min_len = float(min_seg_len_px)
    raw: List[_Seg] = []
    for l in lines.reshape(-1, 4):
        x1, y1, x2, y2 = (float(v) for v in l)
        if math.hypot(x2 - x1, y2 - y1) >= min_len:
            raw.append(_Seg((x1, y1), (x2, y2)))
    # 段数过多时自适应提高下限，保护合并阶段的内存与耗时
    while len(raw) > _MAX_LSD_SEGS:
        min_len *= 1.5
        raw = [s for s in raw if s.length >= min_len]
        logger.info("LSD 段数过多，提高 min_seg_len 至 %.1f（剩 %d 段）",
                    min_len, len(raw))
    merged = merge_segments(raw, gap_px)
    solid, dashed = [], []
    for s in merged:
        span = s.length
        if (s.count >= _DASH_MIN_SEGS and span > 0
                and s.gap_sum / span > _DASH_GAP_RATIO):
            dashed.append(s)
        else:
            solid.append(s)
    return solid, dashed


# --------------------------------------------------------------------------- #
# 最小二乘圆拟合（Kasa 法）+ 弧覆盖角判定
# --------------------------------------------------------------------------- #
def fit_circle(pts: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """Kasa 最小二乘圆拟合。返回 (cx, cy, r, rms残差)，点数不足返回 None。"""
    if len(pts) < 3:
        return None
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r2 = c + cx * cx + cy * cy
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    rms = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
    return float(cx), float(cy), r, rms


def arc_angles(pts: np.ndarray, cx: float, cy: float
               ) -> Optional[Tuple[float, float, float]]:
    """弧覆盖角分析。返回 (dxf_start, dxf_end, coverage_deg)。

    覆盖角 = 360° - 最大相邻角间隙。像素系(y 向下)与 DXF 系(y 向上)互为
    镜像：像素系 CCW 从 b 到 a 的弧，在 DXF 系为 CCW 从 -a 到 -b。
    """
    ang = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)) % 360.0
    ang.sort()
    gaps = np.diff(np.concatenate([ang, ang[:1] + 360.0]))
    i = int(np.argmax(gaps))
    coverage = 360.0 - float(gaps[i])
    a = float(ang[i])                        # 间隙起点角
    b = float(ang[(i + 1) % len(ang)])       # 间隙终点角
    dxf_start = (-a) % 360.0
    dxf_end = (-b) % 360.0
    # 校验：(end - start) % 360 应与 coverage 一致
    if abs((dxf_end - dxf_start) % 360.0 - coverage) > 5.0:
        return None
    return dxf_start, dxf_end, coverage


# --------------------------------------------------------------------------- #
# SVG path 简易解析（vtracer 通道）：M/L/C/Q/Z（含小写相对命令、H/V）
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"([MmLlCcQqZzHhVv])|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
_PATH_RE = re.compile(r'<path[^>]*\sd="([^"]+)"[^>]*>', re.S)
_TRANSLATE_RE = re.compile(r'translate\(\s*(-?[\d.]+)[\s,]+(-?[\d.]+)\s*\)')


def _sample_cubic(p0, p1, p2, p3, n: int = 12) -> List[Tuple[float, float]]:
    ts = np.linspace(0.0, 1.0, n + 1)[1:]
    out = []
    for t in ts:
        mt = 1.0 - t
        x = mt**3 * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t**3 * p3[1]
        out.append((x, y))
    return out


def _sample_quad(p0, p1, p2, n: int = 10) -> List[Tuple[float, float]]:
    ts = np.linspace(0.0, 1.0, n + 1)[1:]
    out = []
    for t in ts:
        mt = 1.0 - t
        x = mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0]
        y = mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1]
        out.append((x, y))
    return out


def parse_svg_paths(svg: str, max_paths: int = 2000
                    ) -> List[Tuple[List[Tuple[float, float]], bool]]:
    """把 SVG 的 path d 属性解析为折线列表。返回 [(点列, 是否闭合)]。"""
    result: List[Tuple[List[Tuple[float, float]], bool]] = []
    for m_path in _PATH_RE.finditer(svg):
        if len(result) >= max_paths:
            break
        d_attr = m_path.group(1)
        tag = m_path.group(0)
        # vtracer 用 transform="translate(x,y)" 放置每个色块，必须叠加
        dx = dy = 0.0
        m_tr = _TRANSLATE_RE.search(tag)
        if m_tr:
            dx, dy = float(m_tr.group(1)), float(m_tr.group(2))
        tokens = _TOKEN_RE.findall(d_attr)
        cmd = None
        cur = (0.0, 0.0)
        start = (0.0, 0.0)
        pts: List[Tuple[float, float]] = []
        nums: List[float] = []

        def flush_path(closed: bool):
            nonlocal pts
            if len(pts) >= 2:
                result.append(([(x + dx, y + dy) for x, y in pts], closed))
            pts = []

        def take(n: int) -> Optional[List[float]]:
            nonlocal nums
            if len(nums) < n:
                return None
            out, nums = nums[:n], nums[n:]
            return out

        i = 0
        while i < len(tokens):
            letter, num = tokens[i]
            if letter:
                cmd = letter
                if cmd in "Zz":
                    flush_path(True)
                    cur = start
                i += 1
                continue
            if cmd is None:
                i += 1
                continue
            nums.append(float(num))
            rel = cmd.islower()
            need = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "Q": 4}.get(cmd.upper())
            vals = take(need)
            i += 1
            if vals is None:
                continue
            if cmd.upper() == "M":
                if pts:                # 前一个子路径未闭合
                    flush_path(False)
                nx, ny = vals
                cur = (cur[0] + nx, cur[1] + ny) if rel else (nx, ny)
                start = cur
                pts = [cur]
                cmd = "l" if rel else "L"   # M 后续坐标按 L 处理
            elif cmd.upper() == "L":
                nx, ny = vals
                cur = (cur[0] + nx, cur[1] + ny) if rel else (nx, ny)
                pts.append(cur)
            elif cmd.upper() == "H":
                cur = (cur[0] + vals[0], cur[1]) if rel else (vals[0], cur[1])
                pts.append(cur)
            elif cmd.upper() == "V":
                cur = (cur[0], cur[1] + vals[0]) if rel else (cur[0], vals[0])
                pts.append(cur)
            elif cmd.upper() == "C":
                p1 = (cur[0] + vals[0], cur[1] + vals[1]) if rel else (vals[0], vals[1])
                p2 = (cur[0] + vals[2], cur[1] + vals[3]) if rel else (vals[2], vals[3])
                p3 = (cur[0] + vals[4], cur[1] + vals[5]) if rel else (vals[4], vals[5])
                pts.extend(_sample_cubic(cur, p1, p2, p3))
                cur = p3
            elif cmd.upper() == "Q":
                p1 = (cur[0] + vals[0], cur[1] + vals[1]) if rel else (vals[0], vals[1])
                p2 = (cur[0] + vals[2], cur[1] + vals[3]) if rel else (vals[2], vals[3])
                pts.extend(_sample_quad(cur, p1, p2))
                cur = p2
        flush_path(False)
    return result


# --------------------------------------------------------------------------- #
# 主转换器
# --------------------------------------------------------------------------- #
class RasterPlusConverter(RasterToDXFConverter):
    """CV+ 增强视觉管线：文字掩膜 + LSD 虚线重建 + 弧拟合 + 可选 vtracer。

    继承 RasterToDXFConverter 复用 _map / _detect_circles / 参数约定；
    骨架化仅对"去字"墨迹进行，文字框区域不再产生乱线折线。
    """

    def __init__(self, dpi: int = 200, do_ocr: bool = True,
                 do_circles: bool = True, gap_px: float = 8.0,
                 min_seg_len_px: float = 6.0, use_vtracer: bool = False,
                 ocr_engine: str = "auto"):
        super().__init__(dpi=dpi, do_ocr=do_ocr, do_circles=do_circles)
        self.gap_px = float(gap_px)
        self.min_seg_len_px = float(min_seg_len_px)
        self.use_vtracer = use_vtracer
        # OCR 引擎选择：'auto'/'tesseract' 走既有 tesseract 4 方向路径
        # （默认行为不变）；显式指定其他引擎（如 'paddle'）时切换
        # ocr_engines.ocr_extract 抽象层（含自动降级）。
        self.ocr_engine = ocr_engine or "auto"
        self.warnings: List[str] = []

    # ------------------------------------------------------------------ #
    def _extract_text_items(self, img_bgr: np.ndarray, h_px: int
                            ) -> Tuple[List[dict], Optional[str]]:
        """按 ocr_engine 选择文字探测路径，返回 (text_items, error)。

        默认/无 GPU 引擎时与既有 detect_text_items（tesseract 4 方向）
        完全一致；仅当显式指定非 tesseract 引擎且可用时走 ocr_engines 抽象层。
        """
        try:
            from core import ocr_engines
        except ImportError:  # 直接以 core/ 为顶层目录运行时
            import ocr_engines
        eng = self.ocr_engine
        if eng in ("tesseract",) or (
                eng == "auto" and "paddle" not in ocr_engines.available_engines()):
            return detect_text_items(img_bgr, h_px)  # 既有默认路径
        raw = ocr_engines.ocr_extract(img_bgr, engine=eng)
        if not raw and not ocr_engines.available_engines():
            return [], "无可用 OCR 引擎（tesseract/paddle 均缺失）"
        # 映射为 detect_text_items 的元素格式（box/rotation/insert/text_h_px/text）
        items = [{"box": it["bbox"], "rotation": it["angle"],
                  "insert": it["insert"], "text_h_px": it["text_h_px"],
                  "text": it["text"]} for it in raw]
        if eng != "tesseract":
            used = eng if eng != "auto" else (
                "paddle" if "paddle" in ocr_engines.available_engines()
                else "tesseract")
            logger.info("OCR 引擎路径: %s（请求 %s），检出 %d 条",
                        used, eng, len(items))
        return items, None

    # ------------------------------------------------------------------ #
    def convert_image(self, img_bgr: np.ndarray, dxf_path: str) -> str:
        t0 = time.perf_counter()
        self.warnings = []
        ink = binarize(img_bgr)
        img_bgr, ink = deskew(img_bgr, ink)
        h_px, w_px = ink.shape

        # ---- 升级 1：文字掩膜分离（二值化后、骨架化之前）----
        text_items, text_err = self._extract_text_items(img_bgr, h_px)
        if text_err is not None:
            self.warnings.append(
                f"OCR 文字探测失败（{text_err}），跳过文字掩膜分离步骤")
            text_items = []
        ink_geo = ink.copy()
        for it in text_items:
            x, y, bw, bh = it["box"]
            x0 = max(0, x - _TEXT_EXPAND_PX)
            y0 = max(0, y - _TEXT_EXPAND_PX)
            x1 = min(w_px, x + bw + _TEXT_EXPAND_PX)
            y1 = min(h_px, y + bh + _TEXT_EXPAND_PX)
            ink_geo[y0:y1, x0:x1] = False
        logger.info("文字掩膜: 抠除 %d 个文字框", len(text_items))

        from skimage.morphology import skeletonize
        skel = skeletonize(ink_geo)
        logger.info("骨架像素: %d / 墨迹像素: %d（去字后 %d）",
                    int(skel.sum()), int(ink.sum()), int(ink_geo.sum()))

        doc = ezdxf.new(DXF_VERSION, setup=True)
        doc.header["$INSUNITS"] = 0
        msp = doc.modelspace()
        doc.layers.add(LAYER_LINE, color=7)
        doc.layers.add(LAYER_TEXT, color=3)
        doc.layers.add(LAYER_DASHED, color=1)
        doc.layers.add(LAYER_VTRACER, color=5)

        stats: Dict[str, object] = {}

        # ---- 升级 2：LSD 线段检测 + 间隙容忍合并（虚线重建）----
        solid, dashed = detect_lines_lsd(ink_geo, self.min_seg_len_px,
                                         self.gap_px)
        s = 72.0 / self.dpi
        n_line = 0
        for seg in solid:
            p0 = self._map(seg.p[0], seg.p[1], h_px)
            p1 = self._map(seg.q[0], seg.q[1], h_px)
            try:
                msp.add_line(p0, p1, dxfattribs={"layer": LAYER_LINE})
                n_line += 1
            except Exception as exc:
                logger.debug("LINE 写入失败: %s", exc)
        n_dash = 0
        for seg in dashed:
            p0 = self._map(seg.p[0], seg.p[1], h_px)
            p1 = self._map(seg.q[0], seg.q[1], h_px)
            try:
                msp.add_line(p0, p1, dxfattribs={"layer": LAYER_DASHED,
                                                 "linetype": "DASHED"})
                n_dash += 1
            except Exception as exc:
                logger.debug("DASHED 写入失败: %s", exc)
        stats["LINE"] = n_line
        stats["DASHED"] = n_dash
        logger.info("LSD: 实线 %d 条，重建虚线 %d 条", n_line, n_dash)

        # ---- 骨架折线 + 升级 3：长路径最小二乘圆拟合改写 ARC ----
        n_poly = 0
        n_arc = 0
        for path in trace_skeleton(skel, self.min_len_px):
            pts = simplify(path)
            if len(pts) < 2:
                continue
            if self.do_circles and self._try_write_arc(path, msp, h_px):
                n_arc += 1
                continue
            mapped = [self._map(x, y, h_px) for x, y in pts]
            closed = len(path) > 2 and path[0] == path[-1]
            try:
                msp.add_lwpolyline(mapped, close=closed,
                                   dxfattribs={"layer": LAYER_LINE})
                n_poly += 1
            except Exception as exc:
                logger.debug("折线写入失败: %s", exc)
        stats["LWPOLYLINE"] = n_poly
        stats["ARC"] = n_arc

        # ---- 圆仍走 Hough 验证路径（复用父类）----
        if self.do_circles:
            stats["CIRCLE"] = self._detect_circles(skel, msp, h_px)

        # ---- 被抠掉的文字照常写入 TEXT 实体（复用 _ocr_text 的思路）----
        if self.do_ocr:
            stats["TEXT"] = self._write_text_items(text_items, msp, h_px)

        # ---- 升级 4：vtracer 可选通道 ----
        if self.use_vtracer:
            stats["VTRACER"] = self._vtracer_paths(ink_geo, msp, h_px)

        del ink, ink_geo, skel
        gc.collect()

        os.makedirs(os.path.dirname(os.path.abspath(dxf_path)), exist_ok=True)
        doc.saveas(dxf_path)
        aud = doc.audit()
        stats["audit_errors"] = len(aud.errors)
        stats["elapsed_sec"] = round(time.perf_counter() - t0, 2)
        stats["warnings"] = list(self.warnings)
        self.stats = stats
        logger.info("已保存 %s: %s", dxf_path,
                    {k: v for k, v in stats.items() if k != "warnings"})
        return dxf_path

    # ------------------------------------------------------------------ #
    def _try_write_arc(self, path: List[Tuple[int, int]], msp, h_px: int
                       ) -> bool:
        """长折线路径尝试最小二乘圆拟合，达标则写 ARC 并返回 True。

        任一条件不满足（路径太短/残差大/覆盖角不在 30°~330°）返回 False，
        调用方保留原折线——绝不减少实体。
        """
        if len(path) < _ARC_MIN_POINTS:
            return False
        pts = np.asarray(path, np.float64)
        seg = np.diff(pts, axis=0)
        if float(np.hypot(seg[:, 0], seg[:, 1]).sum()) < _ARC_MIN_LEN_PX:
            return False
        fit = fit_circle(pts)
        if fit is None:
            return False
        cx, cy, r, rms = fit
        if r < _ARC_MIN_RADIUS_PX or rms > max(1.5, 0.03 * r):
            return False
        ang = arc_angles(pts, cx, cy)
        if ang is None:
            return False
        dxf_start, dxf_end, coverage = ang
        if not (_ARC_MIN_COVER_DEG < coverage < _ARC_MAX_COVER_DEG):
            return False
        cx_dxf, cy_dxf = self._map(cx, cy, h_px)
        try:
            msp.add_arc((cx_dxf, cy_dxf), r * 72.0 / self.dpi,
                        start_angle=dxf_start, end_angle=dxf_end,
                        dxfattribs={"layer": LAYER_LINE})
            return True
        except Exception as exc:
            logger.debug("ARC 写入失败: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    def _write_text_items(self, items: List[dict], msp, h_px: int) -> int:
        """把多方向 OCR 结果写为 TEXT 实体（思路同父类 _ocr_text）。"""
        n = 0
        for it in items:
            h_pt = it["text_h_px"] * 72.0 / self.dpi
            mx, my = self._map(it["insert"][0], it["insert"][1], h_px)
            try:
                msp.add_text(
                    it["text"], height=max(h_pt, 1.0),
                    dxfattribs={"layer": LAYER_TEXT,
                                "rotation": it["rotation"]}
                ).set_placement((mx, my))
                n += 1
            except Exception as exc:
                logger.debug("文本写入失败: %s", exc)
        return n

    # ------------------------------------------------------------------ #
    def _vtracer_paths(self, ink: np.ndarray, msp, h_px: int) -> int:
        """vtracer 可选通道：ink 位图 -> SVG path -> 折线写入 VTRACER 图层。"""
        try:
            import vtracer  # lazy import：可选依赖
        except Exception:
            self.warnings.append("vtracer 不可用（未安装），跳过 vtracer 曲线通道")
            return 0
        tmp_png = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                tmp_png = f.name
            out_svg = tmp_png + ".svg"
            # 白底黑墨位图走文件通道（避免像素元组列表撑爆内存）
            cv2.imwrite(tmp_png, np.where(ink, 0, 255).astype(np.uint8))
            vtracer.convert_image_to_svg_py(tmp_png, out_svg,
                                            colormode="binary", mode="spline")
            with open(out_svg, "r", encoding="utf-8", errors="ignore") as f:
                svg = f.read()
            n = 0
            for pts, closed in parse_svg_paths(svg):
                mapped = [self._map(x, y, h_px) for x, y in pts]
                try:
                    msp.add_lwpolyline(mapped, close=closed,
                                       dxfattribs={"layer": LAYER_VTRACER})
                    n += 1
                except Exception as exc:
                    logger.debug("VTRACER 折线写入失败: %s", exc)
            return n
        except Exception as exc:
            logger.warning("vtracer 通道执行失败: %s", exc)
            self.warnings.append(f"vtracer 通道执行失败（{exc}），已跳过")
            return 0
        finally:
            for p in (tmp_png, (tmp_png or "") + ".svg"):
                if p and os.path.isfile(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
