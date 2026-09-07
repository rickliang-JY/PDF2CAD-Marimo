"""ocr_engines.py —— OCR 引擎抽象层（CPU 默认可用，GPU 引擎 lazy import 探测）。

引擎：
    tesseract  Tesseract OCR（CPU，pytesseract + 系统二进制，默认可用）
    paddle     PaddleOCR PP-OCRv5（GPU 推荐；中文/旋转文字检出率更高，
               lazy import，缺失时自动降级）

统一接口：
    available_engines() -> {'engine_key': '中文显示名', ...}（只含可用项）
    ocr_extract(img_bgr, engine='auto') -> [
        {'text': str, 'bbox': (x, y, w, h),  # 原图像素轴对齐框
         'angle': float,                      # DXF 文字旋转角（度，逆时针为正）
         'conf': float,                       # 置信度 0.0~1.0
         # 以下内部字段供 DXF TEXT 精确定位（baseline 基点）
         'insert': (x, y), 'text_h_px': float}, ...]

engine='auto'：paddle 可用优先（旋转文字强），否则 tesseract。
任何引擎失败都降级到下一可用引擎并记 warning；全不可用返回 []（不抛异常）。
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger("pdf2cad.ocr_engines")

# 引擎显示名（available_engines 返回子集）
_ENGINE_LABELS = {
    "tesseract": "Tesseract（CPU，默认可用）",
    "paddle": "PaddleOCR PP-OCRv5（GPU 推荐）",
}

# auto 模式的优先级顺序
_AUTO_ORDER = ("paddle", "tesseract")


# --------------------------------------------------------------------------- #
# 可用性探测（全部 lazy import，绝不抛异常）
# --------------------------------------------------------------------------- #
def tesseract_available() -> bool:
    """pytesseract 可 import 且 tesseract 系统二进制存在。"""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def paddle_available() -> bool:
    """paddleocr 可 import（lazy 探测；不真正加载模型）。"""
    try:
        import paddleocr  # noqa: F401
        return True
    except Exception:
        return False


def available_engines() -> Dict[str, str]:
    """返回当前可用的 OCR 引擎 {键: 中文显示名}。"""
    out: Dict[str, str] = {}
    if tesseract_available():
        out["tesseract"] = _ENGINE_LABELS["tesseract"]
    if paddle_available():
        out["paddle"] = _ENGINE_LABELS["paddle"]
    return out


# --------------------------------------------------------------------------- #
# Tesseract 路径：4 方向 image_to_data（复用 raster_plus 的旋转探测思路）
# --------------------------------------------------------------------------- #
def _tesseract_extract(img_bgr: np.ndarray) -> List[dict]:
    """tesseract 4 方向文字探测。失败抛异常（由上层统一降级）。"""
    import pytesseract
    # 复用 raster_plus 的旋转逆映射与去重工具（同一约定，避免重造轮子）
    try:
        from core.raster_plus import _rotation_passes, _iou
    except ImportError:  # 直接以 core/ 为顶层目录运行时
        from raster_plus import _rotation_passes, _iou

    h_px = img_bgr.shape[0]
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
            # 与 raster_plus.detect_text_items 相同的过滤口径（保证行为一致）
            if h_i > 0.03 * h_px or h_i < 4 or w_i < 4:
                continue
            if not any(ch.isalnum() for ch in txt):
                continue
            x, y, bw, bh, ix, iy = inv(int(data["left"][i]), int(data["top"][i]),
                                       w_i, h_i)
            box = (int(x), int(y), int(bw), int(bh))
            # 跨方向去重：与已保留框 IoU>0.3 视为同一文字
            if any(_iou(box, it["bbox"]) > 0.3 for it in items):
                continue
            items.append({
                "text": txt,
                "bbox": box,
                "angle": float(rot_deg),          # DXF 旋转角
                "conf": max(0.0, min(1.0, conf / 100.0)),
                "insert": (float(ix), float(iy)),  # baseline 基点（像素）
                "text_h_px": float(h_i),
            })
    if n_pass_fail == 4:
        raise RuntimeError("tesseract 四个方向探测均失败")
    return items


# --------------------------------------------------------------------------- #
# PaddleOCR 路径（lazy import；PP-OCR 旋转框四点 -> bbox + angle）
# --------------------------------------------------------------------------- #
def _paddle_extract(img_bgr: np.ndarray) -> List[dict]:
    """PaddleOCR 文字探测。失败抛异常（由上层统一降级）。

    PP-OCR 返回每项为 [四点旋转框, (文本, 置信度)]，四点按阅读顺序
    （左上/右上/右下/左下）。图像系(y 向下)的文字方向角取反即为 DXF 旋转角。
    """
    from paddleocr import PaddleOCR  # lazy import：重依赖

    ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    result = ocr.ocr(img_bgr, cls=True)
    items: List[dict] = []
    for page in result or []:
        for line in page or []:
            try:
                pts = np.asarray(line[0], dtype=np.float64)  # (4, 2)
                text, conf = str(line[1][0]).strip(), float(line[1][1])
            except Exception:
                continue
            if not text:
                continue
            p0, p1, p3 = pts[0], pts[1], pts[3]
            # 文字方向角（图像系）-> DXF 旋转角（y 翻转取反）
            img_ang = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
            dxf_ang = (-img_ang) % 360.0
            x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
            x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
            items.append({
                "text": text,
                "bbox": (int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
                "angle": dxf_ang,
                "conf": max(0.0, min(1.0, conf)),
                "insert": (float(p3[0]), float(p3[1])),  # baseline 起点
                "text_h_px": float(max(1.0, np.hypot(*(p3 - p0)))),
            })
    return items


# --------------------------------------------------------------------------- #
# 统一入口
# --------------------------------------------------------------------------- #
def ocr_extract(img_bgr: np.ndarray, engine: str = "auto") -> List[dict]:
    """统一 OCR 入口。任何引擎失败都降级到下一可用引擎并记 warning。

    engine: 'auto' | 'tesseract' | 'paddle'。全不可用或全部失败返回 []。
    """
    if engine == "auto":
        order = [e for e in _AUTO_ORDER]
    else:
        # 指定引擎优先，其余按 auto 顺序兜底
        order = [engine] + [e for e in _AUTO_ORDER if e != engine]

    avail = available_engines()
    attempted = False
    for eng in order:
        if eng not in avail:
            if engine != "auto" and eng == engine:
                logger.warning("OCR 引擎 %s 不可用（未安装/缺依赖），尝试降级", eng)
            continue
        attempted = True
        try:
            if eng == "paddle":
                items = _paddle_extract(img_bgr)
            else:
                items = _tesseract_extract(img_bgr)
            if eng != engine and engine != "auto":
                logger.warning("OCR 引擎 %s 失败/不可用，已降级到 %s", engine, eng)
            return items
        except Exception as exc:
            logger.warning("OCR 引擎 %s 执行失败（%s），尝试下一引擎", eng, exc)
            continue
    if not attempted:
        logger.warning("无可用 OCR 引擎（tesseract/paddle 均缺失）")
    return []
