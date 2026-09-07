"""backends.py —— 五个转换后端的统一接口。

后端：
    vector  纯 PDF 矢量提取（封装 PDFToDXFConverter，强制矢量路径）
    cv      经典视觉管线（渲染位图 -> RasterToDXFConverter 骨架矢量化）
    cv+     增强视觉管线（RasterPlusConverter：LSD 虚线重建 + 文字掩膜 + 弧拟合）
    yolo    YOLO 图纸元素检测 + CV 矢量化（ultralytics/权重缺失时优雅回退 cv）
    auto    逐页自动路由（vector 页 -> vector；scan 页 -> yolo/cv+）

统一返回 BackendResult(TypedDict)：
    dxf_path / engine / warnings(中文) / stats
"""
from __future__ import annotations

import gc
import logging
import os
import sys
from pathlib import Path
from typing import Optional, TypedDict

import cv2
import numpy as np
import pymupdf

# 让 core/ 下的既有模块（pdf2dxf.py / raster2dxf.py）能以顶层模块方式互相 import
# （pdf2dxf.py 内部有 `from raster2dxf import ...`，raster2dxf 内部有 `import fitz`）
_CORE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)

from pdf2dxf import PDFToDXFConverter          # noqa: E402 既有矢量提取模块
from raster2dxf import RasterToDXFConverter    # noqa: E402 既有光栅矢量化模块
try:
    from core import detect                     # noqa: E402 包方式导入（项目根在 sys.path）
except ImportError:  # 直接以 core/ 为顶层目录运行时（如 marimo 脚本方式）
    import detect                               # noqa: E402 同目录导入

logger = logging.getLogger("pdf2cad.backends")

# 渲染位图时的内存约束：dpi 上限（3GB 内存机器，单页用完即 del + gc）
MAX_RENDER_DPI = 200


class BackendResult(TypedDict):
    dxf_path: str          # 输出 DXF 路径
    engine: str            # 实际使用的引擎标识
    warnings: list[str]    # 降级/近似等警告（中文）
    stats: dict            # 实体统计


MODELS = {
    "auto": "自动识别（逐页路由）",
    "vector": "纯 PDF 矢量提取",
    "cv": "经典视觉管线（OpenCV 骨架矢量化）",
    "cv+": "增强视觉管线（LSD+文字掩膜+虚线重建）",
    "yolo": "YOLO 视觉模型 + 矢量化（可自动下载安装）",
}

# YOLO 通用预训练权重（留空时自动下载；ultralytics 首次使用会从官方源拉取。
# 注意：COCO 通用权重非图纸专用，正式使用建议训练图纸权重后填入路径。
# 可用环境变量 PDF2CAD_YOLO_WEIGHTS 覆盖默认值）
DEFAULT_YOLO_WEIGHTS = os.environ.get("PDF2CAD_YOLO_WEIGHTS", "yolov8n.pt")


def _pip_install(spec: str, timeout: int = 600) -> bool:
    """尝试 pip 安装（用于自动安装缺失组件），失败返回 False，绝不抛异常。"""
    import subprocess
    try:
        proc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", spec],
                              capture_output=True, text=True, timeout=timeout)
        return proc.returncode == 0
    except Exception as exc:
        logger.warning("pip install %s 异常: %s", spec, exc)
        return False


def _ensure_ultralytics(auto_setup: bool) -> "tuple[bool, Optional[str]]":
    """探测/按需自动安装 ultralytics。返回 (可用?, 说明或 None)。"""
    try:
        import ultralytics  # noqa: F401
        return True, None
    except Exception:
        pass
    if not auto_setup:
        return False, ("未安装 ultralytics（GPU 环境请 "
                       "pip install -r requirements-gpu.txt，"
                       "或在「高级选项」打开「自动下载/安装缺失组件」）")
    if _pip_install("ultralytics"):
        try:
            import ultralytics  # noqa: F401
            return True, "已自动安装 ultralytics"
        except Exception as exc:
            return False, f"自动安装 ultralytics 后导入失败（{exc}）"
    return False, "自动安装 ultralytics 失败（pip install 未成功，可能无网络/权限）"


def _resolve_weights(weights: Optional[str]) -> "tuple[Optional[str], Optional[str]]":
    """权重解析：用户路径有效则用之；路径无效返回 (None, 原因)；
    留空则返回 (通用预训练权重名, 自动下载说明)。"""
    if weights and os.path.isfile(str(weights)):
        return str(weights), None
    if weights:
        return None, f"指定的权重文件不存在：{weights}"
    return DEFAULT_YOLO_WEIGHTS, (
        f"未提供权重，将自动下载通用预训练权重 {DEFAULT_YOLO_WEIGHTS}"
        "（COCO 通用检测，非图纸专用；建议训练图纸权重后填入路径）")


def _out_dxf_path(pdf_path: str, page_num: int, out_dir: str) -> str:
    """统一输出命名约定：{stem}_p{页码:03d}.dxf（页码从 1 起）。"""
    os.makedirs(out_dir, exist_ok=True)
    stem = Path(pdf_path).stem
    return str(Path(out_dir) / f"{stem}_p{page_num + 1:03d}.dxf")


def _render_page_bgr(pdf_path: str, page_num: int, dpi: int) -> np.ndarray:
    """渲染 PDF 页为 BGR 位图（OpenCV 约定）。处理 pix.n 为 3 或 4 的情况。"""
    dpi = max(50, min(int(dpi), MAX_RENDER_DPI))  # 内存硬约束：dpi <= 200
    doc = pymupdf.open(pdf_path)
    try:
        pix = doc[page_num].get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:  # 灰度等罕见情况，统一转 BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img
    finally:
        doc.close()
        gc.collect()


def _ocr_available() -> bool:
    """探测 OCR 是否可用：pytesseract 可 import 且 tesseract 系统二进制存在。"""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# 后端 1：纯 PDF 矢量提取
# --------------------------------------------------------------------------- #
class _ForcedVectorConverter(PDFToDXFConverter):
    """强制走矢量路径的转换器：禁用内部的扫描页自动判定。

    不改 pdf2dxf.py 源码逻辑，仅在子类中把 _looks_like_scan 固定为 False，
    保证 convert_vector 永远不会被路由到内部光栅管线。
    """

    def _looks_like_scan(self, page_num, entities):  # noqa: D102
        return False


def convert_vector(pdf_path: str, page_num: int, out_dir: str) -> BackendResult:
    """纯 PDF 矢量提取（无损路径，不触发内部扫描判定）。"""
    warnings: list[str] = []
    os.makedirs(out_dir, exist_ok=True)
    # 传入 {stem}.dxf 作为基准路径：多页时 PDFToDXFConverter._output_path
    # 会自动追加 _pNNN 后缀；单页时为 {stem}.dxf。转换后统一重命名为
    # 平台约定 {stem}_pNNN.dxf，保证四种后端命名一致。
    base_path = str(Path(out_dir) / f"{Path(pdf_path).stem}.dxf")
    conv = _ForcedVectorConverter(pdf_path, base_path)
    try:
        produced = conv.run(page_num)
    finally:
        conv.close()
    dxf_path = _out_dxf_path(pdf_path, page_num, out_dir)
    if os.path.abspath(produced) != os.path.abspath(dxf_path):
        os.replace(produced, dxf_path)
    stats = dict(conv.stats)
    stats.pop("unconverted", None)
    if stats.get("pdf_drawings", 0) == 0 and stats.get("pdf_text_spans", 0) == 0:
        warnings.append("该页未提取到任何矢量图元或文本（可能为扫描页），输出为空 DXF")
    del conv
    gc.collect()
    return BackendResult(dxf_path=dxf_path, engine="vector",
                         warnings=warnings, stats=stats)


# --------------------------------------------------------------------------- #
# 后端 2：经典视觉管线（OpenCV 骨架矢量化）
# --------------------------------------------------------------------------- #
def convert_cv(pdf_path: str, page_num: int, out_dir: str,
               dpi: int = 200) -> BackendResult:
    """渲染页为位图 -> RasterToDXFConverter 骨架矢量化（近似重建）。"""
    warnings: list[str] = ["光栅矢量化是近似重建，精度受分辨率影响，非坐标级无损"]
    do_ocr = _ocr_available()
    if not do_ocr:
        warnings.append("OCR 不可用（pytesseract/tesseract 缺失），跳过文本识别")
    dpi = max(50, min(int(dpi), MAX_RENDER_DPI))
    img = _render_page_bgr(pdf_path, page_num, dpi)
    try:
        out_path = _out_dxf_path(pdf_path, page_num, out_dir)
        conv = RasterToDXFConverter(dpi=dpi, do_ocr=do_ocr)
        dxf_path = conv.convert_image(img, out_path)
        stats = dict(conv.stats)
    finally:
        del img
        gc.collect()
    return BackendResult(dxf_path=dxf_path, engine="cv",
                         warnings=warnings, stats=stats)


# --------------------------------------------------------------------------- #
# 后端 3：CV+ 增强视觉管线（LSD 虚线重建 + 文字掩膜 + 弧拟合 + 可选 vtracer）
# --------------------------------------------------------------------------- #
def convert_cvplus(pdf_path: str, page_num: int, out_dir: str,
                   dpi: int = 200, ocr_engine: str = "auto") -> BackendResult:
    """渲染页为位图 -> RasterPlusConverter 增强视觉管线（近似重建）。

    相对经典 cv 管线：文字掩膜防乱线、LSD+间隙合并重建虚线、
    长折线最小二乘拟合改写 ARC。OCR 与虚线重建均由管线内部完成。
    ocr_engine 透传 RasterPlusConverter（默认 'auto'，行为与旧版一致）。
    """
    from raster_plus import RasterPlusConverter  # lazy import：与 raster2dxf 同目录
    warnings: list[str] = ["光栅矢量化是近似重建，精度受分辨率影响，非坐标级无损"]
    do_ocr = _ocr_available()
    if not do_ocr:
        warnings.append("OCR 不可用（pytesseract/tesseract 缺失），跳过文本识别与文字掩膜")
    dpi = max(50, min(int(dpi), MAX_RENDER_DPI))
    img = _render_page_bgr(pdf_path, page_num, dpi)
    try:
        out_path = _out_dxf_path(pdf_path, page_num, out_dir)
        conv = RasterPlusConverter(dpi=dpi, do_ocr=do_ocr, ocr_engine=ocr_engine)
        dxf_path = conv.convert_image(img, out_path)
        stats = dict(conv.stats)
        # 管线内部 warning（tesseract/vtracer 降级等）并入后端 warnings
        warnings.extend(stats.pop("warnings", []) or [])
    finally:
        del img
        gc.collect()
    return BackendResult(dxf_path=dxf_path, engine="cv+",
                         warnings=warnings, stats=stats)


# --------------------------------------------------------------------------- #
# 后端 4：YOLO 图纸元素检测 + CV+ 矢量化（可选依赖，优雅降级）
# --------------------------------------------------------------------------- #
# YOLO 类别名 -> 中文标签映射（类别名从模型 results.names 读取，未知名称原样保留）
_YOLO_CLASS_ZH = {
    "frame": "图框", "border": "图框", "drawing_frame": "图框",
    "title_block": "标题栏", "titleblock": "标题栏", "title": "标题栏",
    "table": "表格", "symbol": "符号", "legend": "图例",
    "text": "文字块", "text_block": "文字块", "label": "标注",
    "dimension": "尺寸标注", "drawing": "图形区", "image": "插图",
}


def _zh_label(name: str) -> str:
    """把模型类别名映射为中文标签（先精确后子串匹配，未知原名返回）。"""
    low = str(name).strip().lower()
    if low in _YOLO_CLASS_ZH:
        return _YOLO_CLASS_ZH[low]
    for key, zh in _YOLO_CLASS_ZH.items():
        if key in low:
            return zh
    return str(name)


def _yolo_detect_boxes(img_bgr: np.ndarray, weights: str):
    """用 ultralytics YOLO 检测图纸元素区域。

    返回 [(x0, y0, x1, y1, 类别名, 置信度)]，类别名取自模型 results.names。
    仅在 ultralytics 可 import 且权重文件存在时才会被调用；
    任何失败都抛给上层 convert_yolo 统一回退。
    """
    from ultralytics import YOLO  # lazy import：重依赖，缺失时上层捕获
    model = YOLO(weights)
    results = model.predict(img_bgr, verbose=False)
    boxes = []
    # 类别名优先级：单帧 results.names -> model.names（不同版本 ultralytics 差异）
    for r in results:
        names = getattr(r, "names", None) or getattr(model, "names", {}) or {}
        for b in getattr(r, "boxes", []) or []:
            xyxy = b.xyxy[0].tolist()
            cls_id = int(b.cls[0]) if b.cls is not None else -1
            conf = float(b.conf[0]) if getattr(b, "conf", None) is not None else 0.0
            boxes.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3],
                          str(names.get(cls_id, cls_id)), conf))
    return boxes


def _pick_title_block_bbox(boxes) -> Optional[tuple]:
    """从检测框中挑出标题栏框（类别名含 title，取面积最大者）。"""
    cands = [b for b in boxes if "title" in str(b[4]).lower()]
    if not cands:
        return None
    best = max(cands, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    return (float(best[0]), float(best[1]), float(best[2]), float(best[3]))


def convert_yolo(pdf_path: str, page_num: int, out_dir: str,
                 dpi: int = 200, weights: Optional[str] = None,
                 ocr_engine: str = "auto",
                 auto_setup: bool = False) -> BackendResult:
    """YOLO 检测图纸元素（图框/标题栏/表格/符号），检测框写入 DXF 的
    DETECTION 图层（LWPOLYLINE 矩形框 + 中文类别 TEXT 标注 + 置信度），
    框内墨迹抠除后，框外区域走 CV+ 增强管线（RasterPlusConverter）矢量化，
    两者合并进同一 DXF。标题栏检测框存入 stats['title_block_bbox']。

    auto_setup=True 时：ultralytics 缺失会尝试 pip 自动安装；
    weights 留空时自动下载通用预训练权重（DEFAULT_YOLO_WEIGHTS）。
    组件仍不可用时：warnings 追加中文说明并回退 convert_cv
    （engine='yolo->cv-fallback'）。本函数绝不抛异常。
    """
    fallback_reason: Optional[str] = None
    pre_warnings: list[str] = []
    has_ultra, ultra_note = _ensure_ultralytics(auto_setup)
    if not has_ultra:
        fallback_reason = ultra_note
    else:
        if ultra_note:  # 自动安装成功的提示
            pre_warnings.append(ultra_note)
        resolved, w_note = _resolve_weights(weights)
        if resolved is None:
            fallback_reason = w_note
        else:
            weights = resolved
            if w_note:
                pre_warnings.append(w_note)

    if fallback_reason is not None:
        res = convert_cv(pdf_path, page_num, out_dir, dpi=dpi)
        res["warnings"].insert(0, f"YOLO 后端不可用：{fallback_reason}，已回退到经典视觉管线")
        res["engine"] = "yolo->cv-fallback"
        return res

    # ---- YOLO 可用路径：检测 -> 框内墨迹抠除 -> 框外 CV+ 矢量化
    #      -> 检测框写入 DETECTION 图层（合并进同一 DXF）----
    try:
        from ezdxf import recover as _ezrecover  # ezdxf 1.4 需显式导入子模块
        from raster_plus import RasterPlusConverter  # lazy import：同目录

        dpi = max(50, min(int(dpi), MAX_RENDER_DPI))
        img = _render_page_bgr(pdf_path, page_num, dpi)
        boxes = _yolo_detect_boxes(img, str(weights))
        # 框内区域置白（抠除检测框内墨迹），只对框外区域做 CV+ 矢量化
        masked = img.copy()
        for x0, y0, x1, y1, _name, _conf in boxes:
            cv2.rectangle(masked, (int(x0), int(y0)), (int(x1), int(y1)),
                          (255, 255, 255), thickness=-1)
        del img
        out_path = _out_dxf_path(pdf_path, page_num, out_dir)
        conv = RasterPlusConverter(dpi=dpi, do_ocr=_ocr_available(),
                                   ocr_engine=ocr_engine)
        dxf_path = conv.convert_image(masked, out_path)
        stats = dict(conv.stats)
        warnings: list[str] = ["光栅矢量化是近似重建；检测框位于 DETECTION 图层，"
                               "框内区域未矢量化"]
        warnings.extend(pre_warnings)  # 自动安装/自动下载等提示
        # 管线内部 warning（tesseract/vtracer 降级等）并入后端 warnings
        warnings.extend(stats.pop("warnings", []) or [])
        del masked
        gc.collect()
        # 把检测框追加到 DXF 的 DETECTION 图层（像素 -> pt 换算 + Y 翻转）
        doc, _auditor = _ezrecover.readfile(dxf_path)  # 返回 (Drawing, auditor)
        if "DETECTION" not in doc.layers:
            doc.layers.add("DETECTION", color=1)  # 红色
        msp = doc.modelspace()
        # 页高（像素）用于 Y 翻转
        src = pymupdf.open(pdf_path)
        h_px = int(round(src[page_num].rect.height * dpi / 72.0))
        src.close()
        s = 72.0 / dpi
        for x0, y0, x1, y1, name, conf in boxes:
            pts = [(x0 * s, (h_px - y0) * s), (x1 * s, (h_px - y0) * s),
                   (x1 * s, (h_px - y1) * s), (x0 * s, (h_px - y1) * s)]
            msp.add_lwpolyline(pts, close=True,
                               dxfattribs={"layer": "DETECTION"})
            label = f"{_zh_label(name)} {conf:.2f}"  # 中文类别 + 置信度
            t = msp.add_text(label, height=max(6.0, 10 * s),
                             dxfattribs={"layer": "DETECTION"})
            t.set_placement((x0 * s, (h_px - y0) * s))
        doc.saveas(dxf_path)
        stats["DETECTION"] = len(boxes)
        # 标题栏检测框供 VLM 结构化抽取（任务 3，pipeline 层消费）
        tb = _pick_title_block_bbox(boxes)
        if tb is not None:
            stats["title_block_bbox"] = tb
        return BackendResult(dxf_path=dxf_path, engine="yolo",
                             warnings=warnings, stats=stats)
    except Exception as exc:  # 绝不抛异常：任何意外都回退 cv
        logger.warning("YOLO 管线执行失败，回退 cv: %s", exc)
        res = convert_cv(pdf_path, page_num, out_dir, dpi=dpi)
        res["warnings"].insert(0, f"YOLO 管线执行失败（{exc}），已回退到经典视觉管线")
        res["engine"] = "yolo->cv-fallback"
        return res


# --------------------------------------------------------------------------- #
# 后端 5：自动路由（逐页判定）
# --------------------------------------------------------------------------- #
def convert_auto(pdf_path: str, page_num: int, out_dir: str,
                 ocr_engine: str = "auto",
                 auto_setup: bool = False) -> BackendResult:
    """detect.page_kind：'vector' -> convert_vector；'scan' -> yolo/cv+。

    扫描页路由：ultralytics 可 import（或 auto_setup 开启允许自动安装）时走
    YOLO 视觉管线（框外区域用 CV+ 矢量化）；否则由 CV+ 增强视觉管线兜底。
    ocr_engine 透传到扫描页管线（默认 'auto'，行为与旧版一致）。
    """
    doc = pymupdf.open(pdf_path)
    try:
        kind = detect.page_kind(doc[page_num])
    finally:
        doc.close()
    if kind == "scan":
        has_ultra, _ = _ensure_ultralytics(False)  # 只探测，路由处不触发安装
        if has_ultra or auto_setup:
            res = convert_yolo(pdf_path, page_num, out_dir,
                               ocr_engine=ocr_engine, auto_setup=auto_setup)
            res["warnings"].insert(0, "自动判定：该页为扫描页，路由到 YOLO 视觉管线")
            return res
        res = convert_cvplus(pdf_path, page_num, out_dir, ocr_engine=ocr_engine)
        res["warnings"].insert(
            0, "自动判定：该页为扫描页，YOLO 不可用，路由到 CV+ 增强视觉管线兜底")
        return res
    res = convert_vector(pdf_path, page_num, out_dir)
    return res
