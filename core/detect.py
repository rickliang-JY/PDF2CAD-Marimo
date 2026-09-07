"""detect.py —— 逐页类型判定（矢量页 / 扫描页）。

判定规则（与 SPEC 契约一致）：
    矢量 drawings > 10 或有文本  -> 'vector'
    否则若存在覆盖页面面积 > 50% 的栅格图 -> 'scan'
    默认 -> 'vector'
"""
from __future__ import annotations

import gc
from typing import Any, Dict, List

import pymupdf

# 矢量图元数阈值：超过该数直接判矢量页
DRAWING_THRESHOLD = 10
# 单张栅格图覆盖页面面积的比例阈值：超过则判扫描页
IMAGE_COVERAGE_THRESHOLD = 0.5


def _image_coverage(page: pymupdf.Page) -> float:
    """页面上最大单张栅格图的面积覆盖率（0~1，可因重叠超过 1，截断到 1）。"""
    page_area = abs(page.rect)
    if page_area <= 0:
        return 0.0
    total = 0.0
    for img in page.get_images(full=True):
        try:
            rects = page.get_image_rects(img[0])
        except Exception:
            # 某些损坏的 xref 取不到矩形，跳过该图
            continue
        total += sum(abs(r) for r in rects)
    return min(1.0, total / page_area)


def _page_stats(page: pymupdf.Page) -> Dict[str, Any]:
    """统计单页的图元/文本/图像覆盖率（供 page_kind 与 analyze_pdf 复用）。"""
    try:
        n_drawings = len(page.get_drawings())
    except Exception:
        n_drawings = 0
    n_texts = 0
    try:
        tdict = page.get_text("dict")
        for block in tdict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    # 只计非空白文本
                    if str(span.get("text", "")).strip():
                        n_texts += 1
    except Exception:
        n_texts = 0
    coverage = _image_coverage(page)
    return {
        "n_drawings": n_drawings,
        "n_texts": n_texts,
        "image_coverage": coverage,
        "width_pt": float(page.rect.width),
        "height_pt": float(page.rect.height),
    }


def page_kind(page: pymupdf.Page) -> str:
    """返回 'vector' | 'scan'。

    规则：矢量 drawings>10 或有文本 -> 'vector'；
    否则若有覆盖页面面积>50% 的栅格图 -> 'scan'；默认 'vector'。
    """
    st = _page_stats(page)
    if st["n_drawings"] > DRAWING_THRESHOLD or st["n_texts"] > 0:
        return "vector"
    if st["image_coverage"] > IMAGE_COVERAGE_THRESHOLD:
        return "scan"
    return "vector"


def analyze_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """逐页分析 PDF，每页返回统计字典。

    返回: [{'page': int, 'kind': str, 'n_drawings': int, 'n_texts': int,
            'image_coverage': float, 'width_pt': float, 'height_pt': float}, ...]
    """
    doc = pymupdf.open(pdf_path)
    out: List[Dict[str, Any]] = []
    try:
        for i in range(doc.page_count):
            page = doc[i]
            st = _page_stats(page)
            kind = ("vector" if (st["n_drawings"] > DRAWING_THRESHOLD
                                 or st["n_texts"] > 0)
                    else "scan" if st["image_coverage"] > IMAGE_COVERAGE_THRESHOLD
                    else "vector")
            out.append({"page": i, "kind": kind, **st})
    finally:
        doc.close()
    gc.collect()
    return out
