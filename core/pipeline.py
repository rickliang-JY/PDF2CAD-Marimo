"""pipeline.py —— 多页编排 + 合并输出。

convert_pdf：按模型逐页转换 -> 每页一个 DXF（{stem}_pNNN.dxf）
-> 汇总 conversion.log（引擎/实体数/警告）-> 全部打入 zip。

单页失败不中断整本：该页记 warnings 并继续（dxf_path 为空字符串）。
"""
from __future__ import annotations

import gc
import logging
import os
import time
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pymupdf

try:
    from core import backends, detect
except ImportError:  # 直接以 core/ 为顶层目录运行时
    import backends
    import detect

logger = logging.getLogger("pdf2cad.pipeline")

# 模型 -> 后端函数 分发表
_BACKENDS: Dict[str, Callable] = {
    "auto": backends.convert_auto,
    "vector": backends.convert_vector,
    "cv": backends.convert_cv,
    "cv+": backends.convert_cvplus,
    "yolo": backends.convert_yolo,
}

# 统计中视为“实体计数”的键（其余为过程指标）
_ENTITY_KEYS = ("LINE", "LWPOLYLINE", "CIRCLE", "ELLIPSE", "ARC",
                "SPLINE", "HATCH", "TEXT", "DETECTION", "DASHED", "VTRACER")


def _entity_total(stats: dict) -> int:
    """实体总数：只统计几何/文本实体键，忽略过程指标。"""
    return int(sum(v for k, v in stats.items()
                   if k in _ENTITY_KEYS and isinstance(v, (int, float))))


def _try_extract_title(pdf_path: str, page_num: int, res: dict) -> Optional[dict]:
    """VLM 标题栏结构化抽取（lazy 探测，任何失败返回 None，绝不抛异常）。

    区域：YOLO 检测框 stats['title_block_bbox']（像素坐标，渲染 dpi 与
    后端一致 ≤200）优先，否则右下角启发式裁剪（core/vlm.py 内部处理）。
    """
    try:
        from core import vlm
    except ImportError:  # 直接以 core/ 为顶层目录运行时
        import vlm
    if not vlm.vlm_available():
        return None
    try:
        bbox = (res.get("stats") or {}).get("title_block_bbox")
        # 渲染 dpi 与后端一致（后端内部硬上限 200），保证像素坐标系对齐
        img = backends._render_page_bgr(pdf_path, page_num,
                                        backends.MAX_RENDER_DPI)
        try:
            return vlm.extract_title_block(img, title_bbox=bbox)
        finally:
            del img
            gc.collect()
    except Exception as exc:
        logger.warning("第 %d 页标题栏抽取失败（%s），已跳过", page_num + 1, exc)
        return None


def convert_pdf(pdf_path: str, model: str, out_dir: str,
                pages: Optional[List[int]] = None,
                progress_cb: Optional[Callable[[int, int, str], None]] = None,
                ocr_engine: str = "auto",
                yolo_weights: Optional[str] = None,
                extract_title: bool = False,
                auto_setup: bool = False) -> dict:
    """整本（或指定页集）转换入口。

    参数:
        pdf_path:   输入 PDF 路径
        model:      模型键，∈ MODELS.keys()（auto/vector/cv/yolo）
        out_dir:    输出目录（DXF / conversion.log / zip 均放其下）
        pages:      0 起页码列表；None 表示全部页
        progress_cb: 进度回调 progress_cb(i, n, msg)
        ocr_engine: OCR 引擎（'auto'/'tesseract'/'paddle'），透传 cv+/yolo/auto
        yolo_weights: YOLO 权重路径（仅 model='yolo' 时使用；留空且
            auto_setup 开启时自动下载通用预训练权重）
        extract_title: 开启后每页转换后尝试 VLM 标题栏结构化抽取，
            结果存该页 stats['title_block']，并随 zip 附带 title_block.json
            （VLM 组件不可用时静默跳过，不抛异常）
        auto_setup: 开启后 yolo/auto 路由允许自动 pip 安装缺失的
            ultralytics 并自动下载预训练权重（需联网，首次较慢）

    返回:
        {'pages': [BackendResult...], 'page_kinds': [...], 'zip_path': str,
         'log_path': str, 'elapsed_sec': float}
    """
    t0 = time.perf_counter()
    if model not in backends.MODELS:
        raise ValueError(f"未知模型 {model!r}，可选: {list(backends.MODELS)}")
    os.makedirs(out_dir, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    try:
        n_total = doc.page_count
    finally:
        doc.close()
    page_list = list(range(n_total)) if pages is None else sorted(set(pages))
    # 过滤越界页码，避免单页 IndexError 中断整本
    page_list = [p for p in page_list if 0 <= p < n_total]

    page_kinds = detect.analyze_pdf(pdf_path)
    convert_fn = _BACKENDS[model]

    # 按后端能力组装逐页调用参数（vector/cv 无 OCR/YOLO 参数，保持原签名调用）
    def _convert_page(p: int) -> dict:
        if model == "yolo":
            return convert_fn(pdf_path, p, out_dir, weights=yolo_weights,
                              ocr_engine=ocr_engine, auto_setup=auto_setup)
        if model == "auto":
            return convert_fn(pdf_path, p, out_dir, ocr_engine=ocr_engine,
                              auto_setup=auto_setup)
        if model == "cv+":
            return convert_fn(pdf_path, p, out_dir, ocr_engine=ocr_engine)
        return convert_fn(pdf_path, p, out_dir)

    results: List[dict] = []
    title_blocks: Dict[int, dict] = {}  # 页码 -> VLM 标题栏结构化结果
    log_lines: List[str] = [
        f"# PDF2CAD 转换日志",
        f"输入: {pdf_path}",
        f"模型: {model}（{backends.MODELS[model]}）",
        f"总页数: {n_total}，本次转换页: {[p + 1 for p in page_list]}",
        "",
    ]
    n = len(page_list)
    for i, p in enumerate(page_list):
        kind = page_kinds[p]["kind"] if p < len(page_kinds) else "?"
        if progress_cb:
            progress_cb(i, n, f"正在转换第 {p + 1}/{n_total} 页（判定: {kind}）…")
        try:
            res = _convert_page(p)
        except Exception as exc:
            # 单页失败不中断整本：记录警告并继续
            logger.exception("第 %d 页转换失败", p + 1)
            res = {"dxf_path": "", "engine": f"{model}->error",
                   "warnings": [f"第 {p + 1} 页转换失败：{exc}"], "stats": {}}
        res = dict(res)
        # ---- 可选：VLM 标题栏结构化抽取（GPU 组件，不可用时静默跳过）----
        if extract_title:
            info = _try_extract_title(pdf_path, p, res)
            if info:
                res.setdefault("stats", {})["title_block"] = info
                title_blocks[p] = info
                log_lines.append(f"    标题栏: {info}")
        res["page"] = p
        res["kind"] = kind
        results.append(res)
        ent = _entity_total(res.get("stats", {}))
        log_lines.append(
            f"第 {p + 1} 页 | 判定: {kind} | 引擎: {res['engine']} | "
            f"实体数: {ent} | 文件: {os.path.basename(res['dxf_path']) or '(无)'}")
        for w in res.get("warnings", []):
            log_lines.append(f"    警告: {w}")
        del res
        gc.collect()
        if progress_cb:
            progress_cb(i + 1, n, f"第 {p + 1} 页完成（引擎: {results[-1]['engine']}）")

    # VLM 标题栏结构化结果汇总（开启 extract_title 且有产出时写出）
    tb_json_path = None
    if title_blocks:
        import json
        tb_json_path = str(Path(out_dir) / "title_block.json")
        with open(tb_json_path, "w", encoding="utf-8") as f:
            json.dump({f"page_{p + 1}": info for p, info in
                       sorted(title_blocks.items())},
                      f, ensure_ascii=False, indent=2)
        log_lines.append(f"标题栏结构化结果: title_block.json"
                         f"（{len(title_blocks)} 页）")

    # conversion.log 汇总
    elapsed = time.perf_counter() - t0
    log_lines.append("")
    log_lines.append(f"总耗时: {elapsed:.2f} s")
    log_path = str(Path(out_dir) / "conversion.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    # 全部 DXF + log 打入 zip
    zip_path = str(Path(out_dir) / f"{Path(pdf_path).stem}_dxf_bundle.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for res in results:
            p_dxf = res.get("dxf_path", "")
            if p_dxf and os.path.isfile(p_dxf):
                zf.write(p_dxf, arcname=os.path.basename(p_dxf))
        zf.write(log_path, arcname="conversion.log")
        if tb_json_path and os.path.isfile(tb_json_path):
            zf.write(tb_json_path, arcname="title_block.json")

    gc.collect()
    return {
        "pages": results,
        "page_kinds": page_kinds,
        "zip_path": zip_path,
        "log_path": log_path,
        "elapsed_sec": elapsed,
    }
