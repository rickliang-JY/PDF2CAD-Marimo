#!/usr/bin/env python3
"""test_e2e.py —— 无头端到端测试（直接 python3 运行，非零退出码表示失败）。

验收项（对应 SPEC「测试」节）：
1. 程序化生成测试 PDF：page0 矢量图纸、page1 模拟扫描（渲染 page0 再嵌回）
2. 五个 model 各跑 convert_pdf：DXF 可 recover.readfile 打开且 audit 无错；
   auto 模式 page0->vector、page1->cv+（YOLO 不可用时 CV+ 兜底，warning 非空）；
   vector 模式 page0 实体数 > 50；cv 模式 page1 有 LWPOLYLINE
3. compare_page(page0)：iou ∈ (0,1] 且三张 PNG 非空
4. `marimo check app.py` 通过（subprocess）
5. CV+ 增强管线专项：虚线矩形 + 旋转文字 + 圆的图纸页走 RasterPlusConverter
   ——DASHED 图层存在（虚线被重建）、TEXT ≥1 且文字框邻域无折线顶点、圆/弧 ≥1
6. GPU 可选组件降级：ocr_engines（tesseract 可用/paddle 降级）、VLM 返回 None、
   convert_pdf(extract_title=True) 不崩、convert_yolo 权重缺失回退 cv-fallback、
   RasterPlusConverter(ocr_engine='auto') 回归（虚线/文字/圆弧断言同 5）
7. 打印 PASS/FAIL 汇总
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import traceback

# 项目根目录入 sys.path（本文件在 tests/ 下，直接运行时 sys.path[0] 是 tests/）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2
import numpy as np
import pymupdf
import ezdxf
from ezdxf import recover as ezrecover  # ezdxf 1.4 需显式导入 recover 子模块

from core.pipeline import convert_pdf
from core.compare import compare_page
from core.backends import MODELS

ENTITY_KEYS = ("LINE", "LWPOLYLINE", "CIRCLE", "ELLIPSE", "ARC",
               "SPLINE", "HATCH", "TEXT", "DETECTION", "DASHED", "VTRACER")

RESULTS = []  # (名称, 是否通过, 说明)


def check(name: str, ok: bool, note: str = "") -> None:
    RESULTS.append((name, bool(ok), note))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {note}")


def entity_total(stats: dict) -> int:
    return int(sum(v for k, v in stats.items()
                   if k in ENTITY_KEYS and isinstance(v, (int, float))))


def make_test_pdf(path: str) -> None:
    """生成两页测试 PDF。

    page0：矢量图纸（网格线/矩形/圆/贝塞尔/旋转文字/多颜色）
    page1：模拟扫描（把 page0 渲染成 PNG 再整页 insert_image 嵌入）
    """
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # A4 纵向
    margin = 40
    w, h = 595, 842
    # 1) 网格线：30 竖 + 30 横（保证实体数远超 50）
    for i in range(30):
        x = margin + i * (w - 2 * margin) / 29
        page.draw_line((x, margin), (x, h - margin),
                       color=(0.2, 0.2, 0.6), width=0.5)
    for j in range(30):
        y = margin + j * (h - 2 * margin) / 29
        page.draw_line((margin, y), (w - margin, y),
                       color=(0.2, 0.2, 0.6), width=0.5)
    # 2) 矩形（外框 + 标题栏）
    page.draw_rect(pymupdf.Rect(20, 20, w - 20, h - 20),
                   color=(0, 0, 0), width=2.0)
    page.draw_rect(pymupdf.Rect(20, h - 120, w - 20, h - 20),
                   color=(0, 0, 0), width=1.0)
    # 3) 圆与圆弧（多颜色）
    page.draw_circle((150, 200), 60, color=(0.8, 0, 0), width=1.5)
    page.draw_circle((450, 200), 40, color=(0, 0.6, 0), width=1.0)
    page.draw_circle((300, 500), 90, color=(0, 0, 0.8), width=1.2)
    # 4) 贝塞尔曲线
    page.draw_bezier((100, 650), (200, 550), (350, 750), (480, 620),
                     color=(0.6, 0, 0.6), width=1.5)
    # 5) 文字（水平 + 旋转 90°）
    page.insert_text((60, 100), "PDF2CAD VECTOR TEST SHEET",
                     fontsize=16, color=(0, 0, 0))
    page.insert_text((60, h - 60), "TITLE BLOCK - DWG NO. 001",
                     fontsize=12, color=(0, 0, 0))
    page.insert_text(pymupdf.Point(w - 45, h - 200), "ROTATED TEXT 90",
                     fontsize=10, rotate=90, color=(0.3, 0.3, 0.3))

    # page1：把 page0 渲染成位图再整页嵌回（模拟扫描件）
    pix = page.get_pixmap(dpi=100)
    png_bytes = pix.tobytes("png")
    scan_page = doc.new_page(width=w, height=h)
    scan_page.insert_image(pymupdf.Rect(0, 0, w, h), stream=png_bytes)
    doc.save(path)
    doc.close()


def run_cvplus_checks(tmp: str, ocr_engine: str | None = None,
                      tag: str = "CV+ 专项") -> None:
    """CV+ 专项：虚线矩形 + 旋转文字 + 圆的图纸页走 RasterPlusConverter。

    稳定断言口径：
    - DASHED 图层存在实体（虚线被间隙容忍合并重建）
    - TEXT 实体数 ≥ 1（旋转文字经多方向 OCR 识别）
    - 文字框 8px 邻域内无 LWPOLYLINE 顶点（去字后骨架不再产生乱线）
    - CIRCLE + ARC 实体 ≥ 1

    ocr_engine 非 None 时显式传入 RasterPlusConverter（GPU OCR 抽象层回归）。
    """
    from core.raster_plus import RasterPlusConverter, detect_text_items

    dpi = 200
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # A4 纵向
    # 虚线矩形：dashes="[10 2.5]" → 200dpi 下 dash≈28px、间隙≈7px
    page.draw_rect(pymupdf.Rect(80, 80, 515, 400), color=(0, 0, 0),
                   width=1.2, dashes="[10 2.5] 0")
    # 旋转文字（90° 竖排）
    page.insert_text(pymupdf.Point(120, 620), "ROTATED TEXT 90",
                     fontsize=16, rotate=90, color=(0, 0, 0))
    # 圆
    page.draw_circle((350, 680), 70, color=(0, 0, 0), width=1.2)
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    doc.close()

    dxf_path = os.path.join(tmp, f"cvplus_direct_{tag}.dxf")
    try:
        kwargs = dict(dpi=dpi, do_ocr=True, gap_px=12.0, min_seg_len_px=6.0)
        if ocr_engine is not None:
            kwargs["ocr_engine"] = ocr_engine
        conv = RasterPlusConverter(**kwargs)
        conv.convert_image(img, dxf_path)
        check(f"{tag} 转换运行", True, f"stats={ {k: v for k, v in conv.stats.items() if k != 'warnings'} }")
    except Exception as exc:
        check(f"{tag} 转换运行", False, repr(exc))
        traceback.print_exc()
        return

    try:
        ddoc, auditor = ezrecover.readfile(dxf_path)
        check(f"{tag} DXF recover+audit 无错", len(auditor.errors) == 0,
              f"errors={len(auditor.errors)}")
        msp = ddoc.modelspace()

        # 1) DASHED 图层存在实体（虚线被重建为合并长线）
        n_dashed = len(msp.query('LINE[layer=="DASHED"]'))
        lens = []
        for e in msp.query('LINE[layer=="DASHED"]'):
            lens.append(abs(e.dxf.end - e.dxf.start))
        check(f"{tag} DASHED 图层有实体（虚线重建）", n_dashed >= 1,
              f"DASHED 实体数={n_dashed}, 最长={max(lens) if lens else 0:.1f}pt")

        # 2) TEXT 实体数 ≥ 1
        n_text = len(msp.query("TEXT"))
        check(f"{tag} TEXT 实体数 ≥ 1", n_text >= 1, f"TEXT 数={n_text}")

        # 3) 文字框 8px 邻域内无 LWPOLYLINE 顶点（去字后无乱线）
        h_px = img.shape[0]
        text_items, _ = detect_text_items(img, h_px)
        s = 72.0 / dpi
        tol = 8.0 * s  # 8px 换算到 DXF 单位
        # 文字框 -> DXF 坐标区间（Y 翻转）
        boxes_dxf = []
        for it in text_items:
            x, y, bw, bh = it["box"]
            boxes_dxf.append((x * s, (h_px - y - bh) * s,
                              (x + bw) * s, (h_px - y) * s))
        bad = 0
        for e in msp.query("LWPOLYLINE"):
            for pt in e.get_points("xy"):
                px, py = pt
                for x0, y0, x1, y1 in boxes_dxf:
                    if (x0 - tol <= px <= x1 + tol
                            and y0 - tol <= py <= y1 + tol):
                        bad += 1
                        break
                if bad:
                    break
            if bad:
                break
        check(f"{tag} 文字框 8px 邻域内无折线顶点", bad == 0,
              f"文字框数={len(boxes_dxf)}, 违例顶点={bad}")

        # 4) 圆/弧实体 ≥ 1
        n_ca = len(msp.query("CIRCLE")) + len(msp.query("ARC"))
        check(f"{tag} 圆/弧实体 ≥ 1", n_ca >= 1, f"CIRCLE+ARC={n_ca}")
    except Exception as exc:
        check(f"{tag} DXF 校验", False, repr(exc))
        traceback.print_exc()


def run_gpu_component_checks(tmp: str, pdf_path: str) -> None:
    """GPU 可选组件降级路径验证（本环境无 GPU 组件，验证优雅降级不崩）。

    覆盖：ocr_engines 抽象层 / paddle 降级 / VLM 抽取返回 None /
    convert_pdf(extract_title=True) / convert_yolo 权重缺失降级 /
    title_block_to_dxf_text 写入。
    """
    from core import ocr_engines, vlm
    from core.backends import convert_yolo

    # 测试图：水平 + 90° 旋转文字（dpi 150 足够 tesseract 检出）
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((60, 100), "PDF2CAD OCR ENGINE TEST", fontsize=16,
                     color=(0, 0, 0))
    page.insert_text(pymupdf.Point(120, 620), "ROTATED TEXT 90",
                     fontsize=16, rotate=90, color=(0, 0, 0))
    pix = page.get_pixmap(dpi=150)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    doc.close()

    # 1) available_engines() 至少含 tesseract
    try:
        avail = ocr_engines.available_engines()
        check("OCR 引擎 available_engines 含 tesseract", "tesseract" in avail,
              f"可用引擎={list(avail)}")
    except Exception as exc:
        check("OCR 引擎 available_engines 含 tesseract", False, repr(exc))

    # 2) ocr_extract(engine='auto') 返回非空且字段齐全
    try:
        items = ocr_engines.ocr_extract(img, engine="auto")
        keys_ok = all({"text", "bbox", "angle", "conf"} <= set(it)
                      for it in items) if items else False
        check("OCR 引擎 auto 抽取非空且字段齐全", bool(items) and keys_ok,
              f"条数={len(items)}, 角度={sorted({it['angle'] for it in items})}")
    except Exception as exc:
        check("OCR 引擎 auto 抽取非空且字段齐全", False, repr(exc))

    # 3) ocr_extract(engine='paddle') 无 paddle 环境降级 tesseract 且不抛异常
    try:
        items_p = ocr_engines.ocr_extract(img, engine="paddle")
        if ocr_engines.paddle_available():
            note = f"paddle 真实可用，条数={len(items_p)}"
            ok = isinstance(items_p, list)
        else:
            note = f"paddle 不可用，降级后条数={len(items_p)}"
            ok = isinstance(items_p, list) and len(items_p) > 0
        check("OCR 引擎 paddle 降级不抛异常", ok, note)
    except Exception as exc:
        check("OCR 引擎 paddle 降级不抛异常", False, repr(exc))

    # 4) extract_title_block 本环境返回 None 且不抛异常
    try:
        info = vlm.extract_title_block(img)
        if vlm.vlm_available():
            check("VLM extract_title_block 不抛异常", True,
                  f"组件可用，返回 {type(info).__name__}")
        else:
            check("VLM extract_title_block 不抛异常", info is None,
                  f"组件不可用，返回 {info!r}")
    except Exception as exc:
        check("VLM extract_title_block 不抛异常", False, repr(exc))

    # 5) convert_pdf(extract_title=True) 全程不崩
    try:
        out_et = os.path.join(tmp, "out_extract_title")
        res_et = convert_pdf(pdf_path, "vector", out_et, pages=[0],
                             extract_title=True)
        st = res_et["pages"][0].get("stats", {})
        tb_ok = ("title_block" not in st) or (st.get("title_block") is None) \
            or isinstance(st.get("title_block"), dict)
        check("convert_pdf(extract_title=True) 全程不崩",
              bool(res_et["pages"][0]["dxf_path"]) and tb_ok,
              f"engine={res_et['pages'][0]['engine']}, "
              f"title_block={'有' if st.get('title_block') else '无'}")
    except Exception as exc:
        check("convert_pdf(extract_title=True) 全程不崩", False, repr(exc))
        traceback.print_exc()

    # 6) convert_yolo 降级路径（无 ultralytics/权重缺失 → cv-fallback）
    try:
        out_y = os.path.join(tmp, "out_yolo_direct")
        ry = convert_yolo(pdf_path, 1, out_y, weights="/nonexistent.pt")
        try:
            import ultralytics  # noqa: F401
            has_ultra = True
        except Exception:
            has_ultra = False
        if has_ultra:
            ok = ry["engine"] in ("yolo", "yolo->cv-fallback")
        else:
            ok = (ry["engine"] == "yolo->cv-fallback"
                  and len(ry["warnings"]) > 0)
        ddoc, auditor = ezrecover.readfile(ry["dxf_path"])
        ok = ok and len(auditor.errors) == 0
        check("YOLO 后端降级路径（cv-fallback）", ok,
              f"engine={ry['engine']}, warnings={ry['warnings'][:1]}")
    except Exception as exc:
        check("YOLO 后端降级路径（cv-fallback）", False, repr(exc))
        traceback.print_exc()

    # 7) title_block_to_dxf_text 写 TITLE_INFO 图层
    try:
        doc_t = ezdxf.new()
        n = vlm.title_block_to_dxf_text(
            doc_t.modelspace(),
            {"图名": "测试图", "图号": "A-001", "比例": "1:100"}, (100.0, 200.0))
        check("VLM title_block_to_dxf_text 写入", n == 3,
              f"实体数={n}, 图层 TITLE_INFO={'TITLE_INFO' in doc_t.layers}")
    except Exception as exc:
        check("VLM title_block_to_dxf_text 写入", False, repr(exc))

    # 8) YOLO 可用路径 mock 冒烟（本环境无 ultralytics，用假模块验证完整逻辑：
    #    DETECTION 图层矩形框 + 中文类别 TEXT + 置信度 + title_block_bbox）
    fake_ok, note = False, ""
    try:
        import types

        class _FakeBox:
            def __init__(self, xyxy, cls_id, conf):
                self.xyxy = np.asarray([xyxy], dtype=np.float32)
                self.cls = np.asarray([cls_id])
                self.conf = np.asarray([conf])

        _NAMES = {0: "title_block", 1: "table"}

        class _FakeResult:
            def __init__(self, w, h):
                self.names = _NAMES
                self.boxes = [
                    _FakeBox((0.66 * w, 0.85 * h, 0.97 * w, 0.98 * h), 0, 0.93),
                    _FakeBox((0.10 * w, 0.10 * h, 0.45 * w, 0.30 * h), 1, 0.87),
                ]

        class _FakeYOLO:
            names = _NAMES

            def __init__(self, weights):
                pass

            def predict(self, img, verbose=False):
                h, w = img.shape[:2]
                return [_FakeResult(w, h)]

        _fake = types.ModuleType("ultralytics")
        _fake.YOLO = _FakeYOLO
        _prev = sys.modules.get("ultralytics")
        sys.modules["ultralytics"] = _fake
        try:
            wfile = os.path.join(tmp, "fake_yolo.pt")
            with open(wfile, "wb") as _f:
                _f.write(b"fake")
            out_y2 = os.path.join(tmp, "out_yolo_mock")
            ry2 = convert_yolo(pdf_path, 1, out_y2, dpi=100, weights=wfile)
            assert ry2["engine"] == "yolo", f"engine={ry2['engine']}"
            assert ry2["stats"].get("DETECTION") == 2, ry2["stats"]
            assert "title_block_bbox" in ry2["stats"], "缺 title_block_bbox"
            ddoc2, aud2 = ezrecover.readfile(ry2["dxf_path"])
            assert len(aud2.errors) == 0
            msp2 = ddoc2.modelspace()
            n_rect = len(msp2.query('LWPOLYLINE[layer=="DETECTION"]'))
            texts = [e.dxf.text for e in
                     msp2.query('TEXT[layer=="DETECTION"]')]
            assert n_rect == 2, f"DETECTION 矩形数={n_rect}"
            assert any("标题栏" in t and "0.93" in t for t in texts), texts
            assert any("表格" in t for t in texts), texts
            fake_ok, note = True, (f"DETECTION 矩形={n_rect}, 标注={texts}, "
                                   f"title_block_bbox={ry2['stats']['title_block_bbox']}")
        finally:
            if _prev is None:
                sys.modules.pop("ultralytics", None)
            else:
                sys.modules["ultralytics"] = _prev
    except Exception as exc:
        note = repr(exc)
        traceback.print_exc()
    check("YOLO 可用路径 mock 冒烟（DETECTION 图层+中文标注）", fake_ok, note)


def run_all() -> None:
    tmp = tempfile.mkdtemp(prefix="pdf2cad_e2e_")
    pdf_path = os.path.join(tmp, "e2e_test.pdf")
    make_test_pdf(pdf_path)
    print(f"测试 PDF: {pdf_path}")

    # ---- 验收 1/2：四个模型各跑一遍 convert_pdf ----
    outputs = {}
    for model in MODELS:
        out_dir = os.path.join(tmp, f"out_{model}")
        try:
            outputs[model] = convert_pdf(pdf_path, model, out_dir)
            n_ok = sum(1 for r in outputs[model]["pages"] if r["dxf_path"])
            check(f"convert_pdf[{model}] 运行", True,
                  f"{n_ok}/{len(outputs[model]['pages'])} 页产出 DXF")
        except Exception as exc:
            check(f"convert_pdf[{model}] 运行", False, repr(exc))
            traceback.print_exc()

    # ---- 验收 2a：每个 DXF 可 recover.readfile 打开且 audit 无错 ----
    all_dxf_ok, bad = True, []
    for model, out in outputs.items():
        for res in out["pages"]:
            p = res.get("dxf_path", "")
            if not p or not os.path.isfile(p):
                continue  # 失败页无 DXF，由该页 warnings 记录
            try:
                doc, auditor = ezrecover.readfile(p)
                if len(auditor.errors) != 0:
                    all_dxf_ok = False
                    bad.append(f"{os.path.basename(p)}: {len(auditor.errors)} 个 audit 错误")
            except Exception as exc:
                all_dxf_ok = False
                bad.append(f"{os.path.basename(p)}: 打开失败 {exc!r}")
    check("全部 DXF recover+audit 无错", all_dxf_ok, "; ".join(bad))

    # ---- 验收 2b：auto 模式逐页路由 ----
    if "auto" in outputs:
        pages = outputs["auto"]["pages"]
        p0_ok = pages[0]["engine"] == "vector"
        # 追加规格：scan 页路由到 cv+（本环境无 ultralytics，CV+ 兜底）
        p1_ok = (pages[1]["engine"] == "cv+"
                 and len(pages[1]["warnings"]) > 0)
        check("auto 路由: page0->vector", p0_ok, f"实际 engine={pages[0]['engine']}")
        check("auto 路由: page1->cv+（YOLO 不可用兜底, warning 非空）", p1_ok,
              f"实际 engine={pages[1]['engine']}, warnings={pages[1]['warnings'][:1]}")

    # ---- 验收 2c：vector 模式 page0 实体数 > 50 ----
    if "vector" in outputs:
        n_ent = entity_total(outputs["vector"]["pages"][0]["stats"])
        check("vector 模式 page0 实体数 > 50", n_ent > 50, f"实体数={n_ent}")

    # ---- 验收 2d：cv 模式 page1 有 LWPOLYLINE ----
    if "cv" in outputs:
        dxf_p1 = outputs["cv"]["pages"][1]["dxf_path"]
        n_lwp = 0
        try:
            doc, _ = ezrecover.readfile(dxf_p1)
            n_lwp = len(doc.modelspace().query("LWPOLYLINE"))
        except Exception as exc:
            check("cv 模式 page1 打开", False, repr(exc))
        check("cv 模式 page1 含 LWPOLYLINE", n_lwp > 0, f"LWPOLYLINE 数={n_lwp}")

    # ---- 验收 3：compare_page(page0) iou ∈ (0,1] 且三 PNG 非空 ----
    if "vector" in outputs:
        dxf_p0 = outputs["vector"]["pages"][0]["dxf_path"]
        try:
            cmp_res = compare_page(pdf_path, dxf_p0, 0, dpi=120)
            iou = cmp_res["iou"]
            pngs_ok = all(len(cmp_res[k]) > 100 for k in
                          ("pdf_png", "dxf_png", "overlay_png"))
            iou_ok = isinstance(iou, float) and 0.0 < iou <= 1.0
            check("compare_page iou ∈ (0,1]", iou_ok, f"iou={iou}")
            check("compare_page 三张 PNG 非空", pngs_ok,
                  f"字节数: {len(cmp_res['pdf_png'])}/{len(cmp_res['dxf_png'])}/{len(cmp_res['overlay_png'])}")
        except Exception as exc:
            check("compare_page 运行", False, repr(exc))
            traceback.print_exc()

    # ---- 验收 5：CV+ 增强管线专项（虚线矩形 + 旋转文字 + 圆）----
    run_cvplus_checks(tmp)

    # ---- 验收 5b：RasterPlusConverter(ocr_engine='auto') 回归
    #      （GPU OCR 抽象层接入后默认路径行为不变）----
    run_cvplus_checks(tmp, ocr_engine="auto", tag="OCR auto 回归")

    # ---- 验收 5c：GPU 可选组件降级路径（ocr_engines/vlm/yolo/extract_title）----
    run_gpu_component_checks(tmp, pdf_path)

    # ---- 验收 6：model="cv+" 显式跑通 + DXF recover+audit ----
    if "cv+" in outputs:
        n_ok = sum(1 for res in outputs["cv+"]["pages"]
                   if res.get("dxf_path") and os.path.isfile(res["dxf_path"]))
        check("cv+ 模式产出 DXF", n_ok == len(outputs["cv+"]["pages"]),
              f"{n_ok}/{len(outputs['cv+']['pages'])} 页")
        st = outputs["cv+"]["pages"][0]["stats"]
        check("cv+ 模式统计含 LINE/DASHED 键",
              "LINE" in st and "DASHED" in st, f"stats 键={sorted(st)[:8]}")

    # ---- 验收 7：marimo check app.py ----
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "marimo", "check",
             os.path.join(ROOT, "app.py")],
            capture_output=True, text=True, timeout=300)
        out = (proc.stdout + proc.stderr).strip()
        check("marimo check app.py", proc.returncode == 0,
              f"exit={proc.returncode} {out[-400:]}")
    except Exception as exc:
        check("marimo check app.py", False, repr(exc))

    # ---- 验收 7b：GPU 探测与自动下载/安装逻辑 ----
    from core import gpu_detect
    from core.backends import (DEFAULT_YOLO_WEIGHTS, _ensure_ultralytics,
                               _pip_install, _resolve_weights, convert_yolo)
    gi = gpu_detect.gpu_info()
    check("gpu_info 返回结构完整",
          isinstance(gi, dict) and {"available", "backend", "devices",
                                    "detail"} <= set(gi)
          and isinstance(gi["available"], bool) and gi["detail"],
          f"available={gi['available']} backend={gi['backend']}")

    w, note = _resolve_weights(None)
    check("权重留空 -> 自动下载通用预训练权重",
          w == DEFAULT_YOLO_WEIGHTS and note and "自动下载" in note,
          f"weights={w}")
    w2, note2 = _resolve_weights("/definitely/not/exist.pt")
    check("权重路径无效 -> 返回 None + 原因",
          w2 is None and note2 and "不存在" in note2)
    ok_u, note_u = _ensure_ultralytics(False)
    try:
        import ultralytics  # noqa: F401
        check("_ensure_ultralytics(False) 探测", ok_u is True)
    except ImportError:
        check("_ensure_ultralytics(False) 探测",
              ok_u is False and note_u and "自动下载" in note_u,
              note_u[:60] if note_u else "")
    check("_pip_install 假包返回 False 不抛异常",
          _pip_install("definitely-not-a-real-pkg-pdf2cad-xyz",
                       timeout=90) is False)
    # auto_setup=False 时 yolo 缺包仍走降级（不触发安装）
    ry3 = convert_yolo(pdf_path, 1, os.path.join(tmp, "out_yolo_nosetup"),
                       auto_setup=False)
    check("auto_setup=False 时 yolo 降级不回自动安装",
          ry3["engine"] in ("yolo", "yolo->cv-fallback")
          and ry3["warnings"], f"engine={ry3['engine']}")

    # ---- 验收 7c：marimo dict 下拉必须 {标签: 值}（回归：曾致未知模型错误） ----
    src_app = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
    check("下拉 dict 选项已按 {标签: 值} 反转",
          "options={v: k for k, v in backends.MODELS.items()}" in src_app
          and "options=backends.MODELS" not in src_app
          and "options=_ocr_opts" not in src_app
          and "options=_ocr_dd_opts" in src_app)

    # ---- 验收 8：PASS/FAIL 汇总 ----
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    print("\n========== 汇总 ==========")
    print(f"PASS {n_pass}/{len(RESULTS)}")
    for name, ok, note in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    sys.exit(0 if n_pass == len(RESULTS) else 1)


if __name__ == "__main__":
    run_all()
