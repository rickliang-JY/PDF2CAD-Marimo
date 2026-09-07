# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.24",
#     "pymupdf>=1.26",
#     "ezdxf>=1.4",
#     "opencv-python-headless>=4.10",
#     "numpy>=1.26",
#     "matplotlib>=3.9",
#     "scikit-image>=0.24",
#     "pytesseract>=0.3.13",
#     "vtracer>=0.6",
# ]
# ///
"""app.py —— PDF2CAD 图纸解析平台（marimo 交互应用）。

运行：marimo run app.py
云端：molab 打开时 PEP 723 依赖自动安装，core/ 包自动从 GitHub 仓库引导下载。

功能：上传 PDF（多页、矢量/扫描混合）-> 逐页类型判定 -> 多模型转换
（自动/矢量/CV/YOLO）-> DXF 下载 + PDF/CAD 并排对比（含叠差图与 IoU）。

注意：重活全部放在按钮回调/惰性 cell 中，顶层无副作用（保证 marimo check 通过）。
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import sys
    import tempfile
    from pathlib import Path

    import marimo as mo

    # 保证 core 包可导入（marimo run 的工作目录未必是项目根）
    _ROOT = os.path.dirname(os.path.abspath(__file__))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

    try:
        from core import backends, compare, detect, ocr_engines, pipeline, vlm
    except ImportError:
        # molab / 单文件场景：core/ 不在本地时，从 GitHub 仓库引导下载
        import io
        import zipfile
        import urllib.request

        _REPO_ZIP = ("https://codeload.github.com/rickliang-JY/"
                     "PDF2CAD-Marimo/zip/refs/heads/main")
        _data = urllib.request.urlopen(_REPO_ZIP, timeout=60).read()
        _extract = Path(tempfile.mkdtemp(prefix="pdf2cad_repo_"))
        with zipfile.ZipFile(io.BytesIO(_data)) as _zf:
            _zf.extractall(_extract)
        _repo_root = next(_extract.glob("PDF2CAD-Marimo-*"))
        sys.path.insert(0, str(_repo_root))
        from core import backends, compare, detect, ocr_engines, pipeline, vlm

    return backends, compare, detect, mo, ocr_engines, os, pipeline, tempfile, vlm


@app.cell
def _(mo):
    mo.md(r"""
    # PDF2CAD 图纸解析平台

    上传 PDF 工程图纸（支持多页、矢量/扫描混合），选择转换模型，
    输出 DXF 下载，并提供 **原 PDF / DXF 渲染 / 叠差图** 三图并排对比与 IoU 指标。

    > 运行环境：无 GPU、内存受限。YOLO 后端需要 `ultralytics` 与权重文件，
    > 缺失时自动回退到经典视觉管线（不会报错中断）。
    """)
    return


@app.cell
def _(mo):
    upload = mo.ui.file(filetypes=[".pdf"], kind="area")
    mo.vstack([mo.md("## 1. 上传 PDF 图纸"), upload])
    return (upload,)


@app.cell
def _(mo, os, tempfile, upload):
    """把上传的 PDF 写入临时工作目录（输出放其下 out/）。"""
    pdf_path = None
    work_dir = None
    if upload.value:
        _f = upload.value[0]  # UploadedFile: .name / .contents(bytes)
        work_dir = tempfile.mkdtemp(prefix="pdf2cad_")
        pdf_path = os.path.join(work_dir, _f.name or "input.pdf")
        with open(pdf_path, "wb") as _fh:
            _fh.write(_f.contents)
    _has = pdf_path is not None
    mo.stop(not _has, mo.md("⬆️ 请先上传一个 PDF 文件"))
    return pdf_path, work_dir


@app.cell
def _(detect, mo, pdf_path):
    """上传后立即做逐页判定（轻量解析，不渲染）。"""
    analysis = detect.analyze_pdf(pdf_path)
    _rows = [
        {
            "页码": r["page"] + 1,
            "判定": "矢量" if r["kind"] == "vector" else "扫描",
            "矢量图元数": r["n_drawings"],
            "文本数": r["n_texts"],
            "图像覆盖率": f"{r['image_coverage']:.1%}",
            "页面尺寸(pt)": f"{r['width_pt']:.0f}×{r['height_pt']:.0f}",
        }
        for r in analysis
    ]
    mo.vstack([
        mo.md("## 2. 逐页类型判定"),
        mo.ui.table(_rows, selection=None, show_download=False),
    ])
    return (analysis,)


@app.cell
def _(analysis, backends, mo):
    model_sel = mo.ui.dropdown(
        options=backends.MODELS, value="auto", label="转换模型：")
    page_opts = {
        f"第 {r['page'] + 1} 页（{'矢量' if r['kind'] == 'vector' else '扫描'}）":
            r["page"]
        for r in analysis
    }
    pages_sel = mo.ui.multiselect(
        options=page_opts, label="页面选择（留空 = 全部页）：")
    run_btn = mo.ui.run_button(label="🚀 开始转换")
    mo.vstack([
        mo.md("## 3. 转换设置"),
        mo.hstack([model_sel, pages_sel], justify="start", gap=2),
        run_btn,
    ])
    return model_sel, pages_sel, run_btn


@app.cell
def _(mo, ocr_engines, vlm):
    """GPU 可选组件可用性探测（全部 lazy import，绝不抛异常）。"""
    avail_ocr = ocr_engines.available_engines()
    try:
        import ultralytics  # noqa: F401 仅探测是否安装
        has_ultra = True
    except Exception:
        has_ultra = False
    has_vlm = vlm.vlm_available()
    return avail_ocr, has_ultra, has_vlm


@app.cell
def _(avail_ocr, has_ultra, has_vlm, mo):
    """高级选项（GPU 可选组件）：折叠面板，未安装的组件显示灰色提示而非隐藏。"""
    _ocr_opts = dict(avail_ocr)
    if "tesseract" not in _ocr_opts:  # 极端情况：系统连 tesseract 都缺失
        _ocr_opts["tesseract"] = "Tesseract（当前不可用）"
    ocr_sel = mo.ui.dropdown(options=_ocr_opts, label="OCR 引擎：")
    yolo_weights_txt = mo.ui.text(
        label="YOLO 权重路径（.pt，留空则回退 CV 管线）：",
        placeholder="/path/to/drawing_yolo.pt")
    vlm_sw = mo.ui.switch(label="VLM 标题栏结构化抽取（Qwen2.5-VL）")

    _gpu_hint = ("<span style='color:#999'>{name} 未安装——GPU 环境 "
                 "<code>pip install -r requirements-gpu.txt</code> 后启用</span>")
    _hints = []
    if "paddle" not in avail_ocr:
        _hints.append(mo.md(_gpu_hint.format(name="PaddleOCR 引擎")))
    if not has_ultra:
        _hints.append(mo.md(_gpu_hint.format(name="YOLO 后端（ultralytics）")))
    if not has_vlm:
        _hints.append(mo.md(
            _gpu_hint.format(name="VLM 组件（transformers/qwen-vl-utils/torch）")))
    mo.vstack([
        mo.accordion({
            "⚙️ 高级选项（OCR 引擎 / YOLO 权重 / VLM 标题栏抽取）":
                mo.vstack([ocr_sel, yolo_weights_txt, vlm_sw] + _hints)
        }),
    ])
    return ocr_sel, vlm_sw, yolo_weights_txt


@app.cell
def _(mo, model_sel, ocr_sel, os, pages_sel, pdf_path, pipeline, run_btn,
      vlm_sw, work_dir, yolo_weights_txt):
    mo.stop(not run_btn.value,
            mo.md("⏳ 设置完成后点击「开始转换」"))

    _pages = sorted(pages_sel.value) if pages_sel.value else None
    out_dir = os.path.join(work_dir, "out")
    _n = len(_pages) if _pages else len(
        __import__("pymupdf").open(pdf_path))
    _prog = mo.status.progress_bar(total=_n, title="正在转换", subtitle="准备中…")

    def _cb(i, n, msg):
        # progress_cb(i, n, msg)：驱动 marimo 进度条
        _prog.update(progress=i, subtitle=msg)

    result = pipeline.convert_pdf(
        pdf_path, model_sel.value, out_dir, pages=_pages, progress_cb=_cb,
        ocr_engine=ocr_sel.value,
        yolo_weights=(yolo_weights_txt.value or None),
        extract_title=bool(vlm_sw.value))
    _prog.update(progress=_n, subtitle="全部完成")
    mo.md(f"✅ 转换完成，耗时 **{result['elapsed_sec']:.1f} 秒**，"
          f"产出 {sum(1 for r in result['pages'] if r['dxf_path'])} 个 DXF")
    return (result,)


@app.cell
def _(mo, result):
    _ENTITY_KEYS = ("LINE", "LWPOLYLINE", "CIRCLE", "ELLIPSE", "ARC",
                    "SPLINE", "HATCH", "TEXT", "DETECTION", "DASHED", "VTRACER")
    _rows = []
    for _r in result["pages"]:
        _ent = {k: int(v) for k, v in _r["stats"].items()
                if k in _ENTITY_KEYS and isinstance(v, (int, float)) and v}
        _rows.append({
            "页码": _r["page"] + 1,
            "判定": "矢量" if _r.get("kind") == "vector" else "扫描",
            "引擎": _r["engine"],
            "实体统计": ", ".join(f"{k}:{v}" for k, v in _ent.items()) or "—",
            "警告": "；".join(_r["warnings"]) or "无",
        })
    mo.vstack([
        mo.md("## 4. 逐页转换结果"),
        mo.ui.table(_rows, selection=None, show_download=False),
    ])
    return


@app.cell
def _(mo, result):
    """标题栏结构化抽取结果（开启 VLM 抽取且组件可用的页才有内容）。"""
    _tb_rows = [
        {"页码": _r["page"] + 1, "字段": _k, "内容": _v}
        for _r in result["pages"]
        for _k, _v in ((_r.get("stats") or {}).get("title_block") or {}).items()
    ]
    mo.stop(not _tb_rows, mo.md(""))
    mo.vstack([
        mo.md("### 标题栏结构化抽取结果（VLM）"),
        mo.ui.table(_tb_rows, selection=None, show_download=False),
    ])
    return


@app.cell
def _(mo, os, result):
    _items = [mo.md("## 5. 下载")]
    with open(result["zip_path"], "rb") as _f:
        _zip_bytes = _f.read()
    _items.append(mo.download(
        data=_zip_bytes,
        filename=os.path.basename(result["zip_path"]),
        mimetype="application/zip",
        label="⬇️ 下载整包（全部 DXF + conversion.log）",
    ))
    with open(result["log_path"], "rb") as _f:
        _log_bytes = _f.read()
    _items.append(mo.download(
        data=_log_bytes, filename="conversion.log",
        mimetype="text/plain", label="⬇️ 下载转换日志",
    ))
    for r in result["pages"]:
        if r["dxf_path"] and os.path.isfile(r["dxf_path"]):
            with open(r["dxf_path"], "rb") as _f:
                _dxf_bytes = _f.read()
            _items.append(mo.download(
                data=_dxf_bytes,
                filename=os.path.basename(r["dxf_path"]),
                mimetype="application/dxf",
                label=f"⬇️ 第 {r['page'] + 1} 页 DXF（{r['engine']}）",
            ))
    mo.vstack(_items)
    return


@app.cell
def _(mo, result):
    cmp_page_sel = mo.ui.dropdown(
        options={f"第 {r['page'] + 1} 页（{r['engine']}）": r["page"]
                 for r in result["pages"] if r["dxf_path"]},
        label="对比页：")
    cmp_btn = mo.ui.run_button(label="🔍 生成对比")
    cmp_dpi = mo.ui.slider(start=72, stop=200, step=4, value=120,
                           label="渲染 DPI（越大越清晰越慢）")
    mo.vstack([
        mo.md("## 6. PDF / CAD 对比视图"),
        mo.hstack([cmp_page_sel, cmp_dpi, cmp_btn], justify="start", gap=2),
    ])
    return cmp_btn, cmp_dpi, cmp_page_sel


@app.cell
def _(cmp_btn, cmp_dpi, cmp_page_sel, compare, mo, pdf_path, result):
    mo.stop(not cmp_btn.value,
            mo.md("⏳ 选择页面后点击「生成对比」"))
    _page = cmp_page_sel.value
    _dxf = next(r["dxf_path"] for r in result["pages"] if r["page"] == _page)
    cmp_res = compare.compare_page(pdf_path, _dxf, _page, dpi=cmp_dpi.value)

    _iou_txt = ("渲染失败，无法计算" if cmp_res["iou"] is None
                else f"**{cmp_res['iou']:.3f}**")
    _legend = mo.md(
        f"墨迹 IoU：{_iou_txt}　｜　PDF 墨迹像素：{cmp_res['ink_pdf']}　｜　"
        f"DXF 墨迹像素：{cmp_res['ink_dxf']}\n\n"
        f"叠差图例：🟥 仅 PDF 有　🟩 仅 DXF 有　🟨 两侧重合"
    )
    mo.vstack([
        _legend,
        mo.hstack([
            mo.vstack([mo.md("**原 PDF**"), mo.image(cmp_res["pdf_png"])]),
            mo.vstack([mo.md("**DXF 渲染**"), mo.image(cmp_res["dxf_png"])]),
            mo.vstack([mo.md("**叠差图**"), mo.image(cmp_res["overlay_png"])]),
        ], justify="space-around", gap=1),
    ])
    return


if __name__ == "__main__":
    app.run()
