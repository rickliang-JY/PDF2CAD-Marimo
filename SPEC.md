# SPEC — PDF2CAD 图纸解析平台（marimo）

## 概述
完整 marimo 交互平台：上传任意 PDF（多页、矢量/扫描混合）→ 多模型可选转换 → DXF 下载 + PDF/CAD 并排对比。

运行环境约束：**无 GPU，3GB 内存，3 CPU**。禁止引入 torch/ultralytics 等重依赖为必需项。

## 目录结构
```
/mnt/agents/output/pdf2cad_platform/
├── SPEC.md                 # 本文件
├── README.md               # 安装/运行/模型说明/已知限制
├── requirements.txt
├── app.py                  # marimo 应用入口（marimo run app.py）
├── core/
│   ├── __init__.py
│   ├── pdf2dxf.py          # 复制自 /mnt/agents/output/pdf2dxf/pdf2dxf.py（不改逻辑）
│   ├── raster2dxf.py       # 复制自 /mnt/agents/output/pdf2dxf/raster2dxf.py（不改逻辑）
│   ├── detect.py           # 逐页类型判定
│   ├── backends.py         # 四个后端统一接口
│   ├── pipeline.py         # 多页编排 + 合并输出
│   └── compare.py          # PDF vs DXF 渲染对比
└── tests/
    └── test_e2e.py         # 无头端到端测试（可直接 python 运行）
```

## 接口契约（必须严格实现）

### core/detect.py
```python
def page_kind(page: pymupdf.Page) -> str:
    """返回 'vector' | 'scan'。规则：矢量 drawings>10 或有文本 → 'vector'；
    否则若有覆盖页面面积>50% 的栅格图 → 'scan'；默认 'vector'。"""
def analyze_pdf(pdf_path: str) -> list[dict]:
    """每页返回 {'page': int, 'kind': str, 'n_drawings': int, 'n_texts': int,
    'image_coverage': float, 'width_pt': float, 'height_pt': float}"""
```

### core/backends.py
统一后端协议：
```python
class BackendResult(TypedDict):
    dxf_path: str          # 输出 DXF 路径
    engine: str            # 实际使用的引擎标识
    warnings: list[str]    # 降级/近似等警告（中文）
    stats: dict            # 实体统计

def convert_vector(pdf_path: str, page_num: int, out_dir: str) -> BackendResult
    # 封装 PDFToDXFConverter（强制矢量路径，不触发其内部扫描判定）
def convert_cv(pdf_path: str, page_num: int, out_dir: str, dpi: int = 200) -> BackendResult
    # 渲染页为位图 → RasterToDXFConverter(dpi=dpi, do_ocr=False)
    # do_ocr 默认 False（tesseract 系统依赖可能缺失；尝试 import，可用则开）
def convert_yolo(pdf_path: str, page_num: int, out_dir: str, dpi: int = 200,
                 weights: str | None = None) -> BackendResult
    # 尝试 import ultralytics 且 weights 文件存在 → YOLO 检测图纸元素区域
    # （图框/表格/文字块/符号），检测框写入 DXF 的 DETECTION 图层，
    # 框外区域仍走 CV 矢量化。不可用时：warnings 追加明确中文说明并
    # 回退 convert_cv（engine='yolo->cv-fallback'）。绝不抛异常。
def convert_auto(pdf_path: str, page_num: int, out_dir: str) -> BackendResult
    # detect.page_kind: 'vector'→convert_vector；'scan'→convert_yolo
MODELS = {"auto": "自动识别（逐页路由）", "vector": "纯 PDF 矢量提取",
          "cv": "经典视觉管线（OpenCV 骨架矢量化）",
          "yolo": "YOLO 视觉模型 + 矢量化（需 ultralytics 与权重）"}
```

### core/pipeline.py
```python
def convert_pdf(pdf_path: str, model: str, out_dir: str,
                pages: list[int] | None = None,
                progress_cb: callable | None = None) -> dict:
    """返回 {'pages': [BackendResult...], 'page_kinds': [...], 'zip_path': str,
    'log_path': str, 'elapsed_sec': float}
    - model ∈ MODELS.keys()；pages=None 表示全部页
    - 每页一个 DXF（{stem}_p001.dxf 约定），progress_cb(i, n, msg) 回调
    - conversion.log 汇总每页引擎/实体数/警告；全部 DXF+log 打入 zip_path
    - 单页失败不中断整本：该页记 warnings 并继续"""
```

### core/compare.py
```python
def render_pdf_page(pdf_path: str, page_num: int, dpi: int = 150) -> np.ndarray  # RGB
def render_dxf(dxf_path: str, dpi: int = 150,
               size_hint: tuple[int, int] | None = None) -> np.ndarray
    # ezdxf.addons.drawing + matplotlib Agg 后端渲染为 RGB 数组；
    # size_hint=(h,w) 时缩放到与该尺寸一致（cv2.resize）
def compare_page(pdf_path: str, dxf_path: str, page_num: int,
                 dpi: int = 150) -> dict:
    """返回 {'pdf_png': bytes, 'dxf_png': bytes, 'overlay_png': bytes,
    'iou': float, 'ink_pdf': int, 'ink_dxf': int}
    - overlay：PDF 墨迹红色、DXF 墨迹绿色、重合黄色的 RGB 叠差图
    - iou：二值墨迹交并比；渲染失败时 iou=None 并在 overlay 写错误说明"""
```

### app.py（marimo ≥0.24）
- `mo.ui.file(filetypes=[".pdf"])` 上传；显示 analyze_pdf 的逐页判定表
- `mo.ui.dropdown(options=MODELS)` 模型选择；`mo.ui.multiselect` 或页码范围选择
- 运行按钮 + `mo.status.progress_bar` 进度；逐页：页码/引擎/实体统计/警告
- 对比视图：页选择器 + 三图并排（原 PDF / DXF 渲染 / 叠差）+ IoU 指标
- 下载：`mo.ui.file_download`（或 mo.download）提供整包 zip 与单页 DXF
- 全程中文 UI；上传文件写入临时工作目录（tempfile.mkdtemp），输出放其下 out/
- 避免顶层副作用：重活全部放在按钮回调/惰性 cell 中，保证 `marimo check` 通过

## 测试（tests/test_e2e.py，无头，python 直接跑）
1. 用 pymupdf 程序化生成测试 PDF：page0 矢量图纸（线/矩形/圆/贝塞尔/旋转文字/颜色）、page1 模拟扫描（把 page0 渲染成 PNG 再 insert_image 嵌入）
2. 四个 model 各跑一遍 convert_pdf：
   - 每个 DXF 用 ezdxf.recover.readfile 打开 + audit，errors==0
   - auto 模式：page0 走 vector、page1 走 yolo（无权重→cv-fallback，warning 非空）
   - vector 模式 page0 实体数 > 50；cv 模式 page1 有 LWPOLYLINE
3. compare_page 对 page0：iou 为 (0,1] 浮点且三张 PNG 非空
4. `marimo check app.py` 通过（subprocess 调用）
5. 打印 PASS/FAIL 汇总，非零退出码表示失败

## 硬性规则
- 不修改复制过来的 pdf2dxf.py / raster2dxf.py 内部逻辑（bug 修复除外，需在 README 记录）
- 所有新增代码中文注释/中文日志；UI 文案中文
- 单页转换内存峰值控制：渲染 dpi ≤ 200，用完即 del + gc
- 禁止尝试 pip install torch/ultralytics（环境装不下）；yolo 后端只做 lazy import 探测
- matplotlib 一律 Agg 后端；中文字体 rcParams 显式设置
  ['Noto Sans CJK SC','Noto Sans CJK JP','WenQuanYi Zen Hei']
