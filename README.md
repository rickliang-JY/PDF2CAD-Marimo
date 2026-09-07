# PDF2CAD 图纸解析平台（marimo）

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/rickliang-JY/PDF2CAD-Marimo/blob/main/app.py)

上传任意 PDF（多页、矢量/扫描混合）→ 多模型可选转换 → DXF 下载 + PDF/CAD 并排对比。

仓库：https://github.com/rickliang-JY/PDF2CAD-Marimo

## 在 molab 中打开（云端，免安装）

1. 直接访问镜像链接（GitHub 为 source of truth，推送新 commit 后自动同步）：
   `https://molab.marimo.io/github/rickliang-JY/PDF2CAD-Marimo/blob/main/app.py`
2. `app.py` 顶部带 PEP 723 依赖声明，molab 启动时会自动安装全部基础依赖；
   本地 `core/` 包不在时，首个 cell 会自动从 GitHub 仓库 zip 引导下载，无需手工操作。
3. 启动后点右上角 **Preview / Run as app** 即进入纯应用模式（隐藏代码）。

**molab 环境注意**：
- molab 无系统级 `tesseract` 二进制 → OCR 文字识别自动跳过（转换主流程不受影响，
  warnings 中有说明）；如需文字识别请本地运行或装 GPU 组件用 PaddleOCR。
- molab 免费实例为 CPU：YOLO/PaddleOCR/VLM 三个 GPU 组件仍会降级；
  需要完整 GPU 能力请按「GPU 组件」章节在自有 GPU 机器部署。
- WebAssembly 预览（`/wasm` 后缀）不支持本应用（PyMuPDF/OpenCV 不能在 Pyodide 运行），
  请使用服务器模式预览。

## 安装

```bash
pip install -r requirements.txt
```

可选依赖（缺失时自动降级，不影响主流程）：
- `pytesseract` + 系统二进制 `tesseract`：CV 管线的 OCR 文本识别
- `ultralytics` + YOLO 权重文件：YOLO 图纸元素检测后端
- GPU 组件见下文「GPU 组件」章节（`requirements-gpu.txt`）

> **环境约束**：无 GPU、3GB 内存。**不要** pip install torch/ultralytics 等重依赖；
> 全部 GPU 组件只做 lazy import 探测，缺失时自动回退轻量路径。

## 运行

```bash
marimo run app.py        # 交互应用
marimo check app.py      # 静态检查（CI/验收用）
python3 tests/test_e2e.py  # 无头端到端测试
```

## 目录结构

```
pdf2cad_platform/
├── SPEC.md
├── README.md
├── requirements.txt
├── requirements-gpu.txt  # GPU 可选组件（叠加安装，见「GPU 组件」章节）
├── app.py               # marimo 应用入口
├── core/
│   ├── __init__.py
│   ├── pdf2dxf.py       # 既有矢量提取模块（原样复制，未改逻辑）
│   ├── raster2dxf.py    # 既有光栅矢量化模块（原样复制，未改逻辑）
│   ├── detect.py        # 逐页类型判定（矢量/扫描）
│   ├── backends.py      # 五个后端统一接口（vector/cv/cv+/yolo/auto）
│   ├── raster_plus.py   # CV+ 增强视觉管线（LSD+文字掩膜+虚线重建）
│   ├── ocr_engines.py   # OCR 引擎抽象层（tesseract / PaddleOCR，自动降级）
│   ├── vlm.py           # Qwen2.5-VL 标题栏结构化抽取（GPU 可选组件）
│   ├── pipeline.py      # 多页编排 + conversion.log + zip 打包
│   └── compare.py       # PDF vs DXF 渲染对比（并排/叠差/IoU）
└── tests/
    └── test_e2e.py      # 无头端到端测试
```

## 模型说明

| 模型键 | 说明 | 依赖 |
|---|---|---|
| `auto` | 自动识别：逐页路由，矢量页→vector，扫描页→yolo（YOLO 不可用时由 cv+ 兜底） | 内置 |
| `vector` | 纯 PDF 矢量提取（无损，坐标级精确） | 内置 |
| `cv` | 经典视觉管线：渲染位图 → 二值化/骨架化/追踪 → LWPOLYLINE | OpenCV + scikit-image |
| `cv+` | 增强视觉管线：LSD 线段合并重建虚线 + 文字掩膜防乱线 + 折线拟合 ARC | OpenCV + scikit-image（+pytesseract/vtracer 可选） |
| `yolo` | YOLO 检测图纸元素（图框/标题栏/表格/符号），检测框写入 DXF 的 DETECTION 图层（中文类别+置信度），框内墨迹抠除后框外区域走 CV+ 矢量化 | ultralytics + 权重（可选） |

逐页判定规则（`core/detect.py`）：矢量图元 >10 或有文本 → `vector`；
否则若有覆盖页面 >50% 面积的栅格图 → `scan`；默认 `vector`。

YOLO 降级行为：`ultralytics` 未安装或未提供权重时，`convert_yolo` 追加中文警告
并回退 `convert_cv`，`engine='yolo->cv-fallback'`，**绝不抛异常**。

## CV+ 增强管线（`core/raster_plus.py`）

在经典 CV 管线基础上的三项升级（全部纯 OpenCV/numpy/可选依赖，无 GPU）：

1. **文字掩膜分离**（解决文字粘连图元）：二值化后、骨架化之前，用
   `pytesseract.image_to_data`（原始 + 90°/180°/270° 四个方向，覆盖旋转文字）
   取文字框，把框内墨迹（外扩 3px）从 ink 中抠除；几何提取只对"去字"墨迹进行，
   文字框区域不再被骨架化成乱线。被抠掉的文字照常写入 DXF `TEXT` 实体
   （含旋转角度）。tesseract 探测失败时跳过掩膜步骤并给出中文 warning，不中断。
2. **LSD 线段检测 + 间隙容忍合并**（补充骨架折线）：
   `cv2.createLineSegmentDetector` 在去字墨迹上检测线段，对近共线
   （角度差 <3°、法向距离 <2.5px）且端点间隙 < `gap_px` 的线段按角度分桶
   迭代合并为一条——虚线/点划线的断裂段被重建为单图元。一组合并的原始段数
   ≥3 且总间隙/总长 > 15% 时，写入 `DASHED` 图层并使用 DASHED 线型；
   其余写入 `SCAN_LINE` 图层的 `LINE` 实体。
3. **圆/弧拟合升级**：保留 Hough 圆检测；新增对骨架追踪出的较长折线路径做
   Kasa 最小二乘圆拟合，拟合残差 < 阈值（max(1.5px, 3%·r)）且弧覆盖角在
   30°~330° 之间时改写为 `ARC` 实体（整圆仍走 Hough 验证路径）；失败保留原
   折线，**绝不减少实体**。

另有可选 **vtracer 通道**（`use_vtracer=True`，默认关闭）：lazy import vtracer，
把 ink 位图矢量化为 SVG path，解析 M/L/C/Q/Z（贝塞尔分段采样为折线）写入
`VTRACER` 图层，适合曲线密集区域；库不可用或执行失败时中文 warning 跳过。

### 参数表（`RasterPlusConverter`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `dpi` | 200 | 像素→DXF 单位换算基准（平台侧硬上限 200） |
| `do_ocr` | True | 文字掩膜 + TEXT 实体写入（tesseract 不可用时自动跳过并告警） |
| `do_circles` | True | Hough 圆检测 + 折线弧拟合 |
| `gap_px` | 8.0 | LSD 合并允许的端点间隙（虚线空档大于此值时需调大） |
| `min_seg_len_px` | 6.0 | LSD 线段最短长度（段数超 6000 时自动上调保护内存） |
| `use_vtracer` | False | 开启 vtracer 曲线通道（输出到 VTRACER 图层） |
| `ocr_engine` | `'auto'` | OCR 引擎（`'auto'`/`'tesseract'`/`'paddle'`）；`'auto'` 与 `'tesseract'` 走既有 tesseract 4 方向路径（默认行为不变），显式 `'paddle'` 切换 OCR 抽象层（含自动降级） |

`stats` 含 `LINE/LWPOLYLINE/CIRCLE/ARC/TEXT/DASHED[/VTRACER]` 分段统计 +
`audit_errors` + `warnings`（中文告警列表）。

### 与经典 CV 管线的对比建议

- **直线为主的图纸**（建筑平面图、框线表格、含虚线/点划线的机械图）：用 `cv+`。
  LSD 直线段比骨架折线更平直，虚线能被重建为单条 DASHED 线。
- **曲线密集/自由曲线图纸**（等高线、手绘扫描）：用 `cv`（骨架折线更贴合任意
  曲线），或 `cv+` 程序调用时开 `use_vtracer=True`（贝塞尔分段采样）。
- 注意：cv+ 的 LSD 直线与骨架折线是**并存输出**（直线重建 + 曲线补充），
  直线区域会在两个图层各有一份几何；需要"干净直线版"时可只保留
  `LINE`/`DASHED` 图层。
- **auto 路由说明**：扫描页在 YOLO（ultralytics）可用时仍走 YOLO 管线，
  且 YOLO 检测框外区域保持用经典 cv 矢量化（未改动，改动收益小复杂度高）；
  YOLO 不可用时扫描页由 cv+ 兜底。

## GPU 组件（可选，`requirements-gpu.txt`）

本地轻量环境开箱即用（lazy import + 优雅降级）；GPU 机器叠加安装后自动启用：

```bash
pip install -r requirements.txt
pip install -r requirements-gpu.txt   # 仅 GPU 机器
```

| 组件 | 作用 | 安装包 | 启用后行为变化 |
|---|---|---|---|
| YOLO 图纸元素检测 | 检测图框/标题栏/表格/符号 | `ultralytics` + 自备权重 | `yolo` 后端不再回退：检测框（中文类别 + 置信度）写入 DXF `DETECTION` 图层，框内墨迹抠除、框外走 CV+ 矢量化；标题栏框存入 `stats['title_block_bbox']` 供 VLM 复用 |
| PaddleOCR 引擎 | 中文/旋转文字识别（PP-OCRv5） | `paddlepaddle-gpu` + `paddleocr` | UI「OCR 引擎」下拉出现 paddle 选项；`ocr_engine='paddle'/'auto'` 时文字提取走 PaddleOCR（旋转框四点 → bbox+角度，DXF TEXT 应用旋转） |
| VLM 标题栏抽取 | 标题栏结构化（图名/图号/比例/日期/设计/审核/单位） | `torch` + `transformers` + `qwen-vl-utils` + `accelerate` | UI 打开「VLM 标题栏抽取」开关后，每页转换后调用 Qwen2.5-VL（YOLO 标题栏框优先，否则右下角 40%×25% 启发式裁剪），结果展示在结果表下方、写入 `stats['title_block']` 并随 zip 附带 `title_block.json` |

权重路径、OCR 引擎、VLM 开关均在 UI 的「⚙️ 高级选项」折叠面板中配置；
组件未安装时相应位置显示灰色提示（而非隐藏），告知能力存在。

### 降级逻辑表（组件缺失时各路径退回什么）

| 路径 | 组件缺失时行为 |
|---|---|
| `yolo` 后端 | 无 ultralytics 或无权重 → 回退经典 CV 管线，`engine='yolo->cv-fallback'`，中文 warning，**绝不抛异常** |
| `auto` 路由（扫描页） | ultralytics 不可用 → CV+ 增强管线兜底（`engine='cv+'`，warning 非空） |
| OCR 引擎 `paddle` | paddleocr 不可用或执行失败 → 自动降级 tesseract 4 方向探测（记 warning）；tesseract 也缺失 → 返回空，跳过文字掩膜与 TEXT 实体 |
| OCR 引擎 `auto`（默认） | paddle 可用优先，否则 tesseract——**无 GPU 组件时与旧版行为逐字节一致** |
| VLM 标题栏抽取 | transformers/qwen_vl_utils/torch 任一缺失，或推理/JSON 解析失败 → 返回 `None`，`stats` 无 `title_block` 键，转换主流程不受影响 |

## 输出约定

- 每页一个 DXF：`{文件名}_p001.dxf`（页码从 1 起，三位数字）
- `conversion.log`：逐页引擎/实体数/警告汇总
- 全部 DXF + log 打入 `{文件名}_dxf_bundle.zip`
- 单页失败不中断整本：该页记 warnings 并继续

## 对既有模块的修改记录

`core/pdf2dxf.py` 与 `core/raster2dxf.py` 为 **原样复制，未改任何逻辑**。

`convert_vector` 需要强制走矢量路径（不触发 `PDFToDXFConverter.run()` 内部的
`_looks_like_scan` 扫描判定）。实现方式是**在 wrapper 层**定义子类
`_ForcedVectorConverter`，将 `_looks_like_scan` 固定返回 `False`，
未改动被复制源码的一行。

## 已知限制

1. **YOLO 后端在本环境不可用**（无 GPU/3GB 内存，禁止安装 ultralytics）：
   选择 yolo/auto 且遇到扫描页时自动回退 CV 管线，DXF 为近似重建。
2. **CV/光栅矢量化是近似重建**：精度受渲染分辨率（≤200 dpi）与扫描质量影响，
   不保证矢量管线的坐标级精度；文字只有 OCR 可用时才识别。
3. **OCR 依赖系统 tesseract**：缺失时自动跳过文本识别并在 warnings 中说明。
4. **内存约束**：渲染 dpi 硬上限 200，单页位图用完即释放（del + gc）；
   超大图纸页建议分批选择页面转换。
5. DXF 单位约定：1 单位 = 1 PDF 点（1/72 英寸），`$INSUNITS=0`（unitless），
   导入 CAD 后需按实际比例缩放。
6. 叠差 IoU 基于渲染位图的墨迹像素，受线宽/抗锯齿影响，仅作质量参考指标。
7. **CV+ 管线已知限制**：LSD 合并是启发式（3°/2.5px/gap_px 阈值），密集平行线
   间距 <2.5px 时可能误并；虚线空档 > `gap_px` 时需手动调大参数；文字掩膜依赖
   tesseract 检出率，艺术字/特大字（>页高 3%）不抠除；弧拟合只处理覆盖角
   30°~330° 的路径，整圆交 Hough。

## 未来改进

受环境限制（GitHub/HuggingFace 不可达、无 GPU、3GB 内存），以下方案仅列入规划：
- **PaddleOCR** 替代 tesseract：中文图纸文字与旋转文字检出率更高。
- **Qwen2.5-VL** 等多模态大模型：图纸语义理解（房间/门窗/标注）端到端解析。
- **FloorPlanCAD** 等专用数据集微调的检测模型：替代通用 YOLO 权重。
- vtracer 通道默认开启并做图层融合去重（LSD 直线 + vtracer 曲线互补）。
