# Plan — PDF2CAD 图纸解析平台（marimo 完整版）

## 目标
把现有 pdf2dxf 工作升级为完整的 marimo 交互平台：
- 用户上传任意 PDF（多页、扫描件/矢量件均可，逐页自动识别）
- 多模型可选：① 纯 PDF 矢量提取（pdf2dxf）② 经典视觉管线（OpenCV/raster2dxf）③ YOLO 视觉模型 + 后处理（可选依赖，缺失时优雅降级）④ 自动路由（逐页判定）
- 输出 DXF 下载 + PDF 原页与 CAD 渲染图并排对比 + 叠差图
- 无 GPU、3GB 内存约束下可运行

## Stage 1 — 环境与资产盘点（主代理）
- 技能：vibecoding-general-swarm（编码任务强制）
- 安装：marimo、pymupdf、ezdxf、opencv-python-headless、numpy、matplotlib
- 摸清现有 pdf2dxf.py / raster2dxf.py 的公开接口（函数签名、输入输出）
- 输出：接口清单 + 架构设计

## Stage 2 — 平台实现（coder 子代理）
目录 /mnt/agents/output/pdf2cad_platform/：
- core/detect.py：逐页扫描件判定（矢量图元密度 vs 图像覆盖率）
- core/vector_backend.py：封装 pdf2dxf 为 backend 接口
- core/raster_backend.py：封装 raster2dxf（OpenCV Hough/轮廓）
- core/yolo_backend.py：ultralytics 可用则加载（图纸元素检测→分区→矢量化），不可用则返回明确降级说明
- core/pipeline.py：多页编排、逐页路由、合并 DXF（每页一个 layout/block）、conversion log
- core/compare.py：PDF 页渲染 vs DXF 渲染（ezdxf addons），并排图 + 叠差热力图 + IoU 统计
- app.py：marimo 应用（mo.file_upload 上传、模型选择下拉、逐页预览 carousel、对比视图、DXF/log 下载按钮）
- README.md + requirements.txt

## Stage 3 — 无头验证（verifier/coder）
- 生成测试 PDF：矢量图纸页 + 模拟扫描页（渲染成图再嵌回）
- 跑 pipeline 全路径（四种模型选项），验证 DXF 可 ezdxf recover 打开、audit 无错
- marimo 应用静态检查：marimo check / python -c import 级验证
- 输出：验证报告

## Stage 4 — 交付（主代理）
- README、使用说明、已知限制（YOLO 在无 GPU/3GB 环境的取舍）
- REF 标签交付

## Stage 5 — 借鉴工业方案升级（已完成派发）
- CV+ 增强管线：LSD 线段检测 + 间隙容忍虚线重建 + 文字掩膜分离 + 圆/弧拟合 + vtracer 可选通道
- MODELS 增加 "cv+"，auto 的 scan 分支路由到 cv+

## Stage 6 — GPU 组件双层架构（用户明确要未来 GPU 部署）
- 原则：本地轻量可跑（lazy import + 优雅降级），GPU 机器 pip install -r requirements-gpu.txt 后自动启用
- YOLO 补全：UI 权重路径、检测类别→DETECTION 图层、框外区域 CV+ 矢量化
- OCR 引擎抽象：tesseract（默认 CPU）/ PaddleOCR PP-OCRv5（GPU 可选，旋转框→DXF TEXT rotation）
- Qwen2.5-VL 标题栏结构化抽取：区域定位（YOLO 框或启发式）→ JSON → UI 展示 + DXF 存档
- requirements-gpu.txt：ultralytics / paddlepaddle-gpu / paddleocr / torch / transformers / qwen-vl-utils
- 所有 GPU 路径在本环境验证"降级不崩"，测试全绿
