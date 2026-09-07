"""core 包：PDF2CAD 图纸解析平台核心模块。

- detect.py   逐页类型判定（矢量/扫描）
- backends.py 五个转换后端统一接口（vector/cv/cv+/yolo/auto）
- pipeline.py 多页编排 + 合并输出（DXF + conversion.log + zip）
- compare.py  PDF 原页 vs DXF 渲染对比（并排 + 叠差 + IoU）
- pdf2dxf.py  既有矢量提取模块（原样复制）
- raster2dxf.py 既有光栅矢量化模块（原样复制）
- raster_plus.py CV+ 增强视觉管线（LSD 虚线重建 + 文字掩膜 + 弧拟合 + 可选 vtracer）
- ocr_engines.py OCR 引擎抽象层（tesseract / PaddleOCR，lazy import + 自动降级）
- vlm.py        Qwen2.5-VL 标题栏结构化抽取（GPU 可选组件，lazy import）
- gpu_detect.py GPU 环境自动探测（torch.cuda/mps/nvidia-smi 三级，lazy）
"""
