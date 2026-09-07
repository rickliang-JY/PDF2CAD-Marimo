"""vlm.py —— Qwen2.5-VL 标题栏结构化抽取（GPU 可选组件，lazy import + 优雅降级）。

架构原则：本地轻量可跑——本模块所有重依赖（torch/transformers/qwen_vl_utils）
均为 lazy import 探测；不可用时 extract_title_block 返回 None（不抛异常）。

GPU 机器 `pip install -r requirements-gpu.txt` 后自动启用：
    区域裁剪（YOLO 检测框优先，否则右下角启发式）-> Qwen2.5-VL 结构化 prompt
    -> JSON 解析容错 -> {'图名':..., '图号':..., '比例':..., '日期':...,
    '设计':..., '审核':..., '单位':...}（缺字段省略）
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("pdf2cad.vlm")

# 标题栏结构化字段（固定顺序，写入 DXF 时按此排列）
TITLE_FIELDS = ("图名", "图号", "比例", "日期", "设计", "审核", "单位")

# 启发式裁剪：无 YOLO 检测框时取右下角 40% x 25% 区域（标题栏惯例位置）
_HEURISTIC_W_RATIO = 0.40
_HEURISTIC_H_RATIO = 0.25

# 结构化 prompt：要求严格 JSON 输出
_PROMPT = (
    "这是一张工程图纸的标题栏区域截图。请抽取以下字段并以严格 JSON 输出"
    "（无对应内容则省略该键，不要输出任何 JSON 以外的文字）：\n"
    '{"图名": ..., "图号": ..., "比例": ..., "日期": ..., '
    '"设计": ..., "审核": ..., "单位": ...}'
)

# 生成参数（短输出即可，限制显存/耗时）
_MAX_NEW_TOKENS = 256


def vlm_available() -> bool:
    """lazy 探测 transformers + qwen_vl_utils + torch 三件套是否齐备。"""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import qwen_vl_utils  # noqa: F401
        return True
    except Exception:
        return False


def _crop_title_region(img_bgr: np.ndarray,
                       title_bbox: Optional[Tuple[float, float, float, float]],
                       ) -> np.ndarray:
    """裁剪标题栏区域：title_bbox（像素 x0,y0,x1,y1）优先，否则右下角启发式。"""
    h, w = img_bgr.shape[:2]
    if title_bbox is not None:
        x0, y0, x1, y1 = (int(round(v)) for v in title_bbox)
        x0, x1 = max(0, x0), min(w, x1)
        y0, y1 = max(0, y0), min(h, y1)
        if x1 - x0 >= 8 and y1 - y0 >= 8:
            return img_bgr[y0:y1, x0:x1]
        logger.warning("YOLO 标题栏框过小（%sx%s），改用启发式裁剪",
                       x1 - x0, y1 - y0)
    x0 = int(round(w * (1.0 - _HEURISTIC_W_RATIO)))
    y0 = int(round(h * (1.0 - _HEURISTIC_H_RATIO)))
    return img_bgr[y0:h, x0:w]


def _parse_json_tolerant(text: str) -> Optional[dict]:
    """容错解析模型输出中的 JSON 对象（截取首个 {...}，字段过滤白名单）。"""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception as exc:
        logger.warning("VLM 输出 JSON 解析失败（%s）: %.120s", exc, text)
        return None
    if not isinstance(obj, dict):
        return None
    out = {k: str(v).strip() for k, v in obj.items()
           if k in TITLE_FIELDS and str(v).strip()}
    return out or None


def extract_title_block(
    img_bgr: np.ndarray,
    title_bbox: Optional[Tuple[float, float, float, float]] = None,
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
) -> Optional[dict]:
    """Qwen2.5-VL 标题栏结构化抽取。不可用或任何失败返回 None（不抛异常）。

    参数:
        img_bgr:    整页 BGR 位图
        title_bbox: YOLO 检测到的标题栏框（像素 x0,y0,x1,y1），None 走启发式
        model_id:   HuggingFace 模型 ID（GPU 机器可换 7B 提升精度）
    """
    if not vlm_available():
        logger.info("VLM 组件不可用（transformers/qwen_vl_utils/torch 缺失），"
                    "跳过标题栏结构化抽取")
        return None
    try:
        import cv2
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        crop = _crop_title_region(img_bgr, title_bbox)
        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto" if device == "cuda"
            else None)
        processor = AutoProcessor.from_pretrained(model_id)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": _PROMPT},
            ],
        }]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs,
                           videos=video_inputs, padding=True,
                           return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=_MAX_NEW_TOKENS)
        trimmed = [out[len(inp):] for inp, out in
                   zip(inputs.input_ids, generated)]
        out_text = processor.batch_decode(
            trimmed, skip_special_tokens=True)[0]
        info = _parse_json_tolerant(out_text)
        if info is None:
            logger.warning("VLM 未产出有效结构化结果: %.120s", out_text)
        return info
    except Exception as exc:
        logger.warning("VLM 标题栏抽取失败（%s），已跳过", exc)
        return None


def title_block_to_dxf_text(msp, info: dict, anchor_xy: Tuple[float, float],
                            layer: str = "TITLE_INFO") -> int:
    """把抽取字段写成 DXF TEXT 实体（便于 CAD 内查看），返回实体数。

    anchor_xy: 首行文字基点（DXF 单位）；每行向下排列，行距 1.6 倍字高。
    """
    if not info:
        return 0
    try:
        if layer not in msp.doc.layers:
            msp.doc.layers.add(layer, color=4)  # 青色
    except Exception:
        pass
    height = 8.0
    n = 0
    for i, key in enumerate(TITLE_FIELDS):
        val = info.get(key)
        if not val:
            continue
        try:
            t = msp.add_text(f"{key}: {val}", height=height,
                             dxfattribs={"layer": layer})
            t.set_placement((anchor_xy[0], anchor_xy[1] - i * height * 1.6))
            n += 1
        except Exception as exc:
            logger.debug("TITLE_INFO 写入失败: %s", exc)
    return n
