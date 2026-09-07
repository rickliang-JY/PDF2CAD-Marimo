"""gpu_detect.py —— GPU 环境自动探测（全部 lazy，绝不抛异常）。"""
from __future__ import annotations

import subprocess
from typing import Any, Dict


def gpu_info() -> Dict[str, Any]:
    """探测 GPU 可用性。

    返回 {
        'available': bool,
        'backend': 'cuda' | 'mps' | None,
        'devices': [str, ...],   # 显卡型号列表
        'detail': str,           # 中文说明（用于 UI 展示）
    }
    探测顺序：torch.cuda → torch.mps → nvidia-smi（torch 未装时的兜底）。
    """
    # 1) torch 探测
    try:
        import torch
        if torch.cuda.is_available():
            names = [torch.cuda.get_device_name(i)
                     for i in range(torch.cuda.device_count())]
            return {"available": True, "backend": "cuda", "devices": names,
                    "detail": f"检测到 CUDA GPU：{'、'.join(names)}"}
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return {"available": True, "backend": "mps", "devices": ["Apple Silicon (MPS)"],
                    "detail": "检测到 Apple Silicon GPU（MPS）"}
    except Exception:
        pass

    # 2) nvidia-smi 兜底（torch 未安装但机器有 NVIDIA 显卡）
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if proc.returncode == 0 and proc.stdout.strip():
            names = [ln.strip() for ln in proc.stdout.strip().splitlines() if ln.strip()]
            return {"available": True, "backend": "cuda", "devices": names,
                    "detail": f"检测到 NVIDIA GPU（{'、'.join(names)}），"
                              "但未安装 torch——GPU 组件需要 pip install -r requirements-gpu.txt"}
    except Exception:
        pass

    return {"available": False, "backend": None, "devices": [],
            "detail": "未检测到 GPU，将以 CPU 模式运行（GPU 组件自动降级或按需下载 CPU 版）"}
