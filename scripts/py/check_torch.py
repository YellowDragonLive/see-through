"""torch 2.8.0+cu128 且 CUDA 可用时退出码 0，用于 setup_env.ps1 幂等跳过。"""
try:
    import torch
    ok = (torch.__version__.startswith('2.8.0')
          and torch.version.cuda == '12.8'
          and torch.cuda.is_available())
except Exception:
    ok = False
import sys
sys.exit(0 if ok else 1)
