"""检查两个 NF4 模型是否已在本地 HF 缓存中（供 infer_psd.ps1 -Offline 调用）。

独立成文件而不是 python -c：Windows PowerShell 5.1 会把 -c 参数里的双引号吃掉。
"""
import sys
from huggingface_hub import snapshot_download

REPOS = [
    "24yearsold/seethroughv0.0.2_layerdiff3d_nf4",
    "24yearsold/seethroughv0.0.1_marigold_nf4",
]

missing = []
for repo in REPOS:
    try:
        snapshot_download(repo_id=repo, local_files_only=True)
    except Exception:
        missing.append(repo)

if missing:
    print("MISSING: " + ", ".join(missing))
    sys.exit(1)
print("CACHE OK")
