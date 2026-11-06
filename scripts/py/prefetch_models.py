"""预下载两个 NF4 模型；ST_VERIFY=1 时只校验本地缓存。"""
import os
import sys
import time
from huggingface_hub import snapshot_download

REPOS = [
    "24yearsold/seethroughv0.0.2_layerdiff3d_nf4",
    "24yearsold/seethroughv0.0.1_marigold_nf4",
]
verify = os.environ.get("ST_VERIFY") == "1"
retries = int(os.environ.get("ST_RETRIES", "3"))
endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
print(f"endpoint = {endpoint}  verify_only = {verify}")

failed = []
for repo in REPOS:
    ok = False
    for attempt in range(1, retries + 1):
        try:
            path = snapshot_download(
                repo_id=repo,
                local_files_only=verify,
                max_workers=4,
            )
            size_mb = sum(
                os.path.getsize(os.path.join(root, f))
                for root, _, files in os.walk(path) for f in files
            ) / 1e6
            print(f"OK   {repo}  {size_mb:.1f} MB  -> {path}", flush=True)
            ok = True
            break
        except Exception as e:
            print(f"WARN {repo} attempt {attempt}/{retries}: {type(e).__name__}: {e}", flush=True)
            if attempt < retries:
                time.sleep(3)
    if not ok:
        failed.append(repo)

if failed:
    print("FAILED:", ", ".join(failed))
    sys.exit(1)
print("ALL MODELS READY")
