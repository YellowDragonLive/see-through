#!/usr/bin/env python3
"""
see-through Batch Layered PSD Runner (Windows 11, Pure ASCII)

Folder layout (next to batch_win.bat / batch_win.py, e.g. repo\\workspace\\):
  target\\   Input images (png / jpg / jpeg / webp; Chinese filenames OK)
  output\\   Output PSDs grouped by timestamp
  log\\      ASCII copies (t01/t02...), per-image logs, timings, summary

Usage:
  batch_win.bat                       Default 1024px (uses built-in v5.1 inference_utils)
  batch_win.bat 1280                  Specify resolution (8GB GPU ceiling is ~1280)
  batch_win.bat 1024 --no-preview     Skip preview PNG compositing
  batch_win.bat 1024 --fix            Also run external fix.py after PSD export
  batch_win.bat 1024 --no-fix         Skip external fix.py (default)
  batch_win.bat 1024 --order depth    Sort mode for fix.py
  batch_win.bat 1024 --steps 30       Diffusion steps (default 30)
  batch_win.bat 1024 --depth-res 768  Marigold depth resolution (default 768)
"""
import argparse
import datetime
import glob
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HR = "-" * 68
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
OOM_RE = re.compile(r"out of memory|OutOfMemory|CUDA error|illegal memory", re.I)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="batch_win.bat",
        description="see-through Batch Layered PSD Runner for Windows 11 (Pure ASCII)",
    )
    parser.add_argument(
        "resolution",
        nargs="?",
        type=int,
        default=1024,
        help="Square inference resolution in px (default: 1024; 8GB GPU max: 1280)",
    )
    parser.add_argument(
        "--depth-res",
        type=int,
        default=768,
        help="Marigold depth resolution (default: 768)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Diffusion inference steps (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip PSD composite preview PNG generation",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Also run external fix.py post-processing (default: off, built-in v5.1 handles it)",
    )
    parser.add_argument(
        "--no-fix",
        action="store_true",
        help="Disable external fix.py (default)",
    )
    parser.add_argument(
        "--order",
        default="auto",
        help="Sort mode for external fix.py (auto|depth|raw|classic|arms_front|arms_back|hair_over_top)",
    )
    parser.add_argument(
        "--group-offload",
        action="store_true",
        help="Enable group offload (slower, lower VRAM)",
    )
    args = parser.parse_args(argv)
    if args.no_fix:
        args.fix = False
    return args


def query_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.free", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
        line = out.strip().splitlines()[0]
        name, free_str = [x.strip() for x in line.split(",", 1)]
        return name, int(float(free_str))
    except Exception:
        return None, None


def estimate_vram_mib(res: int) -> int:
    a = 1024 * 1024
    b = 1280 * 1280
    v = 6554 + (7803 - 6554) * ((res * res - a) / (b - a))
    return max(6554, int(v))


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)

    here = Path(__file__).resolve().parent
    repo = here.parent
    target = Path(os.environ.get("TARGET", str(here / "target"))).resolve()
    outroot = Path(os.environ.get("OUTROOT", str(here / "output"))).resolve()
    logroot = Path(os.environ.get("LOGROOT", str(here / "log"))).resolve()
    fix_py = repo / "fix.py"
    infer_script = repo / "inference" / "scripts" / "inference_psd_quantized.py"

    print("=" * 68)
    print(f" see-through Batch PSD   {args.resolution}px / NF4 / {args.steps} steps / Windows 11")
    print("=" * 68)

    # 1. Environment Check
    print(HR)
    print("[1/4] Environment Check")
    print(f"[ OK ] Python: {sys.executable}")
    print(f"[ OK ] Repo  : {repo}")

    if not infer_script.is_file():
        print(f"[FAIL] Inference script not found: {infer_script}")
        return 1

    if not target.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        print(f"[FAIL] target folder did not exist and was created: {target}")
        print("       Put input images into target\\ and run again.")
        return 1

    gpu_name, free_mib = query_gpu()
    if gpu_name is None:
        print("[FAIL] nvidia-smi unavailable or no NVIDIA GPU detected.")
        return 1

    need_mib = estimate_vram_mib(args.resolution)
    print(f"[ OK ] GPU: {gpu_name} (Free VRAM: {free_mib} MiB)")
    print(f"[INFO] Estimated peak VRAM at {args.resolution}px: {need_mib} MiB / Free: {free_mib} MiB")
    if free_mib < need_mib:
        print("[WARN] Free VRAM is below estimated peak; CUDA OOM may occur.")

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    default_hub = Path.home() / ".cache" / "huggingface" / "hub"
    hub_dir = Path(env.get("HUGGINGFACE_HUB_CACHE", str(default_hub)))
    if hub_dir.is_dir() and any(hub_dir.iterdir()):
        env["HUGGINGFACE_HUB_CACHE"] = str(hub_dir)
        env.setdefault("HF_HUB_OFFLINE", "1")
        print(f"[ OK ] HF Cache: {hub_dir} (OFFLINE={env['HF_HUB_OFFLINE']})")
    else:
        print(f"[WARN] Local HF cache not found; models will download via {env['HF_ENDPOINT']}")

    # 2. Collect inputs & copy to ASCII t01/t02...
    print(HR)
    print("[2/4] Collecting Input Images (ASCII copy)")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_batch = outroot / ts
    log_batch = logroot / ts
    inputs_dir = log_batch / "inputs"
    fix_dir = out_batch / "fixed"

    out_batch.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    if args.fix:
        fix_dir.mkdir(parents=True, exist_ok=True)

    srcs = sorted(
        p for p in target.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    if not srcs:
        print(f"[FAIL] No images found in: {target}")
        return 1

    rows = []
    map_lines = []
    for i, src in enumerate(srcs, 1):
        ext = src.suffix.lower()
        asc_name = f"t{i:02d}{ext}"
        dst = inputs_dir / asc_name
        shutil.copy2(src, dst)
        sz_kb = max(1, src.stat().st_size // 1024)
        rows.append((asc_name, src.name, f"{sz_kb}KB"))
        map_lines.append(f"{asc_name}\t{src}\t{sz_kb}KB\n")

    (inputs_dir / "map.tsv").write_text("".join(map_lines), encoding="utf-8")
    print(f"[ OK ] {len(rows)} image(s) -> ASCII copies in {inputs_dir}")
    for asc_name, orig_name, sz_str in rows:
        safe_orig = orig_name.encode("ascii", "replace").decode("ascii")
        print(f"       {asc_name}  ({sz_str})  <-  {safe_orig}")

    # 3. Per-image isolated inference
    print(HR)
    print("[3/4] Running Per-Image Inference")

    timings = []
    failed = []
    fix_ok, fix_ng = 0, 0
    t_all0 = time.time()

    for idx, (asc_name, _orig_name, sz_str) in enumerate(rows, 1):
        stem = Path(asc_name).stem
        src_p = inputs_dir / asc_name
        log_p = log_batch / f"{stem}.log"

        print()
        print(HR)
        print(f"[{idx}/{len(rows)}] {stem}  ({sz_str})")
        print(HR, flush=True)

        cmd = [
            sys.executable,
            str(infer_script),
            "--srcp", str(src_p),
            "--save_dir", str(out_batch),
            "--save_to_psd",
            "--resolution", str(args.resolution),
            "--resolution_depth", str(args.depth_res),
            "--quant_mode", "nf4",
            "--num_inference_steps", str(args.steps),
            "--seed", str(args.seed),
        ]
        if not args.group_offload:
            cmd.append("--no_group_offload")

        t0 = time.time()
        oom = False
        with log_p.open("w", encoding="utf-8", errors="replace") as lf:
            proc = subprocess.Popen(
                cmd,
                cwd=str(repo),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                lf.write(line)
                if OOM_RE.search(line):
                    oom = True
            rc = proc.wait()
        dur = int(time.time() - t0)

        if rc != 0:
            why = "VRAM_OOM" if oom else "NON_VRAM_ERR"
            print(f"[FAIL] {stem} failed (rc={rc} / {why} / {dur}s) -> {log_p}", flush=True)
            failed.append(f"{stem}:{why}")
            timings.append((stem, dur, f"FAIL({why})", ""))
            if oom:
                print("[ABORT] CUDA OOM detected; stopping remaining images.", flush=True)
                break
            continue

        seg = ""
        jp = out_batch / f"{stem}.psd.json"
        if jp.is_file():
            try:
                txt = jp.read_text(encoding="utf-8", errors="ignore")
                seg = " ".join(re.findall(r'"[a-z_]*time[a-z_]*": *[0-9.]+', txt))
            except Exception:
                pass
        print(f"[DONE] {stem}  {dur}s  {seg}", flush=True)
        timings.append((stem, dur, "OK", seg))

        if args.fix:
            if fix_py.is_file():
                fx_log = log_batch / f"fix_{stem}.log"
                fx_out = fix_dir / f"{stem}_fixed.psd"
                fx_cmd = [
                    sys.executable,
                    str(fix_py),
                    str(out_batch / f"{stem}.psd"),
                    "-o", str(fx_out),
                    "--preview",
                    "--order", args.order,
                ]
                with fx_log.open("w", encoding="utf-8", errors="replace") as fl:
                    frc = subprocess.call(fx_cmd, cwd=str(repo), env=env, stdout=fl, stderr=subprocess.STDOUT)
                if frc == 0:
                    fix_ok += 1
                    print(f"[FIX ] {stem} -> fixed\\{stem}_fixed.psd", flush=True)
                else:
                    fix_ng += 1
                    print(f"[WARN] {stem} fix.py failed (see {fx_log}); raw PSD untouched.", flush=True)
            else:
                print(f"[WARN] fix.py not found: {fix_py}; skipping post-fix.")
                args.fix = False

    t_all = int(time.time() - t_all0)

    # 4. Previews & Summary
    print()
    print(HR)
    print("[4/4] Preview & Summary")

    if not args.no_preview:
        try:
            from psd_tools import PSDImage
            for p_str in sorted(glob.glob(str(out_batch / "*.psd"))):
                if p_str.endswith("_depth.psd"):
                    continue
                p = Path(p_str)
                try:
                    img = PSDImage.open(str(p)).composite()
                    out_png = p.with_name(p.stem + "_preview.png")
                    img.save(str(out_png))
                    print(f"  {out_png.name:28s} {img.size[0]}x{img.size[1]}  {out_png.stat().st_size // 1024} KB")
                except Exception as exc:
                    print(f"  [WARN] {p.name} preview failed: {exc}")
        except Exception as exc:
            print(f"  [WARN] psd_tools unavailable, skipping previews: {exc}")

    tsv_lines = [f"{a}\t{b}\t{c}\t{d}\n" for a, b, c, d in timings]
    (log_batch / "timings.tsv").write_text("".join(tsv_lines), encoding="utf-8")

    sum_lines = [
        f"Run time  : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Resolution: {args.resolution}px   Depth: {args.depth_res}px   Images: {len(rows)}",
        f"Total time: {t_all // 60}m {t_all % 60}s",
        "",
        f"{'Image':<8s} {'Time':<8s} {'Status':<18s} Details",
    ]
    for a, b, c, d in timings:
        sum_lines.append(f"{a:<8s} {str(b) + 's':<8s} {c:<18s} {d}")
    sum_txt = "\n".join(sum_lines) + "\n"
    (log_batch / "summary.txt").write_text(sum_txt, encoding="utf-8")
    print()
    print(sum_txt)

    print(HR)
    print(f"Output: {out_batch}")
    print(f"Logs  : {log_batch}")
    if args.fix:
        print(f"fix.py: {fix_ok} OK, {fix_ng} FAIL")
    if failed:
        print()
        print(f"[WARN] Failed ({len(failed)}): " + " ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())