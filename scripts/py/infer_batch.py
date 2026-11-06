# -*- coding: utf-8 -*-
"""批量推理：单个 Python 进程、单线程串联多张图，模型资源只加载一次。

与「循环调 infer_psd.cmd」的本质区别：
  * 循环调 = 每张图都重新起进程、重新 import torch、重新读 5.6GB NF4 权重；
  * 本脚本 = 一次 build LayerDiff -> 逐张 body/head -> 卸载
             -> 一次 build Marigold -> 逐张深度 -> 卸载
             -> 逐张组装 PSD 并复制到目标路径。

两个大模型从不同时驻留显存，峰值与单张一致；固定开销只付一次。
不修改官方仓库任何文件，通过 importlib 加载官方推理模块并复用其函数。

输入图统一复制成 001.png / 002.png ... 短名：用户文件名很长，
官方代码会把图片名再套两层目录，直接用原名容易撞 Windows MAX_PATH。
"""
import time
_T0 = time.time()

import argparse
import gc
import importlib.util
import json
import os
import os.path as osp
import shutil
import sys
from types import SimpleNamespace

import torch

REPO_LAYERDIFF = '24yearsold/seethroughv0.0.2_layerdiff3d_nf4'
REPO_DEPTH = '24yearsold/seethroughv0.0.1_marigold_nf4'


def load_official(repo_root):
    """加载官方 inference_psd_quantized.py，补齐 modules / utils 的搜索路径。"""
    for p in (osp.join(repo_root, 'common'),
              osp.join(repo_root, 'inference', 'scripts')):
        if p not in sys.path:
            sys.path.insert(0, p)
    target = osp.join(repo_root, 'inference', 'scripts', 'inference_psd_quantized.py')
    if not osp.isfile(target):
        raise RuntimeError('官方推理脚本缺失：' + target)
    spec = importlib.util.spec_from_file_location('st_infer_psd_quantized', target)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def say(msg):
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser(description='see-through 单进程批量推理')
    ap.add_argument('--manifest', required=True, help='UTF-8 JSON，含 items:[{src,dst,name}]')
    ap.add_argument('--repo-root', required=True)
    ap.add_argument('--work-dir', required=True, help='本次批量中间产物目录')
    ap.add_argument('--report', required=True, help='耗时报告输出路径')
    ap.add_argument('--resolution', type=int, default=768)
    ap.add_argument('--depth-resolution', type=int, default=720)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu-offload', action='store_true')
    ap.add_argument('--group-offload', action='store_true')
    ap.add_argument('--tblr-split', action='store_true')
    args = ap.parse_args()

    with open(args.manifest, 'r', encoding='utf-8-sig') as f:
        manifest = json.load(f)
    items = manifest['items']
    if not items:
        raise RuntimeError('manifest 里没有图片')

    repo_root = osp.abspath(args.repo_root)
    run_dir = osp.abspath(args.work_dir)
    inputs_dir = osp.join(run_dir, 'inputs')
    work_dir = osp.join(run_dir, 'work')
    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    for i, it in enumerate(items, 1):
        ext = osp.splitext(it['src'])[1].lower()
        if ext not in exts:
            ext = '.png'
        short = '%03d%s' % (i, ext)
        it['short'] = short
        it['stem'] = '%03d' % i
        it['short_path'] = osp.join(inputs_dir, short)
        shutil.copy2(it['src'], it['short_path'])
        os.makedirs(osp.dirname(it['dst']), exist_ok=True)
        for k in ('layerdiff_s', 'marigold_s', 'psd_s'):
            it[k] = None
        it['status'] = 'pending'
        it['error'] = ''

    say('=' * 64)
    say('批量推理：单进程 / 单线程 / %d 张串联 / 模型只加载一次' % len(items))
    say('  分辨率 %d，depth %d，步数 %d，seed %d' %
        (args.resolution, args.depth_resolution, args.steps, args.seed))
    say('=' * 64)

    startup_s = time.time() - _T0
    st = load_official(repo_root)
    qargs = SimpleNamespace(
        quant_mode='nf4',
        repo_id_layerdiff=REPO_LAYERDIFF,
        repo_id_depth=REPO_DEPTH,
        cpu_offload=args.cpu_offload,
        group_offload=args.group_offload,
    )
    torch.cuda.reset_peak_memory_stats()
    total_t0 = time.time()

    # ---------- 阶段 A：一次加载 LayerDiff，逐张跑 body + head ----------
    say('\n[A] 加载 LayerDiff3D（本次批量只加载这一次）...')
    t0 = time.time()
    layerdiff_pipe = st.build_layerdiff_pipeline(qargs)
    build_ld_s = time.time() - t0
    say('    LayerDiff3D 加载完成 %.1f 秒' % build_ld_s)

    for it in items:
        say('\n[A] (%s) %s' % (it['stem'], it['name']))
        t0 = time.time()
        try:
            st.run_layerdiff(layerdiff_pipe, it['short_path'], work_dir,
                             args.seed, args.steps, args.resolution)
            it['layerdiff_s'] = round(time.time() - t0, 2)
            it['status'] = 'layerdiff_ok'
            say('    body+head 完成 %.1f 秒' % it['layerdiff_s'])
        except Exception as e:
            it['error'] = '%s: %s' % (type(e).__name__, e)
            it['status'] = 'failed'
            say('    失败：%s' % it['error'])
            if 'out of memory' in it['error'].lower():
                say('    检测到显存不足，停止后续图片。')
                break
        finally:
            cleanup()

    del layerdiff_pipe
    cleanup()

    # ---------- 阶段 B：卸载 LayerDiff，一次加载 Marigold，逐张跑深度 ----------
    say('\n[B] 卸载 LayerDiff3D，加载 Marigold（本次批量只加载这一次）...')
    t0 = time.time()
    marigold_pipe = st.build_marigold_pipeline(qargs)
    build_mg_s = time.time() - t0
    say('    Marigold 加载完成 %.1f 秒' % build_mg_s)

    for it in items:
        if it['status'] != 'layerdiff_ok':
            continue
        t0 = time.time()
        try:
            st.run_marigold(marigold_pipe, it['short_path'], work_dir,
                            args.seed, args.depth_resolution)
            it['marigold_s'] = round(time.time() - t0, 2)
            it['status'] = 'marigold_ok'
            say('    [%s] 深度完成 %.1f 秒' % (it['stem'], it['marigold_s']))
        except Exception as e:
            it['error'] = '%s: %s' % (type(e).__name__, e)
            it['status'] = 'failed'
            say('    [%s] 深度失败：%s' % (it['stem'], it['error']))
            if 'out of memory' in it['error'].lower():
                break
        finally:
            cleanup()

    del marigold_pipe
    cleanup()

    # ---------- 阶段 C：逐张组装 PSD 并复制到目标路径 ----------
    say('\n[C] 组装 PSD 并输出...')
    for it in items:
        if it['status'] != 'marigold_ok':
            continue
        t0 = time.time()
        try:
            saved = osp.join(work_dir, it['stem'])
            st.further_extr(saved, rotate=False, save_to_psd=True,
                            tblr_split=args.tblr_split)
            psd = osp.join(work_dir, it['stem'] + '.psd')
            if not osp.isfile(psd):
                raise RuntimeError('组装后找不到 PSD：' + psd)
            shutil.copy2(psd, it['dst'])
            it['psd_s'] = round(time.time() - t0, 2)
            it['size_mb'] = round(osp.getsize(it['dst']) / 1024 ** 2, 2)
            it['status'] = 'ok'
            say('    [%s] %s (%.1f MB)' % (it['stem'], it['dst'], it['size_mb']))
        except Exception as e:
            it['error'] = '%s: %s' % (type(e).__name__, e)
            it['status'] = 'failed'
            say('    [%s] 组装失败：%s' % (it['stem'], it['error']))
        finally:
            cleanup()

    total_s = time.time() - total_t0
    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3

    ok = sum(1 for it in items if it['status'] == 'ok')
    failed = len(items) - ok
    report = {
        'mode': 'single_process_sequential',
        'count': len(items),
        'ok': ok,
        'failed': failed,
        'resolution': args.resolution,
        'depth_resolution': args.depth_resolution,
        'steps': args.steps,
        'startup_s': round(startup_s, 2),
        'build_layerdiff_s': round(build_ld_s, 2),
        'build_marigold_s': round(build_mg_s, 2),
        'total_s': round(total_s, 2),
        'peak_vram_gb': round(peak_gb, 3),
        'items': [{k: it[k] for k in ('stem', 'name', 'src', 'dst', 'status',
                                      'layerdiff_s', 'marigold_s', 'psd_s',
                                      'size_mb', 'error') if k in it}
                  for it in items],
    }
    with open(args.report, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    say('\n' + '=' * 64)
    say('批量完成：%d/%d 成功，总耗时 %.1f 秒，峰值显存 %.2f GB' %
        (ok, len(items), total_s, peak_gb))
    say('  进程启动+import %.1f 秒 / LayerDiff 加载 %.1f 秒 / Marigold 加载 %.1f 秒' %
        (startup_s, build_ld_s, build_mg_s))
    say('  报告：%s' % args.report)
    say('=' * 64)
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
