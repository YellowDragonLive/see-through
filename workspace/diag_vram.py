"""一次性诊断：复现官方 build_layerdiff_pipeline 的显存曲线，定位 NF4 路径的失败点。

用法：
    C:\\Users\\13410\\.conda\\envs\\see_through\\python.exe workspace/diag_vram.py [cpu_offload]
不属于交付脚本，仅用于排查。
"""
import os
import sys
import torch

ROOT = r'X:\ai_tool\see-through'
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, 'common'))

REPO = '24yearsold/seethroughv0.0.2_layerdiff3d_nf4'
CPU_OFFLOAD = len(sys.argv) > 1 and sys.argv[1] == 'cpu_offload'
NO_GROUP = len(sys.argv) > 1 and sys.argv[1] == 'nogroup'


def mem(tag):
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated() / 1e6
    reserved = torch.cuda.memory_reserved() / 1e6
    print(f'[MEM] {tag:<28} free={free/1e6:8.1f}MB alloc={alloc:8.1f}MB reserved={reserved:8.1f}MB',
          flush=True)


from modules.layerdiffuse.vae import TransparentVAE
from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel
from modules.layerdiffuse.diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline

mem('start')
unet = UNetFrameConditionModel.from_pretrained(REPO, subfolder='unet')
mem('after unet')
trans_vae = TransparentVAE.from_pretrained(REPO, subfolder='trans_vae')
mem('after trans_vae')
pipe = KDiffusionStableDiffusionXLPipeline.from_pretrained(
    REPO, trans_vae=trans_vae, unet=unet, scheduler=None)
mem('after full pipeline')

if CPU_OFFLOAD:
    pipe.vae.to(dtype=torch.bfloat16)
    pipe.trans_vae.to(dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    mem('after cpu_offload')
    print('offload_device =', getattr(pipe, '_offload_device', None))
    print('text_encoder.device =', pipe.text_encoder.device)
    print('text_encoder hook =', getattr(getattr(pipe.text_encoder, '_hf_hook', None), 'execution_device', None))
    print('unet.device =', pipe.unet.device)
else:
    pipe.vae.to(dtype=torch.bfloat16, device='cuda')
    pipe.trans_vae.to(dtype=torch.bfloat16, device='cuda')
    mem('after vae move')
    if NO_GROUP:
        mem('after group offload (SKIPPED)')
    else:
        pipe.enable_group_offload('cuda', num_blocks_per_group=1)
        mem('after group offload')

try:
    pipe.cache_tag_embeds()
    mem('after cache_tag_embeds OK')
    print('cached tags =', len(pipe._cached_prompt_embeds))
except Exception as e:
    mem('at ERROR')
    print('ERROR:', type(e).__name__, e)
