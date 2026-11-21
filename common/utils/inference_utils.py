import os
import os.path as osp
import gc

from modules.layerdiffuse.diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline, UNetFrameConditionModel
from modules.layerdiffuse.vae import TransparentVAE
from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel
from modules.marigold import MarigoldDepthPipeline
from utils.cv import center_square_pad_resize, img_alpha_blending, pad_rgb, smart_resize, validate_resolution
from utils.torch_utils import seed_everything
from utils.io_utils import json2dict, dict2json, load_parts, save_tmp_img, load_part, save_psd
from utils.torchcv import cluster_inpaint_part

from psd_tools import PSDImage
from safetensors.torch import load_file
import cv2
import numpy as np
import torch
from PIL import Image

VALID_BODY_PARTS_V2 = [
    'hair', 'headwear', 'face', 'eyes', 'eyewear', 'ears', 'earwear', 'nose', 'mouth', 
    'neck', 'neckwear', 'topwear', 'handwear', 'bottomwear', 'legwear', 'footwear', 
    'tail', 'wings', 'objects'
]


layerdiff_pipeline: KDiffusionStableDiffusionXLPipeline = None
def apply_layerdiff(
    imgp: str, pretrained: str, num_inference_steps=30, seed=0, save_dir='workspace/layerdiff_output', target_tag_list=VALID_BODY_PARTS_V2, 
    resolution=1280, vae_ckpt=None, unet_ckpt=None, disable_progressbar=False, cache_tag_embeds=True, group_offload=False):
    
    global layerdiff_pipeline
    if layerdiff_pipeline is None:
        trans_vae = TransparentVAE.from_pretrained(pretrained, subfolder='trans_vae')
        if unet_ckpt is None:
            unet = UNetFrameConditionModel.from_pretrained(pretrained, subfolder='unet')
        else:
            print(f'load unet from {unet_ckpt}')
            unet = UNetFrameConditionModel.from_pretrained(unet_ckpt)
        layerdiff_pipeline = KDiffusionStableDiffusionXLPipeline.from_pretrained(
            pretrained,
            trans_vae=trans_vae, unet=unet,
            scheduler=None
        )

        if vae_ckpt is not None:
            td_sd = {}
            vae_sd = {}
            sd = load_file(vae_ckpt)
            for k, v in sd.items():
                if k.startswith('trans_decoder.'):
                    td_sd[k.lstrip('trans_decoder.')] = v
                elif k.startswith('vae.'):
                    vae_sd[k.replace('vae.', '')] = v

            if len(vae_sd) > 0:
                layerdiff_pipeline.vae.load_state_dict(vae_sd)
                print(f'load vae from {vae_ckpt}')

            if len(td_sd) > 0:
                layerdiff_pipeline.trans_vae.decoder.load_state_dict(td_sd)
                print(f'load vae from {vae_ckpt}')

        layerdiff_pipeline.vae.to(dtype=torch.bfloat16, device='cuda')
        layerdiff_pipeline.trans_vae.to(dtype=torch.bfloat16, device='cuda')
        layerdiff_pipeline.unet.to(dtype=torch.bfloat16, device='cuda')
        layerdiff_pipeline.text_encoder.to(dtype=torch.bfloat16, device='cuda')
        layerdiff_pipeline.text_encoder_2.to(dtype=torch.bfloat16, device='cuda')
        if group_offload:
            layerdiff_pipeline.enable_group_offload('cuda', num_blocks_per_group=1)

    pipeline = layerdiff_pipeline
    if cache_tag_embeds:
        pipeline.cache_tag_embeds()
    pipeline.set_progress_bar_config(disable=disable_progressbar)

    saved = osp.join(save_dir, osp.splitext(osp.basename(imgp))[0])
    os.makedirs(saved, exist_ok=True)
    input_img = np.array(Image.open(imgp).convert('RGBA'))
    fullpage, pad_size, pad_pos = center_square_pad_resize(input_img, resolution, return_pad_info=True)
    scale = pad_size[0] / resolution
    Image.fromarray(fullpage).save(osp.join(saved, 'src_img.png'))

    rng = torch.Generator(device=pipeline.unet.device).manual_seed(seed)

    tag_version = pipeline.unet.get_tag_version()
    if tag_version == 'v2':
        pipeline_output = pipeline(
            strength=1.0,
            num_inference_steps=num_inference_steps,
            batch_size=1,
            generator=rng,
            guidance_scale=1.0,
            prompt=target_tag_list,
            negative_prompt='',
            fullpage=fullpage
        )
        images = pipeline_output.images
        for rst, tag in zip(images, target_tag_list):
            savename = osp.join(saved, f'{tag}.png')
            Image.fromarray(rst).save(savename)

    elif tag_version == 'v3':

        def _crop_head(img, xywh):
            x, y, w, h = xywh
            ih, iw = img.shape[:2]
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            if w < iw // 2:
                px = min(iw - x - w, x, w // 5)
                x1 = min(max(x - px, 0), iw)
                x2 = min(max(x + w + px, 0), iw)
            if h < ih // 2:
                py = min(ih - y - h, y, h // 5)
                y2 = min(max(y + h + py, 0), ih)
                y1 = min(max(y - py, 0), ih)

            return img[y1: y2, x1: x2], (x1, y1, x2, y2)

        body_tag_list = ['front hair', 'back hair', 'head', 'neck', 'neckwear', 'topwear', 'handwear', 'bottomwear', 'legwear', 'footwear', 'tail', 'wings', 'objects']
        pipeline_output = pipeline(
            strength=1.0,
            num_inference_steps=num_inference_steps,
            batch_size=1,
            generator=rng,
            guidance_scale=1.0,
            prompt=body_tag_list,
            negative_prompt='',
            fullpage=fullpage,
            group_index=0
        )
        images = pipeline_output.images
        for rst, tag in zip(pipeline_output.images, body_tag_list):
            savename = osp.join(saved, f'{tag}.png')
            Image.fromarray(rst).save(savename)
        head_img = images[2]
        # head_img = np.array(Image.open(osp.join(saved, 'head.png')))

        head_tag_list = ['headwear', 'face', 'irides', 'eyebrow', 'eyewhite', 'eyelash', 'eyewear', 'ears', 'earwear', 'nose', 'mouth']
        head_mask = (head_img[..., -1] > 15).astype(np.uint8)
        if not np.any(head_mask):
            raise ValueError(
                'Cannot extract head layers: generated head mask is empty '
                f'(head image shape={head_img.shape}, source image shape={input_img.shape}, '
                f'output directory={saved}).'
            )

        hx0, hy0, hw, hh = cv2.boundingRect(cv2.findNonZero(head_mask))
        
        hx = int(hx0 * scale) - pad_pos[0]
        hy = int(hy0 * scale) - pad_pos[1]
        hw = int(hw * scale)
        hh = int(hh * scale)
        src_h, src_w = input_img.shape[:2]
        hx2 = min(hx + hw, src_w)
        hy2 = min(hy + hh, src_h)
        hx = max(hx, 0)
        hy = max(hy, 0)
        if hx2 <= hx or hy2 <= hy:
            raise ValueError(
                'Cannot extract head layers: generated head bounding box does not intersect '
                f'the source image (bbox={(hx, hy, hx2, hy2)}, '
                f'source image size={(src_w, src_h)}, output directory={saved}).'
            )

        input_head, (hx1, hy1, hx2, hy2) = _crop_head(input_img, [hx, hy, hx2 - hx, hy2 - hy])
        if input_head.size == 0 or input_head.shape[0] == 0 or input_head.shape[1] == 0:
            raise ValueError(
                'Cannot extract head layers: clipped head crop is empty '
                f'(bbox={(hx1, hy1, hx2, hy2)}, source image size={(src_w, src_h)}, '
                f'output directory={saved}).'
            )
        hx1 = int(hx1 / scale + pad_pos[0] / scale)
        hy1 = int(hy1 / scale + pad_pos[1] / scale)
        ih, iw = input_head.shape[:2]
        input_head, pad_size, pad_pos = center_square_pad_resize(input_head, resolution, return_pad_info=True)

        Image.fromarray(input_head).save(osp.join(saved, 'src_head.png'))

        pipeline_output = pipeline(
            strength=1.0,
            num_inference_steps=num_inference_steps,
            batch_size=1,
            generator=rng,
            guidance_scale=1.0,
            prompt=head_tag_list,
            negative_prompt='',
            fullpage=input_head,
            group_index=1
        )
        canvas = np.zeros((resolution, resolution, 4), dtype=np.uint8)

        py1, py2, px1, px2 = (np.array([pad_pos[1], pad_pos[1] + ih, pad_pos[0], pad_pos[0] + iw]) / scale).astype(np.int64)
        
        scale_size = (int(pad_size[0] / scale), int(pad_size[1] / scale))

        for rst, tag in zip(pipeline_output.images, head_tag_list):
            rst = smart_resize(rst, scale_size)[py1: py2, px1: px2]
            full = canvas.copy()
            full[hy1: hy1 + rst.shape[0], hx1: hx1 + rst.shape[1]] = rst
            savename = osp.join(saved, f'{tag}.png')
            Image.fromarray(full).save(savename)

    else:
        raise




marigold_pipeline: MarigoldDepthPipeline = None
def apply_marigold(srcp, pretrained: str, num_inference_steps=-1, seed=0, save_dir='workspace/layerdiff_output', target_tag_list=VALID_BODY_PARTS_V2, \
    resolution=768, normalize_depth=False, disable_progressbar=False, cache_tag_embeds=True, group_offload=False):
    global marigold_pipeline
    if marigold_pipeline is None:
        unet = UNetFrameConditionModel.from_pretrained(pretrained, subfolder='unet')
        marigold_pipeline = MarigoldDepthPipeline.from_pretrained(pretrained, unet=unet)
        marigold_pipeline.to(device='cuda', dtype=torch.bfloat16)

        if group_offload:
            marigold_pipeline.enable_group_offload('cuda', num_blocks_per_group=1)

    pipe = marigold_pipeline

    pipe.set_progress_bar_config(disable=disable_progressbar)

    srcname = osp.basename(osp.splitext(srcp)[0])
    saved = osp.join(save_dir, srcname)

    src_img_p = osp.join(saved, 'src_img.png')
    fullpage = np.array(Image.open(src_img_p).convert('RGBA'))

    src_h, src_w = fullpage.shape[:2]
    if isinstance(resolution, int) and resolution == -1:
        resolution = [src_h, src_w]
    resolution = validate_resolution(resolution)
    src_rescaled = resolution[0] != src_h or resolution[1] != src_w

    img_list = []
    caption_list = []
    exist_list = []
    empty_array = np.zeros((src_h, src_w, 4), dtype=np.uint8)
    blended_alpha = np.zeros((src_h, src_w), dtype=np.float32)

    compose_list = {'eyes': ['eyewhite', 'irides', 'eyelash', 'eyebrow'], 'hair': ['back hair', 'front hair']}
    for tag in VALID_BODY_PARTS_V2:
        tagp = osp.join(saved, f'{tag}.png')
        if osp.exists(tagp):
            exist_list.append(True)
            caption_list.append(tag)
            tag_arr = np.array(Image.open(tagp))
            tag_arr[..., -1][tag_arr[..., -1] < 15] = 0
            # blended_alpha += tag_arr[..., -1].astype(np.float32) / 255
            img_list.append(tag_arr)
        else:
            img_list.append(empty_array)
            exist_list.append(False)

    compose_dict = {}
    for c, clist in compose_list.items():
        imlist = []
        taglist = []
        for tag in clist:
            p = osp.join(saved, tag + '.png')
            if osp.exists(p):
                tag_arr = np.array(Image.open(p))
                tag_arr[..., -1][tag_arr[..., -1] < 15] = 0
                imlist.append(tag_arr)
                taglist.append(tag)
        if len(imlist) > 0:
            img = img_alpha_blending(imlist, premultiplied=False)
            img_list[VALID_BODY_PARTS_V2.index(c)] = img
            compose_dict[c] = {'taglist': taglist, 'imlist': imlist}

    for img in img_list:
        blended_alpha += img[..., -1].astype(np.float32) / 255

    blended_alpha = np.clip(blended_alpha, 0, 1) * 255
    blended_alpha = blended_alpha.astype(np.uint8)
    fullpage[..., -1] = blended_alpha
    img_list.append(fullpage)

    img_list_input = img_list
    if src_rescaled:
        img_list_input = [smart_resize(img, resolution) for img in img_list]

    seed_everything(seed)
    pipe_out = pipe(
        # tensor2img(img, 'pil', denormalize=True, mean=127.5, std=127.5),
        color_map=None,
        img_list = img_list_input,
        denoising_steps=num_inference_steps
    )
    depth_pred: np.ndarray = pipe_out.depth_tensor
    
    depth_pred = depth_pred.to(device='cpu', dtype=torch.float32).numpy()
    if src_rescaled:
        depth_pred = [smart_resize(d, (src_h, src_w)) for d in depth_pred]
    drawables = [{'img': img, 'depth': depth} for img, depth in zip(img_list, depth_pred)]
    drawables = drawables[:-1]
    blended = img_alpha_blending(drawables, premultiplied=False)

    infop = osp.join(saved, 'info.json')
    if osp.exists(infop):
        info = json2dict(infop)
    else:
        info = {'parts': {}}

    parts = info['parts']
    for ii, depth in enumerate(depth_pred[:-1]):
        if normalize_depth:
            depth_max, depth_min = depth.max(), depth.min()
            depth = np.clip((depth - depth_min) / (depth_max - depth_min + 1e-7) * 255, 0, 255).astype(np.uint8)
        else:
            depth = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
        # depth = depth[..., None][..., [-1] * 3].copy()
        tag = VALID_BODY_PARTS_V2[ii]
        if tag in compose_dict:
            mask = blended_alpha > 256
            for t, im in zip(compose_dict[tag]['taglist'][::-1], compose_dict[tag]['imlist'][::-1]):
                mask_local = im[..., -1] > 15
                mask_invis = np.bitwise_and(mask, mask_local)
                depth_local = np.full((src_h, src_w), fill_value=255, dtype=np.uint8)
                depth_local[mask_local] = depth[mask_local]
                if np.any(mask_invis):
                    depth_local[mask_invis] = np.median(depth[np.bitwise_and(mask_local, np.bitwise_not(mask_invis))])
                mask = np.bitwise_or(mask, mask_local)

                parts_info = parts.get(t, {})
                savep = osp.join(saved, f'{t}_depth.png')
                Image.fromarray(depth_local).save(savep)
                parts[t] = parts_info
                if normalize_depth:
                    parts_info['depth_max'] = depth_max
                    parts_info['depth_min'] = depth_min
            continue

        parts_info = parts.get(tag, {})
        savep = osp.join(saved, f'{tag}_depth.png')
        Image.fromarray(depth).save(savep)
        parts[tag] = parts_info
        if normalize_depth:
            parts_info['depth_max'] = depth_max
            parts_info['depth_min'] = depth_min

    dict2json(info, infop)
    Image.fromarray(blended).save(osp.join(saved, 'reconstruction.png'))


def label_lr_split(labels, stats, id1, id2):
    label1 = (labels == id1).astype(np.uint8) * 255
    label2 = (labels == id2).astype(np.uint8) * 255

    stats1, stats2 = stats[id1], stats[id2]

    x1 = stats[id1][0] + stats[id1][2] / 2
    x2 = stats[id2][0] + stats[id2][2] / 2

    if x2 < x1:
        return label2, label1, stats2, stats1
    else:
        return label1, label2, stats1, stats2


def save_part(tag, saved, part_dict, crop=True, save_part_info=False, save_to_disk=True):
    img = part_dict.pop('img')
    if 'mask' in part_dict:
        part_dict.pop('mask')

    depth = part_dict.pop('depth')
    mask = img[..., -1] > 10
    depth_median = np.median(depth[mask])

    if crop:
        xywh = cv2.boundingRect(cv2.findNonZero(mask.astype(np.uint8)))
        xyxy = np.array(xywh).copy()
        xyxy[2] += xyxy[0]
        xyxy[3] += xyxy[1]
        depth = depth[xyxy[1]: xyxy[3], xyxy[0]: xyxy[2]]
        img = img[xyxy[1]: xyxy[3], xyxy[0]: xyxy[2]]

        x1, y1, x2, y2 = part_dict['xyxy']
        part_dict['xyxy'] = [x1 + xyxy[0], y1 + xyxy[1], x1 + xyxy[2], y1 + xyxy[3]]

    # dmin, dmax = np.min(depth), np.max(depth)
    depth = np.clip(depth, 0, 1) * 255
    depth = np.round(depth).astype(np.uint8)


    # part_dict['depth_min'] = dmin
    # part_dict['depth_max'] = dmax
    part_dict['depth_median'] = depth_median
    part_dict['tag'] = tag

    if save_to_disk:
        Image.fromarray(img).save(osp.join(saved, tag + '.png'))
        Image.fromarray(depth).save(osp.join(saved, tag + '_depth.png'))
        if save_part_info:
            dict2json(part_dict, osp.join(saved, tag + '.json'))
    else:
        part_dict['img'] = img
        part_dict['depth'] = depth 

    return part_dict


def process_cuts(img, depth, src_xyxy, tgt_bbox, p=5, mask=None):
    tx1, ty1, tx2, ty2 = tgt_bbox[:4]
    tx2 += tx1
    ty2 += ty1
    img = img[ty1: ty2, tx1: tx2].copy()
    depth = depth[ty1: ty2, tx1: tx2]
    
    depth_median = 1
    if mask is not None:
        mask = (mask[ty1: ty2, tx1: tx2].copy() > 15).astype(np.uint8)
        ksize = 1
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1),(ksize, ksize))
        mask = cv2.dilate(mask, element)
        img[..., -1] *= mask
        depth = 1 - (1-depth) * mask
        if np.any(mask):
            depth_median = np.median(depth[mask > 0])
    
    fxyxy = [tx1 + src_xyxy[0], ty1 + src_xyxy[1], tx2 + src_xyxy[0], ty2 + src_xyxy[1]]

    return img, depth, fxyxy, depth_median


def part_lr_split(tag, part_info):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        part_info['mask'].astype(np.uint8) * 255, connectivity=8)
    tag2pinfo = {}
    if len(stats) > 2:
        stats = np.array(stats)
        stats_order = np.argsort(stats[..., -1])[::-1][1:]
        arml_mask, armr_mask, statsl, statsr = label_lr_split(labels, stats, stats_order[0], stats_order[1])
        depth_median = part_info.get('depth_median', 1)

        img, depth, xyxy, depth_median = process_cuts(part_info['img'], part_info['depth'], part_info['xyxy'], statsl, mask=arml_mask)
        arml_mask = arml_mask[xyxy[1]: xyxy[3], xyxy[0]: xyxy[2]]
        tag2pinfo[f'{tag}-r'] = {'img': img, 'xyxy': xyxy, 'depth': depth, 'depth_median': depth_median, 'tag': f'{tag}-r'}

        img, depth, xyxy, depth_median = process_cuts(part_info['img'], part_info['depth'], part_info['xyxy'], statsr, mask=armr_mask)
        armr_mask = armr_mask[xyxy[1]: xyxy[3], xyxy[0]: xyxy[2]]
        tag2pinfo[f'{tag}-l'] = {'img': img, 'xyxy': xyxy, 'depth': depth, 'depth_median': depth_median, 'tag': f'{tag}-l'}

    else:
        tag2pinfo[tag] = part_info
    return tag2pinfo


def tag_lr_split(tag: str, tag2pinfo):
    if tag in tag2pinfo:
        part_info = tag2pinfo.pop(tag)
        tag2pinfo.update(part_lr_split(tag, part_info))




# ==============================================================================
# Plan C: Source-level CV Pipeline (Guided Filter Seam, Underlap Inpaint,
#         Ground-Truth src_img Residual Sorting, Edge Decontamination,
#         Mouth Halo Trim & Eyelash Skin-Fringe Cleanup)
# ==============================================================================

def _norm_part_role(tag: str) -> str:
    s = tag.strip().lower().replace('_', ' ').replace('-', ' ')
    parts = s.split()
    if len(parts) >= 2 and (parts[-1].isdigit() or parts[-1] in ('l', 'r')):
        s = ' '.join(parts[:-1])
    return s


def _part_to_full_canvas(pdict, H: int, W: int):
    x1, y1, x2, y2 = [int(v) for v in pdict['xyxy']]
    img = pdict['img']
    dep = pdict['depth']
    dep_f = (dep.astype(np.float32) / 255.0) if dep.dtype == np.uint8 else dep.astype(np.float32)
    full_img = np.zeros((H, W, 4), dtype=np.uint8)
    full_dep = np.ones((H, W), dtype=np.float32)
    lh, lw = img.shape[:2]
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(W, x1 + lw), min(H, y1 + lh)
    if cx2 > cx1 and cy2 > cy1:
        full_img[cy1:cy2, cx1:cx2] = img[cy1 - y1:cy2 - y1, cx1 - x1:cx2 - x1]
        full_dep[cy1:cy2, cx1:cx2] = dep_f[cy1 - y1:cy2 - y1, cx1 - x1:cx2 - x1]
    return full_img, full_dep


def _guided_filter_mask(guide_rgb: np.ndarray, mask_f: np.ndarray, radius: int = 6, eps: float = 1e-3) -> np.ndarray:
    """Fast pure-OpenCV Guided Filter (He et al.): snaps coarse depth cut masks to crisp RGB ink edges."""
    I = cv2.cvtColor(guide_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    p = mask_f.astype(np.float32)
    ksize = (2 * radius + 1, 2 * radius + 1)
    mean_I = cv2.boxFilter(I, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize)
    corr_I = cv2.boxFilter(I * I, cv2.CV_32F, ksize)
    corr_Ip = cv2.boxFilter(I * p, cv2.CV_32F, ksize)
    var_I = np.maximum(corr_I - mean_I * mean_I, 0.0)
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)
    q = np.clip(mean_a * I + mean_b, 0.0, 1.0)
    return q


def _extend_back_underlap(
    full_img: np.ndarray,
    back_alpha_scale: np.ndarray,
    front_bin: np.ndarray,
    seam_px: int = 8,
    use_lama: bool = False,
) -> tuple:
    """Extend the back sub-layer slightly underneath the front cut seam and inpaint RGB
    so 2.5D swaying/turning never exposes a transparent crack between front & back sub-layers."""
    orig_a = full_img[..., 3]
    back_img = full_img.copy()
    back_a = np.clip(np.round(orig_a.astype(np.float32) * back_alpha_scale), 0, 255).astype(np.uint8)
    back_solid = back_a > 128

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * seam_px + 1, 2 * seam_px + 1))
    # Dilate back_solid into front_bin (only inside original layer silhouette) to create an underlap lip
    underlap = (cv2.dilate(back_solid.astype(np.uint8), k) > 0) & front_bin & (orig_a > 24) & (~back_solid)
    if int(underlap.sum()) < 16:
        back_img[..., 3] = back_a
        return back_img

    # Crop to bounding box of (back_solid | underlap) for fast local inpainting
    union = back_solid | underlap
    ys, xs = np.nonzero(union)
    y1, y2 = max(0, int(ys.min()) - 4), min(full_img.shape[0], int(ys.max()) + 5)
    x1, x2 = max(0, int(xs.min()) - 4), min(full_img.shape[1], int(xs.max()) + 5)
    rgb_crop = full_img[y1:y2, x1:x2, :3].copy()
    hole_crop = (underlap[y1:y2, x1:x2].astype(np.uint8)) * 255

    inpainted_rgb = None
    if use_lama:
        try:
            from annotators.lama_inpainter import apply_inpaint
            inpainted_rgb = apply_inpaint(rgb_crop, hole_crop > 0)
        except Exception:
            inpainted_rgb = None
    if inpainted_rgb is None:
        inpainted_rgb = cv2.inpaint(rgb_crop, hole_crop, inpaintRadius=max(3, seam_px // 2), flags=cv2.INPAINT_TELEA)

    back_img[y1:y2, x1:x2, :3] = np.where(hole_crop[..., None] > 0, inpainted_rgb, rgb_crop)
    # Softly fade the underlap alpha so it stays cleanly tucked behind the front sub-layer
    under_soft = cv2.GaussianBlur(underlap.astype(np.float32), (0, 0), 1.5)
    ext_a = np.clip(np.round(orig_a.astype(np.float32) * np.clip(back_alpha_scale + under_soft * 0.95, 0.0, 1.0)), 0, 255).astype(np.uint8)
    back_img[..., 3] = np.maximum(back_a, ext_a)
    return back_img


def _make_split_part_from_rgba(tag: str, full_rgba: np.ndarray, full_dep: np.ndarray):
    mask = full_rgba[..., 3] > 10
    if not np.any(mask):
        return None
    ys, xs = np.nonzero(mask)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    sub_crop = full_rgba[y1:y2, x1:x2].copy()
    dep_crop = full_dep[y1:y2, x1:x2].copy()
    m_crop = sub_crop[..., 3] > 10
    return {
        'tag': tag,
        'img': sub_crop,
        'depth': dep_crop,
        'mask': m_crop,
        'xyxy': [x1, y1, x2, y2],
        'depth_median': float(np.median(dep_crop[m_crop])),
    }


def _decontaminate_edges(img: np.ndarray) -> np.ndarray:
    """Remove dark/background color bleeding on semi-transparent fringes using alpha-pyramid RGB padding."""
    if img is None or img.size == 0 or img.shape[2] != 4:
        return img
    a = img[..., 3]
    if not np.any((a > 0) & (a < 230)) or not np.any(a >= 230):
        return img
    try:
        # Build clean RGB from opaque core pixels (a >= 180) and apply only to semi-transparent fringe
        core = img.copy()
        core[..., 3] = np.where(a >= 180, a, 0)
        padded_rgb = pad_rgb(core, return_format='rgb', to_uint8=True, keep_ori_pixel=True)
        out = img.copy()
        fringe = (a > 0) & (a < 180)
        out[fringe, :3] = padded_rgb[fringe]
        return out
    except Exception:
        return img


def _depth_split_bimodal_parts(tag2pinfo: dict, frame_size, use_lama: bool = False):
    """Split handwear (disconnected & connected arms) and back hair along depth cliffs vs topwear,
    using RGB Guided Filter edge snapping + back-layer underlap inpainting."""
    H, W = int(frame_size[0]), int(frame_size[1])
    if 'topwear' not in tag2pinfo:
        return
    tw_img, tw_dep = _part_to_full_canvas(tag2pinfo['topwear'], H, W)
    tw_m = tw_img[..., 3] > 32
    tw_med = float(tag2pinfo['topwear'].get('depth_median', np.median(tw_dep[tw_m]) if np.any(tw_m) else 0.7))

    # 1. Split `handwear` into `handwear_1` (behind topwear) + `handwear_2` (in front of topwear)
    if 'handwear' in tag2pinfo:
        hw = tag2pinfo['handwear']
        hw_img, hw_dep = _part_to_full_canvas(hw, H, W)
        hw_m = hw_img[..., 3] > 24
        n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(hw_m.astype(np.uint8), connectivity=4)
        front_c, back_c = [], []
        for cid in range(1, n_lbl):
            if stats[cid, cv2.CC_STAT_AREA] < 200:
                continue
            ov = (labels == cid) & tw_m
            ov_n = int(ov.sum())
            if ov_n < 80:
                continue
            diff = hw_dep[ov] - tw_dep[ov]
            fn, bn = int((diff < -0.02).sum()), int((diff > 0.02).sum())
            if fn > ov_n * 0.55 and fn > bn * 2:
                front_c.append(cid)
            elif bn > ov_n * 0.55 and bn > fn * 2:
                back_c.append(cid)

        back_mask, front_mask, connected_split = None, None, False
        if front_c and back_c:
            back_mask = np.isin(labels, back_c)
            front_mask = (hw_img[..., 3] > 0) & (~back_mask)
        else:
            # Connected/touching arms (e.g. both hands holding an object in 1 component)
            ov = hw_m & tw_m
            ov_n = int(ov.sum())
            if ov_n >= 8000:
                diff = hw_dep - tw_dep
                f_ov = ov & (diff < -0.025)
                b_ov = ov & (diff > 0.025)
                fn, bn = int(f_ov.sum()), int(b_ov.sum())
                if fn >= 4000 and bn >= 4000 and (fn / ov_n) >= 0.18 and (bn / ov_n) >= 0.15:
                    med_f, med_b = float(np.median(hw_dep[f_ov])), float(np.median(hw_dep[b_ov]))
                    if abs(med_b - med_f) >= 0.15:
                        thresh = 0.5 * (med_f + med_b)
                        raw_b = ((hw_m & (hw_dep > thresh)) | b_ov) & (~f_ov)
                        nb, blab, bstats, _ = cv2.connectedComponentsWithStats(raw_b.astype(np.uint8), connectivity=4)
                        clean_b = np.isin(blab, [c for c in range(1, nb) if bstats[c, cv2.CC_STAT_AREA] >= 1500])
                        nf, flab, fstats, _ = cv2.connectedComponentsWithStats((hw_m & (~clean_b)).astype(np.uint8), connectivity=4)
                        clean_f = np.isin(flab, [c for c in range(1, nf) if fstats[c, cv2.CC_STAT_AREA] >= 1500])
                        clean_b = hw_m & (~clean_f)
                        if int(clean_b.sum()) >= 3000 and int(clean_f.sum()) >= 3000:
                            back_mask, front_mask, connected_split = clean_b, clean_f, True

        if back_mask is not None and front_mask is not None:
            if connected_split:
                # Snap cut seam to crisp RGB ink edges via Guided Filter + underlap inpaint on back arm
                soft_f = _guided_filter_mask(hw_img[..., :3], front_mask.astype(np.float32), radius=5, eps=1e-3)
                front_rgba = hw_img.copy()
                front_rgba[..., 3] = np.clip(np.round(hw_img[..., 3].astype(np.float32) * soft_f), 0, 255).astype(np.uint8)
                back_rgba = _extend_back_underlap(hw_img, 1.0 - soft_f, front_mask, seam_px=8, use_lama=use_lama)
            else:
                front_rgba = hw_img.copy()
                front_rgba[~front_mask, 3] = 0
                back_rgba = hw_img.copy()
                back_rgba[~back_mask, 3] = 0
            p_back = _make_split_part_from_rgba('handwear_1', back_rgba, hw_dep)
            p_front = _make_split_part_from_rgba('handwear_2', front_rgba, hw_dep)
            if p_back is not None and p_front is not None:
                p_back['depth_median'] = max(p_back['depth_median'], tw_med + 0.01)
                p_front['depth_median'] = min(p_front['depth_median'], tw_med - 0.01)
                tag2pinfo.pop('handwear')
                tag2pinfo['handwear_1'] = p_back
                tag2pinfo['handwear_2'] = p_front

    # 2. Split `back hair` when front-draped braids/locks sit in front of `topwear`
    if 'back hair' in tag2pinfo:
        bh = tag2pinfo['back hair']
        bh_img, bh_dep = _part_to_full_canvas(bh, H, W)
        bh_m = bh_img[..., 3] > 24
        ov = bh_m & tw_m
        ov_n = int(ov.sum())
        if ov_n >= 4000:
            diff = bh_dep - tw_dep
            f_ov = ov & (diff < -0.025)
            b_ov = ov & (diff > 0.025)
            fn, bn = int(f_ov.sum()), int(b_ov.sum())
            if fn >= 4000 and bn >= 4000 and (fn / ov_n) >= 0.22 and (bn / ov_n) >= 0.18:
                thresh = 0.5 * (float(np.median(bh_dep[f_ov])) + float(np.median(bh_dep[b_ov])))
                raw_f = ((bh_m & (bh_dep < thresh)) | f_ov) & (~b_ov)
                # Exclude neck region so hair behind the neck never jumps in front of neck/chin (fixes fix3)
                if 'neck' in tag2pinfo:
                    nk_img, _ = _part_to_full_canvas(tag2pinfo['neck'], H, W)
                    raw_f = raw_f & (~(nk_img[..., 3] > 32))
                k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                raw_f = cv2.morphologyEx(raw_f.astype(np.uint8), cv2.MORPH_OPEN, k_open) > 0
                nf, flab, fstats, _ = cv2.connectedComponentsWithStats(raw_f.astype(np.uint8), connectivity=4)
                keep_f = [c for c in range(1, nf) if fstats[c, cv2.CC_STAT_AREA] >= 2500]
                if keep_f:
                    clean_f = np.isin(flab, keep_f)
                    soft_f = _guided_filter_mask(bh_img[..., :3], clean_f.astype(np.float32), radius=6, eps=1e-3)
                    front_rgba = bh_img.copy()
                    front_rgba[..., 3] = np.clip(np.round(bh_img[..., 3].astype(np.float32) * soft_f), 0, 255).astype(np.uint8)
                    back_rgba = _extend_back_underlap(bh_img, 1.0 - soft_f, clean_f, seam_px=8, use_lama=use_lama)
                    p_back_h = _make_split_part_from_rgba('back hair_1', back_rgba, bh_dep)
                    p_front_h = _make_split_part_from_rgba('back hair_2', front_rgba, bh_dep)
                    if p_back_h is not None and p_front_h is not None:
                        p_back_h['depth_median'] = max(p_back_h['depth_median'], tw_med + 0.015)
                        p_front_h['depth_median'] = min(p_front_h['depth_median'], tw_med - 0.015)
                        if 'face' in tag2pinfo:
                            p_front_h['depth_median'] = max(p_front_h['depth_median'], float(tag2pinfo['face']['depth_median']) + 0.002)
                        tag2pinfo.pop('back hair')
                        tag2pinfo['back hair_1'] = p_back_h
                        tag2pinfo['back hair_2'] = p_front_h


def _clean_mouth_layer(mouth_dict: dict):
    """Strip isolated border bleed specks (e.g. nose shadow) and trim oversized skin-colored halo around mouth."""
    img = mouth_dict.get('img')
    if img is None or img.size == 0:
        return
    mh, mw = img.shape[:2]
    border = np.zeros((mh, mw), dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    b_ok = border & (img[..., 3] > 200)
    if int(b_ok.sum()) < 12:
        return
    skin = img[b_ok, :3].astype(np.float32).mean(axis=0) / 255.0
    skin_lum = float(0.299 * skin[0] + 0.587 * skin[1] + 0.114 * skin[2])
    rgb = img[..., :3].astype(np.float32) / 255.0
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    feat = (img[..., 3] > 48) & ((np.abs(rgb - skin).sum(axis=2) > 0.14) | ((skin_lum - lum) > 0.11))
    if int(feat.sum()) < 16:
        return

    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(feat.astype(np.uint8), connectivity=4)
    if n_lbl <= 1:
        return
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(n_lbl)]
    main_c = 1 + int(np.argmax(areas[1:]))
    ys_m = np.nonzero(labels == main_c)[0]
    main_cy = float(ys_m.mean())
    tol = max(10.0, float(ys_m.max() - ys_m.min() + 1) * 0.75)

    keep = np.zeros((mh, mw), dtype=np.uint8)
    for cid in range(1, n_lbl):
        ys_c = np.nonzero(labels == cid)[0]
        if cid == main_c or (areas[cid] >= max(6, areas[main_c] * 0.12) and abs(float(ys_c.mean()) - main_cy) <= tol):
            keep[labels == cid] = 255

    # Tightly trim giant skin halo while keeping a smooth 4px soft margin around actual mouth features
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    keep_closed = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, k_close)
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tight_halo = cv2.GaussianBlur(cv2.dilate(keep_closed, k_dil).astype(np.float32) / 255.0, (0, 0), 1.5)
    img[..., 3] = np.clip(np.round(img[..., 3].astype(np.float32) * tight_halo), 0, 255).astype(np.uint8)


def _clean_eyelash_skin_fringe(eyelash_dict: dict, face_dict: dict):
    """Attenuate low-contrast skin-colored fringe pixels accidentally baked into eyelash alpha."""
    img = eyelash_dict.get('img')
    fimg = face_dict.get('img') if face_dict else None
    if img is None or fimg is None or img.size == 0 or fimg.size == 0:
        return
    fm = fimg[..., 3] > 230
    if int(fm.sum()) < 50:
        return
    skin = np.median(fimg[fm, :3].astype(np.float32), axis=0) / 255.0
    rgb = img[..., :3].astype(np.float32) / 255.0
    diff = np.abs(rgb - skin).sum(axis=2)
    fringe_skin = (img[..., 3] > 0) & (img[..., 3] < 160) & (diff < 0.14)
    if np.any(fringe_skin):
        img[fringe_skin, 3] = (img[fringe_skin, 3].astype(np.float32) * 0.25).astype(np.uint8)




def _fill_head_interior_gaps(tag2pinfo: dict, frame_size):
    """Close enclosed interior transparent pockets between front hair, back hair, face and ears (fixes fix2)."""
    bh_key = 'back hair_1' if 'back hair_1' in tag2pinfo else ('back hair' if 'back hair' in tag2pinfo else None)
    if bh_key is None or 'face' not in tag2pinfo:
        return
    H, W = int(frame_size[0]), int(frame_size[1])
    bh_img, bh_dep = _part_to_full_canvas(tag2pinfo[bh_key], H, W)
    union = bh_img[..., 3] > 24
    for k in ('front hair', 'back hair_2', 'face', 'ears', 'headwear'):
        if k in tag2pinfo:
            ki, _ = _part_to_full_canvas(tag2pinfo[k], H, W)
            union = union | (ki[..., 3] > 24)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    closed = cv2.morphologyEx(union.astype(np.uint8), cv2.MORPH_CLOSE, k_close)
    ff = closed.copy()
    cv2.floodFill(ff, np.zeros((H + 2, W + 2), np.uint8), (0, 0), 2)
    bh_near = cv2.dilate((bh_img[..., 3] > 24).astype(np.uint8), k_close) > 0
    holes = ((ff == 0) | ((closed > 0) & (~union))) & (~union) & bh_near
    n_h = int(holes.sum())
    if n_h < 40 or n_h > 40000:
        return
    ys, xs = np.nonzero(holes | (bh_img[..., 3] > 24))
    y1, y2 = max(0, int(ys.min()) - 4), min(H, int(ys.max()) + 5)
    x1, x2 = max(0, int(xs.min()) - 4), min(W, int(xs.max()) + 5)
    rgb_crop = bh_img[y1:y2, x1:x2, :3].copy()
    hole_u8 = (holes[y1:y2, x1:x2].astype(np.uint8)) * 255
    inpainted = cv2.inpaint(rgb_crop, hole_u8, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    bh_img[y1:y2, x1:x2, :3] = np.where(hole_u8[..., None] > 0, inpainted, rgb_crop)
    hole_a = cv2.GaussianBlur(hole_u8.astype(np.float32), (0, 0), 1.0)
    bh_img[y1:y2, x1:x2, 3] = np.maximum(bh_img[y1:y2, x1:x2, 3], np.clip(np.round(hole_a), 0, 255).astype(np.uint8))
    updated = _make_split_part_from_rgba(bh_key, bh_img, bh_dep)
    if updated is not None:
        updated['depth_median'] = tag2pinfo[bh_key]['depth_median']
        tag2pinfo[bh_key] = updated


def _clean_and_split_objects(tag2pinfo: dict, frame_size):
    """Remove isolated floating debris in `objects` (fixes fix6 top-right speck) and split foreground props
    (e.g. laptop screen / coffee cup in front of hands) from background props using depth."""
    if 'objects' not in tag2pinfo:
        return
    H, W = int(frame_size[0]), int(frame_size[1])
    obj = tag2pinfo['objects']
    obj_img, obj_dep = _part_to_full_canvas(obj, H, W)
    obj_m = obj_img[..., 3] > 24
    if int(obj_m.sum()) < 50:
        return

    body_m = np.zeros((H, W), dtype=bool)
    for k, pd in tag2pinfo.items():
        if _norm_part_role(k) != 'objects':
            bi, _ = _part_to_full_canvas(pd, H, W)
            body_m |= (bi[..., 3] > 24)
    k_near = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    body_near = cv2.dilate(body_m.astype(np.uint8), k_near) > 0

    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(obj_m.astype(np.uint8), connectivity=8)
    if n_lbl > 2:
        areas = [int(stats[c, cv2.CC_STAT_AREA]) for c in range(n_lbl)]
        max_a = max(areas[1:])
        for c in range(1, n_lbl):
            if areas[c] < max(8000, int(max_a * 0.04)):
                cm = labels == c
                if not np.any(cm & body_near):
                    obj_img[cm, 3] = 0
                    obj_m[cm] = False

    hw_keys = [k for k in tag2pinfo if _norm_part_role(k) == 'handwear']
    if hw_keys:
        hw_m = np.zeros((H, W), dtype=bool)
        hw_d = np.ones((H, W), dtype=np.float32)
        for hk in hw_keys:
            hi, hd = _part_to_full_canvas(tag2pinfo[hk], H, W)
            hm = hi[..., 3] > 32
            hw_d = np.where(hm & (hd < hw_d), hd, hw_d)
            hw_m |= hm
        ov = obj_m & hw_m
        ov_n = int(ov.sum())
        if ov_n >= 1500:
            diff = obj_dep - hw_d
            f_ov = ov & (diff < -0.025)
            b_ov = ov & (diff > 0.025)
            if int(f_ov.sum()) >= 800 and int(b_ov.sum()) >= 800:
                med_f, med_b = float(np.median(obj_dep[f_ov])), float(np.median(obj_dep[b_ov]))
                if abs(med_b - med_f) >= 0.12:
                    thresh = 0.5 * (med_f + med_b)
                    raw_f = ((obj_m & (obj_dep < thresh)) | f_ov) & (~b_ov)
                    nf, flab, fstats, _ = cv2.connectedComponentsWithStats(raw_f.astype(np.uint8), connectivity=4)
                    clean_f = np.isin(flab, [c for c in range(1, nf) if fstats[c, cv2.CC_STAT_AREA] >= 1200])
                    clean_b = obj_m & (~clean_f)
                    if int(clean_f.sum()) >= 1500 and int(clean_b.sum()) >= 1500:
                        soft_f = _guided_filter_mask(obj_img[..., :3], clean_f.astype(np.float32), radius=5, eps=1e-3)
                        fg_rgba = obj_img.copy()
                        fg_rgba[..., 3] = np.clip(np.round(obj_img[..., 3].astype(np.float32) * soft_f), 0, 255).astype(np.uint8)
                        bg_rgba = _extend_back_underlap(obj_img, 1.0 - soft_f, clean_f, seam_px=6, use_lama=False)
                        p_bg = _make_split_part_from_rgba('objects_1', bg_rgba, obj_dep)
                        p_fg = _make_split_part_from_rgba('objects_2', fg_rgba, obj_dep)
                        if p_bg is not None and p_fg is not None:
                            hw_min = min(float(tag2pinfo[hk]['depth_median']) for hk in hw_keys)
                            p_fg['depth_median'] = min(p_fg['depth_median'], hw_min - 0.01)
                            p_bg['depth_median'] = max(p_bg['depth_median'], hw_min + 0.02)
                            tag2pinfo.pop('objects')
                            tag2pinfo['objects_1'] = p_bg
                            tag2pinfo['objects_2'] = p_fg
                            return

    updated = _make_split_part_from_rgba('objects', obj_img, obj_dep)
    if updated is not None:
        updated['depth_median'] = obj['depth_median']
        tag2pinfo['objects'] = updated


def _sort_parts_by_local_overlap(part_dict_list: list, frame_size, fullpage=None):
    """Ground-truth src_img residual verification + local overlap-region depth sorting,
    with strict anatomical face/eye/garment invariants."""
    H, W = int(frame_size[0]), int(frame_size[1])
    arr = sorted(part_dict_list, key=lambda x: float(x.get('depth_median', 1.0)), reverse=True)
    if not arr or 'img' not in arr[0] or 'depth' not in arr[0]:
        return arr

    full_rgb = {}
    full_m = {}
    full_d = {}
    for p in arr:
        fi, fd = _part_to_full_canvas(p, H, W)
        full_rgb[p['tag']] = fi[..., :3].astype(np.float32)
        full_m[p['tag']] = fi[..., 3] > 32
        full_d[p['tag']] = fd

    src_rgb = None
    if fullpage is not None and fullpage.shape[0] == H and fullpage.shape[1] == W:
        src_rgb = fullpage[..., :3].astype(np.float32)

    # 1. Pairwise ordering: prefer ground-truth src_img residual in A ∩ B, fall back to local pixel depth
    body_sortable = {
        'back hair', 'neck', 'topwear', 'bottomwear', 'legwear', 'footwear',
        'neckwear', 'handwear', 'headwear', 'objects', 'ears', 'earwear', 'tail', 'wings',
    }
    guard = len(arr) * len(arr)
    changed = True
    while changed and guard > 0:
        guard -= 1
        changed = False
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                A, B = arr[i], arr[j]
                rA, rB = _norm_part_role(A['tag']), _norm_part_role(B['tag'])
                allow_pair = (rA in body_sortable and rB in body_sortable) or (
                    'handwear' in (rA, rB) and ('face' in (rA, rB) or 'front hair' in (rA, rB))
                )
                if not allow_pair:
                    continue
                ov = full_m[A['tag']] & full_m[B['tag']]
                if int(ov.sum()) < 40:
                    continue
                a_on_top = None
                if src_rgb is not None:
                    # Ground-truth check: which layer's RGB matches the original image src_img better inside A ∩ B?
                    err_a = float(np.abs(full_rgb[A['tag']][ov] - src_rgb[ov]).mean())
                    err_b = float(np.abs(full_rgb[B['tag']][ov] - src_rgb[ov]).mean())
                    if abs(err_a - err_b) >= 6.0 and min(err_a, err_b) <= max(err_a, err_b) * 0.85:
                        a_on_top = err_a < err_b
                if a_on_top is None:
                    ma = float(full_d[A['tag']][ov].mean())
                    mb = float(full_d[B['tag']][ov].mean())
                    if abs(ma - mb) >= 0.025:
                        a_on_top = ma < mb
                if a_on_top:
                    arr.pop(i)
                    arr.insert(j, A)
                    changed = True
                    break
            if changed:
                break

    # 2. Enforce unambiguous garment order (legwear < bottomwear < topwear < neckwear)
    garment_behind = {
        'wings': {'back hair', 'neck', 'legwear', 'footwear', 'bottomwear', 'topwear', 'neckwear'},
        'tail': {'back hair', 'neck', 'legwear', 'footwear', 'bottomwear', 'topwear', 'neckwear'},
        'legwear': {'footwear', 'bottomwear'},
        'footwear': {'bottomwear'},
        'bottomwear': {'topwear'},
        'topwear': {'neckwear'},
    }
    guard = len(arr) * len(arr)
    changed = True
    while changed and guard > 0:
        guard -= 1
        changed = False
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                ra, rb = _norm_part_role(arr[i]['tag']), _norm_part_role(arr[j]['tag'])
                if ra in garment_behind.get(rb, set()) and int((full_m[arr[i]['tag']] & full_m[arr[j]['tag']]).sum()) > 12:
                    b = arr.pop(j)
                    arr.insert(i, b)
                    changed = True
                    break
            if changed:
                break

    # 3. Lock v3 anatomical face/eye stack: back hair/neck/ears < face < mouth/nose < eyewhite < irides < eyelash < front hair
    face_rank = {
        'facedetail': 1, 'nose': 1, 'mouth': 1,
        'eyewhite': 2, 'irides': 3, 'eyelash': 4, 'eyebrow': 4, 'eyewear': 5, 'headwear': 5,
    }
    above_face = {'front hair', 'side hair', 'ahoge'}
    below_face = {'back hair', 'neck', 'ears'}
    guard = len(arr) * len(arr)
    changed = True
    while changed and guard > 0:
        guard -= 1
        changed = False
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                ra, rb = _norm_part_role(arr[i]['tag']), _norm_part_role(arr[j]['tag'])
                need_swap = False
                if rb == 'face' and (ra in face_rank or ra in above_face):
                    need_swap = True
                elif ra == 'face' and rb in below_face:
                    need_swap = True
                elif ra in face_rank and rb in face_rank and face_rank[ra] > face_rank[rb] and not (ra == 'eyebrow' or rb == 'eyebrow'):
                    need_swap = True
                if need_swap:
                    if ra in above_face:
                        a = arr.pop(i)
                        last = max((k for k, x in enumerate(arr) if _norm_part_role(x['tag']) == 'face' or _norm_part_role(x['tag']) in face_rank), default=len(arr) - 1)
                        arr.insert(last + 1, a)
                    elif rb == 'face':
                        a = arr.pop(i)
                        arr.insert(j, a)
                    else:
                        b = arr.pop(j)
                        arr.insert(i, b)
                    changed = True
                    break
            if changed:
                break

    return arr


def further_extr(srcd: str, rotate=True, save_to_psd=False, tblr_split=True):


    saved = osp.join(srcd, 'optimized')
    # infos = json2dict(osp.join(srcd, 'info.json'))
    os.makedirs(saved, exist_ok=True)

    fullpage, infos, part_dict_list = load_parts(srcd, rotate=rotate)

    # optim_depth(part_dict_list, fullpage)

    tag2pinfo = {}
    for pinfo in part_dict_list:
        tag = pinfo['tag']
        tag2pinfo[tag] = pinfo

    if 'eyes' in tag2pinfo:
        part_info = tag2pinfo.pop('eyes')
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            part_info['mask'].astype(np.uint8) * 255, connectivity=8)
        if len(stats) > 2:
            stats = np.array(stats)
            
            if len(stats[..., -1]) >= 5:
                stats_order = np.argsort(stats[..., -1])[::-1][1:]
                eyel_mask, eyer_mask, statsl, statsr = label_lr_split(labels, stats, stats_order[0], stats_order[1])
                img, depth, xyxy, _ = process_cuts(part_info['img'], part_info['depth'], part_info['xyxy'], statsl)
                tag2pinfo['eyer'] = {'img': img, 'xyxy': xyxy, 'depth': depth}
                img, depth, xyxy, _ = process_cuts(part_info['img'], part_info['depth'], part_info['xyxy'], statsr)
                tag2pinfo['eyel'] = {'img': img, 'xyxy': xyxy, 'depth': depth}

                browl_mask, browr_mask, statsl, statsr = label_lr_split(labels, stats, stats_order[2], stats_order[3])
                img, depth, xyxy, _ = process_cuts(part_info['img'], part_info['depth'], part_info['xyxy'], statsl)
                tag2pinfo['browr'] = {'img': img, 'xyxy': xyxy, 'depth': depth}
                img, depth, xyxy, _ = process_cuts(part_info['img'], part_info['depth'], part_info['xyxy'], statsr)
                tag2pinfo['browl'] = {'img': img, 'xyxy': xyxy, 'depth': depth}
        else:
            tag2pinfo['eyes'] = part_info


    if tblr_split:
        tag_lr_split('handwear', tag2pinfo)

        eyetags_v3 = ['eyewhite', 'irides', 'eyelash', 'eyebrow']
        for tag in eyetags_v3:
            tag_lr_split(tag, tag2pinfo)

        tag_lr_split('ears', tag2pinfo)

    # if 'headwear' in tag2pinfo:
    #     part_info = tag2pinfo.pop('headwear')
    #     tag2pinfo['hair']['img'] = img_alpha_blending([tag2pinfo['hair'], part_info], xyxy=tag2pinfo['hair']['xyxy'], premultiplied=False)
    # if 'headwear' in tag2pinfo:
    #     part_info = tag2pinfo.pop('headwear')
    #     tag2pinfo['hair']['img'] = img_alpha_blending([tag2pinfo['hair'], part_info], xyxy=tag2pinfo['hair']['xyxy'], premultiplied=False)

    # if 'footwear' in tag2pinfo:
    #     part_info = tag2pinfo.pop('footwear')
    #     tag2pinfo['legwear']['img'] = img_alpha_blending([tag2pinfo['legwear'], part_info], xyxy=tag2pinfo['legwear']['xyxy'], premultiplied=False)

        if 'hair' in tag2pinfo:
            part_info = tag2pinfo.pop('hair')
            parts = cluster_inpaint_part(**part_info)
            parts.sort(key=lambda x: x['depth_median'])
            tag2pinfo['hairf'] = parts[0]
            tag2pinfo['hairb'] = parts[1]

    # if 'footwear' in tag2pinfo:
    #     tag2pinfo.pop('footwear')

    if 'nose' in tag2pinfo:
        xyxy = tag2pinfo['nose']['xyxy']
        tag2pinfo['nose']['img'][..., :3] = fullpage[xyxy[1]: xyxy[3], xyxy[0]: xyxy[2], :3]

    if 'mouth' in tag2pinfo:
        xyxy = tag2pinfo['mouth']['xyxy']
        tag2pinfo['mouth']['img'][..., :3] = fullpage[xyxy[1]: xyxy[3], xyxy[0]: xyxy[2], :3]
        _clean_mouth_layer(tag2pinfo['mouth'])

    if 'eyelash' in tag2pinfo and 'face' in tag2pinfo:
        _clean_eyelash_skin_fringe(tag2pinfo['eyelash'], tag2pinfo['face'])

    # Plan C: depth-cliff split (Guided Filter edge snapping + back-layer underlap inpaint)
    _depth_split_bimodal_parts(tag2pinfo, fullpage.shape[:2])
    _clean_and_split_objects(tag2pinfo, fullpage.shape[:2])
    _fill_head_interior_gaps(tag2pinfo, fullpage.shape[:2])

    # Plan C: edge RGB decontamination (eliminates dark/gray halos on transparent fringes)
    for _pdict in tag2pinfo.values():
        if 'img' in _pdict:
            _pdict['img'] = _decontaminate_edges(_pdict['img'])

    part_dict_list = []
    save_dir = osp.dirname(saved)
    psd_savep = osp.join(osp.dirname(save_dir), osp.basename(save_dir) + '.psd')

    for t in tag2pinfo:
        if t not in tag2pinfo:
            print(f'{t} is not valid')
            continue
        part_dict = tag2pinfo[t]
        part_dict = save_part(t, saved, part_dict, save_to_disk=not save_to_psd)
        if save_to_psd:
            part_dict_list.append(part_dict)

    if 'face' in tag2pinfo:
        fd = float(tag2pinfo['face']['depth_median'])
        for idx_t, t in enumerate(['nose', 'mouth', 'eyes', 'eyewhite', 'irides', 'eyelash', 'eyebrow', 'headwear']):
            if t in tag2pinfo:
                target_max = fd - 0.001 * (idx_t + 1)
                if tag2pinfo[t]['depth_median'] > target_max:
                    tag2pinfo[t]['depth_median'] = target_max
        for t in ['earr', 'earl', 'ears']:
            if t in tag2pinfo:
                tag2pinfo[t]['depth_median'] = tag2pinfo['face']['depth_median'] + 0.001

    # if 'hairb' in tag2pinfo:
    #     tag2pinfo['hairb']['depth_median'] = 1.

    frame_size = fullpage.shape[:2]

    if save_to_psd:
        dump_parts_psd(tag2pinfo, frame_size, psd_savep, part_dict_list=part_dict_list, fullpage=fullpage)
        print(f'psd saved to {psd_savep}')
    else:
        dict2json({'parts': tag2pinfo, 'frame_size': frame_size}, osp.join(saved, 'info.json'))


def dump_parts_psd(tag2pinfo, frame_size, psd_savep, part_dict_list=None, fullpage=None):
    if part_dict_list is None:
        part_dict_list = []
        for v in tag2pinfo.values():
            part_dict_list.append(v)
    psd_depth_savep = osp.splitext(psd_savep)[0] + '_depth.psd'
    part_dict_list = _sort_parts_by_local_overlap(part_dict_list, frame_size, fullpage=fullpage)
    save_psd(psd_savep, part_dict_list, frame_size[0], frame_size[1])
    save_psd(psd_depth_savep, part_dict_list, frame_size[0], frame_size[1], mode='L', img_key='depth')
    for pdict in tag2pinfo.values():
        for k in {'img', 'depth', 'mask'}:
            if k in pdict:
                pdict.pop(k)
    dict2json({'parts': tag2pinfo, 'frame_size': frame_size}, psd_savep + '.json')


def psd2partdicts(srcp):
    psd = PSDImage.open(srcp)
    json_path = srcp + '.json'
    partdict = json2dict(json_path)
    tag2part= partdict['parts']
    for layer in psd:
        img = layer.numpy()
        tag2part[layer.name]['img'] = np.round(img * 255).astype(np.uint8)
    depth_path = osp.splitext(srcp)[0] + '_depth.psd'
    psd = PSDImage.open(depth_path)
    for layer in psd:
        img = layer.numpy()
        tag2part[layer.name]['depth'] = img[..., 0]
    return partdict


def seg_wdepth(srcp, *args, **kwargs):
    srcd = osp.dirname(srcp)
    part_dict = load_part(srcp)
    tag = part_dict['tag']
    rst_list = cluster_inpaint_part(**part_dict)
    saved = osp.join(srcd, tag)
    if len(rst_list) > 0:
        os.makedirs(saved, exist_ok=True)
        for ii, part in enumerate(rst_list):
            sub_tag = tag + '-' + str(ii)
            save_part(sub_tag, saved, part, save_part_info=True)
        print(f'sub part saved to {saved}')
    else:
        print(f'seg_wdepth: failed to seg more parts')


def seg_wdepth_psd(srcp, target_tags, savep=None):
    part_infos = psd2partdicts(srcp)
    if savep is None:
        savep = osp.splitext(srcp)[0] + '_wdepth.psd'
    if isinstance(target_tags, str):
        target_tags = target_tags.split(',')
    else:
        assert isinstance(target_tags, list)
    valid_tags = list(part_infos['parts'].keys())
    for tag in target_tags:
        if tag not in part_infos['parts']:
            print(f'{tag} is not in {valid_tags}')
            continue
        part_dict = part_infos['parts'].pop(tag)
        mask = part_dict['img'][..., -1] > 10
        if not np.any(mask):
            continue
        part_dict['mask'] = mask
        rst_list = cluster_inpaint_part(**part_dict)
        if len(rst_list) > 0:
            for ii, part in enumerate(rst_list):
                sub_tag = tag + '-' + str(ii)
                part['tag'] = sub_tag
                part_infos['parts'][sub_tag] = part

    dump_parts_psd(part_infos['parts'], part_infos['frame_size'], savep)
    print(f'psd saved to {savep}')


def seg_wlr(srcp, *args, **kwargs):
    srcd = osp.dirname(srcp)
    part_dict = load_part(srcp)
    tag = part_dict['tag']
    rst_dict = part_lr_split(tag, part_dict)
    saved = osp.join(srcd, tag)
    if len(rst_dict) > 1:
        os.makedirs(saved, exist_ok=True)
        for sub_tag, part in rst_dict.items():
            save_part(sub_tag, saved, part, save_part_info=True)
        print(f'sub part saved to {saved}')
    else:
        print(f'seg_wdepth: failed to seg more parts')



def seg_wlr_psd(srcp, target_tags, savep=None):
    part_infos = psd2partdicts(srcp)
    if savep is None:
        savep = osp.splitext(srcp)[0] + '_lrsplit.psd'
    if isinstance(target_tags, str):
        target_tags = target_tags.split(',')
    else:
        assert isinstance(target_tags, list)
    valid_tags = list(part_infos['parts'].keys())
    for tag in target_tags:
        if tag not in part_infos['parts']:
            print(f'{tag} is not in {valid_tags}')
            continue
        part_dict = part_infos['parts'].pop(tag)
        mask = part_dict['img'][..., -1] > 10
        if not np.any(mask):
            continue
        part_dict['mask'] = mask
        rst_dict = part_lr_split(tag, part_dict)
        part_infos['parts'].update(rst_dict)

    dump_parts_psd(part_infos['parts'], part_infos['frame_size'], savep)
    print(f'psd saved to {savep}')
