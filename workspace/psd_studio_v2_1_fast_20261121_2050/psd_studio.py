#!/usr/bin/env python3
"""
PSD Studio v2 for See-through / Anime2.5DRig (Local Web UI + CLI)
Zero extra pip dependencies beyond numpy, opencv-python, Pillow.
"""
import argparse
import base64
import io
import json
import os
import struct
import sys
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

def _unpack_packbits(buf: bytes, expected_len: int) -> bytes:
    out = bytearray()
    i = 0
    n = len(buf)
    while i < n and len(out) < expected_len:
        hdr = buf[i]
        i += 1
        if hdr == 128:
            continue
        if hdr < 128:
            cnt = hdr + 1
            out.extend(buf[i : i + cnt])
            i += cnt
        else:
            cnt = 257 - hdr
            if i < n:
                out.extend(bytes([buf[i]]) * cnt)
                i += 1
    if len(out) < expected_len:
        out.extend(b"\x00" * (expected_len - len(out)))
    return bytes(out[:expected_len])


def _pack_packbits_row(row: bytes) -> bytes:
    out = bytearray()
    n = len(row)
    i = 0
    while i < n:
        run = 1
        while i + run < n and run < 128 and row[i + run] == row[i]:
            run += 1
        if run >= 3:
            out.append(257 - run)
            out.append(row[i])
            i += run
        else:
            start = i
            lit = 0
            while i < n and lit < 128:
                if i + 2 < n and row[i] == row[i + 1] == row[i + 2]:
                    break
                i += 1
                lit += 1
            out.append(lit - 1)
            out.extend(row[start : start + lit])
    return bytes(out)


def read_psd(path: Path) -> dict[str, Any]:
    """Read an 8-bit RGB or Grayscale PSD file into {width, height, layers:[...]}."""
    data = path.read_bytes()
    f = io.BytesIO(data)
    sig, ver = struct.unpack(">4sH", f.read(6))
    if sig != b"8BPS" or ver != 1:
        raise ValueError(f"Unsupported PSD header: {sig!r} v{ver}")
    f.read(6)
    channels, height, width, depth, color_mode = struct.unpack(">HIIHH", f.read(14))
    if depth != 8 or color_mode not in (1, 3):
        raise ValueError(f"Only 8-bit RGB/Grayscale PSD is supported (got {depth}-bit mode={color_mode})")

    cm_len = struct.unpack(">I", f.read(4))[0]
    f.seek(cm_len, 1)
    ir_len = struct.unpack(">I", f.read(4))[0]
    f.seek(ir_len, 1)

    lm_len = struct.unpack(">I", f.read(4))[0]
    if lm_len == 0:
        return {"width": width, "height": height, "layers": []}
    lm_end = f.tell() + lm_len
    layer_info_len = struct.unpack(">I", f.read(4))[0]
    if layer_info_len == 0:
        return {"width": width, "height": height, "layers": []}
    layer_count = struct.unpack(">h", f.read(2))[0]
    n_layers = abs(layer_count)

    records = []
    for _ in range(n_layers):
        top, left, bottom, right = struct.unpack(">iiii", f.read(16))
        num_ch = struct.unpack(">H", f.read(2))[0]
        ch_info = [struct.unpack(">hI", f.read(6)) for _ in range(num_ch)]
        _, _, opacity, _, flags, _ = struct.unpack(">4s4sBBBB", f.read(12))
        extra_len = struct.unpack(">I", f.read(4))[0]
        extra_start = f.tell()

        mask_len = struct.unpack(">I", f.read(4))[0]
        f.seek(mask_len, 1)
        br_len = struct.unpack(">I", f.read(4))[0]
        f.seek(br_len, 1)

        name_len = struct.unpack(">B", f.read(1))[0]
        raw_name = f.read(name_len)
        pad = (4 - ((name_len + 1) % 4)) % 4
        f.seek(pad, 1)

        uni_name = None
        while f.tell() <= extra_start + extra_len - 12:
            tsig = f.read(4)
            if tsig not in (b"8BIM", b"8B64"):
                break
            key = f.read(4)
            alen = struct.unpack(">I", f.read(4))[0]
            if key == b"luni":
                ulen = struct.unpack(">I", f.read(4))[0]
                uni_name = f.read(ulen * 2).decode("utf-16-be", errors="replace").rstrip("\x00")
                rem = alen - 4 - ulen * 2
                if rem > 0:
                    f.seek(rem, 1)
            else:
                f.seek(alen, 1)
            if alen % 2 == 1:
                f.seek(1, 1)
        f.seek(extra_start + extra_len)

        name = uni_name if uni_name is not None else raw_name.decode("utf-8", errors="replace")
        records.append({
            "name": name,
            "left": left, "top": top, "right": right, "bottom": bottom,
            "w": max(0, right - left), "h": max(0, bottom - top),
            "ch_info": ch_info,
            "opacity": opacity,
            "visible": not bool(flags & 2),
        })

    layers = []
    for rec in records:
        w, h = rec["w"], rec["h"]
        ch_planes: dict[int, np.ndarray] = {}
        for ch_id, ch_len in rec["ch_info"]:
            ch_bytes = f.read(ch_len)
            if w == 0 or h == 0 or ch_len < 2:
                continue
            comp = struct.unpack(">H", ch_bytes[:2])[0]
            body = ch_bytes[2:]
            if comp == 0:
                plane = np.frombuffer(body[: w * h], dtype=np.uint8).reshape((h, w))
            elif comp == 1:
                row_counts = struct.unpack(f">{h}H", body[: 2 * h])
                pos = 2 * h
                rows = []
                for rc in row_counts:
                    rows.append(_unpack_packbits(body[pos : pos + rc], w))
                    pos += rc
                plane = np.frombuffer(b"".join(rows), dtype=np.uint8).reshape((h, w))
            else:
                raise ValueError(f"Unsupported PSD channel compression {comp} in layer {rec['name']!r}")
            ch_planes[ch_id] = plane

        if w > 0 and h > 0:
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            if color_mode == 1 and 0 in ch_planes:
                rgba[:, :, 0] = rgba[:, :, 1] = rgba[:, :, 2] = ch_planes[0]
            else:
                if 0 in ch_planes:
                    rgba[:, :, 0] = ch_planes[0]
                if 1 in ch_planes:
                    rgba[:, :, 1] = ch_planes[1]
                if 2 in ch_planes:
                    rgba[:, :, 2] = ch_planes[2]
            rgba[:, :, 3] = ch_planes[-1] if -1 in ch_planes else 255
        else:
            rgba = np.zeros((0, 0, 4), dtype=np.uint8)

        layers.append({
            "name": rec["name"],
            "left": rec["left"], "top": rec["top"],
            "right": rec["right"], "bottom": rec["bottom"],
            "opacity": rec["opacity"],
            "visible": rec["visible"],
            "rgba": rgba,
        })

    f.seek(lm_end)
    return {"width": width, "height": height, "layers": layers}


def _trim_layer(layer: dict[str, Any], W: int, H: int) -> dict[str, Any] | None:
    rgba = layer["rgba"]
    if rgba.size == 0:
        return None
    ys, xs = np.nonzero(rgba[:, :, 3] > 0)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    sub = np.ascontiguousarray(rgba[y0:y1, x0:x1])
    nl = layer["left"] + x0
    nt = layer["top"] + y0
    return {
        "name": layer["name"],
        "left": nl, "top": nt, "right": nl + (x1 - x0), "bottom": nt + (y1 - y0),
        "opacity": layer.get("opacity", 255),
        "visible": layer.get("visible", True),
        "rgba": sub,
    }


def composite_layers(layers: list[dict[str, Any]], W: int, H: int) -> np.ndarray:
    canvas = np.zeros((H, W, 4), dtype=np.float32)
    for L in layers:
        if not L.get("visible", True) or L["rgba"].size == 0:
            continue
        rgba = L["rgba"]
        lh, lw = rgba.shape[:2]
        lx, ly = L["left"], L["top"]
        x0, y0 = max(0, lx), max(0, ly)
        x1, y1 = min(W, lx + lw), min(H, ly + lh)
        if x1 <= x0 or y1 <= y0:
            continue
        src = rgba[y0 - ly : y1 - ly, x0 - lx : x1 - lx].astype(np.float32) / 255.0
        sa = src[:, :, 3:4] * (L.get("opacity", 255) / 255.0)
        dst = canvas[y0:y1, x0:x1]
        da = dst[:, :, 3:4]
        oa = sa + da * (1.0 - sa)
        safe_oa = np.where(oa > 1e-6, oa, 1.0)
        dst[:, :, :3] = (src[:, :, :3] * sa + dst[:, :, :3] * da * (1.0 - sa)) / safe_oa
        dst[:, :, 3:4] = oa
    return np.clip(np.round(canvas * 255.0), 0, 255).astype(np.uint8)


def write_psd(path: Path, width: int, height: int, layers: list[dict[str, Any]]) -> np.ndarray:
    """Write a standard 8-bit RGBA PSD with PackBits RLE channels, Unicode `luni` names, and composite preview."""
    clean_layers = [t for L in layers if (t := _trim_layer(L, width, height)) is not None]

    encoded_channels: list[list[tuple[int, bytes]]] = []
    for L in clean_layers:
        rgba = L["rgba"]
        h, w = rgba.shape[:2]
        chs = []
        for ch_id, ch_idx in ((-1, 3), (0, 0), (1, 1), (2, 2)):
            plane = rgba[:, :, ch_idx]
            packed_rows = [_pack_packbits_row(plane[y].tobytes()) for y in range(h)]
            row_counts = b"".join(struct.pack(">H", len(r)) for r in packed_rows)
            chs.append((ch_id, struct.pack(">H", 1) + row_counts + b"".join(packed_rows)))
        encoded_channels.append(chs)

    li = io.BytesIO()
    li.write(struct.pack(">h", -len(clean_layers)))
    for L, chs in zip(clean_layers, encoded_channels):
        li.write(struct.pack(">iiii", L["top"], L["left"], L["bottom"], L["right"]))
        li.write(struct.pack(">H", len(chs)))
        for ch_id, ch_blob in chs:
            li.write(struct.pack(">hI", ch_id, len(ch_blob)))
        flags = 0 if L.get("visible", True) else 2
        li.write(struct.pack(">4s4sBBBB", b"8BIM", b"norm", int(L.get("opacity", 255)), 0, flags, 0))

        extra = io.BytesIO()
        extra.write(struct.pack(">I", 0))
        extra.write(struct.pack(">I", 0))
        ascii_name = L["name"].encode("ascii", errors="replace")[:255]
        extra.write(struct.pack(">B", len(ascii_name)) + ascii_name)
        extra.write(b"\x00" * ((4 - ((len(ascii_name) + 1) % 4)) % 4))

        u16 = L["name"].encode("utf-16-be")
        luni_body = struct.pack(">I", len(L["name"])) + u16
        if len(luni_body) % 2 == 1:
            luni_body += b"\x00"
        extra.write(b"8BIMluni" + struct.pack(">I", len(luni_body)) + luni_body)

        extra_bytes = extra.getvalue()
        li.write(struct.pack(">I", len(extra_bytes)) + extra_bytes)

    for chs in encoded_channels:
        for _, ch_blob in chs:
            li.write(ch_blob)

    li_bytes = li.getvalue()
    if len(li_bytes) % 2 == 1:
        li_bytes += b"\x00"

    lm = io.BytesIO()
    lm.write(struct.pack(">I", len(li_bytes)) + li_bytes)
    lm.write(struct.pack(">I", 0))
    lm_bytes = lm.getvalue()

    comp = composite_layers(clean_layers, width, height)
    comp_rows_by_ch = [
        [_pack_packbits_row(comp[:, :, ch_idx][y].tobytes()) for y in range(height)]
        for ch_idx in (0, 1, 2, 3)
    ]
    comp_counts = b"".join(struct.pack(">H", len(r)) for ch_rows in comp_rows_by_ch for r in ch_rows)
    comp_body = b"".join(r for ch_rows in comp_rows_by_ch for r in ch_rows)

    out = io.BytesIO()
    out.write(struct.pack(">4sH6sHIIHH", b"8BPS", 1, b"\x00" * 6, 4, height, width, 8, 3))
    out.write(struct.pack(">I", 0))
    out.write(struct.pack(">I", 0))
    out.write(struct.pack(">I", len(lm_bytes)) + lm_bytes)
    out.write(struct.pack(">H", 1) + comp_counts + comp_body)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out.getvalue())
    return comp




# =====================================================================
#  CV Algorithms (All Tunable Interactively or via CLI)
# =====================================================================

def _to_canvas(layer, H, W):
    c = np.zeros((H, W, 4), dtype=np.uint8)
    x0, y0, x1, y1 = layer["left"], layer["top"], layer["right"], layer["bottom"]
    if x1 > x0 and y1 > y0:
        c[y0:y1, x0:x1] = layer["rgba"]
    return c


def _tight_layer_from_canvas(name, canvas, opacity=255, flags=0, blend=b"norm", depth_canvas=None):
    a = canvas[..., 3]
    ys, xs = np.nonzero(a > 0)
    if len(ys) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    out = {
        "name": name,
        "left": x0, "top": y0, "right": x1, "bottom": y1,
        "rgba": canvas[y0:y1, x0:x1].copy(),
        "opacity": opacity, "flags": flags, "blend": blend,
    }
    if depth_canvas is not None:
        out["depth"] = depth_canvas[y0:y1, x0:x1].copy()
        m = out["rgba"][..., 3] > 16
        out["depth_median"] = float(np.median(out["depth"][m])) if np.any(m) else 0.5
    return out


def clean_alpha_and_specks(canvas, alpha_floor=16, min_area=60, border_px=0):
    """Eliminate NF4 faint alpha haze and tiny floating specks."""
    out = canvas.copy()
    H, W = out.shape[:2]
    a = out[..., 3]
    if alpha_floor > 0:
        a = np.where(a <= alpha_floor, 0, a).astype(np.uint8)
    if border_px > 0:
        b = min(border_px, H // 4, W // 4)
        a[:b, :] = 0
        a[H - b:, :] = 0
        a[:, :b] = 0
        a[:, W - b:] = 0
    if min_area > 0:
        fg = (a > 0).astype(np.uint8)
        n_cc, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        if n_cc > 1:
            keep = np.zeros_like(fg)
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_cid = 1 + int(np.argmax(areas))
            for cid in range(1, n_cc):
                if cid == max_cid or stats[cid, cv2.CC_STAT_AREA] >= min_area:
                    keep[lab == cid] = 1
            a = np.where(keep > 0, a, 0).astype(np.uint8)
    out[..., 3] = a
    return out


def decontaminate_edges(rgba, radius=3):
    """Fast RGB bleed from opaque interior into semi-transparent fringe (200x faster than full-canvas TELEA)."""
    if radius <= 0:
        return rgba
    out = rgba.copy()
    a = out[..., 3]
    ys, xs = np.nonzero(a > 0)
    if len(ys) == 0:
        return out
    H, W = a.shape
    pad = int(radius) + 2
    y0, y1 = max(0, int(ys.min()) - pad), min(H, int(ys.max()) + 1 + pad)
    x0, x1 = max(0, int(xs.min()) - pad), min(W, int(xs.max()) + 1 + pad)

    sub_rgb = out[y0:y1, x0:x1, :3].astype(np.float32)
    sub_a = a[y0:y1, x0:x1]
    solid = (sub_a > 180).astype(np.float32)
    fringe = (sub_a > 0) & (sub_a <= 180)
    if not np.any(solid > 0) or not np.any(fringe):
        return out

    # Iterative normalized 3x3 box-filter bleed from solid pixels outward (radius steps)
    cur_rgb = sub_rgb * solid[..., None]
    cur_w = solid.copy()
    for _ in range(max(1, int(radius))):
        num = cv2.blur(cur_rgb, (3, 3))
        den = cv2.blur(cur_w, (3, 3))
        valid = (cur_w < 0.5) & (den > 1e-4)
        if not np.any(valid):
            break
        cur_rgb = np.where(valid[..., None], num / np.maximum(den[..., None], 1e-4), cur_rgb)
        cur_w = np.where(valid, 1.0, cur_w)

    filled = np.clip(np.round(cur_rgb), 0, 255).astype(np.uint8)
    out[y0:y1, x0:x1, :3] = np.where(fringe[..., None], filled, out[y0:y1, x0:x1, :3])
    return out


def split_layer_by_depth(canvas, depth_canvas, name, threshold=None, feather=1.0, underlap_px=3):
    """Split ANY layer into `<name>_1` (farther, depth > thr) and `<name>_2` (nearer, depth <= thr)."""
    a = canvas[..., 3]
    m = a > 16
    if int(m.sum()) < 400 or depth_canvas is None:
        return None, None, None
    vals = depth_canvas[m]
    if float(np.std(vals)) < 1e-3:
        return None, None, None

    if threshold is None:
        u8 = np.clip(np.round(vals * 255.0), 0, 255).astype(np.uint8)
        thr_u8, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = float(thr_u8) / 255.0
    else:
        thr = float(threshold)

    far_m = m & (depth_canvas > thr)
    near_m = m & (depth_canvas <= thr)
    if int(far_m.sum()) < 150 or int(near_m.sum()) < 150:
        return None, None, thr

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    far_u8 = cv2.morphologyEx(far_m.astype(np.uint8), cv2.MORPH_CLOSE, k5)
    near_u8 = (m & (far_u8 == 0)).astype(np.uint8)

    near_f = cv2.GaussianBlur(near_u8.astype(np.float32), (0, 0), float(feather)) if feather > 0 else near_u8.astype(np.float32)
    if underlap_px > 0:
        k_u = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (underlap_px * 2 + 1, underlap_px * 2 + 1))
        far_ext = cv2.dilate(far_u8, k_u).astype(np.float32)
        if feather > 0:
            far_ext = cv2.GaussianBlur(far_ext, (0, 0), float(feather))
    else:
        far_ext = far_u8.astype(np.float32)

    c_far = canvas.copy()
    c_near = canvas.copy()
    c_far[..., 3] = np.clip(np.round(a.astype(np.float32) * far_ext), 0, 255).astype(np.uint8)
    c_near[..., 3] = np.clip(np.round(a.astype(np.float32) * near_f), 0, 255).astype(np.uint8)
    return c_far, c_near, thr


def refine_neck_topwear(layers_map, depth_map, H, W, mode="auto", depth_offset=0.0, skin_diff=24.0, feather=1.0):
    if mode == "off" or "neck" not in layers_map or "topwear" not in layers_map:
        return layers_map, depth_map, "跳过"

    n_c = layers_map["neck"]
    t_c = layers_map["topwear"]
    n_a, t_a = n_c[..., 3], t_c[..., 3]
    nm, tm = n_a > 24, t_a > 24
    ov = nm & tm
    ov_cnt = int(ov.sum())
    if ov_cnt < 200:
        return layers_map, depth_map, "无重叠"

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    n_rgb = n_c[..., :3].astype(np.float32)
    t_rgb = t_c[..., :3].astype(np.float32)
    diff_mean = np.abs(n_rgb - t_rgb).mean(axis=2)
    diff_max = np.abs(n_rgb - t_rgb).max(axis=2)

    if mode in ("auto", "skin_cut"):
        skin_ov = ov & (diff_mean < float(skin_diff)) & (diff_max < float(skin_diff) * 1.6)
        skin_clean = cv2.morphologyEx(skin_ov.astype(np.uint8), cv2.MORPH_OPEN, k5)
        n_cc, cc_lab, stats, _ = cv2.connectedComponentsWithStats(skin_clean, connectivity=8)
        skin_main = np.zeros_like(skin_clean)
        for cid in range(1, n_cc):
            if stats[cid, cv2.CC_STAT_AREA] >= 500:
                skin_main[cc_lab == cid] = 1
        if int(skin_main.sum()) >= 500:
            k_down = np.zeros((25, 1), dtype=np.uint8); k_down[:13, 0] = 1
            top_band = (cv2.dilate((t_a <= 16).astype(np.uint8), k_down) > 0) & (t_a > 0)
            k_up = np.zeros((25, 5), dtype=np.uint8); k_up[12:, :] = 1
            skin_up = (cv2.dilate(skin_main, k_up) > 0) & top_band & (n_a > 0)
            skin_cut = (skin_main > 0) | skin_up
            keep_t = np.where(skin_cut, 0.0, 1.0).astype(np.float32)
            if feather > 0:
                keep_t = cv2.GaussianBlur(keep_t, (0, 0), float(feather))
            t_new = t_c.copy()
            t_new[..., 3] = np.clip(np.round(t_a.astype(np.float32) * keep_t), 0, 255).astype(np.uint8)
            layers_map["topwear"] = t_new
            return layers_map, depth_map, f"肤色抠空({int(skin_cut.sum())}px)"
        if mode == "skin_cut":
            return layers_map, depth_map, "未检出重复肤色"

    if mode in ("auto", "depth_sandwich"):
        nd = depth_map.get("neck")
        td = depth_map.get("topwear")
        if nd is None or td is None or float(np.std(td[ov])) < 1e-3:
            return layers_map, depth_map, "无有效深度图"
        behind_ov = ov & ((td + float(depth_offset)) > nd)
        front_ov = ov & ((td + float(depth_offset)) <= nd)
        b_cnt, f_cnt = int(behind_ov.sum()), int(front_ov.sum())
        if b_cnt >= 250 and f_cnt >= 250 and (b_cnt / ov_cnt) >= 0.08 and (f_cnt / ov_cnt) >= 0.08:
            nk_d_med = float(np.median(nd[ov]))
            ys_n, xs_n = np.nonzero(nm)
            ny1 = int(ys_n.max()) + 1
            nx0, nx1 = int(xs_n.min()), int(xs_n.max()) + 1
            pad_x = max(12, int((nx1 - nx0) * 0.15))
            col_roi = np.zeros((H, W), dtype=bool)
            col_roi[:ny1, max(0, nx0 - pad_x):min(W, nx1 + pad_x)] = True
            ext_cand = (tm & col_roi & ((td + float(depth_offset)) > nk_d_med)) | behind_ov
            n_cc, cc_lab, stats, _ = cv2.connectedComponentsWithStats(ext_cand.astype(np.uint8), connectivity=8)
            back_mask = np.zeros((H, W), dtype=np.uint8)
            for cid in range(1, n_cc):
                comp = (cc_lab == cid)
                if stats[cid, cv2.CC_STAT_AREA] >= 120 and int((comp & behind_ov).sum()) >= 80:
                    back_mask[comp] = 1
            if int(back_mask.sum()) >= 250:
                back_mask = cv2.morphologyEx(back_mask, cv2.MORPH_CLOSE, k5)
                k_down8 = np.zeros((17, 1), dtype=np.uint8); k_down8[:9, 0] = 1
                top_rim = (cv2.dilate((t_a <= 16).astype(np.uint8), k_down8) > 0) & (t_a > 0)
                k_up8 = np.zeros((17, 3), dtype=np.uint8); k_up8[8:, :] = 1
                back_final = ((back_mask > 0) | ((cv2.dilate(back_mask, k_up8) > 0) & top_rim)) & (t_a > 0)
                bm_u8 = back_final.astype(np.uint8)
                bm_f = cv2.GaussianBlur(bm_u8.astype(np.float32), (0, 0), max(0.5, float(feather))) if feather > 0 else bm_u8.astype(np.float32)
                bm_b = cv2.GaussianBlur(cv2.dilate(bm_u8, k5).astype(np.float32), (0, 0), max(0.5, float(feather))) if feather > 0 else cv2.dilate(bm_u8, k5).astype(np.float32)
                t_back = t_c.copy(); t_front = t_c.copy()
                t_back[..., 3] = np.clip(np.round(t_a.astype(np.float32) * bm_b), 0, 255).astype(np.uint8)
                t_front[..., 3] = np.clip(np.round(t_a.astype(np.float32) * (1.0 - bm_f)), 0, 255).astype(np.uint8)
                del layers_map["topwear"]
                layers_map["topwear_1"] = t_back
                layers_map["topwear_2"] = t_front
                depth_map["topwear_1"] = td.copy()
                depth_map["topwear_2"] = td.copy()
                return layers_map, depth_map, f"伪深度三明治拆分(后领={int(back_final.sum())}px)"
    return layers_map, depth_map, "无需拆分"


def _norm_role(tag):
    t = tag.lower().strip()
    for suf in ("_1", "_2", "_l", "_r", "-l", "-r", "_back", "_front"):
        if t.endswith(suf):
            t = t[:-len(suf)]
    return t


CLASSIC_RANK = {
    "wings": 0, "tail": 1, "back hair": 2, "legwear": 3, "footwear": 4,
    "bottomwear": 5, "topwear": 6, "neck": 7, "neckwear": 8, "handwear": 9,
    "ears": 10, "earwear": 11, "face": 12, "facedetail": 13, "nose": 14,
    "mouth": 15, "eyewhite": 16, "irides": 17, "eyelash": 18, "eyebrow": 19,
    "eyewear": 20, "front hair": 21, "side hair": 22, "ahoge": 23,
    "headwear": 24, "objects": 25,
}


def sort_layers_by_mode(names, layers_map, depth_map, mode="auto"):
    """8 selectable layer-sorting algorithms:
    auto | depth | local_overlap | classic | arms_front | arms_back | hair_over_top | raw
    """
    if mode == "raw":
        return list(names)

    # Downsample masks & depths ONCE to <=256px for 300x faster pairwise overlap comparison
    scores = {}
    sm_masks = {}
    sm_depths = {}
    for idx, name in enumerate(names):
        c = layers_map[name]
        H, W = c.shape[:2]
        step = max(1, max(H, W) // 256)
        a_s = c[::step, ::step, 3]
        m_s = a_s > 24
        sm_masks[name] = m_s
        d = depth_map.get(name)
        if d is not None:
            d_s = d[::step, ::step]
            sm_depths[name] = d_s
            scores[name] = float(np.median(d_s[m_s])) if np.any(m_s) else (0.9 - 0.8 * (idx / max(1, len(names) - 1)))
        else:
            sm_depths[name] = None
            scores[name] = 0.9 - 0.8 * (idx / max(1, len(names) - 1))

    if mode == "depth":
        return sorted(names, key=lambda k: scores[k], reverse=True)

    if mode == "classic":
        def ckey(n):
            r = _norm_role(n)
            base = CLASSIC_RANK.get(r, 15)
            if n == "topwear_1":
                return (6.0, scores[n])
            if n == "neck":
                return (6.5, scores[n])
            if n == "topwear_2":
                return (7.0, scores[n])
            sub = 0.2 if n.endswith("_2") else (-0.2 if n.endswith("_1") else 0.0)
            return (base + sub, -scores[n])
        return sorted(names, key=ckey)

    arr = sorted(names, key=lambda k: scores[k], reverse=True)
    masks = sm_masks

    # Precompute pairwise local overlap preference ONCE (O(N^2) on 256x256 -> <5ms!)
    if mode in ("local_overlap", "auto", "arms_front", "arms_back", "hair_over_top"):
        prefer_above = set()  # (a, b) means `a` should sit ABOVE `b`
        min_ov_s = max(6, int(80 // (step * step))) if names else 6
        for i in range(len(arr)):
            a = arr[i]
            da = sm_depths[a]
            if da is None:
                continue
            ma_m = sm_masks[a]
            for j in range(i + 1, len(arr)):
                b = arr[j]
                db = sm_depths[b]
                if db is None:
                    continue
                ov = ma_m & sm_masks[b]
                if int(ov.sum()) < min_ov_s:
                    continue
                va = float(da[ov].mean())
                vb = float(db[ov].mean())
                if va + 0.015 < vb:
                    prefer_above.add((a, b))
                elif vb + 0.015 < va:
                    prefer_above.add((b, a))

        guard = len(arr) * len(arr)
        changed = True
        while changed and guard > 0:
            guard -= 1
            changed = False
            for i in range(len(arr)):
                for j in range(i + 1, len(arr)):
                    if (arr[i], arr[j]) in prefer_above:
                        a = arr.pop(i)
                        arr.insert(j, a)
                        changed = True
                        break
                if changed:
                    break

    if mode == "local_overlap":
        return arr

    # Anatomical invariants for auto / arms_front / arms_back / hair_over_top
    def ensure_before(a, b):
        if a in arr and b in arr and arr.index(a) > arr.index(b):
            arr.remove(a)
            arr.insert(arr.index(b), a)

    def ensure_after(a, b):
        if a in arr and b in arr and arr.index(a) < arr.index(b):
            arr.remove(a)
            arr.insert(arr.index(b) + 1, a)

    role_map = {n: _norm_role(n) for n in arr}

    # Lock face/eye stack
    face_stack = ["ears", "face", "nose", "mouth", "eyewhite", "irides", "eyelash", "eyebrow", "eyewear", "front hair"]
    for i in range(len(face_stack)):
        for j in range(i + 1, len(face_stack)):
            ra, rb = face_stack[i], face_stack[j]
            for na in [n for n in arr if role_map[n] == ra]:
                for nb in [n for n in arr if role_map[n] == rb]:
                    ensure_before(na, nb)

    # Lock torso/neck/back-hair below face
    for n in list(arr):
        if role_map[n] in ("back hair", "neck", "topwear", "bottomwear", "legwear") and n != "back hair_2":
            for f_n in [x for x in arr if role_map[x] == "face"]:
                ensure_before(n, f_n)

    # Detect background-wall topwear (>35% overlap with face)
    tw_is_bg = False
    if "topwear" in masks and "face" in masks:
        tw_is_bg = int((masks["topwear"] & masks["face"]).sum()) > 0.35 * max(1, int(masks["face"].sum()))

    if tw_is_bg:
        for fg in ("back hair", "back hair_1", "neck"):
            ensure_before("topwear", fg)
    else:
        if mode != "hair_over_top":
            for bh in ("back hair", "back hair_1"):
                for fg in ("topwear_1", "neck", "topwear", "topwear_2"):
                    ensure_before(bh, fg)
        if "neck" in arr and "topwear_1" in arr and "topwear_2" in arr:
            ensure_before("neck", "topwear_2")
            ensure_before("topwear_1", "neck")

    # Mode-specific overrides
    if mode == "arms_front":
        for hw in [n for n in arr if role_map[n] == "handwear"]:
            for tgt in [n for n in arr if role_map[n] in ("topwear", "bottomwear", "front hair")]:
                ensure_after(hw, tgt)
    elif mode == "arms_back":
        for hw in [n for n in arr if role_map[n] == "handwear"]:
            for tgt in [n for n in arr if role_map[n] in ("topwear", "neck", "face")]:
                ensure_before(hw, tgt)
    elif mode == "hair_over_top":
        for bh in [n for n in arr if role_map[n] == "back hair"]:
            for tgt in [n for n in arr if role_map[n] in ("topwear", "bottomwear", "handwear")]:
                ensure_after(bh, tgt)

    return arr


# =====================================================================
#  Stateful Studio Session (for Interactive Web UI)
# =====================================================================

class StudioSession:
    def __init__(self):
        self.lock = threading.Lock()
        self.filename = ""
        self.W = 0
        self.H = 0
        self.orig_order = []
        self.orig_layers = {}
        self.orig_depths = {}
        self.order = []
        self.layers = {}
        self.depths = {}
        self.visible = {}
        self.undo_stack = []

    def load_psd_bytes(self, psd_bytes, filename="input.psd", depth_bytes=None):
        with self.lock:
            tmp_p = Path("/tmp/_studio_in.psd") if os.name != "nt" else Path(os.environ.get("TEMP", ".")) / "_studio_in.psd"
            tmp_p.write_bytes(psd_bytes)
            psd = read_psd(tmp_p)
            W, H, layers = psd["width"], psd["height"], psd["layers"]

            dmap = {}
            if depth_bytes:
                tmp_d = tmp_p.with_name("_studio_in_depth.psd")
                tmp_d.write_bytes(depth_bytes)
                dpsd = read_psd(tmp_d)
                for L in dpsd["layers"]:
                    dc = _to_canvas(L, H, W)[..., 0].astype(np.float32) / 255.0
                    dmap[L["name"]] = dc

            self.filename = filename
            self.W, self.H = W, H
            self.orig_order = []
            self.orig_layers = {}
            self.orig_depths = {}
            for idx, L in enumerate(layers):
                name = L["name"]
                if name in self.orig_layers:
                    k = 2
                    while f"{name}_{k}" in self.orig_layers:
                        k += 1
                    name = f"{name}_{k}"
                self.orig_order.append(name)
                self.orig_layers[name] = _to_canvas(L, H, W)
                if name in dmap:
                    self.orig_depths[name] = dmap[name]
                else:
                    self.orig_depths[name] = np.full((H, W), 0.9 - 0.8 * (idx / max(1, len(layers) - 1)), dtype=np.float32)

            self._reset_to_orig()
            return self.get_state()

    def _reset_to_orig(self):
        self.order = list(self.orig_order)
        self.layers = {k: v.copy() for k, v in self.orig_layers.items()}
        self.depths = {k: v.copy() for k, v in self.orig_depths.items()}
        self.visible = {k: True for k in self.order}
        self.undo_stack = []

    def _bbox_for_names(self, names, pad_ratio=0.35):
        xs_min, ys_min, xs_max, ys_max = self.W, self.H, 0, 0
        found = False
        for n in names:
            if n not in self.layers:
                continue
            step = max(1, max(self.H, self.W) // 256)
            m = (self.layers[n][::step, ::step, 3] > 64).astype(np.uint8)
            n_cc, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            if n_cc <= 1:
                continue
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_cid = 1 + int(np.argmax(areas))
            min_a = max(4, 150 // (step * step))
            for cid in range(1, n_cc):
                if cid == max_cid or stats[cid, cv2.CC_STAT_AREA] >= min_a:
                    found = True
                    x0 = int(stats[cid, cv2.CC_STAT_LEFT]) * step
                    y0 = int(stats[cid, cv2.CC_STAT_TOP]) * step
                    x1 = min(self.W, (int(stats[cid, cv2.CC_STAT_LEFT]) + int(stats[cid, cv2.CC_STAT_WIDTH])) * step)
                    y1 = min(self.H, (int(stats[cid, cv2.CC_STAT_TOP]) + int(stats[cid, cv2.CC_STAT_HEIGHT])) * step)
                    xs_min = min(xs_min, x0); ys_min = min(ys_min, y0)
                    xs_max = max(xs_max, x1); ys_max = max(ys_max, y1)
        if not found:
            return [0, 0, self.W, self.H]
        bw, bh = xs_max - xs_min, ys_max - ys_min
        pad = max(48, int(max(bw, bh) * pad_ratio))
        return [max(0, xs_min - pad), max(0, ys_min - pad), min(self.W, xs_max + pad), min(self.H, ys_max + pad)]

    def get_state(self):
        info = []
        for name in self.order:
            c = self.layers[name]
            a = c[..., 3]
            ys, xs = np.nonzero(a > 16)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(ys) > 0 else [0, 0, self.W, self.H]
            d = self.depths.get(name)
            m = a > 16
            d_med = round(float(np.median(d[m])), 3) if (d is not None and np.any(m)) else 0.5
            d_std = round(float(np.std(d[m])), 4) if (d is not None and np.any(m)) else 0.0
            info.append({
                "name": name,
                "visible": self.visible.get(name, True),
                "pixels": int((a > 10).sum()),
                "haze_pixels": int(((a > 0) & (a <= 10)).sum()),
                "bbox": bbox,
                "depth_median": d_med,
                "has_depth_var": d_std > 0.005,
            })
        rois = {
            "full": [0, 0, self.W, self.H],
            "face": self._bbox_for_names(["face", "eyewhite", "mouth", "front hair"], 0.25),
            "neck": self._bbox_for_names(["neck"], 0.65),
            "head": self._bbox_for_names(["headwear", "front hair", "face"], 0.20),
            "torso": self._bbox_for_names(["topwear", "topwear_1", "topwear_2"], 0.15),
        }
        return {
            "filename": self.filename,
            "width": self.W,
            "height": self.H,
            "layers": info,
            "rois": rois,
            "can_undo": len(self.undo_stack) > 0,
        }

    def render_composite_png(self, use_orig=False, solo=None, head_dx=0, head_dy=0, crop=None, max_dim=1024):
        with self.lock:
            if self.W == 0:
                return b""
            H, W = self.H, self.W
            src_order = self.orig_order if use_orig else self.order
            src_layers = self.orig_layers if use_orig else self.layers
            # Crop first so localized previews only composite the visible ROI!
            if crop is not None:
                cx0, cy0, cx1, cy1 = crop
                cx0, cy0 = max(0, min(W - 1, cx0)), max(0, min(H - 1, cy0))
                cx1, cy1 = max(cx0 + 1, min(W, cx1)), max(cy0 + 1, min(H, cy1))
            else:
                cx0, cy0, cx1, cy1 = 0, 0, W, H
            rh, rw = cy1 - cy0, cx1 - cx0
            out_rgb = np.zeros((rh, rw, 3), dtype=np.float32)
            out_a = np.zeros((rh, rw, 1), dtype=np.float32)
            head_roles = {"neck", "face", "ears", "mouth", "nose", "eyewhite", "irides", "eyelash", "eyebrow", "front hair", "headwear", "eyewear", "earwear"}
            for name in src_order:
                if solo:
                    if name != solo:
                        continue
                elif not use_orig and not self.visible.get(name, True):
                    continue
                c_full = src_layers[name]
                if (head_dx != 0 or head_dy != 0) and _norm_role(name) in head_roles:
                    M = np.float32([[1, 0, head_dx], [0, 1, head_dy]])
                    c_full = cv2.warpAffine(c_full, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
                sub = c_full[cy0:cy1, cx0:cx1]
                sa = sub[..., 3]
                if not np.any(sa):
                    continue
                # Fast integer-bounded tight slice inside ROI
                rows = np.any(sa, axis=1)
                cols = np.any(sa, axis=0)
                y0, y1 = int(np.argmax(rows)), rh - int(np.argmax(rows[::-1]))
                x0, x1 = int(np.argmax(cols)), rw - int(np.argmax(cols[::-1]))
                sl = sub[y0:y1, x0:x1]
                a = sl[..., 3:4].astype(np.float32) * (1.0 / 255.0)
                inv_a = 1.0 - a
                out_rgb[y0:y1, x0:x1] = sl[..., :3].astype(np.float32) * a + out_rgb[y0:y1, x0:x1] * inv_a
                out_a[y0:y1, x0:x1] = a + out_a[y0:y1, x0:x1] * inv_a
            rgba = np.empty((rh, rw, 4), dtype=np.uint8)
            rgba[..., :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
            rgba[..., 3] = np.clip(out_a[..., 0] * 255.0, 0, 255).astype(np.uint8)
            img = Image.fromarray(rgba, mode="RGBA")
            if max(rw, rh) > max_dim:
                s = max_dim / max(rw, rh)
                img = img.resize((int(rw * s), int(rh * s)), Image.BILINEAR)
            buf = io.BytesIO()
            img.save(buf, format="PNG", compress_level=1)
            return buf.getvalue()

    def apply_pipeline(self, params):
        with self.lock:
            if self.W == 0:
                return {"error": "请先加载 PSD 文件"}
            if params.get("reset_first", True):
                self._reset_to_orig()
            alpha_floor = int(params.get("alpha_floor", 16))
            min_area = int(params.get("min_area", 60))
            border_px = int(params.get("border_px", 0))
            decontam = int(params.get("decontam_radius", 0))
            collar_mode = params.get("collar_mode", "auto")
            depth_offset = float(params.get("depth_offset", 0.0))
            skin_diff = float(params.get("skin_diff", 24.0))
            feather = float(params.get("feather", 1.0))
            sort_mode = params.get("sort_mode", "auto")

            logs = []
            for name in list(self.order):
                self.layers[name] = clean_alpha_and_specks(
                    self.layers[name], alpha_floor=alpha_floor, min_area=min_area, border_px=border_px
                )
            logs.append(f"去噪(alpha>{alpha_floor}, 最小块>={min_area})")

            self.layers, self.depths, c_msg = refine_neck_topwear(
                self.layers, self.depths, self.H, self.W,
                mode=collar_mode, depth_offset=depth_offset, skin_diff=skin_diff, feather=feather
            )
            logs.append(f"领口:{c_msg}")

            if "topwear_1" in self.layers and "topwear_2" in self.layers and "topwear" in self.order:
                ti = self.order.index("topwear")
                self.order[ti:ti+1] = ["topwear_1", "topwear_2"]
                self.visible["topwear_1"] = True
                self.visible["topwear_2"] = True

            if decontam > 0:
                for name in list(self.order):
                    self.layers[name] = decontaminate_edges(self.layers[name], radius=decontam)
                logs.append(f"去色晕(r={decontam})")

            self.order = sort_layers_by_mode(self.order, self.layers, self.depths, mode=sort_mode)
            logs.append(f"排序:{sort_mode}")

            st = self.get_state()
            st["log"] = " | ".join(logs)
            return st

    def apply_sort_only(self, sort_mode):
        with self.lock:
            if self.W == 0:
                return {"error": "请先加载 PSD 文件"}
            self.order = sort_layers_by_mode(self.order, self.layers, self.depths, mode=sort_mode)
            st = self.get_state()
            st["log"] = f"已切换排序算法：{sort_mode}"
            return st

    def split_layer(self, name, threshold=None, feather=1.0):
        with self.lock:
            if name not in self.layers:
                return {"error": f"找不到图层 {name}"}
            c_far, c_near, used_thr = split_layer_by_depth(
                self.layers[name], self.depths.get(name), name, threshold=threshold, feather=feather
            )
            if c_far is None:
                return {"error": f"图层 {name} 无法按深度拆分（深度方差过小或区域太小）"}
            n1, n2 = f"{name}_1", f"{name}_2"
            idx = self.order.index(name)
            d_orig = self.depths[name]
            del self.layers[name]
            self.layers[n1] = c_far
            self.layers[n2] = c_near
            self.depths[n1] = d_orig.copy()
            self.depths[n2] = d_orig.copy()
            self.visible[n1] = True
            self.visible[n2] = True
            self.order[idx:idx+1] = [n1, n2]
            st = self.get_state()
            st["log"] = f"已拆分 {name} -> {n1}（远）/ {n2}（近），阈值={used_thr:.3f}"
            return st

    def brush_stroke(self, name, mode, radius, points):
        with self.lock:
            if name not in self.layers or not points:
                return self.get_state()
            cur = self.layers[name]
            self.undo_stack.append((name, cur.copy()))
            if len(self.undo_stack) > 25:
                self.undo_stack.pop(0)
            mask = np.zeros((self.H, self.W), dtype=np.uint8)
            r = max(1, int(radius))
            pts = [(int(round(p[0])), int(round(p[1]))) for p in points]
            for i, pt in enumerate(pts):
                cv2.circle(mask, pt, r, 255, -1)
                if i > 0:
                    cv2.line(mask, pts[i - 1], pt, 255, r * 2)
            m = mask > 0
            if mode == "erase":
                cur[m, 3] = 0
            elif mode == "restore" and name in self.orig_layers:
                cur[m] = self.orig_layers[name][m]
            return self.get_state()

    def undo(self):
        with self.lock:
            if self.undo_stack:
                name, prev = self.undo_stack.pop()
                if name in self.layers:
                    self.layers[name] = prev
            return self.get_state()

    def export_psd_bytes(self):
        with self.lock:
            out_layers = []
            for name in self.order:
                if not self.visible.get(name, True):
                    continue
                L = _tight_layer_from_canvas(name, self.layers[name])
                if L is not None:
                    out_layers.append(L)
            tmp_out = Path("/tmp/_studio_out.psd") if os.name != "nt" else Path(os.environ.get("TEMP", ".")) / "_studio_out.psd"
            write_psd(tmp_out, self.W, self.H, out_layers)
            return tmp_out.read_bytes()


SESSION = StudioSession()


class StudioHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send_json(self, obj, code=200):
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_bytes(self, raw, ctype, filename=None):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            html_p = Path(__file__).with_name("psd_studio.html")
            self._send_bytes(html_p.read_bytes(), "text/html; charset=utf-8")
        elif u.path == "/api/state":
            self._send_json(SESSION.get_state())
        elif u.path == "/api/preview":
            use_orig = qs.get("orig", ["0"])[0] == "1"
            solo = qs.get("solo", [""])[0] or None
            hdx = int(qs.get("hdx", ["0"])[0])
            hdy = int(qs.get("hdy", ["0"])[0])
            crop = None
            if "crop" in qs and qs["crop"][0]:
                parts = [int(v) for v in qs["crop"][0].split(",")]
                if len(parts) == 4:
                    crop = parts
            png = SESSION.render_composite_png(use_orig=use_orig, solo=solo, head_dx=hdx, head_dy=hdy, crop=crop)
            self._send_bytes(png, "image/png")
        elif u.path == "/api/export/psd":
            raw = SESSION.export_psd_bytes()
            stem = Path(SESSION.filename or "output.psd").stem
            self._send_bytes(raw, "application/octet-stream", filename=f"{stem}_edited.psd")
        elif u.path == "/api/export/png":
            raw = SESSION.render_composite_png(max_dim=4096)
            stem = Path(SESSION.filename or "output.psd").stem
            self._send_bytes(raw, "image/png", filename=f"{stem}_preview.png")
        else:
            self.send_error(404)

    def do_POST(self):
        cl = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(cl) if cl > 0 else b"{}"
        u = urllib.parse.urlparse(self.path)
        try:
            data = json.loads(body.decode("utf-8"))
        except Exception:
            self._send_json({"error": "JSON 解析失败"}, 400)
            return

        if u.path == "/api/load":
            psd_b = base64.b64decode(data["psd_b64"])
            dep_b = base64.b64decode(data["depth_b64"]) if data.get("depth_b64") else None
            self._send_json(SESSION.load_psd_bytes(psd_b, data.get("filename", "input.psd"), dep_b))
        elif u.path == "/api/apply":
            self._send_json(SESSION.apply_pipeline(data))
        elif u.path == "/api/sort":
            self._send_json(SESSION.apply_sort_only(data.get("sort_mode", "auto")))
        elif u.path == "/api/reorder":
            with SESSION.lock:
                new_order = [n for n in data.get("order", []) if n in SESSION.layers]
                if len(new_order) == len(SESSION.order):
                    SESSION.order = new_order
            self._send_json(SESSION.get_state())
        elif u.path == "/api/visibility":
            with SESSION.lock:
                n = data.get("name")
                if n in SESSION.visible:
                    SESSION.visible[n] = bool(data.get("visible", True))
            self._send_json(SESSION.get_state())
        elif u.path == "/api/split":
            thr = data.get("threshold", None)
            thr = None if (thr == "" or thr is None) else float(thr)
            self._send_json(SESSION.split_layer(data["name"], threshold=thr, feather=float(data.get("feather", 1.0))))
        elif u.path == "/api/brush":
            self._send_json(SESSION.brush_stroke(data["name"], data.get("mode", "erase"), data.get("radius", 24), data.get("points", [])))
        elif u.path == "/api/undo":
            self._send_json(SESSION.undo())
        elif u.path == "/api/reset":
            with SESSION.lock:
                SESSION._reset_to_orig()
            self._send_json(SESSION.get_state())
        else:
            self.send_error(404)


def run_cli(args):
    psd_p = Path(args.cli)
    dep_p = Path(args.depth) if args.depth else psd_p.with_name(psd_p.stem + "_depth.psd")
    dep_b = dep_p.read_bytes() if dep_p.is_file() else None
    SESSION.load_psd_bytes(psd_p.read_bytes(), psd_p.name, dep_b)
    st = SESSION.apply_pipeline({
        "alpha_floor": args.alpha_floor,
        "min_area": args.min_area,
        "border_px": args.border_px,
        "decontam_radius": args.decontam,
        "collar_mode": args.collar,
        "depth_offset": args.depth_offset,
        "skin_diff": args.skin_diff,
        "feather": args.feather,
        "sort_mode": args.sort,
    })
    out_p = Path(args.output) if args.output else psd_p.with_name(psd_p.stem + "_clean.psd")
    out_p.write_bytes(SESSION.export_psd_bytes())
    print(f"[ OK ] {psd_p.name} -> {out_p} | {st.get('log', '')}")
    if args.preview:
        png_p = out_p.with_suffix(".png")
        png_p.write_bytes(SESSION.render_composite_png(max_dim=2048))
        print(f"[ OK ] Preview -> {png_p}")


def main():
    p = argparse.ArgumentParser(description="PSD Studio v2 for See-through / Anime2.5DRig (Web UI + CLI)")
    p.add_argument("--port", type=int, default=7865, help="Web UI port (default: 7865)")
    p.add_argument("--no-browser", action="store_true", help="Do not auto-open browser")
    p.add_argument("--cli", help="CLI mode: input .psd path")
    p.add_argument("-o", "--output", help="CLI mode: output .psd path")
    p.add_argument("--depth", help="CLI mode: optional *_depth.psd path")
    p.add_argument("--alpha-floor", type=int, default=16)
    p.add_argument("--min-area", type=int, default=60)
    p.add_argument("--border-px", type=int, default=0)
    p.add_argument("--decontam", type=int, default=0)
    p.add_argument("--collar", default="auto", choices=["auto", "skin_cut", "depth_sandwich", "off"])
    p.add_argument("--sort", default="auto", choices=["auto", "depth", "local_overlap", "classic", "arms_front", "arms_back", "hair_over_top", "raw"])
    p.add_argument("--depth-offset", type=float, default=0.0)
    p.add_argument("--skin-diff", type=float, default=24.0)
    p.add_argument("--feather", type=float, default=1.0)
    p.add_argument("--preview", action="store_true")
    args = p.parse_args()

    if args.cli:
        run_cli(args)
        return

    server = ThreadingHTTPServer(("127.0.0.1", args.port), StudioHandler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"PSD Studio v2 running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
