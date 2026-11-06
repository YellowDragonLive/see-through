"""验收分层 PSD 并生成白底合成图与图层拼版。

用法: python verify_psd.py <psd路径> [预览输出目录]
输出若干 KEY=VALUE 行，由 verify_psd.ps1 解析。
"""
import sys, os
from psd_tools import PSDImage
from PIL import Image, ImageDraw

psd_path = sys.argv[1]
out_dir = sys.argv[2] if len(sys.argv) > 2 else ""
psd = PSDImage.open(psd_path)
if psd is None:
    raise SystemExit("cannot open PSD")


def collect(layers, depth=0):
    out = []
    for l in layers:
        if l.is_group():
            out.extend(collect(l, depth + 1))
        else:
            out.append((l, depth))
    return out


flat = collect(psd)
print("PSD_SIZE=%dx%d" % (psd.size[0], psd.size[1]))
print("TOP_LEVEL=%d" % len(psd))
print("PIXEL_LAYERS=%d" % len(flat))


def layer_image(l):
    for fn in ("topil", "composite"):
        f = getattr(l, fn, None)
        if f is None:
            continue
        try:
            im = f()
            if im is not None:
                return im.convert("RGBA")
        except Exception:
            pass
    return None


def on_white(im):
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, im).convert("RGB")


try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS

imgs = []
nonempty = 0
for l, d in flat:
    im = layer_image(l)
    if im is not None and im.getchannel("A").getbbox() is not None:
        nonempty += 1
    imgs.append((l, im))

print("NONEMPTY=%d" % nonempty)
print("NAMES=" + " | ".join((l.name or "unnamed") for l, _ in flat))

if out_dir:
    stem = os.path.splitext(os.path.basename(psd_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    # psd.composite() 在本版本返回 RGB，透明区被填成黑色；
    # 这里按图层顺序手动叠到白底，更接近在 PS 里看到的效果。
    W, H = psd.size
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    for l, im in imgs:
        if im is None or not l.visible:
            continue
        patch = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        patch.paste(im, l.offset)
        canvas = Image.alpha_composite(canvas, patch)
    p = os.path.join(out_dir, stem + "_composite.png")
    canvas.convert("RGB").save(p)
    print("COMPOSITE=" + p)

    cols, cw, ch, thumb = 4, 250, 270, 230
    rows = max(1, (len(imgs) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * cw, rows * ch), (244, 245, 248))
    draw = ImageDraw.Draw(sheet)
    for i, (l, im) in enumerate(imgs):
        x, y = (i % cols) * cw, (i // cols) * ch
        draw.rectangle([x + 1, y + 1, x + cw - 2, y + ch - 2], outline=(214, 216, 222))
        if im is not None:
            t = on_white(im)
            t.thumbnail((thumb, thumb), RESAMPLE)
            sheet.paste(t, (x + (cw - t.width) // 2, y + 8 + (thumb - t.height) // 2))
        draw.text((x + 10, y + 244), (l.name or "unnamed")[:32], fill=(35, 38, 45))
    p = os.path.join(out_dir, stem + "_layers.png")
    sheet.save(p)
    print("SHEET=" + p)
