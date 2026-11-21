#!/usr/bin/env python3
"""render_previews.py — 把 fix 后的 PSD 渲染成可直接查看的 PNG 预览 + HTML 索引页

用途：PSD 无法直接肉眼验收。本脚本复用 fix.py 的 read_psd / composite_layers，
      逐张合成 PNG（棋盘底，透明区可见），并额外生成「图层条带」大图，
      一眼就能看清拆层效果（handwear_1/2、back hair_1/2、五官顺序）。

用法：
  python render_previews.py                       # 扫描 workspace 下所有 *_fixed.psd
  python render_previews.py --size 640            # 主图长边 640
  python render_previews.py --no-layers           # 不出图层条带（快）
  python render_previews.py --root /path --out /path/out
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from fix import composite_layers, read_psd  # noqa: E402


def checker(W: int, H: int, cell: int = 16) -> Image.Image:
    """浅灰棋盘底，让透明区域一眼可辨。"""
    yy, xx = np.mgrid[0:H, 0:W]
    mask = ((yy // cell) + (xx // cell)) % 2 == 0
    arr = np.where(mask[:, :, None], np.array([255, 255, 255], np.uint8), np.array([226, 226, 230], np.uint8))
    return Image.fromarray(arr, "RGB")


def flatten(comp: np.ndarray) -> Image.Image:
    H, W = comp.shape[:2]
    bg = checker(W, H)
    fg = Image.fromarray(comp, "RGBA")
    bg.paste(fg, (0, 0), fg)
    return bg


def _font(size: int) -> ImageFont.ImageFont:
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def layer_strip(layers: list[dict], W: int, H: int, cols: int = 8, tw: int = 150, th: int = 120) -> Image.Image:
    """把每一层单独渲染成小格，横向铺开，标上图层名与像素占比。"""
    items = [L for L in layers if L["rgba"].size and L.get("visible", True)]
    if not items:
        return Image.new("RGB", (tw, th), (240, 240, 242))
    rows = (len(items) + cols - 1) // cols
    pad, label_h = 8, 20
    cell_w, cell_h = tw + pad, th + label_h + pad
    sheet = Image.new("RGB", (cols * cell_w + pad, rows * cell_h + pad), (248, 248, 250))
    d = ImageDraw.Draw(sheet)
    f = _font(11)

    total_px = sum(int((L["rgba"][:, :, 3] > 0).sum()) for L in items) or 1
    for i, L in enumerate(items):
        cx = pad + (i % cols) * cell_w
        cy = pad + (i // cols) * cell_h
        full = np.zeros((H, W, 4), np.uint8)
        rgba = L["rgba"]
        lx, ly = max(0, L["left"]), max(0, L["top"])
        x1 = min(W, L["left"] + rgba.shape[1])
        y1 = min(H, L["top"] + rgba.shape[0])
        if x1 > lx and y1 > ly:
            full[ly:y1, lx:x1] = rgba[ly - L["top"]: y1 - L["top"], lx - L["left"]: x1 - L["left"]]
        tile = flatten(full)
        ys, xs = np.nonzero(full[:, :, 3] > 0)
        if xs.size:
            box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            tile = tile.crop(box)
        tile.thumbnail((tw, th), Image.LANCZOS)
        bx = cx + (tw - tile.width) // 2
        by = cy + (th - tile.height) // 2
        sheet.paste(tile, (bx, by))
        d.rectangle([cx, cy, cx + tw, cy + th], outline=(214, 214, 220))
        pct = 100.0 * int((rgba[:, :, 3] > 0).sum()) / total_px
        pct_s = "<0.1%" if pct < 0.1 else (f"{pct:.1f}%" if pct < 10 else f"{pct:.0f}%")
        name = L["name"] if len(L["name"]) <= 20 else L["name"][:19] + "…"
        d.text((cx + 2, cy + th + 3), f"{i + 1}.{name} {pct_s}", fill=(60, 60, 66), font=f)
    return sheet


def flat_name(rel: Path) -> str:
    s = str(rel.with_suffix("")).replace("/", "__").replace(" ", "_")
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--size", type=int, default=560, help="主图长边像素")
    ap.add_argument("--no-layers", action="store_true")
    ap.add_argument("--include-old", action="store_true", help="把 old_psd/ 归档区也算进来（默认跳过）")
    ap.add_argument("--title", default="see-through · fix 后全量预览")
    args = ap.parse_args()

    root = args.root.resolve()
    out = (args.out or root / "preview" / "fixall").resolve()
    out.mkdir(parents=True, exist_ok=True)

    files = []
    for p in root.rglob("*_fixed.psd"):
        rel = p.relative_to(root)
        # 只跳过 root 底下的 old_psd 归档区（root 自身就是 old_psd 时照常处理）
        if not args.include_old and rel.parts and rel.parts[0] == "old_psd":
            continue
        files.append(p)
    files.sort()
    if not files:
        print("没有找到 *_fixed.psd"); return 1
    print(f"共 {len(files)} 张 → {out}")

    cards = []
    for i, p in enumerate(files, 1):
        rel = p.relative_to(root)
        name = flat_name(rel)
        try:
            psd = read_psd(p)
            W, H = psd["width"], psd["height"]
            comp = composite_layers(psd["layers"], W, H)
            img = flatten(comp)
            img.thumbnail((args.size, args.size), Image.LANCZOS)
            img.save(out / f"{name}.png")

            strip_rel = ""
            if not args.no_layers:
                st = layer_strip(psd["layers"], W, H)
                if st.width > 1600:
                    k = 1600 / st.width
                    st = st.resize((1600, max(1, int(st.height * k))), Image.LANCZOS)
                st.save(out / f"{name}__layers.png")
                strip_rel = f"{name}__layers.png"

            cards.append({
                "rel": str(rel), "name": name, "w": W, "h": H,
                "n": len(psd["layers"]),
                "order": " < ".join(L["name"] for L in psd["layers"]),
                "strip": strip_rel,
            })
            print(f"  [{i}/{len(files)}] {rel}  {W}x{H}  {len(psd['layers'])} 层")
        except Exception as e:
            print(f"  [FAIL] {rel}: {e}")

    # ---------- HTML ----------
    rows = []
    for c in cards:
        strip_html = (
            f'<details><summary>展开 {c["n"]} 层分层预览</summary>'
            f'<img class="strip" loading="lazy" src="{c["strip"]}" alt="layers"></details>'
            if c["strip"] else ""
        )
        rows.append(f"""
    <article class="card">
      <a href="{c['name']}.png" target="_blank"><img class="main" loading="lazy" src="{c['name']}.png" alt=""></a>
      <h3>{html.escape(c['rel'])}</h3>
      <div class="meta"><span class="tag">{c['w']}×{c['h']}</span><span class="tag">{c['n']} 层</span></div>
      <div class="order">{html.escape(c['order'])}</div>
      {strip_html}
    </article>""")

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(args.title)}</title>
<style>
  :root{{--bg:#f6f7f9;--fg:#1c1d21;--sub:#6b6d76;--line:#e3e4e9;--card:#fff;--accent:#3b6ef5}}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--fg);
    font:14px/1.6 -apple-system,"Segoe UI","Noto Sans CJK SC","PingFang SC",sans-serif}}
  header{{padding:28px 32px 18px;border-bottom:1px solid var(--line);background:var(--card)}}
  h1{{margin:0 0 6px;font-size:20px;font-weight:650;letter-spacing:.2px}}
  .sub{{color:var(--sub);font-size:13px}}
  main{{padding:24px 32px 60px;display:grid;gap:20px;
    grid-template-columns:repeat(auto-fill,minmax(320px,1fr))}}
  .card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;
    box-shadow:0 1px 2px rgba(16,18,24,.04);transition:box-shadow .15s}}
  .card:hover{{box-shadow:0 6px 20px rgba(16,18,24,.08)}}
  .main{{width:100%;border-radius:8px;display:block;background:#fafafa}}
  h3{{margin:10px 2px 4px;font-size:12.5px;font-weight:600;word-break:break-all;color:#33353c}}
  .meta{{display:flex;gap:6px;margin:0 2px}}
  .tag{{font-size:11px;color:#4a4d57;background:#eef0f4;border-radius:5px;padding:1px 7px}}
  .order{{margin:8px 2px 0;font-size:11px;color:var(--sub);line-height:1.5;word-break:break-word}}
  details{{margin-top:8px}}
  summary{{cursor:pointer;font-size:11.5px;color:var(--accent);outline:none}}
  .strip{{width:100%;margin-top:8px;border:1px solid var(--line);border-radius:6px}}
</style></head><body>
<header>
  <h1>{html.escape(args.title)}</h1>
  <div class="sub">{len(cards)} 张 · 棋盘格=透明区 · 点主图看原尺寸 · 展开可逐层验收拆层与排序</div>
</header>
<main>{''.join(rows)}</main>
</body></html>"""

    idx = out / "index.html"
    idx.write_text(doc, encoding="utf-8")
    print(f"\n完成：{len(cards)} 张 → {idx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
