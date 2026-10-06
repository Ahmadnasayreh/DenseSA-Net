"""Combine per-dataset figures into one figure with (a), (b), (c) labels.

    python scripts/merge_panels.py --inputs d1.png d2.png d3.png --output fig.png --layout v

For very tall panels (e.g. the saliency maps) use --layout h with
--max-width, which pastes the panels at full resolution and downsamples
the result once at the end.
"""
import argparse

from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None


def font(size):
    for f in ("DejaVuSerif-Bold.ttf", "timesbd.ttf", "arialbd.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def merge(images, layout, label_size=62, gap=40, margin=30):
    fnt = font(label_size)
    band = label_size + 14
    labels = [f"({chr(97 + i)})" for i in range(len(images))]
    if layout == "v":
        w = max(im.width for im in images)
        images = [im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
                  if im.width != w else im for im in images]
        h = 2 * margin + sum(im.height for im in images) + band * len(images) \
            + gap * (len(images) - 1)
        canvas = Image.new("RGB", (w + 2 * margin, h), "white")
        draw, y = ImageDraw.Draw(canvas), margin
        for im, lab in zip(images, labels):
            draw.text((margin, y), lab, fill="black", font=fnt)
            y += band
            canvas.paste(im, (margin, y))
            y += im.height + gap
    else:
        # panels keep their own height and are aligned on the top edge
        w = 2 * margin + sum(im.width for im in images) + gap * (len(images) - 1)
        h = 2 * margin + band + max(im.height for im in images)
        canvas = Image.new("RGB", (w, h), "white")
        draw, x = ImageDraw.Draw(canvas), margin
        for im, lab in zip(images, labels):
            draw.text((x, margin), lab, fill="black", font=fnt)
            canvas.paste(im, (x, margin + band))
            x += im.width + gap
    return canvas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--layout", choices=["v", "h"], default="v")
    p.add_argument("--max-width", type=int, default=None)
    p.add_argument("--dpi", type=int, default=600)
    a = p.parse_args()
    canvas = merge([Image.open(f).convert("RGB") for f in a.inputs], a.layout)
    if a.max_width and canvas.width > a.max_width:
        s = a.max_width / canvas.width
        canvas = canvas.resize((a.max_width, round(canvas.height * s)), Image.LANCZOS)
    canvas.save(a.output, dpi=(a.dpi, a.dpi), optimize=True)
    print(f"{a.output}: {canvas.width} x {canvas.height}")


if __name__ == "__main__":
    main()
