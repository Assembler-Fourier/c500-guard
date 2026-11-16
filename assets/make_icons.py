"""BarrierWatcher app icons: the dashboard's barrier logo (lime post, striped arm) on a dark tile."""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\c500-guard\assets")
target.mkdir(parents=True, exist_ok=True)
BG, EDGE = (16, 18, 16, 255), (58, 66, 52, 255)
LIME, WHITE, RED, DARK = (216, 247, 138, 255), (238, 240, 233, 255), (255, 107, 87, 255), (16, 18, 16, 255)
GREY_POST, GREY_ARM, GREY_STRIPE, POWER = (120, 128, 112, 255), (170, 174, 165, 255), (110, 116, 106, 255), (255, 137, 111, 255)
S = 256 / 40      # the logo is drawn on a 40x40 grid, like the dashboard SVG


def box(x, y, w, h):
    return [round(x * S), round(y * S), round((x + w) * S), round((y + h) * S)]


def logo(draw, post, arm, stripe, offset=(0, 0)):
    ox, oy = offset

    def r(x, y, w, h, radius, fill):
        draw.rounded_rectangle(box(x + ox, y + oy, w, h), radius=round(radius * S), fill=fill)
    r(4, 7, 8, 27, 2.5, post)
    r(2, 32, 12, 5, 2, post)
    r(12, 10, 26, 8, 2.5, arm)
    for x, w in ((16.5, 4.5), (25.5, 4.5)):
        draw.rectangle(box(x + ox, 10 + oy, w, 8), fill=stripe)
    r(34, 10, 4, 8, 1.5, stripe)
    draw.ellipse(box(8 - 2.2 + ox, 14 - 2.2 + oy, 4.4, 4.4), fill=DARK)


def tile():
    image = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle([6, 6, 250, 250], radius=50, fill=BG, outline=EDGE, width=6)
    return image, draw


def save(image, name):
    image.save(target / name, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])


# The logo, scaled into the tile with a margin.
image, draw = tile()
inner = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
logo(ImageDraw.Draw(inner), LIME, WHITE, RED)
inner = inner.resize((196, 196), Image.LANCZOS)
image.alpha_composite(inner, (30, 34))
save(image, "barrierwatcher.ico")

image, draw = tile()
inner = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
logo(ImageDraw.Draw(inner), GREY_POST, GREY_ARM, GREY_STRIPE)
inner = inner.resize((196, 196), Image.LANCZOS)
image.alpha_composite(inner, (30, 34))
draw = ImageDraw.Draw(image)
cx, cy, rad = 186, 186, 42
draw.ellipse([cx - rad - 12, cy - rad - 12, cx + rad + 12, cy + rad + 12], fill=BG)
draw.arc([cx - rad, cy - rad, cx + rad, cy + rad], start=-60, end=240, fill=POWER, width=14)
draw.line([cx, cy - rad - 6, cx, cy + 4], fill=POWER, width=14)
save(image, "barrierwatcher-off.ico")
print("icons written to", target)
