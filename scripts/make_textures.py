"""
Generates the printed artwork used by the bike: gauge dial faces (speedometer, tachometer, LCD)
and the tank emblem mask. Pure Python + Pillow, no Blender needed.

    python3 scripts/make_textures.py            # writes bike/textures/*.png
"""
import math
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "bike", "textures")
os.makedirs(OUT, exist_ok=True)
SS = 2                      # supersampling factor
N = 1024
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_M = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
for f in (FONT_B, FONT_R, FONT_M):
    if not os.path.exists(f):
        raise SystemExit("missing font " + f)


def font(path, px):
    return ImageFont.truetype(path, int(px * SS))


def polar(c, r, deg):
    a = math.radians(deg)
    return c[0] + r * math.cos(a), c[1] - r * math.sin(a)


def tick(d, c, r0, r1, deg, w, col):
    d.line([polar(c, r0, deg), polar(c, r1, deg)], fill=col, width=int(w * SS))


def text_c(d, xy, s, f, col):
    b = d.textbbox((0, 0), s, font=f)
    d.text((xy[0] - (b[2] - b[0]) / 2 - b[0], xy[1] - (b[3] - b[1]) / 2 - b[1]), s, font=f, fill=col)


def finish(img, name):
    img = img.resize((N, N), Image.LANCZOS)
    img.save(os.path.join(OUT, name))
    print("wrote", name)


def dial_base():
    S = N * SS
    img = Image.new("RGB", (S, S), (14, 14, 15))
    d = ImageDraw.Draw(img)
    c = (S / 2, S / 2)
    # subtle vignette
    for i in range(40):
        r = S * 0.5 * (1 - i / 40)
        g = int(14 + 10 * (i / 40) ** 2)
        d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=(g, g, g + 1))
    d.ellipse([c[0] - S * 0.495, c[1] - S * 0.495, c[0] + S * 0.495, c[1] + S * 0.495], outline=(90, 90, 92), width=4 * SS)
    return img, d, c, S


def speedo():
    img, d, c, S = dial_base()
    emit = Image.new("RGB", (S, S), (0, 0, 0))
    de = ImageDraw.Draw(emit)
    R = S * 0.5
    white = (236, 234, 224)
    amber = (232, 150, 40)
    vmax = 180

    def ang(v):
        return 225 - 270 * v / vmax
    for v in range(0, vmax + 1, 2):
        if v % 20 == 0:
            tick(d, c, R * 0.80, R * 0.94, ang(v), 9, white)
        elif v % 10 == 0:
            tick(d, c, R * 0.85, R * 0.94, ang(v), 6, white)
        elif v % 5 == 0 or True:
            tick(d, c, R * 0.895, R * 0.94, ang(v), 3, (170, 170, 165))
    fb = font(FONT_B, 66)
    for v in range(0, vmax + 1, 20):
        text_c(d, polar(c, R * 0.67, ang(v)), str(v), fb, white)
    # inner mph scale
    mmax = vmax / 1.609
    fs = font(FONT_B, 30)
    for m in range(0, int(mmax) + 1, 10):
        a = ang(m * 1.609)
        tick(d, c, R * 0.52, R * 0.57, a, 4, amber)
        if m % 20 == 0:
            text_c(d, polar(c, R * 0.45, a), str(m), fs, amber)
    d.arc([c[0] - R * 0.57, c[1] - R * 0.57, c[0] + R * 0.57, c[1] + R * 0.57], start=-225, end=45 - 270 * (1 - mmax * 1.609 / vmax),
          fill=amber, width=2 * SS)
    text_c(d, (c[0], c[1] - R * 0.30), "km/h", font(FONT_B, 40), white)
    text_c(d, (c[0], c[1] - R * 0.20), "mph", font(FONT_R, 26), amber)
    text_c(d, (c[0], c[1] + R * 0.86), "WASTELAND  MOTORWORKS", font(FONT_B, 22), (120, 120, 118))
    # LCD odometer window
    x0, x1, y0, y1 = c[0] - R * 0.34, c[0] + R * 0.34, c[1] + R * 0.38, c[1] + R * 0.68
    d.rounded_rectangle([x0, y0, x1, y1], radius=10 * SS, fill=(22, 30, 25), outline=(70, 70, 72), width=3 * SS)
    de.rounded_rectangle([x0, y0, x1, y1], radius=10 * SS, fill=(6, 12, 8))
    lcd = (150, 225, 170)
    fm = font(FONT_M, 64)
    for dd in (d, de):
        dd.text((x0 + R * 0.04, y0 + R * 0.025), "012847", font=fm, fill=lcd)
        dd.text((x0 + R * 0.04, y0 + R * 0.20), "ODO", font=font(FONT_B, 22), fill=lcd)
        dd.text((x0 + R * 0.26, y0 + R * 0.20), "km", font=font(FONT_B, 22), fill=lcd)
        # fuel bar segments
        for k in range(8):
            fill = lcd if k < 5 else (40, 60, 48)
            bx = x0 + R * 0.42 + k * R * 0.028
            dd.rectangle([bx, y0 + R * 0.205, bx + R * 0.018, y0 + R * 0.255], fill=fill)
    d.ellipse([c[0] - R * 0.11, c[1] - R * 0.11, c[0] + R * 0.11, c[1] + R * 0.11], fill=(28, 28, 30))
    finish(img, "dial_speedo.png")
    finish(emit, "dial_speedo_emit.png")


def tacho():
    img, d, c, S = dial_base()
    R = S * 0.5
    white = (236, 234, 224)
    red = (215, 35, 25)
    rmax = 10

    def ang(v):
        return 225 - 270 * v / rmax
    d.arc([c[0] - R * 0.9, c[1] - R * 0.9, c[0] + R * 0.9, c[1] + R * 0.9], start=-ang(8), end=-ang(10), fill=red, width=int(R * 0.09))
    for i in range(0, rmax * 4 + 1):
        v = i / 4
        if i % 4 == 0:
            tick(d, c, R * 0.76, R * 0.94, ang(v), 10, white)
        elif i % 2 == 0:
            tick(d, c, R * 0.83, R * 0.94, ang(v), 6, white)
        else:
            tick(d, c, R * 0.88, R * 0.94, ang(v), 4, (170, 170, 165))
    fb = font(FONT_B, 92)
    for v in range(0, rmax + 1):
        text_c(d, polar(c, R * 0.60, ang(v)), str(v), fb, red if v >= 8 else white)
    text_c(d, (c[0], c[1] + R * 0.40), "x1000 r/min", font(FONT_B, 40), white)
    d.ellipse([c[0] - R * 0.13, c[1] - R * 0.13, c[0] + R * 0.13, c[1] + R * 0.13], fill=(28, 28, 30))
    finish(img, "dial_tacho.png")


def emblem():
    """Tank badge mask: ring + crown + three jewels + base band (white on black)."""
    S = N * SS
    img = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(img)
    ext = 0.08          # design units across the image

    def P(x, y):
        return ((x / ext + 0.5) * S, (0.5 - (y - 0.012) / ext) * S)
    r = 0.034
    w = int(0.0047 / ext * S)
    cx, cy = P(0, 0.012)
    rr = r / ext * S
    d.ellipse([cx - rr - w / 2, cy - rr - w / 2, cx + rr + w / 2, cy + rr + w / 2], outline=255, width=w)
    crown = [(-0.02, -0.004), (-0.023, 0.024), (-0.01, 0.009), (0.0, 0.031), (0.01, 0.009), (0.023, 0.024), (0.02, -0.004)]
    pts = [P(*p) for p in crown]
    d.line(pts + [pts[0]], fill=255, width=int(0.0042 / ext * S), joint="curve")
    d.line([P(-0.021, -0.011), P(0.021, -0.011)], fill=255, width=int(0.0042 / ext * S))
    for jx, jy in ((-0.023, 0.028), (0.0, 0.035), (0.023, 0.028)):
        x, y = P(jx, jy)
        jr = 0.0032 / ext * S
        d.ellipse([x - jr, y - jr, x + jr, y + jr], fill=255)
    img = img.filter(ImageFilter.GaussianBlur(SS * 0.6))
    img.resize((N, N), Image.LANCZOS).save(os.path.join(OUT, "emblem_crown_mask.png"))
    print("wrote emblem_crown_mask.png")


if __name__ == "__main__":
    speedo()
    tacho()
    emblem()
