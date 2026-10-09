"""
Pillow frame renderer — draws a SnakeGame state to a PIL image using the same
dark flat-UI palette as the original pygame renderer (game.py). No pygame, so
it runs on a headless server (Hugging Face Spaces) and streams to the browser.
"""
import math

from PIL import Image, ImageDraw, ImageFont

from snake_game import BLOCK_SIZE, HUD_HEIGHT, Direction

# --- palette copied from the original game.py ---
BG_DARK = (13, 15, 20)
TILE_DARK = (19, 21, 27)
TILE_LIGHT = (25, 28, 35)
PANEL_BG = (17, 19, 25)
DIVIDER = (40, 43, 51)
TEXT_MUTED = (118, 124, 138)
TEXT_BRIGHT = (232, 234, 238)
ACCENT_GREEN = (46, 204, 113)

SNAKE_TAIL = (23, 111, 68)
SNAKE_BODY = (39, 174, 96)
SNAKE_HEAD = (88, 214, 141)
SNAKE_HEAD_SHINE = (170, 240, 200)
SNAKE_OUTLINE = (12, 58, 38)
EYE_DOT = (8, 10, 13)

FOOD_RED_GLOW = (196, 34, 52)
FOOD_RED_MID = (226, 55, 68)
FOOD_RED_CORE = (255, 84, 92)
FOOD_RED_HOT = (255, 154, 148)


def _load_font(size, bold=False):
    """Try a few common fonts; fall back to PIL's default if none found."""
    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""),
        "/System/Library/Fonts/Supplemental/Arial%s.ttf" % (" Bold" if bold else ""),
        "arial.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            continue
    return ImageFont.load_default()


LABEL_FONT = _load_font(13)
VALUE_FONT = _load_font(20, bold=True)


def _lerp_rgb(a, b, t):
    t = max(0.0, min(1.0, float(t)))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def render(game):
    """Return a PIL.Image (RGB) of the current game state."""
    img = Image.new("RGB", (game.w, game.h), BG_DARK)
    d = ImageDraw.Draw(img, "RGBA")

    # checkerboard playfield
    for x in range(0, game.w, BLOCK_SIZE):
        for y in range(HUD_HEIGHT, game.h, BLOCK_SIZE):
            color = TILE_DARK if (x // BLOCK_SIZE + y // BLOCK_SIZE) % 2 == 0 else TILE_LIGHT
            d.rectangle([x, y, x + BLOCK_SIZE, y + BLOCK_SIZE], fill=color)

    nseg = len(game.snake)

    # snake body (tail -> head gradient), draw body first so head sits on top
    for i, pt in enumerate(game.snake):
        px, py = int(pt.x), int(pt.y)
        cell = [px, py, px + BLOCK_SIZE - 1, py + BLOCK_SIZE - 1]
        if i == 0:
            continue
        t = (i - 1) / max(nseg - 2, 1)
        body_rgb = _lerp_rgb(SNAKE_TAIL, SNAKE_BODY, t)
        radius = 5 if i < nseg - 1 else 4
        d.rounded_rectangle(cell, radius=radius, fill=body_rgb, outline=SNAKE_OUTLINE, width=1)

    # snake head
    if nseg > 0:
        hpt = game.snake[0]
        px, py = int(hpt.x), int(hpt.y)
        cell = [px, py, px + BLOCK_SIZE - 1, py + BLOCK_SIZE - 1]
        d.rounded_rectangle(cell, radius=7, fill=SNAKE_HEAD, outline=SNAKE_OUTLINE, width=1)

        left, top, right, bottom = cell
        cx, cy = (left + right) // 2, (top + bottom) // 2
        # leading shine
        if game.direction == Direction.RIGHT:
            d.line([(left + 3, top + 4), (left + 3, bottom - 4)], fill=SNAKE_HEAD_SHINE, width=2)
        elif game.direction == Direction.LEFT:
            d.line([(right - 4, top + 4), (right - 4, bottom - 4)], fill=SNAKE_HEAD_SHINE, width=2)
        elif game.direction == Direction.UP:
            d.line([(left + 4, bottom - 4), (right - 4, bottom - 4)], fill=SNAKE_HEAD_SHINE, width=2)
        else:
            d.line([(left + 4, top + 3), (right - 4, top + 3)], fill=SNAKE_HEAD_SHINE, width=2)

        # eyes
        if game.direction == Direction.RIGHT:
            centers = [(right - 6, cy - 4), (right - 6, cy + 4)]
        elif game.direction == Direction.LEFT:
            centers = [(left + 6, cy - 4), (left + 6, cy + 4)]
        elif game.direction == Direction.UP:
            centers = [(cx - 4, top + 6), (cx + 4, top + 6)]
        else:
            centers = [(cx - 4, bottom - 6), (cx + 4, bottom - 6)]
        for ex, ey in centers:
            d.ellipse([ex - 3, ey - 3, ex + 3, ey + 3], fill=EYE_DOT)

    # food with soft pulsing glow
    _draw_food(img, game)

    # HUD panel
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, game.w, HUD_HEIGHT], fill=PANEL_BG)
    d.line([(0, HUD_HEIGHT - 2), (game.w, HUD_HEIGHT - 2)], fill=DIVIDER)
    d.line([(0, HUD_HEIGHT - 1), (game.w, HUD_HEIGHT - 1)], fill=(0, 0, 0))

    stats = [
        ("SCORE", str(game.score), ACCENT_GREEN),
        ("GEN", str(game.generation), TEXT_BRIGHT),
        ("LENGTH", str(len(game.snake)), TEXT_BRIGHT),
    ]
    x = 18
    cy = HUD_HEIGHT // 2
    for i, (label, value, value_color) in enumerate(stats):
        lw, lh = _text_size(d, label, LABEL_FONT)
        vw, vh = _text_size(d, value, VALUE_FONT)
        d.text((x, cy - vh - 1), label, font=LABEL_FONT, fill=TEXT_MUTED)
        d.text((x, cy + 3), value, font=VALUE_FONT, fill=value_color)
        x += max(lw, vw) + 26
        if i < len(stats) - 1:
            d.line([(x - 13, 12), (x - 13, HUD_HEIGHT - 12)], fill=DIVIDER, width=1)

    return img


def _draw_food(img, game):
    fx = int(game.food.x) + BLOCK_SIZE // 2
    fy = int(game.food.y) + BLOCK_SIZE // 2
    beat = 0.5 + 0.5 * math.sin(game.frame_iteration * 0.18)
    pulse = 0.35 + 0.65 * beat
    r_core = max(4, int(5 * pulse) + 3)
    r_glow = r_core + 9

    glow_size = r_glow * 2 + 6
    glow = Image.new("RGBA", (glow_size, glow_size), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    c = glow_size // 2

    def circle(draw, cx, cy, r, color):
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

    circle(gd, c, c, r_glow, (*FOOD_RED_GLOW, 45))
    circle(gd, c, c, int(r_glow * 0.65), (*FOOD_RED_MID, 90))
    circle(gd, c, c, r_core, (*FOOD_RED_CORE, 255))
    circle(gd, c - 2, c - 2, max(2, r_core // 2), (*FOOD_RED_HOT, 235))

    img.paste(glow, (fx - glow_size // 2, fy - glow_size // 2), glow)
