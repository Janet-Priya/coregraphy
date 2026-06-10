"""Hand-authored pixel art for the cozy dashboard.

Art is defined as char grids; PALETTE maps each char to a hex colour (or None
for transparent). Grids render to crisp SVG (<rect> per pixel) for the web and
can also be rendered to PNG via PIL for visual verification.
"""
from __future__ import annotations

# char -> hex (None = transparent)
PALETTE = {
    ".": None,
    "K": "#14110d",  # hard outline
    "o": "#2a2620",  # soft outline
    "G": "#83a85f",  # leaf light
    "e": "#6b8c4a",  # leaf mid
    "g": "#52703c",  # leaf dark
    "W": "#6e5238",  # wood
    "w": "#4f3a28",  # wood dark
    "P": "#c87f56",  # terracotta
    "p": "#9d5f3c",  # terracotta dark
    "B": "#48473c",  # monitor body
    "b": "#2e2d26",  # monitor dark
    "S": "#22323d",  # screen frame
    "s": "#1a2730",  # screen bg
    "i": "#d98b78",  # rose node
    "N": "#d8b25e",  # gold node
    "c": "#8fae7a",  # green node
    "L": "#3f5d72",  # node line
    "M": "#d98b78",  # mug
    "m": "#b06b58",  # mug dark
    "C": "#ece4d2",  # cream / steam / highlight
    "F": "#e0a6a0",  # flower petal
    "f": "#c97f9a",  # flower petal dark
    "y": "#d8b25e",  # flower centre
    "r": "#c0584f",  # mushroom cap
    "R": "#8f3f39",  # mushroom cap dark
    "u": "#ece4d2",  # mushroom stem / spot
}

# --- art ---------------------------------------------------------------

PLANT = """
....g..g....
...gGggGg...
..gGGGGGGg..
.gGGeGGeGGg.
.gGGGGGGGGg.
..gGGeeGGg..
...gGGGGg...
....geeg....
.....gg.....
.....KK.....
....PPPP....
...PPPPPP...
...PppppP...
...PppppP...
...pPPPPp...
....pppp....
"""

MUG = """
........
.CCCC...
CMMMMMC.
MMMMMmM.
MMMMMmM.
mMMMMmm.
.mmmm...
........
"""

FLOWER = """
..F.F..
.FfyfF.
.FyyyyF
.FfyfF.
..FfF..
...g...
..ge...
"""

MUSHROOM = """
.rrrrr.
rRururR
rrrrrrr
.RuuuR.
..uuu..
..uuu..
"""

LEAF = """
....g....
..gGGGg..
.gGeGGGg.
gGGeGGGGg
.gGGGGGg.
..gGGGg..
....g....
"""

STAR = """
..N..
N.N.N
.NNN.
N.N.N
..N..
"""

# Monitor: '#' cells are the live screen region (filled procedurally).
MONITOR = """
..................
...oBBBBBBBBBBo...
..oBSSSSSSSSSSBo..
..BS##########SB..
..BS##########SB..
..BS##########SB..
..BS##########SB..
..BS##########SB..
..BS##########SB..
..oBSSSSSSSSSSBo..
...oBBBBBBBBBBo...
......BBBB........
......BwwB........
....BBBBBBBBBB....
..................
"""


def grid(art: str) -> list[str]:
    rows = [r for r in art.splitlines() if r != ""]
    w = max(len(r) for r in rows)
    return [r.ljust(w, ".") for r in rows]


def screen_nodes(rows: list[str]):
    """Procedural node-map drawn into the monitor '#' region.

    Returns (nodes, edges, bbox) in grid-cell coordinates. Deterministic — no
    RNG (kept reproducible for resume/builds)."""
    cells = [(x, y) for y, row in enumerate(rows) for x, ch in enumerate(row) if ch == "#"]
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    # hand-placed constellation, colours cycle rose/gold/green
    rel = [
        (0.10, 0.20, "i"), (0.30, 0.08, "c"), (0.18, 0.55, "N"),
        (0.45, 0.35, "i"), (0.38, 0.78, "c"), (0.62, 0.18, "N"),
        (0.72, 0.52, "i"), (0.58, 0.68, "c"), (0.88, 0.30, "N"),
        (0.82, 0.80, "i"), (0.95, 0.62, "c"), (0.05, 0.85, "N"),
    ]
    nodes = []
    for fx, fy, col in rel:
        gx = x0 + fx * (x1 - x0)
        gy = y0 + fy * (y1 - y0)
        nodes.append((gx, gy, col))
    edges = [(0, 1), (0, 2), (1, 5), (3, 5), (3, 6), (6, 8), (6, 7),
             (7, 9), (2, 4), (4, 7), (8, 10), (2, 11), (4, 11)]
    return nodes, edges, (x0, x1, y0, y1)


def to_svg(art: str, scale: int = 1, with_screen: bool = False) -> str:
    rows = grid(art)
    h = len(rows)
    w = len(rows[0])
    parts = [
        f'<svg viewBox="0 0 {w} {h}" width="{w*scale}" height="{h*scale}" '
        f'xmlns="http://www.w3.org/2000/svg" shape-rendering="crispEdges" '
        f'style="image-rendering:pixelated">'
    ]
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            key = "s" if ch == "#" else ch
            col = PALETTE.get(key)
            if col:
                parts.append(f'<rect x="{x}" y="{y}" width="1" height="1" fill="{col}"/>')
    if with_screen:
        nodes, edges, _ = screen_nodes(rows)
        for a, b in edges:
            x1, y1, _ = nodes[a]
            x2, y2, _ = nodes[b]
            parts.append(
                f'<line x1="{x1+0.5:.2f}" y1="{y1+0.5:.2f}" x2="{x2+0.5:.2f}" '
                f'y2="{y2+0.5:.2f}" stroke="{PALETTE["L"]}" stroke-width="0.35"/>'
            )
        for gx, gy, col in nodes:
            parts.append(
                f'<rect x="{gx:.2f}" y="{gy:.2f}" width="1" height="1" fill="{PALETTE[col]}"/>'
            )
    parts.append("</svg>")
    return "".join(parts)


def to_png(art: str, scale: int, path: str, with_screen: bool = False):
    """Render to PNG via PIL for visual verification."""
    from PIL import Image
    rows = grid(art)
    h, w = len(rows), len(rows[0])
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    px = img.load()
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            key = "s" if ch == "#" else ch
            col = PALETTE.get(key)
            if col:
                px[x, y] = _hex(col)
    if with_screen:
        nodes, edges, _ = screen_nodes(rows)
        from PIL import ImageDraw
        d = ImageDraw.Draw(img)
        for a, b in edges:
            x1, y1, _ = nodes[a]
            x2, y2, _ = nodes[b]
            d.line([(x1 + 0.5, y1 + 0.5), (x2 + 0.5, y2 + 0.5)], fill=_hex(PALETTE["L"]), width=1)
        for gx, gy, col in nodes:
            ix, iy = int(round(gx)), int(round(gy))
            px[ix, iy] = _hex(PALETTE[col])
    img = img.resize((w * scale, h * scale), Image.NEAREST)
    img.save(path)


def _hex(h: str):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)


if __name__ == "__main__":
    # contact sheet for visual verification
    from PIL import Image
    items = [
        ("MONITOR", MONITOR, True), ("PLANT", PLANT, False), ("MUG", MUG, False),
        ("FLOWER", FLOWER, False), ("MUSHROOM", MUSHROOM, False),
        ("LEAF", LEAF, False), ("STAR", STAR, False),
    ]
    scale = 12
    pad = 16
    rendered = []
    for name, art, scr in items:
        rows = grid(art)
        h, w = len(rows), len(rows[0])
        sub = Image.new("RGBA", (w * scale, h * scale), (0, 0, 0, 0))
        tmp = "/tmp/_pa.png"
        to_png(art, scale, tmp, with_screen=scr)
        rendered.append((name, Image.open(tmp).copy()))
    sheet_w = sum(im.width for _, im in rendered) + pad * (len(rendered) + 1)
    sheet_h = max(im.height for _, im in rendered) + pad * 2
    sheet = Image.new("RGBA", (sheet_w, sheet_h), (22, 19, 16, 255))
    x = pad
    for name, im in rendered:
        sheet.paste(im, (x, pad), im)
        x += im.width + pad
    sheet.save("/tmp/pixel_contact_sheet.png")
    print("wrote /tmp/pixel_contact_sheet.png")
