"""OCR-confirmed redraw of receipt values.

Small receipt text in a chat screenshot can be too few pixels for any filter
to make it crisp.  This module reads each value with an offline OCR engine
(RapidOCR, ONNX), lets the user confirm or correct every value, and only then
re-typesets the confirmed values in a clean font at their original position,
size and colour.  Nothing is redrawn without an explicit confirmation, and
values the user does not tick keep their enhanced pixels.

Coordinates in every item are in *source* pixels; `redraw_values` maps them to
the enhanced image with its scale factor.
"""

from __future__ import annotations

import base64
import glob
import io
import os
import re
import sys
import threading
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OCR_MAX_SIDE = 2600          # longest side of a region after OCR upscaling
LOW_CONFIDENCE = 0.80        # below this the UI marks a value for extra checking
_ENGINE = None
_ENGINE_LOCK = threading.Lock()

FONT_CANDIDATES = {
    "serif": [
        "timesbd.ttf", "Times New Roman Bold.ttf", "LiberationSerif-Bold.ttf",
        "Tinos-Bold.ttf", "DejaVuSerif-Bold.ttf", "NotoSerif-Bold.ttf",
    ],
    "serif-regular": [
        "times.ttf", "Times New Roman.ttf", "LiberationSerif-Regular.ttf",
        "Tinos-Regular.ttf", "DejaVuSerif.ttf", "NotoSerif-Regular.ttf",
    ],
    "sans": [
        "arialbd.ttf", "Arial Bold.ttf", "LiberationSans-Bold.ttf",
        "Arimo-Bold.ttf", "DejaVuSans-Bold.ttf", "NotoSans-Bold.ttf",
    ],
}
FONT_DIRS = [
    os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
    "/usr/share/fonts", "/usr/local/share/fonts", "/Library/Fonts",
    "/System/Library/Fonts", os.path.expanduser("~/.fonts"),
]


class OcrUnavailable(RuntimeError):
    pass


def ocr_available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def _engine():
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as error:
                raise OcrUnavailable(
                    "OCR redraw needs the 'rapidocr-onnxruntime' package. "
                    "Run: pip install -r requirements-desktop.txt"
                ) from error
            _ENGINE = RapidOCR()
        return _ENGINE


def _run_ocr(rgb: np.ndarray):
    """Return [(quad_points, text, confidence)] for every text line found."""
    engine = _engine()
    with _ENGINE_LOCK:  # onnxruntime sessions are shared; keep calls serial
        result, _ = engine(np.ascontiguousarray(rgb))
    if not result:
        return []
    return [(np.asarray(q, np.float32), str(t), float(c)) for q, t, c in result]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _ink_mask(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """Boolean ink mask of a small crop and its paper luma."""
    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    paper = float(np.percentile(luma, 90))
    ink_level = float(np.percentile(luma, 0.5))
    if paper - ink_level < 40:
        return np.zeros(luma.shape, bool), paper
    return luma < (paper + ink_level) / 2.0, paper


def _measure(rgb_img: np.ndarray, box, force_underline: bool = False):
    """Tighten a box to its glyphs, find an underline, and measure colours.

    OCR boxes in bordered fields (Date, TO, table headers) include parts of
    the border, so long thin components spanning the crop are treated as
    rules and ignored; a thin horizontal stroke just under the text and about
    as wide as it is kept as the value's underline.
    """
    h, w = rgb_img.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(0, x0 - 2), max(0, y0 - 2)
    x1, y1 = min(w, x1 + 2), min(h, y1 + 2)
    crop = rgb_img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    ink, _ = _ink_mask(crop)
    if not ink.any():
        return None
    H, W = ink.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    text = np.zeros_like(ink)
    bars = []          # thin horizontal strokes: underline, rule or hyphen
    for i in range(1, count):
        cx, cy, cw, ch, area = stats[i]
        spans_x = cx <= 0 and cx + cw >= W            # touches both padded crop edges
        spans_y = cy <= 0 and cy + ch >= H
        if (ch >= 0.85 * H and cw <= max(2, 0.2 * W)) or (spans_x and ch <= 0.5 * H) or (spans_y and cw <= 0.5 * W):
            continue                                   # rule / box edge (the crop is padded, glyphs never span it)
        if spans_x or spans_y:
            # A rule with glyphs touching it: drop the rule rows/columns and
            # keep whatever is left as text.
            comp = labels == i
            if spans_x:
                comp[comp.sum(axis=1) >= 0.8 * W, :] = False
            if spans_y:
                comp[:, comp.sum(axis=0) >= 0.8 * H] = False
            n, lab, st, _ = cv2.connectedComponentsWithStats(comp.astype(np.uint8), connectivity=8)
            for j in range(1, n):
                if st[j, cv2.CC_STAT_AREA] >= 4:        # specks left by the rule are not text
                    text |= lab == j
            continue
        if ch <= max(2, 0.25 * H) and cw >= max(3, 3 * ch):
            bars.append((i, cx, cy, cw, ch))
        else:
            text |= labels == i
    if not text.any():
        if not bars:
            return None
        for i, *_ in bars:                             # text made only of bars ("-", "=")
            text |= labels == i
        bars = []
    ys, xs = np.where(text)
    tx0, tx1, ty0, ty1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1

    underline = None
    tw = tx1 - tx0
    paper_luma = float(np.percentile(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), 90))
    for i, cx, cy, cw, ch in bars:
        below = cy >= ty0 + (ty1 - ty0) * 0.6
        # An underline is separated from the glyphs by at least one paper row;
        # a bar touching them is part of a glyph (the base of a "2").
        # Faint diagonals can split a glyph in the mask, so the text above the
        # bar must also be most of the height down to it (not the top half of a "2").
        separated = cy > ty1 and (ty1 - ty0) >= 0.65 * (cy - ty0)
        if below and separated and underline is None and 0.6 * tw <= cw <= 1.4 * tw + 4:
            candidate = [x0 + float(cx), y0 + float(cy), x0 + float(cx + cw), y0 + float(cy + ch)]
            if not _continues_outside(rgb_img, candidate, paper_luma):
                underline = candidate
                continue
        if cx >= tx0 - 1 and cx + cw <= tx1 + 1:
            text |= labels == i                        # a hyphen, dash or glyph bar inside the text
        # anything else (box edges, table rules) is ignored
    ys, xs = np.where(text)
    tx0, tx1, ty0, ty1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    tw = tx1 - tx0
    # Underline fused with descenders: a nearly full ink row in the lower half.
    # A lone glyph's bottom bar (the base of a "2") is not an underline, so
    # for short text the line must stick out past the glyphs above it.
    if underline is None and tw >= 4:
        rows = text[:, tx0:tx1].mean(axis=1)
        lower = [r for r in range(ty0 + (ty1 - ty0) // 2, ty1) if rows[r] >= 0.85]
        if lower:
            u0, u1 = min(lower), max(lower) + 1
            above = [r for r in range(ty0, u0) if rows[r] > 0]
            fused = [x0 + float(tx0), y0 + float(u0), x0 + float(tx1), y0 + float(u1)]
            above_cols = np.where(text[ty0:u0].any(axis=0))[0]
            sticks_out = above_cols.size and (above_cols.max() + 1 - above_cols.min()) + 1 < tw
            wordlike = tw >= 2.5 * (ty1 - ty0)
            if above and (wordlike or sticks_out or force_underline) and not _continues_outside(rgb_img, fused, paper_luma):
                underline = fused
                text[u0:u1, :] = False
                ys, xs = np.where(text)
                if xs.size:
                    tx0, tx1, ty0, ty1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1

    region = crop[ty0:ty1, tx0:tx1]
    glyph_px = region[text[ty0:ty1, tx0:tx1]].astype(np.float32)
    luma = glyph_px @ np.array([0.299, 0.587, 0.114], np.float32) if glyph_px.size else np.zeros(0)
    dark = glyph_px[luma <= np.percentile(luma, 30)] if luma.size else glyph_px
    light = crop[~ink]
    ink_rgb = np.median(dark, axis=0) if dark.size else np.array([20, 20, 20], np.float32)
    paper_rgb = np.median(light, axis=0) if light.size else np.array([255, 255, 255], np.float32)
    return {
        "box": [x0 + float(tx0), y0 + float(ty0), x0 + float(tx1), y0 + float(ty1)],
        "underline": underline,
        "ink": [int(v) for v in ink_rgb],
        "paper": [int(v) for v in paper_rgb],
        "word_gaps": _word_gaps(crop[ty0:ty1, tx0:tx1]),
    }


def _continues_outside(rgb_img: np.ndarray, line, paper_luma: float) -> bool:
    """True if a horizontal line keeps going past both ends (a table rule or box edge)."""
    lx0, ly0, lx1, ly1 = [int(round(v)) for v in line]
    h, w = rgb_img.shape[:2]
    rows = slice(max(0, ly0), min(h, max(ly1, ly0 + 1)))
    reach = 4
    left = rgb_img[rows, max(0, lx0 - reach):max(0, lx0 - 1)]
    right = rgb_img[rows, min(w, lx1 + 1):min(w, lx1 + reach)]
    if left.size == 0 or right.size == 0:
        return True
    def dark(px):
        return float((px.reshape(-1, 3) @ np.array([0.299, 0.587, 0.114])).min()) < paper_luma - 40
    return dark(left) and dark(right)


def _word_gaps(text_rgb: np.ndarray):
    """Fractional x positions (0..1) of the gaps between words in a text crop.

    Measured on a 4x enlargement: a column is "open" when even its darkest
    pixel (top 70 %, so underlines and descenders do not bridge words) is
    near paper.  Open runs fall into two groups, letter gaps and word gaps;
    the split is the largest jump in their sorted widths.
    """
    if text_rgb.size == 0 or min(text_rgb.shape[:2]) < 3:
        return []
    up = cv2.resize(text_rgb, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    luma = cv2.cvtColor(up, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = luma.shape
    darkest = luma[: max(2, int(h * 0.7))].min(axis=0)
    paper = float(np.percentile(luma, 90))
    ink = float(luma.min())
    open_cols = darkest > paper - 0.2 * (paper - ink)
    runs, start = [], None
    for x, is_open in enumerate(open_cols):
        if is_open and start is None:
            start = x
        elif not is_open and start is not None:
            if start > 0:
                runs.append((start, x))
            start = None
    if not runs:
        return []
    widths = sorted(b - a for a, b in runs)
    if len(widths) == 1:
        threshold = max(0.3 * h, 1)
    else:
        ratios = [(widths[i + 1] / max(widths[i], 1), i) for i in range(len(widths) - 1)]
        ratio, i = max(ratios)
        threshold = widths[i + 1] if ratio >= 2.0 else float("inf")
    threshold = max(threshold, 0.2 * h)
    return [(a + b) / 2.0 / w for a, b in runs if b - a >= threshold]


# Words that appear on receipts, rate lists and invoices; used only to place
# re-inserted spaces on the right character boundary.
COMMON_WORDS = set("""
a all an and any are as at be by for from in is it no not of on or the to with without
rate rates for site material materials measurement truck trucks consider final note mode
payment advance only hidden tax taxes withholding sale sales etc type types category
charges charge service carriage crush sub base name date ref to sr total amount qty
quantity price unit per ft road south north east west general order orders suppliers
supplier transporters contractors manufacturers company stone plant crushing website
email cell tel phone proprietor house chak no interchange lahore sargodha faisalabad
pull invoice receipt bill paid cash balance due delivery customer address
""".split())


def _joined(a: str, b: str) -> bool:
    """Characters that never have a space between them (inside numbers)."""
    numeric = set("0123456789.,/-:+")
    return a in numeric and b in numeric


def restore_spaces(text: str, gaps, family: str = "serif") -> str:
    """Re-insert spaces the OCR dropped, using the measured word gaps.

    The text is laid out with real font metrics, so a gap at fraction f of
    the ink box is matched to the character boundary nearest to f.  Existing
    spaces are kept, and numbers are never split; the user still confirms
    the final text.
    """
    if not gaps or not text:
        return text
    if text.count(" ") >= len(gaps):
        return text
    chars = list(text.replace(" ", ""))
    font = _font(family, 64)
    full = max(font.getlength("".join(chars)), 1e-6)
    edges = np.array([font.getlength("".join(chars[:i])) for i in range(len(chars) + 1)]) / full
    cut_points = set()
    last = 0
    for g in sorted(gaps):
        nearest = int(np.argmin(np.abs(edges - g)))
        best, best_score = None, None
        # Font metrics differ a little from the receipt's font, so look one or
        # two characters either side and prefer a cut that ends a known word
        # or falls on a case/punctuation boundary.
        for d in (0, -1, 1, -2, 2):
            cut = nearest + d
            if not (last < cut < len(chars)) or _joined(chars[cut - 1], chars[cut]):
                continue
            if chars[cut] in ":.,;)!?%" or chars[cut - 1] in "(":
                continue
            word = "".join(chars[last:cut]).lower().strip("().,:;!?+")
            score = abs(d)
            if word in COMMON_WORDS:
                score -= 3
            if chars[cut - 1].islower() and chars[cut].isupper():
                score -= 2
            if not chars[cut - 1].isalnum() or not chars[cut].isalnum():
                score -= 1
            if best_score is None or score < best_score:
                best, best_score = cut, score
        if best is not None:
            cut_points.add(best)
            last = best
    out, last = [], 0
    for cut in sorted(cut_points):
        out.append("".join(chars[last:cut]))
        last = cut
    out.append("".join(chars[last:]))
    return " ".join(part for part in out if part)


def _rule_on_left(rgb: np.ndarray, box, reach: int = 6) -> bool:
    """True when a vertical rule runs just left of the text (left-aligned text in a box).

    Such text is redrawn left-aligned with a little padding from the rule
    instead of centred on its old position.
    """
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    h = rgb.shape[0]
    rows = slice(max(0, y0), min(h, max(y1, y0 + 1)))
    strip = rgb[rows, max(0, x0 - reach):max(0, x0 - 1)]
    if strip.size == 0:
        return False
    luma = strip.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    paper = float(np.percentile(rgb[rows, x0:x1].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32), 90))
    dark_cols = (luma < paper - 60).mean(axis=0)
    return bool(dark_cols.size and dark_cols.max() >= 0.8)


def _crop_data_url(img: Image.Image, box, zoom: int = 4) -> str:
    x0, y0, x1, y1 = box
    pad = 3
    crop = img.crop((max(0, int(x0) - pad), max(0, int(y0) - pad),
                     min(img.width, int(x1) + pad + 1), min(img.height, int(y1) + pad + 1)))
    crop = crop.resize((crop.width * zoom, crop.height * zoom), Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _ocr_box(rgb: np.ndarray, offset, factor: float):
    """OCR one crop (already upscaled by `factor`) and map boxes back to source pixels."""
    found = []
    for quad, text, conf in _run_ocr(rgb):
        text = text.strip()
        if not text:
            continue
        xs, ys = quad[:, 0] / factor + offset[0], quad[:, 1] / factor + offset[1]
        found.append({"box": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                      "text": text, "confidence": conf})
    return found


def _upscaled(rgb: np.ndarray, factor: float) -> np.ndarray:
    return cv2.resize(rgb, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def detect_values(img: Image.Image, regions=None, cells=None, tables=None) -> list[dict]:
    """Read text inside receipt regions (whole image if none) and value cells.

    Returns items sorted top-to-bottom, left-to-right.  Each item carries the
    OCR text, its confidence, the tight ink box, an optional underline box,
    measured ink/paper colours and an enlarged crop of the *original* pixels
    so the user can confirm the reading.
    """
    rgb = np.asarray(img.convert("RGB"))
    regions = list(regions or []) or [(0, 0, img.width, img.height)]
    raw = []
    for x0, y0, x1, y1 in regions:
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(img.width, int(x1)), min(img.height, int(y1))
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        factor = min(3.0, OCR_MAX_SIDE / max(x1 - x0, y1 - y0))
        raw += _ocr_box(_upscaled(rgb[y0:y1, x0:x1], factor), (x0, y0), factor)

    # Small values ("1", "A", "56") are often missed on a whole region; read
    # every table value cell that has no detection on its own.
    for cx0, cy0, cx1, cy1 in cells or []:
        if any(cx0 <= (r["box"][0] + r["box"][2]) / 2 <= cx1 and cy0 <= (r["box"][1] + r["box"][3]) / 2 <= cy1
               for r in raw):
            continue
        crop = rgb[cy0:cy1, cx0:cx1]
        if crop.size == 0 or not _ink_mask(crop)[0].any():
            continue
        # Tiny values read best enlarged a lot, with a white margin around them.
        ink, _ = _ink_mask(crop)
        ys, xs = np.where(ink)
        tx0, ty0 = max(0, xs.min() - 2), max(0, ys.min() - 2)
        tight = crop[ty0:ys.max() + 3, tx0:xs.max() + 3]
        found = []
        for factor in (6.0, 8.0, 4.0):
            big = _upscaled(tight, factor)
            pad_y, pad_x = 30, 60
            big = cv2.copyMakeBorder(big, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT,
                                     value=[int(v) for v in np.median(crop.reshape(-1, 3), axis=0)])
            found = [
                {"box": [b[0] - pad_x / factor, b[1] - pad_y / factor, b[2] - pad_x / factor, b[3] - pad_y / factor],
                 "text": f["text"], "confidence": f["confidence"]}
                for f in _ocr_box(big, (0, 0), factor)
                for b in [f["box"]]
            ]
            if found:
                for f in found:
                    f["box"] = [f["box"][0] + cx0 + tx0, f["box"][1] + cy0 + ty0,
                                f["box"][2] + cx0 + tx0, f["box"][3] + cy0 + ty0]
                break
        if not found:
            # Not readable by OCR: still offer the cell so the user can type it.
            found = [{"box": [cx0 + float(xs.min()), cy0 + float(ys.min()),
                              cx0 + float(xs.max() + 1), cy0 + float(ys.max() + 1)],
                      "text": "", "confidence": 0.0}]
        raw += found

    # Overlapping regions read the same text twice; keep the more confident one.
    raw.sort(key=lambda r: -r["confidence"])
    kept = []
    for r in raw:
        if all(_iou(r["box"], k["box"]) < 0.4 for k in kept):
            kept.append(r)

    items = []
    for r in kept:
        m = _measure(rgb, r["box"])
        if m is None:
            continue
        text = restore_spaces(r["text"], m.pop("word_gaps"))
        items.append({
            "text": text,
            "ocr_text": r["text"],
            "confidence": round(r["confidence"], 3),
            "low_confidence": r["confidence"] < LOW_CONFIDENCE,
            "unread": not r["text"],
            **m,
            "font": "serif",
            "crop_url": _crop_data_url(img, m["box"] if m["underline"] is None else
                                       [m["box"][0], m["box"][1], m["underline"][2], m["underline"][3]]),
        })
    # A lone underlined glyph ("2" with its underline touching) is taller than
    # the other values on its row; re-measure it treating the bottom bar as
    # an underline when that brings it back to the row's text height.
    for it in items:
        short = len(it["text"].strip()) <= 3 and not any(ch in "gjpqy,;()[]{}|/" for ch in it["text"])
        if it["underline"] is not None or not short:
            continue
        h = it["box"][3] - it["box"][1]
        cy = (it["box"][1] + it["box"][3]) / 2
        peers = [o["box"][3] - o["box"][1] for o in items
                 if o is not it and abs((o["box"][1] + o["box"][3]) / 2 - cy) < h
                 and len(o["text"].strip()) <= 6 and not any(ch in "gjpqy,;()[]{}|/" for ch in o["text"])]
        if len(peers) >= 2 and h >= float(np.median(peers)) + 2:
            typical = float(np.median(peers))
            x0, y0, x1, y1 = it["box"]
            it["box"] = [x0, y0, x1, y0 + typical]
            it["underline"] = [x0 - 0.5, y0 + typical + 1, x1 + 0.5, y1]
    # Values inside table data cells share one font size when redrawn, and so
    # do the column titles (the table rows above the first data cell).
    data_top = {}
    for t in tables or []:
        inside_cells = [c for c in cells or [] if t[0] <= (c[0] + c[2]) / 2 <= t[2] and t[1] <= c[1] <= t[3]]
        if inside_cells:
            data_top[tuple(t)] = min(c[1] for c in inside_cells)
    for it in items:
        cx = (it["box"][0] + it["box"][2]) / 2
        cy = (it["box"][1] + it["box"][3]) / 2
        it["group"] = None
        if any(c[0] <= cx <= c[2] and c[1] <= cy <= c[3] for c in cells or []):
            it["group"] = "table"
        elif any(t[0] <= cx <= t[2] and t[1] <= cy < top for t, top in data_top.items()):
            it["group"] = "header"
        # Free text sitting against a box's left rule (the Note lines) is
        # left-aligned; table values and titles stay centred in their cells.
        it["align"] = "left" if it["group"] is None and _rule_on_left(rgb, it["box"]) else "centre"
    items.sort(key=lambda it: (round(it["box"][1] / 8), it["box"][0]))
    for index, it in enumerate(items):
        it["id"] = index
    return items


# ---------------------------------------------------------------------------
# Redraw
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _font_path(family: str):
    names = [n.lower() for n in FONT_CANDIDATES.get(family, FONT_CANDIDATES["serif"])]
    for directory in FONT_DIRS:
        if not os.path.isdir(directory):
            continue
        files = {os.path.basename(p).lower(): p
                 for p in glob.glob(os.path.join(directory, "**", "*.tt[fc]"), recursive=True)}
        for name in names:
            if name in files:
                return files[name]
    return None


def _font(family: str, size: int):
    path = _font_path(family)
    if path:
        return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


SUPERSAMPLE = 4
SIZE_SPAN = 1.5              # values on one line join a group only if their sizes are this close


def _ink_extent(text: str, family: str, size: float):
    """Ink box (left, top, right, bottom) of `text` relative to its baseline origin."""
    font = _font(family, max(1, int(round(size))))
    return font.getbbox(text, anchor="ls")


def _fitted_size(text: str, family: str, ink_height: float) -> float:
    """Font size (px) whose rendered ink is `ink_height` tall for this text."""
    ref = 100.0
    left, top, right, bottom = _ink_extent(text, family, ref)
    return ref * ink_height / max(bottom - top, 1)


VALUE_WEIGHT = 0.015    # small stroke on the regular font for table values: about weight 450


def _glyph_mask(text: str, family: str, size: float, weight: float = 0.0):
    """Anti-aliased coverage mask of `text` and the baseline-origin offset inside it.

    Drawn at SUPERSAMPLE x and reduced, so even small sizes get smooth,
    evenly weighted strokes.  The text is never stretched: every value uses
    the font's own proportions.
    """
    big = max(1, int(round(size * SUPERSAMPLE)))
    font = _font(family, big)
    stroke = int(round(weight * big))          # widens every stroke evenly: heavier, still crisp
    left, top, right, bottom = font.getbbox(text, anchor="ls", stroke_width=stroke)
    pad = SUPERSAMPLE * 2
    width = right - left + 2 * pad
    height = bottom - top + 2 * pad
    width += (-width) % SUPERSAMPLE
    height += (-height) % SUPERSAMPLE
    canvas = Image.new("L", (width, height), 0)
    origin = (pad - left, pad - top)
    ImageDraw.Draw(canvas).text(origin, text, font=font, fill=255, anchor="ls",
                                stroke_width=stroke, stroke_fill=255)
    small = canvas.resize((width // SUPERSAMPLE, height // SUPERSAMPLE), Image.Resampling.BOX)
    return np.asarray(small, np.float32) / 255.0, (origin[0] / SUPERSAMPLE, origin[1] / SUPERSAMPLE)


def _layout(items, scale: float):
    """Give confirmed values consistent typography.

    Each value's own size is fitted to its measured ink height (the text is
    never stretched).  All values in the table body share one size (their
    median), and so do values side by side on one line, such as a header
    row; a group shrinks together if one member would not fit its cell.  The baseline is taken from the value's own fit, so
    rows stay aligned; the text is centred on its original position.
    """
    laid = []
    for it in items:
        text = str(it.get("text", "")).strip()
        if not it.get("confirmed") or not text:
            continue
        family = it.get("font", "serif")
        x0, y0, x1, y1 = [float(v) * scale for v in it["box"]]
        own = _fitted_size(text, family, max(1.0, y1 - y0))
        _, top, _, _ = _ink_extent(text, family, own)
        pad = 0.0
        if it.get("align") == "left":
            pad = 2.5 * scale                      # breathing room from the box's left rule
        laid.append({"item": it, "text": text, "family": family, "own": own,
                     "baseline": y0 - top, "centre": (x0 + x1) / 2, "box_w": x1 - x0 - pad,
                     "left": x0 + pad if pad else None})

    # Groups that must look alike: every value in the table body, and values
    # sitting side by side on one line (a header row).  Everything else keeps
    # its own size.
    groups, rest = {}, []
    for e in laid:
        role = e["item"].get("group")
        if role in ("table", "header"):
            groups.setdefault((role, e["family"]), []).append(e)
        else:
            rest.append(e)
    rest.sort(key=lambda e: e["baseline"])
    for e in rest:
        e["size"] = e["own"]
        height = e["own"] * 0.7
        for key, members in groups.items():
            if key[0] != "line" or key[1] != e["family"]:
                continue
            ref = members[0]
            if abs(ref["baseline"] - e["baseline"]) < 0.5 * height and \
                    max(ref["own"], e["own"]) / min(ref["own"], e["own"]) < SIZE_SPAN:
                members.append(e)
                break
        else:
            groups[("line", e["family"], len(groups))] = [e]

    table_size = None
    # Table values first, so the column titles can be capped at their size.
    for key in sorted(groups, key=lambda k: k[0] != "table"):
        members = groups[key]
        size = float(np.median([e["own"] for e in members]))
        if key[0] == "header":
            # Column titles: one small size for all of them (the smaller end
            # of their measurements, never larger than the table values).
            size = float(np.percentile([e["own"] for e in members], 25))
            if table_size:
                size = min(size, table_size)
        # One shrink factor for the whole group: a value that would spill out
        # of its cell makes the group smaller, not just itself.  One- and
        # two-character values sit in narrow boxes and do not count.
        factor = 1.0
        for e in members:
            if len(e["text"]) <= 2:
                continue
            left, _, right, _ = _ink_extent(e["text"], e["family"], size)
            limit = e["box_w"] * 1.02 + scale
            factor = min(factor, limit / max(right - left, 1))
        factor = max(factor, 0.7)            # a misread box must not shrink everything
        for e in members:
            e["size"] = size * factor
        if key[0] == "table":
            table_size = size * factor

    for e in laid:
        if len(e["text"]) <= 2:              # single digits/letters: their cell has room
            continue
        left, _, right, _ = _ink_extent(e["text"], e["family"], e["size"])
        limit = e["box_w"] * 1.08 + scale    # group sizing already aims at 1.02; this only catches outliers
        if right - left > limit:             # never spill out of the original text area
            e["size"] *= limit / (right - left)
    return laid


def solid_ink(ink, paper) -> np.ndarray:
    """Full-strength ink colour from a washed-out measurement.

    Tiny strokes never reach their real colour after downscaling and JPEG
    chroma loss, so the measurement lies part-way to the paper.  Grey or
    black ink becomes solid black; coloured ink keeps its hue and gets full saturation
    (a pinkish measurement becomes print red); already dark ink (navy text on
    a blue bar) is kept.
    """
    ink = np.asarray(ink, np.float32)
    paper = np.asarray(paper, np.float32)
    luma = float(ink @ np.array([0.299, 0.587, 0.114], np.float32))
    hsv = cv2.cvtColor(np.clip(ink, 0, 255).astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0]
    saturation = float(hsv[1]) / 255.0
    if saturation < 0.18 and luma < 200:
        return np.array([0.0, 0.0, 0.0], np.float32)        # black / grey ink -> solid black
    if luma < 70:
        return ink                                           # already dark coloured ink (navy)
    hsv = np.array([hsv[0], 235, max(int(hsv[2]), 200)], np.uint8).reshape(1, 1, 3)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0].astype(np.float32)


def _rule_pixels(patch: np.ndarray, background: np.ndarray, span_w: int, span_h: int) -> np.ndarray:
    """Pixels of table rules that run straight through the erase area.

    "Dark" is measured against the local background, so a coloured bar
    (the blue TO: strip) is background, not a rule.  Only lines at least as
    long as the erased area (span_w / span_h) count, so glyph strokes (the
    bars of an E, the stem of an l) are never kept.
    """
    to_luma = np.array([0.299, 0.587, 0.114], np.float32)
    luma = np.clip(patch, 0, 255) @ to_luma
    bg_luma = np.clip(background, 0, 255) @ to_luma
    dark = (luma < bg_luma - 50).astype(np.uint8)
    horiz = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, span_w), 1)))
    vert = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, span_h))))
    return (horiz | vert) > 0


def _dominant_colour(pixels: np.ndarray, share: float = 0.45):
    """The flat fill colour of a plain area, or None if the area is not plain.

    The most common coarse colour is refined to the median of the pixels
    near it; at least `share` of all pixels must lie within 30 levels of it.
    """
    px = np.asarray(pixels, np.float32).reshape(-1, 3)
    if px.shape[0] < 20:
        return None
    bins = np.clip(px // 24, 0, 10).astype(np.int32)
    keys = bins[:, 0] * 121 + bins[:, 1] * 11 + bins[:, 2]
    top = int(np.argmax(np.bincount(keys)))
    colour = np.median(px[keys == top], axis=0)
    for _ in range(3):                    # bins split a colour; pull in its neighbours
        near = np.linalg.norm(px - colour, axis=1) < 30
        colour = np.median(px[near], axis=0)
    if near.mean() < share:
        return None
    return colour


UNDERLINE_RATIO = 0.09   # redrawn underline thickness as a share of the text height
HALO_RING = 3.0    # source pixels around an erased value checked for leftover tint


def _cell_bounds(out: np.ndarray, box, scale: float):
    """Inner bounds of the ruled cell holding a value, or None if it is not boxed in.

    From the value's box, scan outwards (up to ~40 source pixels) for a row /
    column of dark rule pixels spanning the value.
    """
    h, w = out.shape[:2]
    ex0, ey0, ex1, ey1 = box
    to_luma = np.array([0.299, 0.587, 0.114], np.float32)
    reach = int(round(40 * scale))

    def is_rule(line: np.ndarray) -> bool:
        luma = line @ to_luma
        return line.size > 0 and float((luma < 160).mean()) >= 0.7

    bounds = []
    for axis, start, step, limit in ((1, ex0 - 1, -1, 0), (1, ex1, 1, w),
                                     (0, ey0 - 1, -1, 0), (0, ey1, 1, h)):
        pos, found = start, None
        for _ in range(reach):
            if not (0 <= pos < (w if axis == 1 else h)):
                break
            line = out[ey0:ey1, pos] if axis == 1 else out[pos, ex0:ex1]
            if is_rule(line):
                found = pos - step          # last pixel inside the cell
                break
            pos += step
        if found is None:
            return None
        bounds.append(found)
    left, right, top, bottom = bounds
    return left, top, right + 1, bottom + 1


def _cell_colour(out: np.ndarray, box, scale: float):
    """Flat fill colour of the ruled cell around a value, or None.

    The colour is the dominant colour of the whole cell (its background
    outnumbers the text in it).  Text outside a ruled cell returns None and
    keeps the local background estimate.
    """
    bounds = _cell_bounds(out, box, scale)
    if bounds is None:
        return None
    cx0, cy0, cx1, cy1 = bounds
    px = out[cy0:cy1, cx0:cx1].reshape(-1, 3)
    if px.shape[0] < 20:
        return None
    luma = px @ np.array([0.299, 0.587, 0.114], np.float32)
    px = px[luma > np.percentile(luma, 25)]             # drop the darker text pixels
    return _dominant_colour(px, share=0.3)


def _clean_halo(out: np.ndarray, box, fill, ink, ring: int) -> None:
    """Remove the soft tint an old glyph left just outside its erased box.

    In a ring around the box, pixels lying on the line between the cell
    colour and the old ink colour (a pink haze around red text on a blue
    cell) are reset to the cell colour.  Rules and anything not on that line
    (other colours, solid ink) are left alone.
    """
    h, w = out.shape[:2]
    ex0, ey0, ex1, ey1 = box
    rx0, ry0 = max(0, ex0 - ring), max(0, ey0 - ring)
    rx1, ry1 = min(w, ex1 + ring), min(h, ey1 + ring)
    region = out[ry0:ry1, rx0:rx1]
    fill = np.asarray(fill, np.float32)
    direction = np.asarray(ink, np.float32) - fill
    norm2 = float(direction @ direction)
    if norm2 < 400:                       # ink too close to the fill colour to tell apart
        return
    d = region - fill
    t = (d @ direction) / norm2
    resid = np.linalg.norm(d - t[..., None] * direction, axis=2)
    haze = (t > 0.03) & (t < 0.75) & (resid < 28)
    haze[ey0 - ry0:ey1 - ry0, ex0 - rx0:ex1 - rx0] = False      # inside: already clean
    region[haze] = fill


def _background(patch: np.ndarray, kernel: int) -> np.ndarray:
    """Local background under dark glyphs: per-channel max filter, smoothed."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    bg = cv2.dilate(patch, k)
    return cv2.GaussianBlur(bg, (0, 0), kernel / 3.0)


def redraw_values(enhanced: Image.Image, scale: float, items) -> Image.Image:
    """Re-typeset confirmed values on the enhanced image.

    Only items with `confirmed` true and non-empty `text` are drawn.  Inside
    the tight text box (plus a one-source-pixel margin, so table rules
    survive) the old glyphs are replaced by the local background, then the
    confirmed text is drawn in the measured ink colour with consistent
    typography (see _layout: one font size per group of similar values, real
    font proportions, original baselines); underlines are redrawn as solid
    bars at their measured position.
    """
    out = np.asarray(enhanced.convert("RGB")).astype(np.float32).copy()
    h, w = out.shape[:2]
    laid = _layout(items, scale)

    # Pass 1: remove every old value (and its coloured halo) before drawing
    # anything, so cleaning one value never eats into another's new text.
    for entry in laid:
        it = entry["item"]
        x0, y0, x1, y1 = [float(v) * scale for v in it["box"]]
        margin = max(1.0, 1.5 * scale)     # also removes the old glyphs' soft fringe
        if it.get("group") == "header":
            margin = max(1.0, 2.2 * scale)  # titles shrink, so clear more of the old, larger glyphs
        boxes = [(x0, y0, x1, y1)]
        if it.get("underline"):
            boxes.append(tuple(float(v) * scale for v in it["underline"]))
        stroke = max(3, int(round(2.5 * scale)) | 1)
        for index, (bx0, by0, bx1, by1) in enumerate(boxes):
            ex0, ey0 = max(0, int(np.floor(bx0 - margin))), max(0, int(np.floor(by0 - margin)))
            ex1, ey1 = min(w, int(np.ceil(bx1 + margin))), min(h, int(np.ceil(by1 + margin)))
            if ex1 <= ex0 or ey1 <= ey0:
                continue
            # Estimate background from a slightly larger window so it is not
            # dominated by the glyphs being removed.
            px0, py0 = max(0, ex0 - stroke), max(0, ey0 - stroke)
            px1, py1 = min(w, ex1 + stroke), min(h, ey1 + stroke)
            window = out[py0:py1, px0:px1].copy()
            bg = _background(window, stroke)
            # Table rules survive the erase; the old underline itself is not a
            # rule (it is redrawn under the new text), so only vertical rules
            # are kept in its box.
            span_w = window.shape[1] + 1 if index == 1 else ex1 - ex0
            keep = _rule_pixels(window, bg, span_w, ey1 - ey0)
            sub = (slice(ey0 - py0, ey1 - py0), slice(ex0 - px0, ex1 - px0))
            # A plain cell (white, or the light blue title cells) is filled with
            # its own flat colour: the max-filter estimate picks up the pink of
            # old red glyphs and leaves a tinted patch behind the new text.
            # Scan for the cell's rules from the text itself: the erase margin
            # can already reach past a rule.
            text_box = (int(np.floor(bx0)), int(np.floor(by0)), int(np.ceil(bx1)), int(np.ceil(by1)))
            flat = _cell_colour(out, text_box, scale) if index == 0 else None
            fill = bg[sub] if flat is None else np.broadcast_to(flat, bg[sub].shape)
            out[ey0:ey1, ex0:ex1] = np.where(keep[sub][..., None], window[sub], fill)
            if flat is not None:
                _clean_halo(out, (ex0, ey0, ex1, ey1), flat, it.get("ink", [20, 20, 20]),
                            max(2, int(round(HALO_RING * scale))))

    # Pass 2: draw the confirmed text and underlines.
    for entry in laid:
        it = entry["item"]
        x0, y0, x1, y1 = [float(v) * scale for v in it["box"]]
        ink = solid_ink(it.get("ink", [20, 20, 20]), it.get("paper", [255, 255, 255]))
        weight = VALUE_WEIGHT if it.get("group") == "table" else 0.0
        mask, (ox, oy) = _glyph_mask(entry["text"], entry["family"], entry["size"], weight)
        left, _, right, _ = _ink_extent(entry["text"], entry["family"], entry["size"])
        if entry["left"] is not None:            # left-aligned text in a box: start after the padding
            mx0 = int(round(entry["left"] - (ox + left)))
        else:                                    # centre the ink, not the mask padding
            mx0 = int(round(entry["centre"] - (ox + (left + right) / 2)))
        my0 = int(round(entry["baseline"] - oy))
        sx0, sy0 = max(0, mx0), max(0, my0)
        sx1, sy1 = min(w, mx0 + mask.shape[1]), min(h, my0 + mask.shape[0])
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        a = mask[sy0 - my0:sy1 - my0, sx0 - mx0:sx1 - mx0][..., None]
        out[sy0:sy1, sx0:sx1] = out[sy0:sy1, sx0:sx1] * (1 - a) + ink * a

        if it.get("underline"):
            # Same depth and thickness as the original underline, but spanning
            # the new text (which may have moved, e.g. after left padding).
            ux0, uy0, ux1, uy1 = [float(v) * scale for v in it["underline"]]
            # A blurred underline measures 2-3 source pixels thick; drawn at
            # that size it looks heavy, so keep it in proportion to the text.
            text_h = max(1.0, y1 - y0)
            thickness = max(1, min(int(round(uy1 - uy0)), int(round(UNDERLINE_RATIO * text_h))))
            uy = int(round((uy0 + uy1) / 2 - thickness / 2))
            overhang = max(0.0, ((ux1 - ux0) - (x1 - x0)) / 2)
            tx0 = mx0 + ox + left - overhang
            tx1 = mx0 + ox + right + overhang
            out[max(0, uy):min(h, uy + thickness), max(0, int(round(tx0))):min(w, int(round(tx1)))] = ink
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Comparing two readings (used to check an AI-redrawn image)
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch in "./+-:@")


def compare_readings(original_items, other_items):
    """List original readings that cannot be found in another image's text.

    Matching is by text, not position (a generative redraw may move things).
    A reading counts as found when its normalised text occurs anywhere in the
    other image's text.  Everything else is returned (values with digits
    first) with the closest text in the other image, so the user can check
    those spots.  Because the original is blurry, a flag means "could not be
    confirmed", not necessarily "changed".
    """
    import difflib

    others = [_norm(o["text"]) for o in other_items if _norm(o["text"])]
    joined = "|".join(others)
    flags = []
    for it in original_items:
        n = _norm(it["text"])
        if len(n) < 2 or n in joined:
            continue
        best, best_ratio = "", 0.0
        for o, on in zip(other_items, [_norm(o["text"]) for o in other_items]):
            ratio = difflib.SequenceMatcher(None, n, on).ratio()
            if ratio > best_ratio:
                best, best_ratio = o["text"], ratio
        flags.append({
            "original": it["text"],
            "other": best,
            "similarity": round(best_ratio, 2),
            "has_digits": any(ch.isdigit() for ch in n),
            "box": it["box"],
        })
    flags.sort(key=lambda f: (not f["has_digits"], f["similarity"]))
    return flags


def read_image(img: Image.Image) -> list[dict]:
    """Plain OCR of a whole image (boxes in that image's pixels)."""
    rgb = np.asarray(img.convert("RGB"))
    factor = min(2.0, OCR_MAX_SIDE / max(img.width, img.height))
    return [r for r in _ocr_box(_upscaled(rgb, factor) if factor != 1 else rgb, (0, 0), factor)]


# ---------------------------------------------------------------------------
# Automatic table-value redraw (one click, no confirmation step)
# ---------------------------------------------------------------------------

DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARSETS = {
    "number": DIGITS + "./-,",
    # A category cell can hold a slash or dash ("not applicable"); without them
    # in the set a "/" is forced into the nearest letter and read as "l"/"I".
    "letter": LETTERS + "abcdefghijklmnopqrstuvwxyz" + "/-",
    "text": DIGITS + LETTERS + "abcdefghijklmnopqrstuvwxyz" + " ./-+&:,()%#'",
}
VERIFY_MIN = 0.60               # reading must look like the original glyphs this well
VERIFY_LOW_CONF = 0.80          # ... and better when the reader itself was unsure (< 0.7)
READ_RAW = True                 # also read the plain crop (not only the contrast channel)
READ_HEIGHTS = (32, 48, 64)     # line heights the recogniser is fed
TEMPLATE_WEIGHT = 1.0           # single glyphs: weight of the font-template match ...
VOTE_WEIGHT = 1.0               # ... and of the recogniser's votes
AUTO_MIN_CONFIDENCE = 0.25      # values read less surely than this keep their enhanced pixels


def column_kind(header: str) -> str:
    """Which characters a column's values can contain, from its title.

    Rates, charges and serial numbers are numbers; a category is a letter
    code; everything else (site name, material) is free text.  Constraining
    the reader this way is what lets 5-pixel-tall "A", "56" or "1" be read.
    """
    h = header.lower().replace(" ", "")
    if any(k in h for k in ("rate", "charge", "chare", "carri", "srno", "sr.", "sno", "qty", "quantity",
                            "amount", "price", "total", "/ft", "weight", "no:")):
        return "number"
    if "categ" in h or "grade" in h or "class" in h:
        return "letter"
    return "text"


class _StrongRecogniser:
    """PP-OCRv6 text-line recogniser (from the free `rapidocr` package, offline).

    Much better than the older bundled model on tiny, blurred receipt digits
    ("45", "3", "0" at 5-7 pixels tall).  Loaded once, on first use.
    """

    def __init__(self, model_path: str):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        self.chars = ["blank"] + meta["character"].split("\n") + [" "]
        self.input = self.session.get_inputs()[0].name

    def logits(self, img: np.ndarray) -> np.ndarray:
        h = 48
        w = max(8, int(np.ceil(img.shape[1] * h / img.shape[0])))
        x = cv2.resize(img, (w, h)).astype(np.float32).transpose(2, 0, 1) / 255.0
        x = (x - 0.5) / 0.5
        return self.session.run(None, {self.input: x[None]})[0][0]


_STRONG = None
_STRONG_TRIED = False


def strong_recogniser():
    """The PP-OCRv6 recogniser (models/ folder), or None to use the bundled reader."""
    global _STRONG, _STRONG_TRIED
    with _ENGINE_LOCK:
        if not _STRONG_TRIED:
            _STRONG_TRIED = True
            for path in _strong_model_paths():
                if os.path.exists(path):
                    try:
                        _STRONG = _StrongRecogniser(path)
                        break
                    except Exception:  # unreadable model: try the next, else the bundled reader
                        _STRONG = None
        return _STRONG


def _strong_model_paths():
    """Where the PP-OCRv6 model may be: the app's models folder (also inside the
    built .exe), then an installed `rapidocr` package."""
    import sys
    name = "PP-OCRv6_rec_small.onnx"
    bases = [getattr(sys, "_MEIPASS", None), os.path.dirname(os.path.abspath(__file__))]
    paths = [os.path.join(b, "models", name) for b in bases if b]
    try:
        import importlib.util
        spec = importlib.util.find_spec("rapidocr")
        if spec and spec.origin:
            paths.append(os.path.join(os.path.dirname(spec.origin), "models", name))
    except (ImportError, ValueError):
        pass
    return paths



def _decode_logits(preds: np.ndarray, chars, charset: str, key):
    allowed = _ALLOWED_CACHE.get(key)
    if allowed is None:
        allowed = np.array([0] + [i for i, c in enumerate(chars) if c in set(charset)])
        _ALLOWED_CACHE[key] = allowed
    sub = preds[:, allowed]
    idx, prob = sub.argmax(axis=1), sub.max(axis=1)
    text, conf, prev = [], [], -1
    for i, p in zip(idx, prob):
        if i != prev and i != 0:
            text.append(chars[allowed[i]])
            conf.append(float(p))
        prev = i
    return "".join(text).strip(), (float(np.mean(conf)) if conf else 0.0)


def _constrained_decode(img: np.ndarray, charset: str):
    """Run the recogniser on one text-line image, allowing only `charset`."""
    strong = strong_recogniser()
    if strong is not None:
        with _ENGINE_LOCK:
            preds = strong.logits(img)
        return _decode_logits(preds, strong.chars, charset, ("v6", charset))
    recogniser = _engine().text_recognizer
    chars = recogniser.postprocess_op.character
    key = (id(recogniser), charset)
    allowed = _ALLOWED_CACHE.get(key)
    if allowed is None:
        allowed = np.array([0] + [i for i, c in enumerate(chars) if c in set(charset)])
        _ALLOWED_CACHE[key] = allowed
    norm = recogniser.resize_norm_img(img, img.shape[1] / img.shape[0])[None].astype(np.float32)
    with _ENGINE_LOCK:
        preds = recogniser.session(norm)[0][0]
    sub = preds[:, allowed]
    idx, prob = sub.argmax(axis=1), sub.max(axis=1)
    text, conf, prev = [], [], -1
    for i, p in zip(idx, prob):
        if i != prev and i != 0:
            text.append(chars[allowed[i]])
            conf.append(float(p))
        prev = i
    return "".join(text).strip(), (float(np.mean(conf)) if conf else 0.0)


_ALLOWED_CACHE: dict = {}


def _contrast(rgb: np.ndarray) -> np.ndarray:
    """Black-on-white version of a crop using the channel where the ink is darkest.

    Red ink is pale in plain grey (luma) but almost black in the green
    channel, so coloured text gets far more contrast this way.
    """
    px = rgb.astype(np.float32)
    paper = np.percentile(px.reshape(-1, 3), 90, axis=0)
    ink = np.percentile(px.reshape(-1, 3), 3, axis=0)
    channel = int(np.argmax(paper - ink))
    grey = px[..., channel]
    lo, hi = np.percentile(grey, 2), np.percentile(grey, 98)
    grey = np.clip((grey - lo) / max(hi - lo, 1.0) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB)


def _coverage(rgb: np.ndarray) -> np.ndarray:
    """0..1 ink coverage of a small crop (distance from its paper colour)."""
    px = rgb.astype(np.float32)
    paper = np.percentile(px.reshape(-1, 3), 95, axis=0)
    dist = np.linalg.norm(px - paper, axis=2)
    return dist / max(float(dist.max()), 1.0)


def _render_coverage(text: str, height: int) -> np.ndarray:
    """Coverage of `text` in the receipt font, ink exactly `height` source pixels tall."""
    ss = 8
    font = _font("serif", 100)
    left, top, right, bottom = font.getbbox(text)
    font = _font("serif", max(4, int(round(100 * height * ss / max(bottom - top, 1)))))
    left, top, right, bottom = font.getbbox(text)
    width = right - left + 2 * ss
    width += (-width) % ss
    canvas = Image.new("L", (width, height * ss), 0)
    ImageDraw.Draw(canvas).text((ss - left, -top), text, font=font, fill=255)
    return np.asarray(canvas.resize((width // ss, height), Image.Resampling.BOX), np.float32) / 255.0


def _render_underlined(text: str, height: int) -> np.ndarray:
    """`text` with a 1 px underline below it, `height` pixels in total.

    Underlined category marks ("/" with its red underline) blur into one
    shape that looks like an "L"; comparing against the underlined glyph lets
    the slash win when that is what was printed.
    """
    glyph = _render_coverage(text, max(3, height - 2))
    out = np.zeros((height, glyph.shape[1]), np.float32)
    out[:glyph.shape[0]] = glyph
    cols = np.where(glyph.max(axis=0) > 0.2)[0]
    if cols.size:
        out[-1, cols.min():cols.max() + 1] = 1.0
    return out


def _template_score(cov: np.ndarray, text: str, underlined: bool = False) -> float:
    """How well `text` drawn in the receipt font matches a tiny glyph crop."""
    h, w = cov.shape
    best = -1.0
    for dh in (-1, 0, 1):
        if h + dh < 3:
            continue
        tm = _render_underlined(text, h + dh) if underlined else _render_coverage(text, h + dh)
        if tm.shape[1] > w + 4:
            continue
        canvas = np.zeros((max(h, tm.shape[0]) + 6, max(w, tm.shape[1]) + 6), np.float32)
        canvas[3:3 + h, 3:3 + w] = cov
        best = max(best, float(cv2.matchTemplate(canvas, tm, cv2.TM_CCOEFF_NORMED).max()))
    return best


def read_short_value(rgb: np.ndarray, m: dict, kind: str, info: dict = None):
    """Read a short value (a code, a number) from its measured ink box.

    Several crops and contrasts are read with the column's character set and
    the readings vote.  A single glyph is also compared with every allowed
    character drawn in the receipt font.  Returns (text, confidence 0..1).
    """
    charset = CHARSETS[kind]
    x0, y0, x1, y1 = [int(round(v)) for v in m["box"]]
    bottom = int(round(m["underline"][1])) if m.get("underline") else y1 + 2
    votes, tries = {}, 0
    # Faint strokes (the thin leg of an "A") fall outside the measured ink box,
    # so the crops reach out by a share of the text height, not fixed pixels.
    text_h = max(1, y1 - y0)
    reach_x = sorted({1, 2, 3, max(2, round(0.3 * text_h))})
    reach_y = sorted({1, 2})
    for px in reach_x:
        for py in reach_y:
            crop = rgb[max(0, y0 - py):min(y1 + py, bottom), max(0, x0 - px):x1 + px]
            if crop.size == 0:
                continue
            for variant in ((crop, _contrast(crop)) if READ_RAW else (_contrast(crop),)):
                for height in READ_HEIGHTS:
                    f = height / variant.shape[0]
                    big = cv2.resize(variant, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
                    big = cv2.copyMakeBorder(big, 6, 6, 12, 12, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                    text, conf = _constrained_decode(big, charset)
                    tries += 1
                    if text:
                        votes[text] = votes.get(text, 0.0) + conf
    if not tries:
        return "", 0.0
    scores = {t: v / tries for t, v in votes.items()}
    if info is not None:
        info["top_vote"] = max(scores, key=scores.get) if scores else ""

    reach = max(2, round(0.45 * text_h))
    crop = rgb[max(0, y0 - 1):min(y1 + 1, bottom), max(0, x0 - reach):x1 + reach]
    cov = _coverage(crop) if crop.size else np.zeros((1, 1), np.float32)
    rows, cols = np.where(cov.max(axis=1) > 0.35)[0], np.where(cov.max(axis=0) > 0.25)[0]
    if rows.size and cols.size:
        cov = cov[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
        if cov.shape[1] <= 0.9 * cov.shape[0] and cov.shape[0] >= 4:   # one glyph
            singles = [c for c in charset if c.strip() and c not in "./-,'()%#:+&"]
            if kind == "letter":
                singles = [c for c in singles if c.isupper()] + ["/", "-"]
            combined = {c: TEMPLATE_WEIGHT * _template_score(cov, c) + VOTE_WEIGHT * scores.get(c, 0.0)
                        for c in singles}
            if kind == "letter":
                for mark in ("/", "-"):          # an underlined mark fuses into an "L"-like shape
                    combined[mark] = max(combined[mark], TEMPLATE_WEIGHT * _template_score(cov, mark, underlined=True)
                                         + VOTE_WEIGHT * scores.get(mark, 0.0))
            best = max(combined, key=combined.get)
            runner = sorted(combined.values())[-2] if len(combined) > 1 else 0.0
            margin = combined[best] - runner
            return best, float(np.clip(0.5 + margin * 2 + scores.get(best, 0.0) / 2, 0, 1))
    if not scores:
        return "", 0.0
    # The readings that got most votes are compared with the original glyphs
    # (drawn in the receipt font); e.g. "56" beats a truncated "5".
    top = sorted(scores, key=scores.get, reverse=True)[:3]
    total = sum(scores[t] for t in top)
    ranked = sorted(top, key=lambda t: verify_reading(rgb, {"box": m["box"], "text": t}) + 0.3 * scores[t] / total,
                    reverse=True)
    best = ranked[0]
    return best, float(max(scores[best], scores[best] / total * 0.8))


def _header_texts(rgb: np.ndarray, band, columns):
    """OCR each column title on its own (a whole row reads as one run-on line)."""
    _, y0, _, y1 = band
    texts = []
    for xa, xb in columns:
        crop = rgb[y0 + 2:y1 - 2, xa + 3:xb - 3]
        if crop.size == 0 or not _ink_mask(crop)[0].any():
            texts.append("")
            continue
        big = cv2.copyMakeBorder(_upscaled(crop, 4.0), 30, 30, 60, 60, cv2.BORDER_CONSTANT,
                                 value=(255, 255, 255))
        found = _ocr_box(big, (0, 0), 4.0)
        found.sort(key=lambda f: (round(f["box"][1] / 6), f["box"][0]))
        texts.append(" ".join(f["text"] for f in found))
    return texts


# ---------------------------------------------------------------------------
# Table grid detection
# ---------------------------------------------------------------------------

def _line_runs(idx, gap=2):
    runs=[]
    for i in idx:
        if runs and i-runs[-1][1]<=gap: runs[-1][1]=i
        else: runs.append([i,i])
    return [tuple(r) for r in runs]
def find_table_grids(img):
    """Find ruled tables by their own lines (no dependence on colours or the TO: bar).

    Column rules are long vertical lines that start on the same row; the
    table's row rules are the horizontal lines spanning most of the width
    between them.  Returns [{"columns": [(x0, x1)...], "rows": [(y0, y1)...]}],
    row 0 being the column titles.  Works at any screenshot size.
    """
    rgb=np.asarray(img.convert('RGB')); gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    h,w=gray.shape
    block=max(15,(w//40)|1)
    bin_=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,block,12)
    bin_|=((gray<120)*255).astype(np.uint8)
    hk=max(20,w//10); vk=max(12,w//55)
    hor=cv2.morphologyEx(bin_,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(hk,1)))
    ver=cv2.morphologyEx(bin_,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(1,vk)))
    n,lab,st,_=cv2.connectedComponentsWithStats(ver,8)
    minlen=max(2.5*vk, w//18)
    segs=[(st[i,0]+st[i,2]/2, st[i,1], st[i,1]+st[i,3]) for i in range(1,n)
          if st[i,2]<=max(5,w//120) and st[i,3]>=minlen]
    tol=max(6,w//90)
    groups=[]
    for s in sorted(segs,key=lambda s:s[1]):
        for g in groups:
            if abs(g['y0']-s[1])<=tol:
                g['segs'].append(s); break
        else:
            groups.append({'segs':[s],'y0':s[1]})
    grids=[]
    for g in groups:
        segs_g=g['segs']
        if len(segs_g)<4: continue
        y0=int(np.median([s[1] for s in segs_g])); y1=int(np.median([s[2] for s in segs_g]))
        xs=sorted(x for x,_,_ in segs_g)
        cols=[]
        for x in xs:
            if cols and x-cols[-1][-1]<=tol: cols[-1].append(x)
            else: cols.append([x])
        cols=[float(np.mean(c)) for c in cols]
        if len(cols)<4 or cols[-1]-cols[0] < w*0.3: continue
        x0,x1=cols[0],cols[-1]
        band=hor[max(0,y0-tol):min(h,y1+tol+1), int(x0):int(x1)+1]
        rowsel=np.where((band>0).mean(axis=1)>0.6)[0]+max(0,y0-tol)
        lines=[(a+b)/2 for a,b in _line_runs(rowsel,2)]
        lines=[l for l in lines if y0-tol<=l<=y1+tol]
        rows=[(int(round(a)),int(round(b))) for a,b in zip(lines[:-1],lines[1:]) if b-a>=6]
        if len(rows)<2: continue
        columns=[(int(round(a)),int(round(b))) for a,b in zip(cols[:-1],cols[1:])]
        typical=float(np.median([b-a for a,b in columns]))
        columns=[c for c in columns if c[1]-c[0] >= 0.3*typical]   # slivers from a bubble edge
        if len(columns)>=3:
            grids.append({'columns':columns,'rows':rows})
    return grids


def read_table_values(img: Image.Image, grids) -> list[dict]:
    """Read every value in the data rows of each detected table.

    `grids` are dicts with "columns" [(x0, x1)...] and "rows" [(y0, y1)...],
    the first row being the column titles (never returned).  Long text is
    read by the full OCR, short values with the column's character set.
    Every returned item is ready for redraw_values; items read less surely
    than AUTO_MIN_CONFIDENCE are returned unconfirmed.
    """
    import value_context

    rgb = np.asarray(img.convert("RGB"))
    items = []
    # Column titles: normally the first row, but a thin band between double
    # rules can come first, and a table cut off at the top of the screenshot
    # has no title row at all - it then borrows the titles of a table with
    # the same columns elsewhere on the screenshot.
    layouts = []
    for grid in grids:
        columns, rows = grid["columns"], grid["rows"]
        header_row, headers = None, [""] * len(columns)
        for index, (ya, yb) in enumerate(rows[:3]):
            texts = _header_texts(rgb, (columns[0][0], ya, columns[-1][1], yb), columns)
            if any(re.search(r"iban|account|bank", t, re.I) for t in texts):
                header_row, headers = 0, _header_texts(rgb, (columns[0][0], rows[0][0], columns[-1][1],
                                                              rows[0][1]), columns)
                break                           # bank details: label/value pairs, first row as before
            if value_context.header_score(texts) >= max(2, len(columns) // 3):
                header_row = index
                headers = value_context.canonical_headers(sys.modules[__name__], rgb, columns,
                                                          (columns[0][0], ya, columns[-1][1], yb), texts)
                break
        layouts.append([grid, header_row, headers])
    for layout in layouts:
        grid, header_row, headers = layout
        if header_row is None:
            donor = next((other for other in layouts if other[1] is not None
                          and len(other[0]["columns"]) == len(grid["columns"])), None)
            if donor is not None:
                layout[2] = list(donor[2])
            layout[1] = -1                      # every row holds values
    doc_texts = _title_texts(rgb, grids)
    for grid_index, (grid, header_row, headers) in enumerate(layouts):
        columns, rows = grid["columns"], grid["rows"]
        if len(rows) - (header_row + 1) < 1:
            continue
        for (ya, yb) in rows[header_row + 1:]:
            for ci, (xa, xb) in enumerate(columns):
                cx0, cy0, cx1, cy1 = xa + 3, ya + 3, xb - 3, yb - 3
                crop = rgb[cy0:cy1, cx0:cx1]
                if crop.size == 0:
                    continue
                ink, _ = _ink_mask(crop)
                if not ink.any():
                    continue
                kind = column_kind(headers[ci])
                row_items = _read_cell(rgb, (cx0, cy0, cx1, cy1), kind, headers[ci])
                for it in row_items:
                    it["row"] = (ya, yb)
                    it["table"] = grid_index
                    it["cell"] = (cx0, cy0, cx1, cy1)
                    it["headless"] = header_row == -1   # table cut off at the top: numbering unknown
                items += row_items
    _fix_fused_underlines(rgb, items)
    _apply_serial_numbers(rgb, items)
    heights = [it["box"][3] - it["box"][1] for it in items]
    tiny = bool(heights) and float(np.median(heights)) < 6      # text only a few pixels tall
    for it in items:
        lone_mark = it["text"] in ("/", "-") and it["kind"] != "text"
        if not lone_mark and not any(ch.isalnum() for ch in it["text"]):
            it["confirmed"] = False            # a lone "-" or "." is a fragment, not a value
            it["match"] = 0.0
            continue
        if lone_mark:                          # "/" or "-" as a whole cell value ("not applicable")
            it["match"] = round(verify_reading(rgb, it), 3)
            it["confirmed"] = it["match"] >= VERIFY_MIN and it["confidence"] >= 0.5
            continue
        it["match"] = round(verify_reading(rgb, it), 3)
        ok = it["match"] >= VERIFY_MIN and (it["confidence"] >= 0.7 or it["match"] >= VERIFY_LOW_CONF)
        if tiny:                               # at this size readings are shakier: ask for more
            ok = ok and it["match"] >= 0.7 and it["confidence"] >= 0.7
            # ... and the text reader and the font comparison must agree
            ok = ok and it.get("top_vote", it["text"]) == it["text"]
        # Multi-digit numbers in a tiny, partly blurred font match the drawn
        # font loosely even when read right ("56"); the reader's own vote
        # then has to be reasonably clear.
        if not ok and not tiny and it["kind"] == "number" and len(it["text"]) >= 2 and it.get("serial") is None:
            ok = it["confidence"] >= 0.4 and it["match"] >= 0.5
        # Words read by the full detector with high confidence are reliable even
        # when the receipt uses a font unlike the one used for the check.
        if not ok and it["kind"] == "text" and len(it["text"]) >= 4:
            ok = it["confidence"] >= 0.85 and it["match"] >= 0.40
        if it.get("serial") is not None:
            ok = True                          # position in the table decides it
        if not ok:
            it["confirmed"] = False            # probably misread: keep the enhanced pixels
    items = value_context.refine(sys.modules[__name__], rgb, items, doc_texts)
    for index, it in enumerate(items):
        it["id"] = index
    return items


def _title_texts(rgb: np.ndarray, grids) -> list[str]:
    """Text printed just above each table ("Rate For <site> Site Material", "TO: <name>").

    It is larger and clearer than the table values, so it is read once and
    used as vocabulary for the site-name cells.
    """
    texts = []
    for grid in grids:
        columns, rows = grid["columns"], grid["rows"]
        row_h = max(8, int(np.median([b - a for a, b in rows])))
        y1 = rows[0][0]
        y0 = max(0, y1 - 4 * row_h)
        x0, x1 = max(0, columns[0][0] - 10), columns[-1][1] + 10
        crop = rgb[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        try:
            texts += [f["text"] for f in _ocr_box(_upscaled(crop, 2.0), (0, 0), 2.0) if f["confidence"] >= 0.6]
        except Exception:      # titles are only a help; never fail the reading for them
            pass
    return texts


def _normalise_number(text: str) -> str:
    """'157,00' -> '157.00': at this size a decimal point often reads as a comma."""
    import re
    return re.sub(r",(\d{2})$", r".\1", text.strip())


def _is_serial_column(header: str) -> bool:
    h = header.lower().replace(" ", "").replace(".", "")
    return h.startswith(("srno", "sno", "serial", "sr#", "no:")) or h in ("sr", "no", "#")


def _apply_serial_numbers(rgb: np.ndarray, items) -> None:
    """Sr No values are 1, 2, 3... down each table; use that when the glyph agrees.

    The expected number replaces the reading only if it matches the original
    pixels at least about as well, so a table that skips or restarts numbers
    keeps what is printed.
    """
    tables = {}
    for it in items:
        if _is_serial_column(it.get("column", "")) and not it.get("headless"):
            tables.setdefault(it.get("table"), []).append(it)
    for rows in tables.values():
        rows.sort(key=lambda it: it["box"][1])
        for index, it in enumerate(rows, start=1):
            expected = str(index)
            if it["text"] == expected:
                it["serial"] = index
                continue
            if not it["text"].isdigit():
                continue
            got = verify_reading(rgb, it)
            want = verify_reading(rgb, {"box": it["box"], "text": expected})
            if want >= got - 0.05 and want >= 0.6:
                it.update(text=expected, ocr_text=expected, serial=index)


def _fix_fused_underlines(rgb: np.ndarray, items) -> None:
    """A lone underlined glyph ("2") whose underline touches it measures too tall.

    Compared with the other values on its row, such a value is split into
    the glyph (the row's usual text height) and an underline, then re-read.
    """
    for it in items:
        if it.get("underline") is not None or len(it["text"]) > 3 or it["kind"] == "text":
            continue
        peers = [o["box"][3] - o["box"][1] for o in items
                 if o is not it and o.get("row") == it.get("row") and o["kind"] != "text"]
        h = it["box"][3] - it["box"][1]
        if len(peers) < 2 or h < float(np.median(peers)) + 2:
            continue
        typical = float(np.median(peers))
        x0, y0, x1, y1 = it["box"]
        cut = y0 + typical
        # Put the cut exactly above the bar: the lowest run of (nearly) full ink rows.
        crop = rgb[int(y0):int(y1), max(0, int(x0) - 1):int(x1) + 1]
        ink, _ = _ink_mask(crop) if crop.size else (np.zeros((0, 0), bool), 0)
        if ink.size:
            full = ink.mean(axis=1) >= 0.7
            r = len(full) - 1
            while r >= 0 and not full[r]:
                r -= 1
            if r >= 0:
                top = r
                while top > 0 and full[top - 1]:
                    top -= 1
                if top >= 0.5 * len(full):
                    cut = y0 + top
        it["box"] = [x0, y0, x1, cut]
        it["underline"] = [x0 - 0.5, cut, x1 + 0.5, y1]
        info = {}
        text, conf = read_short_value(rgb, it, it["kind"], info)
        it.update(text=text, ocr_text=text, confidence=round(conf, 3), top_vote=info.get("top_vote", text),
                  confirmed=bool(text) and conf >= AUTO_MIN_CONFIDENCE)


def _read_line(rgb: np.ndarray, box, kind: str):
    """Read one whole text line with the column's character set (a few crops vote)."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    votes, tries = {}, 0
    for py in (1, 2):
        crop = rgb[max(0, y0 - py):y1 + py, max(0, x0 - 2):x1 + 2]
        if crop.size == 0:
            continue
        for variant in (crop, _contrast(crop)):
            for height in (32, 48):
                f = height / variant.shape[0]
                big = cv2.resize(variant, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
                big = cv2.copyMakeBorder(big, 6, 6, 12, 12, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                text, conf = _constrained_decode(big, CHARSETS[kind])
                tries += 1
                if text:
                    votes[text] = votes.get(text, 0.0) + conf
    if not votes:
        return "", 0.0
    best = max(votes, key=votes.get)
    return best, votes[best] / tries


def _merge_line_fragments(rgb: np.ndarray, lines, kind: str):
    """Join pieces of one text line that the detector split (e.g. "...Interchange" + "Labore")."""
    lines = sorted(lines, key=lambda line: (line[0][1], line[0][0]))
    merged = []
    for box, text, conf in lines:
        if merged:
            pb, pt, pc = merged[-1]
            overlap = min(pb[3], box[3]) - max(pb[1], box[1])
            if overlap > 0.5 * min(pb[3] - pb[1], box[3] - box[1]):
                union = [min(pb[0], box[0]), min(pb[1], box[1]), max(pb[2], box[2]), max(pb[3], box[3])]
                text2, conf2 = _read_line(rgb, union, kind)
                merged[-1] = (union, text2 or (pt + " " + text), conf2 if text2 else min(pc, conf))
                continue
        merged.append((box, text, conf))
    return merged


def verify_reading(rgb: np.ndarray, it) -> float:
    """How well the read text, drawn in the receipt font, matches the original pixels.

    A wrong reading ("20" for "70", "3" for "1") usually matches poorly; such
    values are left enhanced instead of being redrawn wrongly.
    """
    x0, y0, x1, y1 = [int(round(v)) for v in it["box"]]
    h = y1 - y0
    if h < 3 or not it.get("text"):
        return 0.0
    reach = max(2, round(0.45 * h))
    crop = rgb[max(0, y0 - 1):y1 + 1, max(0, x0 - reach):x1 + reach]
    if crop.size == 0:
        return 0.0
    cov = cv2.copyMakeBorder(_coverage(crop), 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0).astype(np.float32)
    best = -1.0
    for dh in (-1, 0, 1):
        if h + dh < 3:
            continue
        tm = _render_coverage(it["text"], h + dh)
        cols = np.where(tm.max(axis=0) > 0.15)[0]
        if cols.size == 0:
            continue
        tm = tm[:, cols.min():cols.max() + 1]
        for wscale in (0.85, 1.0, 1.15):
            tw = max(1, int(round((x1 - x0) * wscale)))
            t2 = cv2.resize(tm, (tw, tm.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
            if t2.shape[0] > cov.shape[0] or t2.shape[1] > cov.shape[1]:
                continue
            best = max(best, float(cv2.matchTemplate(cov, t2, cv2.TM_CCOEFF_NORMED).max()))
    return best


def _ink_line_bands(crop: np.ndarray):
    """Row ranges of the text lines in a cell, each including its underline."""
    ink, _ = _ink_mask(crop)
    runs, start = [], None
    for r in range(ink.shape[0] + 1):
        on = r < ink.shape[0] and ink[r].any()
        if on and start is None:
            start = r
        elif not on and start is not None:
            runs.append([start, r])
            start = None
    if not runs:
        return []
    tallest = max(b - a for a, b in runs)
    bands = []
    for a, b in runs:
        # A second line must be about as tall as text; anything thinner is an
        # underline or a split-off glyph bar and stays with the line above.
        if bands and (b - a < max(4, 0.6 * tallest)):
            bands[-1][1] = min(ink.shape[0], b + 1)
        elif b - a > max(1, 0.35 * tallest):
            bands.append([max(0, a - 1), min(ink.shape[0], b + 1)])
    return [tuple(b) for b in bands]


def _read_cell(rgb: np.ndarray, cell, kind: str, header: str) -> list[dict]:
    cx0, cy0, cx1, cy1 = cell
    crop = rgb[cy0:cy1, cx0:cx1]
    allowed = set(CHARSETS[kind])
    lines = []
    if kind == "text":
        pad = cv2.copyMakeBorder(_upscaled(crop, 4.0), 30, 30, 60, 60, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        for f in _ocr_box(pad, (0, 0), 4.0):
            b = f["box"]
            box = [b[0] - 15 + cx0, b[1] - 7.5 + cy0, b[2] - 15 + cx0, b[3] - 7.5 + cy0]
            if f["confidence"] >= 0.6 and all(ch in allowed for ch in f["text"]):
                lines.append((box, f["text"], f["confidence"]))
        lines = _merge_line_fragments(rgb, lines, kind)
    if len(lines) > 1:
        # Several lines in one cell: the detector's boxes overlap, so take each
        # line's rows from the ink itself (runs of inked rows) when they match.
        ink, _ = _ink_mask(crop)
        inked = np.where(ink.any(axis=1))[0]
        bands, start = [], None
        for r in range(ink.shape[0] + 1):
            on = r < ink.shape[0] and ink[r].any()
            if on and start is None:
                start = r
            elif not on and start is not None:
                if r - start >= 2:
                    bands.append((cy0 + start, cy0 + r))
                start = None
        if inked.size and len(bands) == len(lines):
            lines.sort(key=lambda line: line[0][1])
            lines = [([b[0], band[0], b[2], band[1]], t, c) for (b, t, c), band in zip(lines, bands)]
    if not lines:
        # Values read glyph by glyph: one per line of ink in the cell (a
        # category can be "Normal" over "Black").  Thin bands are underlines
        # and stay with the line above them.
        for band in _ink_line_bands(crop):
            sub = crop[band[0]:band[1]]
            ink, _ = _ink_mask(sub)
            ys, xs = np.where(ink)
            if xs.size == 0:
                continue
            box = [cx0 + float(xs.min()), cy0 + band[0] + float(ys.min()),
                   cx0 + float(xs.max() + 1), cy0 + band[0] + float(ys.max() + 1)]
            lines.append((box, None, 0.0))

    out = []
    for box, text, conf in lines:
        m = _measure(rgb, box)
        if m is None:
            continue
        if len(lines) > 1:               # keep each line inside its own rows
            m["box"][1] = max(m["box"][1], box[1])
            m["box"][3] = min(m["box"][3], box[3])
        gaps = m.pop("word_gaps", [])
        info = {}
        if text is None:
            text, conf = read_short_value(rgb, m, kind, info)
        if kind == "number":
            text = _normalise_number(text)
        elif " " not in text and len(text) > 6:
            text = restore_spaces(text, gaps)    # only for run-together words ("RateForThokar")
        out.append({
            **m,
            "text": text,
            "ocr_text": text,
            "confidence": round(float(conf), 3),
            "column": header,
            "kind": kind,
            "font": "serif-regular",        # regular + VALUE_WEIGHT stroke, about weight 450
            "group": "table",
            "align": "centre",
            "confirmed": bool(text) and conf >= AUTO_MIN_CONFIDENCE,
            "top_vote": info.get("top_vote", text),
        })

    return out
