import io
import base64
import binascii
import hashlib
import os
import sys
import threading
import urllib.error
import urllib.request
import zipfile
import zlib
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse
from flask import Flask, request, jsonify, send_file, send_from_directory
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pypdfium2 as pdfium

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from image_processor import EnhancementSettings, encode_png, load_image, process_image
from text_clarity import (
    ClarityConfig,
    analyze_clarity,
    enhance_luma,
    make_config,
    refine_edges,
)


def resource_path(relative_path):
    """Locate files both in development and inside a PyInstaller EXE."""
    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base_path / relative_path


PUBLIC_DIR = resource_path("public")
app = Flask(__name__, static_folder=str(PUBLIC_DIR), static_url_path='')

AI_MODEL_COMMIT = "06c7bd65b0305c2955328f8f2721ea86c341f660"
AI_MODELS = {
    2: ("EDSR_x2.pb", 38490162, "7087df167eae8865b25b4f6032c819aacde25b8a"),
    3: ("EDSR_x3.pb", 38524784, "9b9082ceca0d5a2fc6d6bae7456e2c7934b3aa4d"),
    4: ("EDSR_x4.pb", 38573255, "09b872b26a6ba8d43abe433e65e65477ff8714ea"),
}
AI_MODEL_LOCK = threading.Lock()

def _ai_model_path(model_scale):
    filename, expected_size, expected_git_hash = AI_MODELS[model_scale]
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        cache_dir = Path(local_app_data) / "ScreenshotHDPrintEnhancer" / "models"
    else:
        cache_dir = Path.home() / ".cache" / "screenshot-hd-print-enhancer" / "models"

    model_path = cache_dir / filename
    if model_path.is_file() and model_path.stat().st_size == expected_size:
        return model_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    partial_path = model_path.with_suffix(model_path.suffix + ".part")
    url = (
        "https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/"
        f"{AI_MODEL_COMMIT}/models/{filename}"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "ScreenshotHDPrintEnhancer"})
    digest = hashlib.sha1(f"blob {expected_size}\0".encode("ascii"))
    total_size = 0

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with partial_path.open("wb") as model_file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total_size += len(chunk)
                    if total_size > expected_size:
                        raise RuntimeError("The AI model download was larger than expected.")
                    digest.update(chunk)
                    model_file.write(chunk)

        if total_size != expected_size or digest.hexdigest() != expected_git_hash:
            raise RuntimeError("The AI model download failed its integrity check.")
        partial_path.replace(model_path)
    except (OSError, urllib.error.URLError) as error:
        raise RuntimeError(
            "Could not download the local AI model. Check your internet connection and try again."
        ) from error
    finally:
        if partial_path.exists():
            partial_path.unlink()

    return model_path


@lru_cache(maxsize=3)
def _load_ai_model(model_scale):
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "AI reconstruction needs OpenCV. Rebuild or reinstall the desktop app dependencies."
        ) from error

    if not hasattr(cv2, "dnn_superres"):
        raise RuntimeError(
            "AI reconstruction needs the OpenCV contrib modules. Rebuild or reinstall the desktop app."
        )

    model = cv2.dnn_superres.DnnSuperResImpl_create()
    model.readModel(str(_ai_model_path(model_scale)))
    model.setModel("edsr", model_scale)
    return model


def _reconstruct_image(img, width, height, model_scale):
    import cv2
    import numpy as np

    rgb = np.asarray(img.convert("RGB"))
    with AI_MODEL_LOCK:
        reconstructed = _load_ai_model(model_scale).upsample(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    reconstructed = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2RGB)
    if reconstructed.shape[1] != width or reconstructed.shape[0] != height:
        reconstructed = cv2.resize(
            reconstructed,
            (width, height),
            interpolation=cv2.INTER_LANCZOS4,
        )
    return Image.fromarray(reconstructed)


def _adaptive_binarize(img):
    import cv2
    import numpy as np

    gray = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    smallest_side = min(gray.shape)
    if smallest_side < 3:
        thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    else:
        block_size = min(31, smallest_side if smallest_side % 2 else smallest_side - 1)
        thresholded = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            10,
        )
    return Image.fromarray(np.repeat(thresholded[..., None], 3, axis=2))


def _group_runs(indices, gap=2):
    runs = []
    for value in indices:
        if runs and value - runs[-1][1] <= gap:
            runs[-1][1] = int(value)
        else:
            runs.append([int(value), int(value)])
    return runs


def _find_receipt_value_regions(img):
    """Locate the table (header and data rows) of each receipt in a chat screenshot.

    A receipt is found by its blue "TO:" bar. Below it, the table's horizontal
    lines are located, and the rows are kept for as long as they contain the
    vertical column lines of the grid. The Note rows, which have none, and the
    rest of the chat are left out. Returns (x0, y0, x1, y1) boxes.
    """
    import numpy as np

    a = np.asarray(img.convert("RGB")).astype(np.int16)
    height, width = a.shape[:2]
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    blue = (b > 140) & (b - r > 60) & (b - g > 20)
    peak = a.max(axis=2)
    dark, soft = peak < 115, peak < 165

    regions = []
    for y0, y1 in _group_runs(np.where(blue.mean(axis=1) > 0.2)[0], 3):
        if y1 - y0 < width * 0.008:
            continue
        cols = np.where(blue[y0:y1 + 1].mean(axis=0) > 0.5)[0]
        x0, x1 = int(cols.min()), int(cols.max())
        span = x1 - x0
        if span < width * 0.3:
            continue
        line_share = dark[:, x0:x1 + 1].mean(axis=1)
        limit = min(height, y1 + int(span * 1.6))
        lines = _group_runs([y for y in range(y1 + 4, limit) if line_share[y] > 0.7])
        start = end = None
        for i in range(len(lines) - 1):
            top, bottom = lines[i][1] + 2, lines[i + 1][0] - 2
            if bottom - top < span * 0.03:
                continue
            column_share = soft[top:bottom + 1, x0:x1 + 1].mean(axis=0)
            inner = [c for c in _group_runs(np.where(column_share > 0.75)[0])
                     if c[0] > span * 0.03 and span - c[1] > span * 0.03]
            if len(inner) >= 4:
                start = lines[i][0] if start is None else start
                end = lines[i + 1][1]
            elif start is not None:
                break
        if start is not None:
            regions.append((x0, start, x1 + 1, end + 1))
    return regions


def _find_ruled_table_regions(img):
    """Find large tables from their continuous horizontal and vertical rules."""
    import cv2
    import numpy as np

    rgb = np.asarray(img.convert("RGB"))
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        8,
    )
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(8, width // 12), 1),
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(8, height // 15)),
    )
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    horizontal_runs = _group_runs(
        np.where(horizontal.mean(axis=1) > 35)[0],
        max(2, height // 100),
    )
    vertical_columns = np.where(vertical.mean(axis=0) > 35)[0]
    vertical_runs = _group_runs(vertical_columns, max(2, width // 300))
    if len(horizontal_runs) < 3 or len(vertical_runs) < 3:
        return []

    y0, y1 = horizontal_runs[0][0], horizontal_runs[-1][1] + 1
    x0, x1 = vertical_runs[0][0], vertical_runs[-1][1] + 1
    if x1 - x0 < width * 0.3 or y1 - y0 < height * 0.15:
        return []
    return [(x0, y0, x1, y1)]


def _boxes_overlap(box1, box2):
    """Check if two bounding boxes have substantial overlap."""
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2
    inter_x0 = max(x0_1, x0_2)
    inter_y0 = max(y0_1, y0_2)
    inter_x1 = min(x1_1, x1_2)
    inter_y1 = min(y1_1, y1_2)
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return False
    inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    return inter_area > 0.3 * min(area1, area2)


def _find_receipt_card_regions(img):
    """Detect rectangular receipt containers, cards, and slips in chat screenshots."""
    import cv2
    import numpy as np

    rgb = np.asarray(img.convert("RGB"))
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 20, 80)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw >= width * 0.25 and ch >= height * 0.08:
            if cw <= width * 0.99 or ch <= height * 0.99:
                candidates.append((max(0, x - 2), max(0, y - 2), min(width, x + cw + 2), min(height, y + ch + 2)))

    median_bg = np.median(gray)
    diff = np.abs(gray.astype(np.float32) - median_bg)
    mask = (diff > 18).astype(np.uint8) * 255
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
    contours2, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours2:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw >= width * 0.25 and ch >= height * 0.08:
            box = (max(0, x - 2), max(0, y - 2), min(width, x + cw + 2), min(height, y + ch + 2))
            if not any(_boxes_overlap(box, c) for c in candidates):
                candidates.append(box)

    return candidates


def _find_receipt_regions(img):
    """Expand detected receipt table rows to the receipt's visible header and footer."""
    import numpy as np

    rgb = np.asarray(img.convert("RGB")).astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    blue = (b > 140) & (b - r > 60) & (b - g > 20)
    dark = rgb.max(axis=2) < 115
    height, width = rgb.shape[:2]
    regions = []

    table_regions = _find_receipt_value_regions(img)
    if not table_regions:
        return _find_ruled_table_regions(img)

    for x0, table_y0, x1, table_y1 in table_regions:
        span = x1 - x0
        header_rows = np.where(blue[:, x0:x1].mean(axis=1) > 0.2)[0]
        header_bands = [
            (start, end)
            for start, end in _group_runs(header_rows, 3)
            if end < table_y0 and table_y0 - end <= span * 0.5
        ]
        top = min((start for start, _ in header_bands), default=table_y0)
        top = max(0, top - max(2, int(span * 0.03)))

        line_share = dark[:, x0:x1].mean(axis=1)
        footer_limit = min(height, table_y1 + int(span * 0.35))
        footer_lines = _group_runs(
            [y for y in range(table_y1, footer_limit) if line_share[y] > 0.7]
        )
        bottom = (
            min(height, footer_lines[-1][1] + max(3, int(span * 0.07)))
            if footer_lines
            else table_y1
        )
        regions.append((max(0, x0 - 3), top, min(width, x1 + 3), bottom))
    return regions


def _find_all_receipt_regions(img):
    """Locate all receipt areas (blue header tables, ruled tables, and chat cards)."""
    regions = []
    try:
        blue_regs = _find_receipt_regions(img)
        if blue_regs:
            regions.extend(blue_regs)
    except Exception:
        pass

    try:
        ruled = _find_ruled_table_regions(img)
        if ruled:
            for r in ruled:
                if not any(_boxes_overlap(r, existing) for existing in regions):
                    regions.append(r)
    except Exception:
        pass

    try:
        cards = _find_receipt_card_regions(img)
        if cards:
            for c in cards:
                if not any(_boxes_overlap(c, existing) for existing in regions):
                    regions.append(c)
    except Exception:
        pass

    return regions


_MODE_PROFILES = {"text_focused": "crisp", "print_text": "bw"}


def _clarity_profile_for_mode(mode):
    """Natural colour is the default; crisp text and B&W print are explicit modes."""
    return _MODE_PROFILES.get(mode, "natural")


def _new_clarity_report(config):
    applied = config.resolved()
    return {
        "profile": applied.profile,
        "enabled": applied.enabled,
        "safe_mode": applied.safe_mode,
        "deblur": applied.deblur,
        "edge_refine": applied.edge_refine,
        "threshold": applied.threshold,
        "regions": 0,
        "blur_sigma": None,
        "deblur_applied": False,
        "warnings": [],
    }


def _record_clarity(report, analysis, config):
    """Fold one region's blur analysis into the request-level report."""
    if report is None:
        return
    report["regions"] += 1
    sigma = analysis["blur_sigma"]
    if sigma is not None and (report["blur_sigma"] is None or sigma > report["blur_sigma"]):
        report["blur_sigma"] = sigma
    applied = config.resolved()
    if analysis["deblur_eligible"] and applied.enabled and applied.deblur > 0:
        report["deblur_applied"] = True
    if analysis["warning"] and analysis["warning"] not in report["warnings"]:
        report["warnings"].append(analysis["warning"])


def _source_clarity_analysis(img):
    import cv2
    import numpy as np

    return analyze_clarity(cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY))


def _enhance_receipt_patch_uniform(patch, scale_factor, config=None, blur_sigma=None):
    """Clarify receipt text in one upscaled patch without growing the ink.

    Operates on luminance only, so blue headers, green badges and red debits
    keep their original chroma.  The text_clarity pipeline runs in a fixed
    order (edge-aware denoise, restrained Wiener deblur, gated local contrast,
    half-level edge refinement) and every output pixel stays inside the local
    value range of the input, so there are no halos and stroke width and
    character counters are preserved.
    """
    import cv2
    import numpy as np

    config = config or make_config()
    patch_rgb = np.asarray(patch.convert("RGB"))
    ycbcr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2YCrCb)
    luma = enhance_luma(ycbcr[:, :, 0], scale_factor, blur_sigma, config)
    ycbcr[:, :, 0] = np.clip(luma + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2RGB))



def _crisp_text_edges(img, scale_factor, steepness=6.0):
    """Steepen text edges about the local half level (no thickening, no halos)."""
    import cv2
    import numpy as np

    ycc = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2YCrCb)
    y = refine_edges(ycc[..., 0].astype(np.float32), scale_factor, 1.0, steepness)
    ycc[..., 0] = np.clip(y + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB))


def _clarify_text_ink_selective(img, scale_factor, config=None, blur_sigma=None):
    """Clarify text strokes selectively without altering background or wallpaper.

    Only pixels with real ink/paper contrast are modified (low-contrast
    wallpaper is gated out), and results never leave the input's local value
    range, so strokes are not thickened and counters are not filled.
    """
    import numpy as np

    config = config or make_config()
    y, cb, cr = img.convert("YCbCr").split()
    luma = enhance_luma(np.asarray(y), scale_factor, blur_sigma, config)
    y_out = Image.fromarray(np.clip(luma + 0.5, 0, 255).astype(np.uint8))
    return Image.merge("YCbCr", (y_out, cb, cr)).convert("RGB")

def _sharpen_receipt_values(resized, source, scale_factor, config=None, report=None):
    """Clarify receipt text only; the rest of the screenshot is left untouched.

    Regions come from the existing receipt detectors.  Blur is measured on the
    *source* crop of each region so deconvolution is only used where a blur is
    actually present.
    """
    import cv2
    import numpy as np

    config = config or make_config()
    if not config.enabled:
        return resized

    regions = _find_all_receipt_regions(source)
    new_w, new_h = resized.size

    if regions:
        for x0, y0, x1, y1 in regions:
            bx0 = max(0, int(x0 * scale_factor))
            by0 = max(0, int(y0 * scale_factor))
            bx1 = min(new_w, int(x1 * scale_factor))
            by1 = min(new_h, int(y1 * scale_factor))
            if bx1 - bx0 >= 10 and by1 - by0 >= 10:
                crop_box = (bx0, by0, bx1, by1)
                patch = resized.crop(crop_box)
                analysis = analyze_clarity(
                    cv2.cvtColor(np.asarray(source.crop((x0, y0, x1, y1)).convert("RGB")), cv2.COLOR_RGB2GRAY)
                )
                _record_clarity(report, analysis, config)
                crisp = _enhance_receipt_patch_uniform(patch, scale_factor, config, analysis["blur_sigma"])

                pw, ph = crisp.size
                feather = min(4, pw // 8, ph // 8)
                if feather > 1:
                    mask_arr = np.full((ph, pw), 255, dtype=np.uint8)
                    if bx0 > 0:
                        for i in range(feather):
                            mask_arr[:, i] = int(255 * (i + 1) / (feather + 1))
                    if bx1 < new_w:
                        for i in range(feather):
                            mask_arr[:, pw - 1 - i] = int(255 * (i + 1) / (feather + 1))
                    if by0 > 0:
                        for i in range(feather):
                            mask_arr[i, :] = np.minimum(mask_arr[i, :], int(255 * (i + 1) / (feather + 1)))
                    if by1 < new_h:
                        for i in range(feather):
                            mask_arr[ph - 1 - i, :] = np.minimum(mask_arr[ph - 1 - i, :], int(255 * (i + 1) / (feather + 1)))
                    mask = Image.fromarray(mask_arr)
                    resized.paste(crisp, crop_box[:2], mask)
                else:
                    resized.paste(crisp, crop_box[:2])
        return resized
    else:
        analysis = _source_clarity_analysis(source)
        _record_clarity(report, analysis, config)
        return _clarify_text_ink_selective(resized, scale_factor, config, analysis["blur_sigma"])


def _find_receipt_value_cells(img):
    """Find data-row value cells, excluding table headings and description columns."""
    import cv2
    import numpy as np

    rgb = np.asarray(img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cells = []

    for x0, y0, x1, y1 in _find_receipt_value_regions(img):
        table = gray[y0:y1, x0:x1]
        table_h, table_w = table.shape
        binary = cv2.adaptiveThreshold(
            table,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            8,
        )
        horizontal = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, table_w // 12), 1)),
        )
        vertical = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, table_h // 4))),
        )
        row_lines = _group_runs(np.where(horizontal.mean(axis=1) > 50)[0], 3)
        if (
            len(row_lines) > 3
            and row_lines[0][1] < table_h * 0.04
            and row_lines[1][0] - row_lines[0][1] < table_h * 0.18
        ):
            row_lines = row_lines[1:]
        column_lines = [
            line
            for line in _group_runs(np.where(vertical.mean(axis=0) > 50)[0], 3)
            if line[0] > table_w * 0.02 and table_w - line[1] > table_w * 0.02
        ]
        if len(row_lines) < 3 or len(column_lines) < 3:
            continue

        x_bounds = [x0] + [x0 + (start + end) // 2 for start, end in column_lines] + [x1]
        y_bounds = [y0 + (start + end) // 2 for start, end in row_lines]
        if row_lines[0][0] < table_h * 0.02:
            y_bounds[0] = y0
        if table_h - row_lines[-1][1] < table_h * 0.03:
            y_bounds[-1] = y1
        column_count = len(x_bounds) - 1
        first_value_column = 3 if column_count >= 7 else 2

        for row_index in range(1, len(y_bounds) - 1):
            for column_index in range(first_value_column, column_count):
                cell = (
                    x_bounds[column_index] + 3,
                    y_bounds[row_index] + 3,
                    x_bounds[column_index + 1] - 3,
                    y_bounds[row_index + 1] - 3,
                )
                if cell[2] - cell[0] >= 6 and cell[3] - cell[1] >= 6:
                    cells.append(cell)

    return cells


def _ai_sharpen_receipt_values(resized, source, scale_factor):
    """Apply AI-derived luminance detail only to text in detected receipt value cells."""
    import cv2
    import numpy as np

    cells = _find_receipt_value_cells(source)
    if not cells:
        raise RuntimeError(
            "No receipt value cells were detected. Use a clear screenshot containing a ruled receipt table."
        )

    model_scale = min(max(scale_factor, 2), 4)
    for x0, y0, x1, y1 in cells:
        source_cell = source.crop((x0, y0, x1, y1))
        cell_gray = cv2.cvtColor(np.asarray(source_cell), cv2.COLOR_RGB2GRAY)
        _, ink = cv2.threshold(
            cell_gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        if np.count_nonzero(ink) < 2:
            continue
        ink = cv2.dilate(ink, np.ones((3, 3), dtype=np.uint8), iterations=1)

        target_size = (source_cell.width * scale_factor, source_cell.height * scale_factor)
        ai_size = (source_cell.width * model_scale, source_cell.height * model_scale)
        reconstructed = _reconstruct_image(source_cell, *ai_size, model_scale)
        if reconstructed.size != target_size:
            reconstructed = reconstructed.resize(target_size, Image.Resampling.LANCZOS)

        target_box = (x0 * scale_factor, y0 * scale_factor, x1 * scale_factor, y1 * scale_factor)
        original = resized.crop(target_box).convert("YCbCr")
        predicted = reconstructed.convert("YCbCr")
        original_y, cb, cr = original.split()
        predicted_y = predicted.getchannel("Y")

        mask = Image.fromarray(ink).resize(target_size, Image.Resampling.NEAREST)

        original_luma = np.asarray(original_y).astype(np.float32)
        ai_luma = np.asarray(predicted_y).astype(np.float32)
        refined_luma = np.clip(original_luma * 0.25 + ai_luma * 0.75, 0, 255).astype(np.uint8)
        mask_array = np.asarray(mask) > 0
        original_luma[mask_array] = refined_luma[mask_array]

        enhanced = Image.merge(
            "YCbCr",
            (Image.fromarray(original_luma.astype(np.uint8)), cb, cr),
        ).convert("RGB")
        resized.paste(enhanced, target_box[:2], mask)

    return resized


def _denoise_source(img):
    """Remove JPEG/compression noise from the source before upscaling.

    A gentle bilateral filter smooths compression blocks while keeping
    text edges sharp and preserving exact color fidelity.
    """
    import cv2
    import numpy as np

    rgb = np.asarray(img.convert("RGB"))
    denoised = cv2.bilateralFilter(rgb, d=3, sigmaColor=20, sigmaSpace=20)
    return Image.fromarray(denoised)



def _upscale_ycbcr_split(img, new_w, new_h):
    """Upscale luminance with LANCZOS and chrominance with NEAREST.

    Standard RGB LANCZOS upscaling averages *colour* values across text
    edges, producing visible colour fringing (e.g. green/pink halos around
    black text on a coloured chat bubble).  By splitting into YCbCr first:

    - Y (luminance) is upscaled with LANCZOS for smooth tonal gradations.
    - Cb and Cr (chrominance) are upscaled with NEAREST so colour
      boundaries stay razor-sharp and never bleed into adjacent pixels.

    The result is recombined into an RGB image with clean colour edges.
    """
    ycbcr = img.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y_up = y.resize((new_w, new_h), Image.Resampling.LANCZOS)
    cb_up = cb.resize((new_w, new_h), Image.Resampling.NEAREST)
    cr_up = cr.resize((new_w, new_h), Image.Resampling.NEAREST)
    return Image.merge("YCbCr", (y_up, cb_up, cr_up))


def _sharpen_text_aggressive(ycbcr_img, scale_factor, strength="high", config=None, blur_sigma=None):
    """Compatibility wrapper; clarity is selective and bounded, never aggressive."""
    rgb = ycbcr_img.convert("RGB")
    return _clarify_text_ink_selective(rgb, scale_factor, config, blur_sigma)


def _remove_phone_system_bars(img):
    """Crop the Android status bar and navigation bar from tall phone screenshots."""
    import numpy as np

    width, height = img.size
    if height / width < 1.7:
        return img

    rgb = np.asarray(img.convert("RGB"))
    margin = max(1, int(width * 0.04))
    edges = np.concatenate([rgb[:, :margin], rgb[:, -margin:]], axis=1).min(axis=(1, 2))

    # The navigation bar is a solid white band; chat wallpaper above it is not.
    bottom = height
    while bottom > 0 and edges[bottom - 1] >= 250:
        bottom -= 1
    nav_height = height - bottom
    if not (height * 0.015 <= nav_height <= height * 0.1):
        nav_height = 0

    # The status bar is white like the app bar below it, so use its standard proportion.
    top = int(round(height * 0.043))
    return img.crop((0, top, width, height - nav_height))


def enhance_image(img, scale_factor=3, dpi=300, mode="auto", clarity=None):
    """
    Multi-stage enhancement pipeline supporting WhatsApp screenshots,
    scanned documents, PDFs, and standard screen captures.

    Text-bearing modes share one character-clarity pipeline (text_clarity):
      1. Denoises the source (bilateral filter) before upscaling.
      2. Upscales with a YCbCr colour-split so chrominance edges stay
         sharp and never bleed colours into adjacent text.
      3. Clarifies luminance only: restrained Wiener deblur (when a blur is
         measured), gated local contrast and half-level edge refinement.

    `clarity` is an optional ClarityConfig; by default the profile follows the
    mode (natural colour, crisp text, or black-and-white print).  Nothing is
    recognised or redrawn.
    """
    import numpy as np

    clarity = clarity or make_config(_clarity_profile_for_mode(mode))
    report = _new_clarity_report(clarity)
    # Clean conversion from RGBA/Palette/Transparencies to RGB
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "RGBA":
            background.paste(img, mask=img.split()[3])
        else:
            background.paste(img.convert("RGB"))
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    orig_w, orig_h = img.size
    new_w, new_h = orig_w * scale_factor, orig_h * scale_factor

    if mode == "ai_reconstruction":
        model_scale = min(max(scale_factor, 2), 4)
        final_img = _reconstruct_image(img, new_w, new_h, model_scale)

    elif mode == "document":
        # Adaptive thresholding handles uneven scan and camera lighting better
        # than a single global black/white cutoff.
        final_img = _adaptive_binarize(img).resize(
            (new_w, new_h),
            Image.Resampling.NEAREST,
        )

    elif mode == "print_text":
        # High-contrast black and white for printing.  The source is lightly
        # denoised so JPEG blocks do not turn into specks, then the shared
        # clarity pipeline refines edges and a local (Bernsen) threshold
        # separates ink from paper.  No global cutoff, giant unsharp radius or
        # contrast boost, so strokes keep their real thickness and counters.
        clean = _denoise_source(img)
        analysis = _source_clarity_analysis(img)
        _record_clarity(report, analysis, clarity)
        resized = clean.resize((new_w, new_h), Image.Resampling.LANCZOS)
        luma = np.asarray(ImageOps.grayscale(resized)).astype(np.float32)
        luma = enhance_luma(luma, scale_factor, analysis["blur_sigma"], clarity)
        final_img = Image.fromarray(np.clip(luma, 0, 255).astype(np.uint8)).convert("RGB")

    else:
        # Clean upscale preserves 100% natural colors, wallpaper, and background.
        clean = _denoise_source(img)
        resized_rgb = _upscale_ycbcr_split(clean, new_w, new_h).convert("RGB")


        if mode in ("auto", "chat", "receipt_chat", "receipt_print", "receipt_values"):
            # Target receipts and text values specifically!
            # The WhatsApp wallpaper and surrounding screenshot stay 100% natural,
            # clean, and authentic; only detected receipt regions are clarified.
            final_img = _sharpen_receipt_values(resized_rgb, img, scale_factor, clarity, report)

        elif mode == "ai_receipt_values":
            resized_rgb = _ai_sharpen_receipt_values(resized_rgb, img, scale_factor)
            final_img = resized_rgb

        elif mode == "general":
            # General screenshot / photo: selective text clarity without background alteration
            if clarity.enabled:
                analysis = _source_clarity_analysis(img)
                _record_clarity(report, analysis, clarity)
                final_img = _clarify_text_ink_selective(
                    resized_rgb, scale_factor, clarity, analysis["blur_sigma"]
                )
            else:
                final_img = resized_rgb

        else:
            final_img = _sharpen_receipt_values(resized_rgb, img, scale_factor, clarity, report)

    # Preserve the physical size of the source at the requested print density.
    # Example: a 1080px source for 300 PPI is a 3.6in print; after 3x
    # resampling its 3240px copy must be tagged 900 DPI, not 300 DPI.
    embedded_dpi = dpi * scale_factor
    out_buf = io.BytesIO()
    # PNG is lossless at every compression level. Level 3 is much quicker
    # than optimize=True while preserving identical pixels.
    final_img.save(out_buf, format="PNG", dpi=(embedded_dpi, embedded_dpi), optimize=False, compress_level=3)
    out_bytes = out_buf.getvalue()

    b64_data = base64.b64encode(out_bytes).decode("utf-8")
    return {
        "data_url": f"data:image/png;base64,{b64_data}",
        "width": new_w,
        "height": new_h,
        "orig_width": orig_w,
        "orig_height": orig_h,
        "dpi": embedded_dpi,
        "source_dpi": dpi,
        "print_width_inches": round(orig_w / dpi, 2),
        "print_height_inches": round(orig_h / dpi, 2),
        "clarity": report,
    }

@app.route('/')
def serve_index():
    return send_from_directory(str(PUBLIC_DIR), 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(str(PUBLIC_DIR), path)


A4_POINTS = (595.276, 841.890)


def enhanced_image_to_pdf(image_bytes):
    """Stretch the image to fill one A4 page in a lossless PDF.

    Pillow stores RGB PDF images as JPEG, which smears tiny text, so the PDF is
    written directly with Flate (lossless) compression instead. Stretching
    independently to the page dimensions avoids letterboxing.
    """
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
    width, height = image.size

    page_w, page_h = A4_POINTS if height >= width else A4_POINTS[::-1]
    draw_w, draw_h = page_w, page_h
    offset_x = offset_y = 0

    content = f"q {draw_w:.3f} 0 0 {draw_h:.3f} {offset_x:.3f} {offset_y:.3f} cm /Im0 Do Q".encode("ascii")
    pixels = zlib.compress(image.tobytes(), 6)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w:.3f} {page_h:.3f}] "
            "/Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>"
        ).encode("ascii"),
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        (
            f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
            f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
            f"/Length {len(pixels)} >>\nstream\n"
        ).encode("ascii") + pixels + b"\nendstream",
    ]
    pdf = io.BytesIO()
    pdf.write(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(pdf.tell())
        pdf.write(f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n")
    xref_start = pdf.tell()
    pdf.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets:
        pdf.write(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode("ascii")
    )
    return pdf.getvalue()


def _optional_fraction(name):
    """Read an optional 0..1 form value; blank or missing means 'use the profile default'."""
    raw = request.form.get(name, "")
    return float(raw) if raw.strip() else None


@app.route("/api/process", methods=["POST"])
def process_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename or "file"
    try:
        scale = int(request.form.get("scale", 2))
        dpi = int(request.form.get("dpi", 300))
        denoise = float(request.form.get("denoise", 0))
        contrast = float(request.form.get("contrast", 0))
        brightness = float(request.form.get("brightness", 0))
        sharpen = float(request.form.get("sharpen", 0.35))
        sharpen_sigma = float(request.form.get("sharpen_sigma", 1))
        deblur = _optional_fraction("deblur")
        edge_refine = _optional_fraction("edge_refine")
        text_threshold = _optional_fraction("text_threshold")
    except (TypeError, ValueError) as error:
        return jsonify({"error": "Enhancement settings contain an invalid number."}), 400
    mode = request.form.get("mode", "standard")
    interpolation = request.form.get("interpolation", "lanczos")
    clarity_off = request.form.get("clarity", "on").lower() == "off"
    safe_mode = request.form.get("safe_mode", "true").lower() != "false"

    if scale not in (1, 2, 3, 4, 6):
        return jsonify({"error": "Scale must be 1x, 2x, 3x, 4x, or 6x."}), 400
    if dpi not in (300, 600):
        return jsonify({"error": "Print density must be 300 or 600 PPI."}), 400
    if mode not in (
        "auto",
        "chat",
        "document",
        "print_text",
        "receipt_chat",
        "receipt_values",
        "general",
        "ai_reconstruction",
        "ai_receipt_values",
        "receipt_print",
        "standard",
        "text_focused",
    ):
        return jsonify({"error": "Unknown enhancement profile."}), 400
    if mode in ("ai_reconstruction", "ai_receipt_values"):
        if scale == 1:
            return jsonify({"error": "AI enhancement requires an upscale multiplier of 2x or higher."}), 400
    if mode in ("ai_reconstruction", "ai_receipt_values"):
        hostname = urlparse(request.host_url).hostname
        if hostname not in ("localhost", "127.0.0.1", "::1"):
            return jsonify({
                "error": "AI enhancement is only available in the desktop app or a local server."
            }), 400

    file_bytes = file.read()
    results = []

    try:
        clarity = make_config(
            _clarity_profile_for_mode(mode),
            deblur=deblur,
            edge_refine=edge_refine,
            threshold=text_threshold,
            enabled=not clarity_off,
            safe_mode=safe_mode,
        )
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    try:
        # PDF Rendering
        if filename.lower().endswith(".pdf") or file.content_type == "application/pdf":
            pdf = pdfium.PdfDocument(file_bytes)
            for idx, page in enumerate(pdf):
                render_scale = (dpi / 72.0)
                pil_img = page.render(scale=render_scale).to_pil()
                pdf_scale = (
                    4 if mode == "ai_reconstruction"
                    else scale if mode == "ai_receipt_values"
                    else 1
                )
                processed = enhance_image(
                    pil_img,
                    scale_factor=pdf_scale,
                    dpi=dpi,
                    mode=mode,
                    clarity=clarity,
                )
                base_name = filename.rsplit(".", 1)[0]
                processed["filename"] = f"{base_name}_page_{idx + 1}_HD.png"
                results.append(processed)
        else:
            # Universal Image Handling (PNG, JPG, JPEG, WEBP, BMP, TIFF, GIF)
            img = load_image(file_bytes)
            if mode in ("standard", "text_focused"):
                settings = EnhancementSettings(
                    scale=scale if scale != 6 else 4,
                    interpolation=interpolation,
                    denoise=denoise,
                    contrast=contrast,
                    brightness=brightness,
                    sharpen=sharpen,
                    sharpen_sigma=sharpen_sigma,
                    text_focus=mode == "text_focused",
                    threshold_variant=request.form.get("threshold", "false").lower() == "true",
                )
                settings.validate()
                try:
                    enhanced, threshold = process_image(img, settings)
                except ValueError as error:
                    if "too much memory" in str(error):
                        requested_width = img.width * settings.scale
                        requested_height = img.height * settings.scale
                        raise ValueError(
                            f"{error} Requested output: {requested_width}x{requested_height} "
                            f"({requested_width * requested_height:,} pixels). "
                            "Choose a lower scale, usually 2x for phone screenshots."
                        ) from error
                    raise
                report = _new_clarity_report(clarity)
                if mode in ("standard", "text_focused"):
                    # Reuse the existing receipt detector for table/card
                    # regions. It only changes luminance inside detected
                    # regions and never redraws or OCR-replaces their values.
                    enhanced = _sharpen_receipt_values(enhanced, img, settings.scale, clarity, report)
                if threshold is not None and settings.threshold_variant:
                    # The checkbox requests a monochrome primary result, not
                    # only an additional optional download.
                    enhanced = threshold
                encoded = encode_png(enhanced, dpi)
                processed = {
                    "data_url": f"data:image/png;base64,{base64.b64encode(encoded).decode('ascii')}",
                    "width": enhanced.width,
                    "height": enhanced.height,
                    "orig_width": img.width,
                    "orig_height": img.height,
                    "dpi": dpi,
                    "source_dpi": dpi,
                    "print_width_inches": round(img.width / dpi, 2),
                    "print_height_inches": round(img.height / dpi, 2),
                    "clarity": report,
                }
                if threshold is not None:
                    threshold_bytes = encode_png(threshold, dpi)
                    processed["threshold_data_url"] = (
                        f"data:image/png;base64,{base64.b64encode(threshold_bytes).decode('ascii')}"
                    )
            else:
                processed = enhance_image(img, scale_factor=scale, dpi=dpi, mode=mode, clarity=clarity)
            base_name = filename.rsplit(".", 1)[0]
            processed["filename"] = f"{base_name}_HD.png"
            results.append(processed)

        return jsonify({"success": True, "items": results})

    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/zip", methods=["POST"])
def create_zip():
    """Create result ZIPs locally instead of depending on an online JS library."""
    payload = request.get_json(silent=True) or {}
    items = payload.get("items", [])
    if not isinstance(items, list) or not items:
        return jsonify({"error": "No enhanced files were supplied."}), 400

    try:
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output_zip:
            for item in items:
                image_name = Path(str(item.get("filename", "enhanced-image.png"))).name
                filename = Path(image_name).stem + ".pdf"
                data_url = str(item.get("data_url", ""))
                if "," not in data_url:
                    raise ValueError("An enhanced image is invalid.")
                image_bytes = base64.b64decode(data_url.split(",", 1)[1], validate=True)
                output_zip.writestr(filename, enhanced_image_to_pdf(image_bytes))
        archive.seek(0)
        return send_file(
            archive,
            mimetype="application/zip",
            as_attachment=True,
            download_name="HD_Screenshots_Ready_To_Print.zip",
        )
    except (ValueError, TypeError, binascii.Error, OSError) as error:
        return jsonify({"error": str(error)}), 400


@app.route("/api/pdf", methods=["POST"])
def create_pdf():
    """Convert an enhanced PNG into a print-sized PDF."""
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "An enhanced image is required."}), 400
    data_url = payload.get("data_url")
    if not isinstance(data_url, str) or "," not in data_url:
        return jsonify({"error": "An enhanced image is required."}), 400

    try:
        image_bytes = base64.b64decode(data_url.split(",", 1)[1], validate=True)
        pdf = io.BytesIO(enhanced_image_to_pdf(image_bytes))
        return send_file(
            pdf,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="enhanced-image.pdf",
        )
    except (ValueError, TypeError, binascii.Error, OSError) as error:
        return jsonify({"error": f"Could not create PDF: {error}"}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
