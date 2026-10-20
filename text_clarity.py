"""Conservative character-level text clarity for receipts and chat screenshots.

Everything here works on luminance only (chroma is left to the caller), never
redraws or recognises characters, and keeps every output pixel inside the local
value range of its input.  That bound is what prevents ringing halos and
overshoot; the half-level (50% ink) contour is a fixed point of the edge
refinement, so stroke thickness is preserved rather than grown.

Pipeline (each stage is optional and strength-controlled):
  1. edge-aware denoise            (bilateral, tiny kernel)
  2. regularised deblur            (Wiener, only if a blur is measured)
  3. controlled local contrast     (CLAHE, gated to real text contrast)
  4. ink/paper boundary refinement (local-level sigmoid about the half level)
  5. optional hard binarisation    (Bernsen local threshold, B&W print only)

Receipt regions additionally go through `crisp_ink` (colour output only):
small text that was downscaled into a chat screenshot reaches us as soft,
grey, over-wide strokes.  crisp_ink re-renders each stroke from its own
measured ink and paper levels with an anti-aliased edge placed at the
stroke's half-depth contour, so blur spread is trimmed (strokes get thinner,
not bolder) and edges become sharp, while counters and gaps may never darken.
"""

from dataclasses import dataclass, replace
import math

import cv2
import numpy as np

SHARP_EDGE_FLOOR = 0.8      # measured slope-sigma of a pixel-sharp step edge
BLUR_CALIBRATION = 1.3
MIN_DEBLUR_SIGMA = 0.4      # calibrated source-pixel blur below this is treated as already sharp
IRRECOVERABLE_SIGMA = 1.8   # blur this strong destroys the detail needed to recover glyphs
MAX_WIENER_SIGMA = 2.5      # never model more than this much blur (output pixels)
WIENER_SIGMA_FACTOR = 0.7   # deliberately under-correct the measured blur
GATE_LOW, GATE_RAMP = 20.0, 30.0

PROFILE_DEFAULTS = {
    "natural": dict(deblur=0.25, edge_refine=0.35, threshold=0.0, denoise=0.3, local_contrast=0.1, ink_crisp=1.0),
    "crisp": dict(deblur=0.4, edge_refine=0.5, threshold=0.0, denoise=0.3, local_contrast=0.1, ink_crisp=1.0),
    "bw": dict(deblur=0.35, edge_refine=0.5, threshold=1.0, denoise=0.3, local_contrast=0.1, ink_crisp=0.0, binarize=True),
}
# Caps applied while safe_mode is on so a slider can never reach aggressive territory.
SAFE_LIMITS = {"deblur": 0.5, "edge_refine": 0.6, "threshold": 0.5, "local_contrast": 0.2}
_STRENGTHS = ("deblur", "edge_refine", "threshold", "denoise", "local_contrast", "ink_crisp")


@dataclass(frozen=True)
class ClarityConfig:
    profile: str = "natural"
    enabled: bool = True
    deblur: float = 0.25
    edge_refine: float = 0.35
    threshold: float = 0.0
    denoise: float = 0.3
    local_contrast: float = 0.2
    ink_crisp: float = 0.0
    binarize: bool = False
    safe_mode: bool = True

    def validate(self) -> None:
        if self.profile not in PROFILE_DEFAULTS:
            raise ValueError("Text clarity profile must be natural, crisp, or bw.")
        for name in _STRENGTHS:
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"Text clarity '{name}' must be between 0 and 1.")

    def resolved(self) -> "ClarityConfig":
        """Return the config actually applied (safe-mode caps included)."""
        self.validate()
        if not self.safe_mode:
            return self
        caps = {k: min(getattr(self, k), v) for k, v in SAFE_LIMITS.items()}
        if self.binarize:
            caps["threshold"] = self.threshold
        return replace(self, **caps)


def make_config(profile="natural", **overrides) -> ClarityConfig:
    """Build a config from a profile preset; overrides that are None are ignored."""
    if profile not in PROFILE_DEFAULTS:
        raise ValueError("Text clarity profile must be natural, crisp, or bw.")
    values = dict(PROFILE_DEFAULTS[profile])
    values.update({k: v for k, v in overrides.items() if v is not None})
    config = ClarityConfig(profile=profile, **values)
    config.validate()
    return config


def _odd(value) -> int:
    return max(3, int(value)) | 1


def _window(scale: float) -> int:
    """Small clamp window that grows with the output scale."""
    return 2 * max(1, int(round(scale / 2))) + 1


def _ellipse(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _text_gate(y: np.ndarray, k: int) -> np.ndarray:
    """0..1 weight that is non-zero only where real ink/paper contrast exists."""
    kernel = _ellipse(k)
    span = cv2.dilate(y, kernel) - cv2.erode(y, kernel)
    return np.clip((span - GATE_LOW) / GATE_RAMP, 0.0, 1.0)


def estimate_noise_sigma(luma: np.ndarray) -> float:
    """Immerkaer fast noise estimate (robust to text edges via MAD)."""
    y = luma.astype(np.float32)
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    response = cv2.filter2D(y, cv2.CV_32F, kernel)
    return float(1.4826 * np.median(np.abs(response - np.median(response))) / 6.0)


def estimate_blur_sigma(luma: np.ndarray, min_contrast: float = 40.0, min_samples: int = 30):
    """Estimate the Gaussian blur sigma (in this image's pixels) from text edges.

    For a step of height c blurred by sigma, the peak slope is c/(sigma*sqrt(2*pi)).
    Returns None when there are not enough clean edges, so callers skip
    deconvolution instead of guessing.
    """
    y = luma.astype(np.float32)
    if min(y.shape) < 8:
        return None
    gx = cv2.Scharr(y, cv2.CV_32F, 1, 0) / 32.0
    gy = cv2.Scharr(y, cv2.CV_32F, 0, 1) / 32.0
    magnitude = cv2.magnitude(gx, gy)
    kernel = np.ones((7, 7), np.uint8)
    contrast = cv2.dilate(y, kernel) - cv2.erode(y, kernel)
    peaks = (magnitude >= cv2.dilate(magnitude, np.ones((3, 3), np.uint8))) & (magnitude > 1.0)
    samples = peaks & (contrast >= min_contrast)
    if int(samples.sum()) < min_samples:
        return None
    sigmas = contrast[samples] / (math.sqrt(2.0 * math.pi) * magnitude[samples])
    # A pixel-sharp edge already measures ~0.8 because the derivative spans two
    # pixels; remove that floor.  Thin strokes also compress the measurement, so
    # a modest factor compensates while still under-estimating (conservative).
    measured = float(np.median(sigmas))
    return BLUR_CALIBRATION * math.sqrt(max(measured ** 2 - SHARP_EDGE_FLOOR ** 2, 0.0))


def analyze_clarity(source_luma: np.ndarray) -> dict:
    """Describe how recoverable the text in a source-resolution crop is."""
    sigma = estimate_blur_sigma(source_luma)
    report = {
        "blur_sigma": None if sigma is None else round(sigma, 2),
        "deblur_eligible": sigma is not None and sigma >= MIN_DEBLUR_SIGMA,
        "recoverable": True,
        "warning": None,
    }
    if sigma is not None and sigma >= IRRECOVERABLE_SIGMA:
        report["recoverable"] = False
        report["warning"] = (
            "Some text is too blurred to recover. Sharpening cannot restore detail that "
            "the original screenshot does not contain, so unclear characters must be "
            "checked against the original. Redrawing text needs the separate OCR "
            "workflow with your confirmation."
        )
    return report


def denoise_edge_aware(y: np.ndarray, strength: float, scale: float) -> np.ndarray:
    if strength <= 0:
        return y
    y8 = np.clip(y, 0, 255).astype(np.uint8)
    d = 3 if scale <= 2 else 5
    filtered = cv2.bilateralFilter(y8, d, 8 + 22 * strength, 0.8 + 0.4 * scale).astype(np.float32)
    weight = min(1.0, strength * 2.0)
    return y + weight * (filtered - y)


def wiener_deblur(y: np.ndarray, sigma: float, nsr: float, band: int = 1024) -> np.ndarray:
    """Wiener deconvolution with a Gaussian PSF.

    Assumes a roughly shift-invariant Gaussian blur and additive noise.  The
    image is reflect-padded and processed in row bands so large upscaled
    screenshots stay within memory.  DC gain is kept at exactly 1.
    """
    sigma = float(sigma)
    pad = int(math.ceil(6 * sigma)) + 2
    src = cv2.copyMakeBorder(y.astype(np.float32), pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    height, width = y.shape
    out = np.empty((height, width), dtype=np.float32)
    for r0 in range(0, height, band):
        r1 = min(height, r0 + band)
        slab = src[r0:r1 + 2 * pad]
        sh = cv2.getOptimalDFTSize(slab.shape[0])
        sw = cv2.getOptimalDFTSize(slab.shape[1])
        fy = np.fft.fftfreq(sh)[:, None]
        fx = np.fft.rfftfreq(sw)[None, :]
        otf = np.exp(-2.0 * math.pi ** 2 * sigma ** 2 * (fx ** 2 + fy ** 2))
        wiener = otf / (otf ** 2 + nsr) * (1.0 + nsr)
        restored = np.fft.irfft2(np.fft.rfft2(slab, s=(sh, sw)) * wiener, s=(sh, sw))
        out[r0:r1] = restored[pad:pad + (r1 - r0), pad:pad + width]
    return out


def local_contrast(y: np.ndarray, amount: float, scale: float, gate: np.ndarray) -> np.ndarray:
    if amount <= 0:
        return y
    tile = max(32, int(32 * scale))
    grid = (max(1, y.shape[0] // tile), max(1, y.shape[1] // tile))
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(grid[1], grid[0]))
    equalised = clahe.apply(np.clip(y, 0, 255).astype(np.uint8)).astype(np.float32)
    return y + amount * gate * (equalised - y)


def refine_edges(y: np.ndarray, scale: float, strength: float, steepness: float = 6.0) -> np.ndarray:
    """Steepen ink/paper transitions about the local half level.

    Values are remapped inside the local [ink, paper] range with a sigmoid whose
    centre is the half level, so edges get steeper without moving (stroke width
    is unchanged) and nothing leaves the local range (no halos).
    """
    if strength <= 0:
        return y
    k = _odd(3 * scale)
    kernel = _ellipse(k)
    sigma = k / 3.0
    eroded, dilated = cv2.erode(y, kernel), cv2.dilate(y, kernel)
    # Smoothing hides window-edge steps, but it must never raise the ink level
    # (or lower the paper level) near a stroke: that would shift the half-level
    # contour and thicken the glyph.
    lo = np.minimum(cv2.GaussianBlur(eroded, (0, 0), sigma), eroded)
    hi = np.maximum(cv2.GaussianBlur(dilated, (0, 0), sigma), dilated)
    span = np.maximum(hi - lo, 1.0)
    t = np.clip((y - lo) / span, 0.0, 1.0)
    s = 1.0 / (1.0 + np.exp(-steepness * (t - 0.5)))
    s0 = 1.0 / (1.0 + math.exp(steepness * 0.5))
    s1 = 1.0 / (1.0 + math.exp(-steepness * 0.5))
    mapped = lo + (s - s0) / (s1 - s0) * span
    gate = np.clip((span - GATE_LOW) / GATE_RAMP, 0.0, 1.0)
    return y + strength * gate * (mapped - y)


def adaptive_binarize(luma: np.ndarray, scale: float = 1.0, min_contrast: float = 30.0) -> np.ndarray:
    """Bernsen local threshold; returns a uint8 0/255 map (255 = paper).

    Windows with real contrast split at their own midpoint; flat windows
    (blank paper, solid header blocks) fall back to the global Otsu level.
    Only a window-wide min/max is used, so glyph shapes are not eroded or dilated.
    """
    y8 = np.clip(luma, 0, 255).astype(np.uint8)
    kernel = np.ones((_odd(max(9, 5 * scale)),) * 2, np.uint8)
    lo = cv2.erode(y8, kernel).astype(np.float32)
    hi = cv2.dilate(y8, kernel).astype(np.float32)
    global_level = cv2.threshold(y8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    level = np.where(hi - lo >= min_contrast, (hi + lo) / 2.0, global_level)
    return np.where(y8.astype(np.float32) > level, 255, 0).astype(np.uint8)


def enhance_luma(luma: np.ndarray, scale: float, blur_sigma=None, config: ClarityConfig = None) -> np.ndarray:
    """Run the full clarity pipeline on a luminance plane at output resolution.

    blur_sigma is the blur measured at *source* resolution (see
    estimate_blur_sigma); None or a sharp value skips deconvolution.
    """
    cfg = (config or make_config()).resolved()
    original = luma.astype(np.float32)
    y = original

    if cfg.enabled:
        y = denoise_edge_aware(y, cfg.denoise, scale)
        gate = _text_gate(y, _window(scale) + 2)

        if cfg.deblur > 0 and blur_sigma is not None and blur_sigma >= MIN_DEBLUR_SIGMA:
            sigma = min(WIENER_SIGMA_FACTOR * blur_sigma * scale, MAX_WIENER_SIGMA)
            noise = estimate_noise_sigma(y)
            nsr = float(np.clip(0.03 + 40.0 * (noise / max(float(y.std()), 1.0)) ** 2, 0.03, 0.25))
            restored = wiener_deblur(y, sigma, nsr)
            kernel = np.ones((_window(scale),) * 2, np.uint8)
            restored = np.clip(restored, cv2.erode(y, kernel), cv2.dilate(y, kernel))
            y = y + cfg.deblur * (restored - y)

        y = local_contrast(y, cfg.local_contrast, scale, gate)
        y = refine_edges(y, scale, cfg.edge_refine)
        if cfg.threshold > 0 and not cfg.binarize:
            y = refine_edges(y, scale, cfg.threshold, steepness=16.0)

    y = _protect_shapes(y, original, scale)

    if cfg.binarize:
        return adaptive_binarize(y, scale).astype(np.float32)
    return y


# Coverage (0 = paper, 1 = the stroke's own darkest ink) at which a crisp edge
# is placed.  The half-depth contour of each stroke trims the grey blur spread
# that makes downscaled receipt text look bold, while thin strokes keep their
# centre line (the level is relative to each stroke's own depth).
INK_EDGE_LEVEL = 0.5
INK_EDGE_SOFTNESS = 0.05    # anti-aliasing width, in coverage units
INK_NEUTRAL_BOOST = 0.25    # extra darkness for black/grey ink (coloured ink keeps its tone)
INK_MIN_SPAN = 60.0         # ink/paper contrast below this is not treated as text
RIDGE_MIN_DEPTH = 40.0      # a stroke centre this much darker than paper is never erased


def _weighted_mean(channel: np.ndarray, weight: np.ndarray, sigma: float):
    num = cv2.GaussianBlur(channel * weight, (0, 0), sigma)
    den = cv2.GaussianBlur(weight, (0, 0), sigma)
    return num / np.maximum(den, 1e-4), den


def crisp_ink(rgb: np.ndarray, scale: float, strength: float = 1.0) -> np.ndarray:
    """Sharp, thin, solid text strokes for an upscaled RGB receipt patch.

    For every pixel the local paper level (max over about one glyph) and the
    nearest stroke's own ink level (min over about one source pixel) are
    measured.  Coverage between them is pushed through a narrow sigmoid
    centred on the half level, then the pixel is re-mixed from the
    local paper colour and the stroke's ink colour.  Effects:

    * soft grey fringes from the downscale/JPEG become paper -> no ink spread;
    * stroke interiors become solid ink -> no washed-out cores;
    * edges become ~1 output pixel wide (anti-aliased, not jagged);
    * colours come only from nearby paper/ink, so red text stays red and
      coloured fringes (pink halos around red text) are removed.

    Safeguards: only dark-ink-on-light-paper areas with real contrast where
    ink is the minority are touched (photos, solid colour bars and white text
    on colour are left alone), and a pixel brighter than its neighbourhood
    (a counter or a gap between characters) is never darkened.
    """
    if strength <= 0:
        return rgb
    ycc = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y = ycc[..., 0]
    kb = _odd(4 * scale + 1)    # about one glyph: local paper
    ks = _odd(2 * scale + 1)    # about one source stroke: that stroke's ink

    paper = cv2.dilate(y, _ellipse(kb))
    paper = np.maximum(cv2.GaussianBlur(paper, (0, 0), kb / 3.0), paper)
    ink = cv2.erode(y, _ellipse(ks))
    ink = np.minimum(cv2.GaussianBlur(ink, (0, 0), ks / 3.0), ink)
    span = paper - ink

    coverage = np.clip((paper - y) / np.maximum(span, 1.0), 0.0, 1.0)
    s = 1.0 / (1.0 + np.exp(-(coverage - INK_EDGE_LEVEL) / INK_EDGE_SOFTNESS))
    s0 = 1.0 / (1.0 + math.exp(INK_EDGE_LEVEL / INK_EDGE_SOFTNESS))
    s = np.clip((s - s0) / (1.0 - s0), 0.0, 1.0)

    paper_w = (coverage < 0.12).astype(np.float32)
    ink_w = (coverage > 0.8).astype(np.float32)
    out = ycc.copy()
    ink_chroma = []
    for c in (1, 2):
        pc, pd = _weighted_mean(ycc[..., c], paper_w, kb / 2.0)
        ic, idn = _weighted_mean(ycc[..., c], ink_w, ks)
        pc = np.where(pd > 0.02, pc, ycc[..., c])
        ic = np.where(idn > 0.02, ic, ycc[..., c])
        ink_chroma.append(ic)
        out[..., c] = pc * (1.0 - s) + ic * s

    iy, idn = _weighted_mean(y, ink_w, ks)
    ink_y = np.where(idn > 0.02, np.minimum(iy, ink + 25.0), ink)
    neutral = np.clip(1.0 - np.hypot(ink_chroma[0] - 128.0, ink_chroma[1] - 128.0) / 20.0, 0.0, 1.0)
    ink_y = ink_y * (1.0 - INK_NEUTRAL_BOOST * neutral)
    out[..., 0] = paper * (1.0 - s) + ink_y * s

    # Counters and gaps between characters may only get lighter.
    neighbourhood = cv2.GaussianBlur(y, (0, 0), max(1.0, 0.75 * scale))
    out[..., 0] = np.where(y >= neighbourhood, np.maximum(out[..., 0], y), out[..., 0])

    # Never erase a stroke centre line (the darkest pixel across a stroke).
    # A faint hairline next to a darker stroke (the thin leg of a red "A"
    # above its underline, the flag of a "1") measures a low coverage and
    # would be cut; such pixels keep their original value.  Nothing is
    # darkened here, so characters cannot be joined.
    lr = _odd(2 * scale + 1)
    ridge = (y <= cv2.erode(y, np.ones((1, lr), np.uint8)) + 0.5) | (
        y <= cv2.erode(y, np.ones((lr, 1), np.uint8)) + 0.5
    )
    ridge &= (paper - y >= RIDGE_MIN_DEPTH) & (s < 0.5)
    out = np.where(ridge[..., None], ycc, out)

    gate = np.clip((span - INK_MIN_SPAN) / 40.0, 0.0, 1.0)
    gate *= np.clip((paper - 150.0) / 40.0, 0.0, 1.0)          # light paper only
    ink_share = cv2.blur(coverage, (3 * kb, 3 * kb))
    gate *= np.clip((0.55 - ink_share) / 0.15, 0.0, 1.0)      # ink must be the minority
    gate *= float(strength)
    out = ycc + gate[..., None] * (out - ycc)
    return cv2.cvtColor(np.clip(out + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)


def _protect_shapes(y: np.ndarray, original: np.ndarray, scale: float) -> np.ndarray:
    """Final guarantee that processing only steepens edges.

    * A pixel brighter than its neighbourhood (counter, gap between digits)
      may not get darker; a pixel darker than its neighbourhood (thin stroke,
      decimal point) may not get lighter.  Counters therefore cannot fill,
      adjacent characters cannot merge, and thin ink cannot be erased.
    * No pixel may leave the local value range of the original (no halos).
    """
    sigma = max(1.0, 0.75 * scale)
    neighbourhood = cv2.GaussianBlur(original, (0, 0), sigma)
    y = np.where(original >= neighbourhood, np.maximum(y, original), np.minimum(y, original))
    kernel = np.ones((_window(scale),) * 2, np.uint8)
    return np.clip(y, cv2.erode(original, kernel), cv2.dilate(original, kernel))


# ---------------------------------------------------------------------------
# Evaluation helpers: enlarged crops, edge profiles, and accept/reject checks.
# ---------------------------------------------------------------------------

def edge_profile(luma: np.ndarray, row: int, x0: int, x1: int) -> np.ndarray:
    """Intensity profile of one row segment, for plotting or width measurement."""
    return luma[row, x0:x1].astype(np.float32)


def rise_width(profile: np.ndarray) -> float:
    """10-90% transition width (pixels) of a monotone-ish edge profile."""
    p = np.asarray(profile, dtype=np.float32)
    lo, hi = float(p.min()), float(p.max())
    if hi - lo < 1e-3:
        return 0.0
    n = (p - lo) / (hi - lo)
    if n[0] > n[-1]:
        n = 1.0 - n
    x = np.arange(len(n), dtype=np.float32)
    monotone = np.maximum.accumulate(n)
    return float(np.interp(0.9, monotone, x) - np.interp(0.1, monotone, x))


def enlarged_crop(image: np.ndarray, box, zoom: int = 8) -> np.ndarray:
    """Nearest-neighbour enlargement so individual pixels stay inspectable."""
    x0, y0, x1, y1 = box
    return cv2.resize(image[y0:y1, x0:x1], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)


def _ink_mask(luma: np.ndarray, level: float) -> np.ndarray:
    return (luma < level).astype(np.uint8)


def _count_holes(mask: np.ndarray) -> int:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    return 0 if hierarchy is None else int(sum(1 for h in hierarchy[0] if h[3] != -1))


def _enclosed_paper(reference: np.ndarray) -> np.ndarray:
    """Boolean mask of paper pixels fully enclosed by ink (character counters)."""
    level = (np.percentile(reference, 2) + np.percentile(reference, 98)) / 2.0
    paper = (reference >= level).astype(np.uint8)
    flood = paper.copy()
    padded = cv2.copyMakeBorder(flood, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(padded, mask, (0, 0), 2)
    return padded[1:-1, 1:-1] == 1


def evaluate_clarity(before: np.ndarray, after: np.ndarray, scale: float = 1.0, reference: np.ndarray = None) -> dict:
    """Accept or reject a processed luminance crop against its baseline.

    `before` and `after` share one size/scale (`before` is the plain upscale).
    Rejected when the result is thicker, merges characters, closes counters or
    gaps, or overshoots the local range (halos).  If a ground-truth
    `reference` is available, thickness is also measured at the reference's
    physical half level, and counters are checked by their brightness.
    """
    before = before.astype(np.float32)
    after = after.astype(np.float32)
    level = (np.percentile(before, 2) + np.percentile(before, 98)) / 2.0
    ink_before, ink_after = _ink_mask(before, level), _ink_mask(after, level)

    k = np.ones((_window(scale),) * 2, np.uint8)
    over = np.maximum(after - cv2.dilate(before, k), cv2.erode(before, k) - after)
    metrics = {
        "thickness_ratio": float(ink_after.sum()) / max(float(ink_before.sum()), 1.0),
        "components_before": int(cv2.connectedComponents(ink_before)[0] - 1),
        "components_after": int(cv2.connectedComponents(ink_after)[0] - 1),
        "counters_before": _count_holes(ink_before),
        "counters_after": _count_holes(ink_after),
        "overshoot": float(max(over.max(), 0.0)),
    }
    ok = (
        metrics["thickness_ratio"] <= 1.05
        and metrics["components_after"] >= metrics["components_before"]
        and metrics["overshoot"] <= 2.0
    )
    required_counters = metrics["counters_before"]
    if reference is not None:
        reference = reference.astype(np.float32)
        ref_level = (np.percentile(reference, 2) + np.percentile(reference, 98)) / 2.0
        ref_area = max(float((reference < ref_level).sum()), 1.0)
        metrics["thickness_vs_reference"] = float((after < ref_level).sum()) / ref_area
        # Spurious holes in a noisy baseline are not counters the glyphs really have.
        required_counters = min(required_counters, _count_holes(_ink_mask(reference, ref_level)))
        # Only counters the baseline actually resolved can be "closed" by processing.
        resolved = _enclosed_paper(reference) & (before >= level)
        if resolved.any():
            metrics["counter_brightness_change"] = float(after[resolved].mean() - before[resolved].mean())
            ok = ok and metrics["counter_brightness_change"] >= -1.0
        ok = ok and metrics["thickness_vs_reference"] <= 1.05
    ok = ok and metrics["counters_after"] >= required_counters
    metrics["accepted"] = bool(ok)
    return metrics
