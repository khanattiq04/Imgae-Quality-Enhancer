"""Local, non-generative image enhancement primitives.

The module deliberately keeps the source image separate from every output.  It
is usable from the Flask API and independently testable without starting the
desktop window.
"""

from dataclasses import dataclass
from io import BytesIO
from typing import BinaryIO

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from text_clarity import adaptive_binarize


SUPPORTED_FORMATS = {"JPEG", "PNG", "BMP", "TIFF", "WEBP", "GIF"}
INTERPOLATIONS = {
    "lanczos": cv2.INTER_LANCZOS4,
    "cubic": cv2.INTER_CUBIC,
    "linear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
}


@dataclass(frozen=True)
class EnhancementSettings:
    scale: int = 2
    interpolation: str = "lanczos"
    denoise: float = 0.0
    contrast: float = 0.0
    brightness: float = 0.0
    sharpen: float = 0.35
    sharpen_sigma: float = 1.0
    text_focus: bool = False
    threshold_variant: bool = False
    # A 64 MP ceiling supports common phone screenshots at 2x or 3x while
    # preventing accidental multi-hundred-megapixel allocations.
    max_pixels: int = 64_000_000

    def validate(self) -> None:
        if self.scale not in (1, 2, 3, 4, 6):
            raise ValueError("Scale must be 1x, 2x, 3x, 4x, or 6x.")
        if self.interpolation not in INTERPOLATIONS:
            raise ValueError("Interpolation must be lanczos, cubic, linear, or nearest.")
        if not 0 <= self.denoise <= 10:
            raise ValueError("Denoising strength must be between 0 and 10.")
        if -1 <= self.contrast <= 1 and -1 <= self.brightness <= 1:
            pass
        else:
            raise ValueError("Contrast and brightness must be between -1 and 1.")
        if not 0 <= self.sharpen <= 2:
            raise ValueError("Sharpening amount must be between 0 and 2.")
        if not 0.1 <= self.sharpen_sigma <= 5:
            raise ValueError("Sharpening sigma must be between 0.1 and 5.")
        if self.max_pixels < 1:
            raise ValueError("Maximum output pixels must be positive.")


def load_image(source: bytes | BinaryIO, max_pixels: int = 40_000_000) -> Image.Image:
    """Load and orient a supported image without modifying the input bytes."""
    try:
        image = Image.open(BytesIO(source) if isinstance(source, bytes) else source)
        image.load()
    except (OSError, ValueError, SyntaxError) as error:
        raise ValueError("The file is not a valid supported image.") from error
    if image.format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image format: {image.format or 'unknown'}.")
    if image.width * image.height > max_pixels:
        raise ValueError("The image is too large to process safely.")
    return ImageOps.exif_transpose(image)


def _as_array(image: Image.Image) -> tuple[np.ndarray, bool]:
    has_alpha = "A" in image.getbands()
    if has_alpha:
        rgba = np.asarray(image.convert("RGBA"))
        return rgba[:, :, :3], True
    return np.asarray(image.convert("RGB")), False


def _denoise(rgb: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return rgb
    value = int(round(3 + strength * 2))
    return cv2.fastNlMeansDenoisingColored(rgb, None, value, value, 7, 21)


def _clahe(rgb: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0:
        return rgb
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    luminance, a_channel, b_channel = cv2.split(lab)
    clip_limit = 2.0 + abs(amount) * 2.0
    enhanced = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(luminance)
    if amount < 0:
        enhanced = cv2.addWeighted(luminance, 1 - abs(amount), enhanced, abs(amount), 0)
    lab = cv2.merge((enhanced, a_channel, b_channel))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _unsharp(rgb: np.ndarray, amount: float, sigma: float) -> np.ndarray:
    if amount <= 0:
        return rgb
    blurred = cv2.GaussianBlur(rgb, (0, 0), sigma)
    return np.clip(cv2.addWeighted(rgb, 1 + amount, blurred, -amount, 0), 0, 255).astype(np.uint8)


def _resize_luminance_only(
    rgb: np.ndarray, width: int, height: int, interpolation: int
) -> np.ndarray:
    """Resize luminance smoothly while keeping text-edge colors from bleeding."""
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    luminance = cv2.resize(ycrcb[:, :, 0], (width, height), interpolation=interpolation)
    chroma_r = cv2.resize(ycrcb[:, :, 1], (width, height), interpolation=cv2.INTER_NEAREST)
    chroma_b = cv2.resize(ycrcb[:, :, 2], (width, height), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(cv2.merge((luminance, chroma_r, chroma_b)), cv2.COLOR_YCrCb2RGB)


def _unsharp_luminance_only(rgb: np.ndarray, amount: float, sigma: float) -> np.ndarray:
    if amount <= 0:
        return rgb
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    blurred = cv2.GaussianBlur(ycrcb[:, :, 0], (0, 0), sigma)
    ycrcb[:, :, 0] = np.clip(
        cv2.addWeighted(ycrcb[:, :, 0], 1 + amount, blurred, -amount, 0),
        0,
        255,
    ).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def _threshold(rgb: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Local ink/paper separation for the black-and-white variant.

    Uses a Bernsen local threshold from text_clarity.  There is deliberately no
    morphological closing: it fills counters and merges adjacent digits.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    output = adaptive_binarize(gray, scale)
    return np.repeat(output[:, :, None], 3, axis=2)


def process_image(image: Image.Image, settings: EnhancementSettings) -> tuple[Image.Image, Image.Image | None]:
    """Return the enhanced image and an optional separate threshold variant."""
    settings.validate()
    source = ImageOps.exif_transpose(image)
    rgb, has_alpha = _as_array(source)
    output_width = source.width * settings.scale
    output_height = source.height * settings.scale
    if output_width * output_height > settings.max_pixels:
        raise ValueError("The requested output would require too much memory.")

    rgb = _denoise(rgb, settings.denoise)
    rgb = _clahe(rgb, settings.contrast)
    interpolation = INTERPOLATIONS[settings.interpolation]
    if settings.text_focus:
        rgb = _resize_luminance_only(rgb, output_width, output_height, interpolation)
        # Respect the selected amount.  Adding a hidden extra sharpening pass
        # makes small receipt characters grow into one another.
        rgb = _unsharp_luminance_only(rgb, settings.sharpen, settings.sharpen_sigma)
    else:
        rgb = cv2.resize(rgb, (output_width, output_height), interpolation=interpolation)
        rgb = _unsharp(rgb, settings.sharpen, settings.sharpen_sigma)
    if settings.brightness or settings.contrast < 0:
        adjusted = Image.fromarray(rgb, "RGB")
        adjusted = ImageEnhance.Brightness(adjusted).enhance(1 + settings.brightness)
        rgb = np.asarray(adjusted)

    alpha = None
    if has_alpha:
        alpha = cv2.resize(np.asarray(source.convert("RGBA"))[:, :, 3], (output_width, output_height), interpolation=INTERPOLATIONS[settings.interpolation])
    output = Image.fromarray(rgb, "RGB")
    if alpha is not None:
        output.putalpha(Image.fromarray(alpha, "L"))
    threshold_variant = (
        Image.fromarray(_threshold(rgb, settings.scale), "RGB") if settings.threshold_variant else None
    )
    return output, threshold_variant


def encode_png(image: Image.Image, dpi: int = 300) -> bytes:
    # An upscaled image keeps its intended physical print size by carrying a
    # proportionally higher metadata DPI (for example, 3x at 300 PPI = 900).
    if not 72 <= dpi <= 3600:
        raise ValueError("DPI must be between 72 and 3600.")
    buffer = BytesIO()
    image.save(buffer, format="PNG", dpi=(dpi, dpi), optimize=False)
    return buffer.getvalue()


def encode_jpeg(image: Image.Image, quality: int = 95, dpi: int = 300) -> bytes:
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be between 1 and 100.")
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, dpi=(dpi, dpi), optimize=True)
    return buffer.getvalue()
