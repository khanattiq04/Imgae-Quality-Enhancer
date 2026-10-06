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
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse
from flask import Flask, request, jsonify, send_file, send_from_directory
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pypdfium2 as pdfium


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

def enhance_image(img, scale_factor=3, dpi=300, mode="chat"):
    """
    Multi-stage enhancement pipeline supporting WhatsApp screenshots, 
    scanned documents, PDFs, and standard screen captures.
    """
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
    else:
        # A screenshot has a fixed amount of source detail. Enlarging it helps a
        # printer render smooth edges, but must not make it print physically larger.
        # The output DPI metadata below keeps its intended physical size intact.
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    if mode == "ai_reconstruction":
        # The generative model output already produced final_img above.
        pass
    elif mode == "chat":
        # CHAT PRINT PIPELINE: enhance luminance only so small text gains
        # separation without changing bubble or emoji colours.
        y, cb, cr = resized.convert("YCbCr").split()
        y = ImageOps.autocontrast(y, cutoff=0.4)
        # One controlled pass avoids the light/dark halos caused by stacking
        # sharpen filters, which are especially obvious on printed text.
        y = y.filter(ImageFilter.UnsharpMask(radius=1.15, percent=165, threshold=3))
        y = ImageEnhance.Contrast(y).enhance(1.10)
        final_img = Image.merge("YCbCr", (y, cb, cr)).convert("RGB")

    elif mode == "document":
        # DOCUMENT & SCAN MODE (Binarized High Contrast)
        sharpened = resized.filter(ImageFilter.UnsharpMask(radius=1.5, percent=180, threshold=1))
        contrast_enhancer = ImageEnhance.Contrast(sharpened)
        contrast_img = contrast_enhancer.enhance(1.25)
        sharp_enhancer = ImageEnhance.Sharpness(contrast_img)
        final_img = sharp_enhancer.enhance(1.20)

    elif mode == "print_text":
        # Maximum text legibility for printing: convert chat/document text to
        # solid black and white. This cannot recover missing characters, but
        # it gives the printer the clearest possible edges from the source.
        text_layer = ImageOps.grayscale(resized)
        text_layer = ImageOps.autocontrast(text_layer, cutoff=0.5)
        text_layer = text_layer.filter(ImageFilter.UnsharpMask(radius=1.0, percent=180, threshold=2))
        text_layer = ImageEnhance.Contrast(text_layer).enhance(1.7)
        final_img = text_layer.point(lambda pixel: 255 if pixel >= 160 else 0).convert("RGB")

    else:
        # GENERAL SCREENSHOT & PHOTO MODE
        sharpened = resized.filter(ImageFilter.UnsharpMask(radius=1.8, percent=150, threshold=2))
        contrast_enhancer = ImageEnhance.Contrast(sharpened)
        final_img = contrast_enhancer.enhance(1.08)

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
    }

@app.route('/')
def serve_index():
    return send_from_directory(str(PUBLIC_DIR), 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(str(PUBLIC_DIR), path)

@app.route("/api/process", methods=["POST"])
def process_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename or "file"
    scale = int(request.form.get("scale", 3))
    dpi = int(request.form.get("dpi", 300))
    mode = request.form.get("mode", "chat")

    if scale not in (1, 2, 3, 4, 6):
        return jsonify({"error": "Scale must be 1x, 2x, 3x, 4x, or 6x."}), 400
    if dpi not in (300, 600):
        return jsonify({"error": "Print density must be 300 or 600 PPI."}), 400
    if mode not in ("chat", "document", "print_text", "general", "ai_reconstruction"):
        return jsonify({"error": "Unknown enhancement profile."}), 400
    if mode == "ai_reconstruction":
        if scale == 1:
            return jsonify({"error": "AI reconstruction requires an upscale multiplier of 2x or higher."}), 400
        hostname = urlparse(request.host_url).hostname
        if hostname not in ("localhost", "127.0.0.1", "::1"):
            return jsonify({
                "error": "AI reconstruction is only available in the desktop app or a local server."
            }), 400

    file_bytes = file.read()
    results = []

    try:
        # PDF Rendering
        if filename.lower().endswith(".pdf") or file.content_type == "application/pdf":
            pdf = pdfium.PdfDocument(file_bytes)
            for idx, page in enumerate(pdf):
                render_scale = (dpi / 72.0)
                pil_img = page.render(scale=render_scale).to_pil()
                pdf_scale = 4 if mode == "ai_reconstruction" else 1
                processed = enhance_image(pil_img, scale_factor=pdf_scale, dpi=dpi, mode=mode)
                base_name = filename.rsplit(".", 1)[0]
                processed["filename"] = f"{base_name}_page_{idx + 1}_HD.png"
                results.append(processed)
        else:
            # Universal Image Handling (PNG, JPG, JPEG, WEBP, BMP, TIFF, GIF)
            img = Image.open(io.BytesIO(file_bytes))
            processed = enhance_image(img, scale_factor=scale, dpi=dpi, mode=mode)
            base_name = filename.rsplit(".", 1)[0]
            processed["filename"] = f"{base_name}_HD.png"
            results.append(processed)

        return jsonify({"success": True, "items": results})

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
                filename = Path(str(item.get("filename", "enhanced-image.png"))).name
                data_url = str(item.get("data_url", ""))
                if "," not in data_url:
                    raise ValueError("An enhanced image is invalid.")
                image_bytes = base64.b64decode(data_url.split(",", 1)[1], validate=True)
                output_zip.writestr(filename, image_bytes)
        archive.seek(0)
        return send_file(
            archive,
            mimetype="application/zip",
            as_attachment=True,
            download_name="HD_Screenshots_Ready_To_Print.zip",
        )
    except (ValueError, TypeError, binascii.Error) as error:
        return jsonify({"error": str(error)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
