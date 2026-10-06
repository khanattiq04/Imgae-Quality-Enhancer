import io
import base64
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pypdfium2 as pdfium

app = Flask(__name__, static_folder='../public', static_url_path='')

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

    # A screenshot has a fixed amount of source detail. Enlarging it helps a
    # printer render smooth edges, but must not make it print physically larger.
    # The output DPI metadata below keeps its intended physical size intact.
    # 1. High-Order Lanczos Anti-Aliased Resampling
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    if mode == "chat":
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
    final_img.save(out_buf, format="PNG", dpi=(embedded_dpi, embedded_dpi), optimize=True)
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
    return send_from_directory('../public', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('../public', path)

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
    if mode not in ("chat", "document", "general"):
        return jsonify({"error": "Unknown enhancement profile."}), 400

    file_bytes = file.read()
    results = []

    try:
        # PDF Rendering
        if filename.lower().endswith(".pdf") or file.content_type == "application/pdf":
            pdf = pdfium.PdfDocument(file_bytes)
            for idx, page in enumerate(pdf):
                render_scale = (dpi / 72.0)
                pil_img = page.render(scale=render_scale).to_pil()
                processed = enhance_image(pil_img, scale_factor=1, dpi=dpi, mode=mode)
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

if __name__ == "__main__":
    app.run(debug=True, port=5000)
