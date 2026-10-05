import io
import base64
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageEnhance, ImageFilter
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

    # 1. High-Order Lanczos Anti-Aliased Resampling
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    if mode == "chat":
        # WHATSAPP & CHAT ENHANCEMENT PIPELINE
        # Pass 1: Fine micro-sharpening for tiny chat text, timestamps, & ticks
        micro_sharp = resized.filter(ImageFilter.UnsharpMask(radius=0.8, percent=220, threshold=1))
        # Pass 2: Macro structural sharpening for chat bubbles & borders
        macro_sharp = micro_sharp.filter(ImageFilter.UnsharpMask(radius=2.2, percent=130, threshold=2))
        
        # Micro-contrast boost so light grey subtext (#8696a0) prints crisp
        contrast_enhancer = ImageEnhance.Contrast(macro_sharp)
        contrast_img = contrast_enhancer.enhance(1.18)
        
        # Sharpness pass for readable chat text
        sharp_enhancer = ImageEnhance.Sharpness(contrast_img)
        final_img = sharp_enhancer.enhance(1.30)

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

    # Save with embedded print DPI headers (300/600 DPI)
    out_buf = io.BytesIO()
    final_img.save(out_buf, format="PNG", dpi=(dpi, dpi), optimize=True)
    out_bytes = out_buf.getvalue()

    b64_data = base64.b64encode(out_bytes).decode("utf-8")
    return {
        "data_url": f"data:image/png;base64,{b64_data}",
        "width": new_w,
        "height": new_h,
        "orig_width": orig_w,
        "orig_height": orig_h,
        "dpi": dpi
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