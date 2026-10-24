# Hugging Face Space for Screenshot HD Print Enhancer (with OCR).
# The Space only holds this Dockerfile and README.md. The app code is pulled
# from GitHub at build time, so it is never published in the Space itself.
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
        fonts-liberation fonts-dejavu-core libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

ARG REPO=https://github.com/khanattiq04/Imgae-Quality-Enhancer.git
ARG BRANCH=main

# GITHUB_TOKEN (a Space secret) is only needed when the GitHub repo is private.
RUN --mount=type=secret,id=GITHUB_TOKEN,mode=0444,required=false \
    TOKEN="$(cat /run/secrets/GITHUB_TOKEN 2>/dev/null || true)"; \
    URL="$REPO"; \
    if [ -n "$TOKEN" ]; then URL="$(echo "$REPO" | sed "s#https://#https://x-access-token:${TOKEN}@#")"; fi; \
    git clone --depth 1 --branch "$BRANCH" "$URL" /app \
 && rm -rf /app/.git \
 && test -f /app/models/PP-OCRv6_rec_small.onnx

WORKDIR /app

# rapidocr is installed without its own dependencies because it would add a
# second OpenCV package that clashes with opencv-contrib-python-headless.
RUN pip install --no-cache-dir -r requirements.txt gunicorn \
        onnxruntime pyclipper PyYAML shapely six \
 && pip install --no-cache-dir --no-deps rapidocr-onnxruntime \
 && python -c "import cv2, rapidocr_onnxruntime, onnxruntime; print('OCR ok')" \
 && chown -R user:user /app

USER user
ENV HOME=/home/user PYTHONUNBUFFERED=1
EXPOSE 7860

CMD ["gunicorn", "--chdir", "api", "--bind", "0.0.0.0:7860", \
     "--workers", "1", "--threads", "4", "--timeout", "300", "index:app"]
