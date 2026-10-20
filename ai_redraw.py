"""Optional generative redraw through the OpenAI image-editing API.

This is the "like ChatGPT" mode: an image model repaints the screenshot with
crisp text.  It is generative, so it can silently change names and numbers.
The API endpoint therefore always returns it next to the original together
with an OCR comparison of both images, and the UI shows a warning.

Requires an OpenAI API key (OPENAI_API_KEY environment variable or entered in
the desktop app; it is never stored).  The model can be changed with the
OPENAI_IMAGE_MODEL environment variable.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
import uuid

OPENAI_EDITS_URL = "https://api.openai.com/v1/images/edits"
DEFAULT_MODEL = "gpt-image-1"

PROMPT = (
    "Restore this phone screenshot to sharp, high-definition, print quality. "
    "Keep the exact same layout, colours, chat bubbles, background and proportions. "
    "Copy every word, name, number, date, phone number, symbol and punctuation mark "
    "exactly as it appears in the original, character for character. Do not change, "
    "add, remove, translate, re-spell or correct anything, even if it looks like a "
    "spelling mistake. Only make blurry text crisp, clean and easy to read."
)


class AiRedrawError(RuntimeError):
    pass


def model_name(requested: str | None = None) -> str:
    return (requested or os.environ.get("OPENAI_IMAGE_MODEL") or DEFAULT_MODEL).strip()


def _multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    for name, (filename, data, content_type) in files.items():
        parts.append(
            (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
             f"Content-Type: {content_type}\r\n\r\n").encode() + data + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def ai_redraw(png_bytes: bytes, api_key: str | None = None, model: str | None = None,
              timeout: int = 300) -> bytes:
    """Send the screenshot to the image-edit API and return the PNG it draws."""
    key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise AiRedrawError(
            "AI redraw needs an OpenAI API key. Enter it in the app or set the "
            "OPENAI_API_KEY environment variable."
        )
    body, content_type = _multipart(
        {"model": model_name(model), "prompt": PROMPT, "size": "auto", "quality": "high", "n": "1"},
        {"image": ("screenshot.png", png_bytes, "image/png")},
    )
    request = urllib.request.Request(
        OPENAI_EDITS_URL,
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": content_type},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            pass
        raise AiRedrawError(f"The image API refused the request ({error.code}): {detail}") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise AiRedrawError(f"Could not reach the image API: {error}") from error

    try:
        b64 = payload["data"][0]["b64_json"]
    except (KeyError, IndexError, TypeError) as error:
        raise AiRedrawError("The image API returned no image.") from error
    return base64.b64decode(b64)
