"""Windows desktop entry point for Screenshot HD Print Enhancer.

The processing server is intentionally bound to 127.0.0.1 only: uploaded
screenshots never leave the computer. PyWebView presents it in an app window
instead of the user's browser.
"""
import base64
import binascii
import threading
from pathlib import Path

import webview
from werkzeug.serving import make_server

from api.index import app


def save_result(filename, data_url):
    """Save a download through Windows, bypassing WebView download handling."""
    safe_filename = Path(str(filename)).name or "enhanced-image.pdf"
    extension = Path(safe_filename).suffix.lower()
    file_types_by_extension = {
        ".pdf": ("PDF document (*.pdf)",),
        ".png": ("PNG image (*.png)",),
        ".zip": ("ZIP archive (*.zip)",),
    }
    file_types = file_types_by_extension.get(extension, ("All files (*.*)",))
    selected_path = app_window.create_file_dialog(
        webview.SAVE_DIALOG,
        save_filename=safe_filename,
        file_types=file_types,
    )
    if not selected_path:
        return {"success": False, "cancelled": True}

    if isinstance(selected_path, (list, tuple)):
        selected_path = selected_path[0]
    try:
        encoded_data = str(data_url).split(",", 1)[1]
        Path(selected_path).write_bytes(base64.b64decode(encoded_data, validate=True))
        return {"success": True}
    except (IndexError, ValueError, TypeError, binascii.Error, OSError) as error:
        return {"success": False, "error": str(error)}


def start_local_server():
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main():
    global app_window
    server = start_local_server()
    url = f"http://127.0.0.1:{server.server_port}"
    app_window = webview.create_window(
        "Screenshot HD Print Enhancer",
        url,
        width=1120,
        height=820,
        min_size=(780, 620),
    )
    app_window.expose(save_result)
    try:
        webview.start()
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
