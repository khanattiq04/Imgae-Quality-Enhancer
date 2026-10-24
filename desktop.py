"""Windows desktop entry point for Screenshot HD Print Enhancer.

The processing server is intentionally bound to 127.0.0.1 only: uploaded
screenshots never leave the computer. PyWebView presents it in an app window
instead of the user's browser.
"""
import base64
import binascii
import html
import threading
import time
from pathlib import Path

import webview
from werkzeug.serving import make_server

from api.index import app
import license_check

TITLE = "Screenshot HD Print Enhancer"


def blocked_page(message):
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
    body {{ margin:0; height:100vh; display:flex; align-items:center; justify-content:center;
           background:#0b1222; color:#e2e8f0; font-family:Segoe UI, Arial, sans-serif; }}
    .box {{ max-width:520px; padding:2rem 2.2rem; border-radius:16px; text-align:center;
           background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1); }}
    h1 {{ font-size:1.3rem; margin:0 0 0.8rem; color:#fbbf24; }}
    p {{ line-height:1.55; color:#cbd5e1; }}
    code {{ background:#111a2e; padding:0.15rem 0.45rem; border-radius:6px; color:#fff; }}
    </style></head><body><div class="box"><h1>App turned off</h1>
    <p>{html.escape(message)}</p>
    <p>Device ID: <code>{license_check.device_id()}</code></p></div></body></html>"""


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


def watch_licence(server):
    """Re-check the owner's on/off switch while the app is open."""
    while True:
        time.sleep(license_check.CHECK_EVERY_SECONDS)
        allowed, message = license_check.check()
        if not allowed:
            try:
                app_window.load_html(blocked_page(message))
            finally:
                server.shutdown()
            return


def main():
    global app_window
    title = f"{TITLE}  -  Device ID {license_check.device_id()}"
    allowed, message = license_check.check()
    if not allowed:
        app_window = webview.create_window(title, html=blocked_page(message), width=720, height=480)
        webview.start()
        return

    server = start_local_server()
    url = f"http://127.0.0.1:{server.server_port}"
    app_window = webview.create_window(
        title,
        url,
        width=1120,
        height=820,
        min_size=(780, 620),
    )
    app_window.expose(save_result)
    threading.Thread(target=watch_licence, args=(server,), daemon=True).start()
    try:
        webview.start()
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
