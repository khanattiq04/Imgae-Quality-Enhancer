# Screenshot HD Print Enhancer

Local screenshot enhancement for chat screenshots, receipts, invoices, tables,
and scanned documents. Standard processing uses OpenCV and Pillow only; it
does not upload images or redraw text.

## Run in VS Code on Windows

```powershell
py -3 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements-desktop.txt
.\venv\Scripts\python.exe desktop.py
```

The desktop window is backed by a Flask server bound to `127.0.0.1`. Source
images are read in memory and are never overwritten.

## Modes and settings

- **Standard Enhancement** preserves color and applies optional denoising,
  CLAHE luminance contrast, interpolation-based scaling, and controlled
  unsharp masking.
- **Text-Focused Enhancement** adds conservative text sharpening. A separate
  adaptive threshold variant can be generated through the API; the normal
  output remains colored.
- Existing receipt/chat and document profiles are retained for compatibility.
- Lanczos, cubic, and linear interpolation change pixel dimensions but cannot
  recover detail absent from the source.
- PNG is the recommended lossless export. DPI is print metadata, not extra
  image detail.
- AI profiles are optional and explicitly labeled. They may reconstruct or
  alter characters and numbers and are never used by standard processing.

The browser UI processes multiple selected files and reports individual
failures. The desktop build uses the native Save As dialog.

## Text clarity pipeline

Receipt text is processed only inside detected receipt regions, on luminance
only, so wallpaper, colored headers and red text keep their original colors.
The pipeline (`text_clarity.py`) estimates blur conservatively, applies a
regularized Wiener deconvolution only when blur is measurable, uses edge-aware
denoising, gated local contrast, half-level edge steepening, and (for the
black-and-white profile) adaptive ink/paper thresholding. It never erodes or
dilates glyphs, and a final guard prevents thicker strokes, closed counters,
merged characters or halos.

Controls: deblur strength, edge refinement, ink/paper thresholding, Safe mode
(on by default; caps aggressive values) and "Disable text clarity processing".
Leave a control on "Profile default" for safe values. Each result has
"Show original" and a before/after zoom crop.

Limits: sharpening cannot recover detail that the source screenshot does not
contain. If text is too blurred, the response carries a warning; check those
values against the original. Nothing is ever redrawn automatically.

## Tests

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Troubleshooting

- If OpenCV cannot be imported, reinstall `requirements-desktop.txt` in the
  same interpreter used to start the application.
- AI modes require the OpenCV contrib package and download their pinned model
  only when explicitly selected.
- Very large images may be rejected to prevent excessive memory allocation.
  Use 2x for large phone screenshots; the application reports the requested
  output dimensions when the safety limit is reached.
