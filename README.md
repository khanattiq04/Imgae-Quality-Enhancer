# Screenshot HD Print Enhancer

A complete Python & Web-based tool designed to upscale, sharpen, and convert screenshots into high-resolution 300 DPI print-ready images.

## Project Structure
- `api/index.py`: Python Flask backend engine with PIL and pypdfium2 upscaling logic.
- `public/index.html`: Responsive drag-and-drop web GUI with batch processing and instant download.
- `requirements.txt`: Python package dependencies.
- `vercel.json`: Vercel serverless deployment routing config.

## How to Run Locally
1. Install requirements:
   `pip install -r requirements.txt`
2. Launch backend server:
   `python api/index.py`
3. Open `http://127.0.0.1:5000` in your browser.

## How to Deploy on Vercel
1. Upload/Push this project folder to GitHub.
2. Link your GitHub repository in Vercel.
3. Deploy! Vercel will automatically read `vercel.json` and host your tool live.

## Windows Desktop App (offline processing)
The web version uploads each image to the server, so its speed depends on your
internet connection and server capacity. The desktop version processes files on
the computer only: no screenshots are uploaded and no internet connection is
needed after it is built.

1. Install Python 3 for Windows and select **Add python.exe to PATH** during
   setup.
2. Double-click `build-windows-exe.bat`.
3. When the build finishes, open `dist\\ScreenshotHDPrintEnhancer.exe`.

You can copy that one EXE to another Windows PC and run it there; Python is not
required on the receiving PC. Keep the web deployment as an alternative for
occasions when you need to use the tool from another device.

## Enhancement profiles
- **AI reconstruction:** makes the attached preview look much clearer for printing, but may introduce incorrect fine text/numbers.
- **Maximum Print Text (Black & White):** prioritizes hard black/white text edges for the clearest print readability.
- **WhatsApp & Chat Screenshot:** balances text sharpening while keeping chat colors natural.
- **Document & Scan:** boosts contrast and edge definition for scanned or document-like pages.
- **General Screenshot / Graphic:** moderate enhancement for mixed screenshots and graphics.

> Note: AI reconstruction is intended for local desktop use where the model (~38MB) is downloaded once and then reused from local storage.
