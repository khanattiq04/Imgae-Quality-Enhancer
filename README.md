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
