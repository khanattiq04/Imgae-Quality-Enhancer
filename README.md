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

Inside receipt regions a final "sharp receipt text" stage (`crisp_ink`)
re-renders each stroke from its own ink and paper colours with a ~1 px
anti-aliased edge at the stroke's half-depth contour. This trims the grey
blur spread that makes small receipt values look bold and smudged, removes
pink/grey fringes, and keeps red text red. Counters and gaps can only get
lighter, and white-on-colour bars, photos and the chat wallpaper are skipped.
Set "Sharp receipt text" to Off to get the previous softer output.

Controls: deblur strength, edge refinement, ink/paper thresholding, Safe mode
(on by default; caps aggressive values) and "Disable text clarity processing".
Leave a control on "Profile default" for safe values. Each result has
"Show original" and a before/after zoom crop.

Limits: sharpening cannot recover detail that the source screenshot does not
contain. If text is too blurred, the response carries a warning; check those
values against the original. Nothing is ever redrawn automatically.

## Generate OCR (one click)

Select screenshots and click **Generate OCR (sharp table values)**. For
each screenshot the app enhances it (Receipt / Table in Chat profile, at the
chosen Upscale Multiplier and DPI), reads every value in the data rows of the
receipt table (Sr No, Site Name, Material, Category, Material Rate, Carriage,
Service Charges, Rate/FT ...), redraws those values sharp in the receipt's
font and colours, and shows the result as a card; use its Download PDF or
Download PNG button to save it. Column
titles and the rest of the screenshot are not redrawn.

Values are read with the PP-OCRv6 text recogniser (free, offline; the model
file is in `models/` and is built into the .exe), which handles tiny blurred
digits far better than the small default reader. Each column's title decides
what its values may contain (rates, charges and
Sr No are numbers, Category is a letter code), which is what allows values
only a few pixels tall to be read. A value that still cannot be read
reasonably surely is left enhanced (not guessed). The result card then shows
a zoomed crop of each such value with a box: type what it says and click
**Apply**, and it is redrawn sharp with the rest.

### Reading every value by itself (value_context.py)

The recogniser alone confuses tiny, blurred, underlined values ("77" read
as "11", "/" as "L"). Each reading is therefore re-decided with the rest of
the receipt, the way a person would read it:

- **The receipt's own arithmetic.** Material Rate + Carriage + Service
  Charges = Rate/FT (and CFT x Rate = Total Amount). The candidate values of
  a row that add up are chosen together; when the recogniser's own guesses do
  not add up, every value with the right number of digits that makes the sum
  right is scored against the pixels. A row whose readings cannot be made to
  add up is never redrawn without asking.
- **Known names.** Categories (A, B, /, Black, Normal Black, Asphalt Quality,
  Carriage Only, NLC) and materials (Crush 1/2+3/8, Sub Base, Rora, Khaka
  Black ...) are a small vocabulary, and the site name is printed large in
  the receipt's title ("Rate For Qila Kalar Wala Site Material"). A known
  name wins only when the pixels support it about as well as the raw reading
  (character probabilities from the recogniser + comparison with the font).
- **Shape and format.** A lone category glyph that is narrow and leans right
  is "/"; a Rate/FT column written "157.00" turns "16400" into "164.00"; a
  line ending in "-" before a continuation line keeps its hyphen.
- **Layout.** The title row is found even under a double rule, a table cut
  off at the top borrows the titles of a table with the same columns, lines
  the detector missed ("mm" under "Khaka 0.5") are read, an underline blurred
  into its value is separated, and each column keeps one colour and
  underline style.
- **Learning.** Values you type in the check box are saved to
  `%APPDATA%\ScreenshotHDPrintEnhancer\learned_values.json` and are known
  from then on.

On the sample receipt screenshots used in testing, every rate-table value was
read automatically except one cell where two numbers share a cell. Bank
detail tables (IBAN / account numbers) keep the plain reading and are still
listed for checking.

## Redraw values (OCR, confirmed)

When receipt text is only a few pixels tall, no filter can make it as crisp
as a re-typed document. After processing an image, **Redraw values (OCR)**
reads every value with an offline OCR engine (RapidOCR) and lists it next to
an enlarged crop of the original. Correct anything that is wrong, tick
**Confirm**, then **Redraw confirmed values**: only ticked values are
re-typeset in a sharp font (Times/Arial Bold on Windows) at the same place
and colour, with underlines redrawn and table rules kept. Text is never
stretched: every value in the table body gets one shared font size, the
column titles (Sr No, Site Name, Material, ...) share one small size that is
never larger than the values, and "Same font for all" sets one font family
for every value. Text that sits against a box's left rule (the Note lines) is
redrawn left-aligned with a little padding, and its underline follows it. Unticked
values keep their enhanced pixels, and **Undo redraw** restores the enhanced
image. Values the OCR could not read are still listed (yellow) so you can
type them.

## AI redraw (optional, generative)

**AI redraw (GPT)** sends the screenshot to the OpenAI image-editing API
(desktop/local only; needs an API key entered in the dialog or the
`OPENAI_API_KEY` environment variable; model from `OPENAI_IMAGE_MODEL`,
default `gpt-image-1`). The result is shown next to the original with a list
of original texts that OCR could not find in the AI image. Generative models
can change names and numbers, so check every value before using it.

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
