"""Context-aware reading of receipt table values.

The text recogniser alone reads each value in isolation.  A person reading a
blurred receipt uses more than the pixels of one cell, and so does this module:

* Vocabulary - categories, materials and site names repeat from receipt to
  receipt ("Crush 1/2+3/8", "Normal Black", "/").  Site names also appear in
  large, clear type in the receipt's own title ("Rate For Qila Kalar Wala
  Site Material").  Each candidate is scored against the recogniser's
  character probabilities (CTC), so a vocabulary entry is only chosen when the
  pixels support it about as well as the free reading.
* Arithmetic - on rate receipts Material Rate + Carriage + Service Charges =
  Rate/FT, and on summaries CFT x Rate = Total Amount.  The candidate values
  of a row that satisfy the receipt's own arithmetic are chosen together.
* Format - a Rate/FT column that reads "157.00", "148.00" ... turns "16400"
  into "164.00".
* Learning - values the user types in are saved and become vocabulary for the
  next screenshot.

Nothing here invents a value: every accepted value is one the recogniser
itself proposes (or a vocabulary/title entry) that matches the original
pixels, and that agrees with the rest of the receipt.
"""

from __future__ import annotations

import difflib
import itertools
import json
import os
import re
import threading

import cv2
import numpy as np

NEG_INF = -1e9
_LOCK = threading.Lock()

DEFAULT_VOCABULARY = {
    "category": ["A", "B", "C", "D", "/", "-", "Black", "Normal Black", "Normal", "Asphalt Quality", "Asphalt",
                 "Quality", "Carriage Only", "NLC"],
    "material": ["Crush", "Crush 1/2+3/8", "1/2+3/8", "Crush 2/8", "Crush 1/2 , 3/8", "Crush 5-10 mm",
                 "Crush 5-10", "Crush 8-", "12,10-19,19-", "Crush 8-12,10-19,19-", "25,25-38 mm",
                 "Crush 8-12,10-19,19-25,25-38 mm",
                 "Sub Base", "Rora", "Khaka Black", "Khaka 0.5 mm", "Khaka 0.5", "0.5 Khaka", "mm", "Aggregate",
                 "Bajjar Saaf", "Bolder", "5-10 MM"],
    "site": [],
    "header": ["Sr No:", "Site Name", "Material", "Category", "Material Rate", "Carriage", "Service Charges",
               "Rate/FT", "Rate/TON", "CFT", "Total Amount", "Page Number", "Quantity Of Trucks", "Trucks"],
}

HEADER_WORDS = ("sr", "no", "site", "name", "material", "category", "rate", "carriage", "service", "charges",
                "cft", "total", "amount", "page", "quantity", "trucks", "bank", "iban", "account", "title",
                "date", "/ft", "ton")


# ---------------------------------------------------------------------------
# Vocabulary (defaults + values learned from the user's corrections)
# ---------------------------------------------------------------------------

def _learned_path() -> str:
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    return os.path.join(base, "ScreenshotHDPrintEnhancer", "learned_values.json")


def load_vocabulary() -> dict:
    vocab = {k: list(v) for k, v in DEFAULT_VOCABULARY.items()}
    try:
        with open(_learned_path(), encoding="utf-8") as fh:
            learned = json.load(fh)
        for role, values in learned.items():
            vocab.setdefault(role, [])
            vocab[role] += [v for v in values if isinstance(v, str) and v not in vocab[role]]
    except (OSError, ValueError):
        pass
    return vocab


def learn_values(pairs) -> int:
    """Remember values the user typed: [(column title, text)...]. Returns how many were new."""
    pairs = [(column_role(c), str(t).strip()) for c, t in pairs if str(t).strip()]
    pairs = [(r, t) for r, t in pairs if r in ("category", "material", "site") and len(t) <= 60]
    if not pairs:
        return 0
    with _LOCK:
        path = _learned_path()
        try:
            with open(path, encoding="utf-8") as fh:
                learned = json.load(fh)
        except (OSError, ValueError):
            learned = {}
        added = 0
        for role, text in pairs:
            values = learned.setdefault(role, [])
            if text not in values and text not in DEFAULT_VOCABULARY.get(role, []):
                values.append(text)
                added += 1
        if added:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(learned, fh, ensure_ascii=False, indent=1)
            except OSError:
                return 0
        return added


def document_phrases(texts) -> list[str]:
    """Site names and other phrases printed clearly elsewhere on the screenshot."""
    phrases = []
    for text in texts:
        m = re.search(r"rate\s*for\s*(.+?)[\s_]*sit\w*\s*mat", text, re.I)
        if m:
            phrases.append(m.group(1))
    out = []
    for p in phrases:
        p = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", p.replace("_", " "))   # "QilaKalarWala" -> "Qila Kalar Wala"
        p = re.sub(r"\s+", " ", p).strip(" .:")
        if len(p) >= 3:
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------

def column_role(header: str) -> str:
    h = (header or "").lower().replace(" ", "")
    if "material" in h and "rate" in h:
        return "material_rate"
    if "carri" in h:
        return "carriage"
    if "servic" in h or "charg" in h or "chare" in h:
        return "service"
    if "cft" in h:
        return "cft"
    if "total" in h or "amount" in h:
        return "total"
    if "rate" in h or "/ft" in h or "/ton" in h:
        return "rate"
    if "categ" in h or "categ" in h.replace("r", "g"):
        return "category"
    if "site" in h:
        return "site"
    if "material" in h or "materlal" in h:
        return "material"
    if h.startswith(("srno", "sno", "sr.", "5rno")) or h in ("sr", "no", "#"):
        return "serial"
    if "truck" in h or "quant" in h:
        return "trucks"
    return ""


def header_score(texts) -> int:
    """How much a row of cell texts looks like column titles."""
    score = 0
    for t in texts:
        low = (t or "").lower()
        if not low or re.search(r"\d{2,}", low):
            continue
        if any(w in low for w in HEADER_WORDS):
            score += 1
    return score


# ---------------------------------------------------------------------------
# CTC scoring with the PP-OCR recogniser's character probabilities
# ---------------------------------------------------------------------------

def _logsumexp(*arrays):
    stacked = np.stack(arrays)
    m = stacked.max(axis=0)
    m_safe = np.where(m <= NEG_INF / 2, 0.0, m)
    with np.errstate(divide="ignore"):
        out = m_safe + np.log(np.exp(stacked - m_safe).sum(axis=0))
    return np.where(m <= NEG_INF / 2, NEG_INF, out)


def ctc_logp(probs: np.ndarray, labels) -> float:
    """log P(labels | probs) under CTC (blank = index 0)."""
    if not labels:
        return float(np.log(probs[:, 0] + 1e-12).sum())
    ext = [0]
    for lab in labels:
        ext += [lab, 0]
    ext = np.array(ext)
    S = len(ext)
    lp = np.log(probs[:, ext] + 1e-12)
    can_skip = np.zeros(S, bool)
    for s in range(2, S):
        can_skip[s] = ext[s] != 0 and ext[s] != ext[s - 2]
    alpha = np.full(S, NEG_INF)
    alpha[0] = lp[0, 0]
    alpha[1] = lp[0, 1]
    for t in range(1, probs.shape[0]):
        prev1 = np.concatenate([[NEG_INF], alpha[:-1]])
        prev2 = np.concatenate([[NEG_INF, NEG_INF], alpha[:-2]])
        prev2 = np.where(can_skip, prev2, NEG_INF)
        alpha = _logsumexp(alpha, prev1, prev2) + lp[t]
    return float(_logsumexp(np.array([alpha[-1]]), np.array([alpha[-2]]))[0])


def ctc_beam(probs: np.ndarray, allowed: np.ndarray, width: int = 8):
    """Prefix beam search restricted to `allowed` class indices -> [(label tuple, logp)]."""
    beams = {(): (0.0, NEG_INF)}   # prefix -> (log p ending in blank, log p ending in non-blank)
    lp = np.log(probs + 1e-12)
    for t in range(probs.shape[0]):
        row = lp[t]
        top = allowed[np.argsort(row[allowed])[-6:]]
        nxt = {}

        def add(prefix, pb, pnb):
            ob, onb = nxt.get(prefix, (NEG_INF, NEG_INF))
            nxt[prefix] = (np.logaddexp(ob, pb), np.logaddexp(onb, pnb))

        for prefix, (pb, pnb) in beams.items():
            total = np.logaddexp(pb, pnb)
            add(prefix, total + row[0], NEG_INF)
            for c in top:
                if c == 0:
                    continue
                if prefix and prefix[-1] == c:
                    add(prefix + (c,), NEG_INF, pb + row[c])
                    add(prefix, NEG_INF, pnb + row[c])
                else:
                    add(prefix + (c,), NEG_INF, total + row[c])
        beams = dict(sorted(nxt.items(), key=lambda kv: -np.logaddexp(*kv[1]))[:width])
    return [(p, float(np.logaddexp(*v))) for p, v in beams.items()]


class Scorer:
    """Character probabilities of one value, from several crops and contrasts."""

    def __init__(self, rgb: np.ndarray, box, recogniser, contrast_fn, lock, bottom=None):
        self.rec = recogniser
        self.probs = []
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        h = max(1, y1 - y0)
        # Never let the crops reach into the value's underline: a "0" sitting
        # on its red bar reads as "2".
        limit = int(round(bottom)) if bottom is not None else None
        for py in (1, 2):
            for px in sorted({2, max(2, round(0.3 * h))}):
                y_end = y1 + py if limit is None else max(y1, min(y1 + py, limit))
                crop = rgb[max(0, y0 - py):y_end, max(0, x0 - px):x1 + px]
                if crop.size == 0:
                    continue
                for variant in (crop, contrast_fn(crop)):
                    for height in (32, 48):
                        f = height / variant.shape[0]
                        big = cv2.resize(variant, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
                        big = cv2.copyMakeBorder(big, 6, 6, 12, 12, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                        with lock:
                            self.probs.append(recogniser.logits(big))
        self.index = {c: i for i, c in enumerate(recogniser.chars)}
        self._cache = {}

    def labels(self, text: str):
        out = []
        for ch in text:
            if ch not in self.index:
                return None
            out.append(self.index[ch])
        return out

    def score(self, text: str) -> float:
        """Mean log-probability of `text` over all crops (higher is better)."""
        if text in self._cache:
            return self._cache[text]
        labels = self.labels(text)
        if labels is None or not self.probs:
            value = NEG_INF
        else:
            value = float(np.mean([ctc_logp(p, labels) for p in self.probs]))
        self._cache[text] = value
        return value

    def proposals(self, charset: str, per_crop: int = 6):
        allowed = np.array([0] + [self.index[c] for c in charset if c in self.index])
        seen = set()
        for p in self.probs[::2]:
            for prefix, _ in ctc_beam(p, allowed, width=per_crop)[:per_crop]:
                text = "".join(self.rec.chars[i] for i in prefix).strip()
                if text:
                    seen.add(text)
        return seen


def posterior(scores: dict) -> dict:
    """Share of probability each candidate gets among the candidates."""
    if not scores:
        return {}
    keys = list(scores)
    vals = np.array([scores[k] for k in keys])
    vals = np.exp(vals - vals.max())
    vals /= vals.sum()
    return dict(zip(keys, vals.tolist()))


def similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower().replace(" ", ""), b.lower().replace(" ", "")).ratio()


def as_number(text: str):
    t = text.replace(",", "")
    if not re.fullmatch(r"\d+(\.\d+)?", t):
        return None
    return float(t)


def number_variants(text: str, decimal_column: bool):
    """Same digits in the receipt's number format ("16400" -> "164.00", "155,00" -> "155.00")."""
    out = {text}
    t = re.sub(r",(\d{2})$", r".\1", text)
    out.add(t)
    digits = t.replace(".", "").replace(",", "")
    if decimal_column and digits.isdigit() and len(digits) >= 3:
        out.add(digits[:-2] + "." + digits[-2:])
    if not decimal_column and t.endswith(".00"):
        out.add(t[:-3])
    return {v for v in out if v}


# ---------------------------------------------------------------------------
# Refining a table's readings
# ---------------------------------------------------------------------------

NUMBER_ROLES = ("material_rate", "carriage", "service", "rate", "cft", "total", "trucks")
DIGITS = "0123456789"
TEXT_WEIGHT = 12.0          # pixel-template evidence vs. log-probability units
LEXICON_BONUS = 3.0         # prior for a value already known (vocabulary); doubled for categories
TITLE_BONUS = 6.0           # a site name printed in the receipt's own title


def _pixel_match(T, rgb, it, text: str) -> float:
    """How well `text` drawn in the receipt font matches the value's pixels."""
    score = T.verify_reading(rgb, {"box": it["box"], "text": text})
    if len(text) == 1:                              # a lone glyph may have its underline fused on
        x0, y0, x1, y1 = [int(round(v)) for v in it["box"]]
        h = max(1, y1 - y0)
        reach = max(2, round(0.45 * h))
        crop = rgb[max(0, y0 - 1):y1 + 1, max(0, x0 - reach):x1 + reach]
        if crop.size:
            cov = T._coverage(crop)
            rows, cols = np.where(cov.max(axis=1) > 0.35)[0], np.where(cov.max(axis=0) > 0.25)[0]
            if rows.size and cols.size:
                cov = cov[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
                score = max(score, T._template_score(cov, text), T._template_score(cov, text, underlined=True))
    return float(score)


def _widen_to_ink(T, rgb, it) -> None:
    """Stretch a word box that the detector cut short ("Sub B" of "Sub Base") to the line's ink."""
    cell = it.get("cell")
    if not cell:
        return
    x0, y0, x1, y1 = it["box"]
    cx0, _, cx1, _ = cell
    band = rgb[int(round(y0)):int(round(y1)), int(cx0):int(cx1)]
    if band.size == 0:
        return
    ink, _ = T._ink_mask(band)
    cols = np.where(ink.any(axis=0))[0]
    if not cols.size:
        return
    h = max(1.0, y1 - y0)
    # Only ink that continues the word (no gap wider than a word space) counts.
    left, right = int(round(x0 - cx0)), int(round(x1 - cx0))
    gap = max(3, int(round(1.2 * h)))
    on = set(cols.tolist())
    while any(c in on for c in range(right, right + gap)):
        right = max(c for c in range(right, right + gap) if c in on) + 1
    while any(c in on for c in range(left - gap, left)):
        left = min(c for c in range(left - gap, left) if c in on)
    nx0, nx1 = cx0 + left, cx0 + right
    if max(x1, nx1) - min(x0, nx0) > 1.8 * max(1.0, x1 - x0):
        return                                  # that is more than the rest of one word
    if nx0 < x0 - max(1.5, 0.2 * h) or nx1 > x1 + max(1.5, 0.2 * h):
        it["box"] = [float(min(x0, nx0)), y0, float(max(x1, nx1)), y1]
        if it.get("underline"):
            u = it["underline"]
            it["underline"] = [min(u[0], it["box"][0]), u[1], max(u[2], it["box"][2]), u[3]]


def glyph_shape(T, rgb, box):
    """(width / height, slant) of a lone glyph's ink; slant > 0 leans right like "/"."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    h = max(1, y1 - y0)
    reach = max(2, round(0.45 * h))
    crop = rgb[max(0, y0 - 1):y1 + 1, max(0, x0 - reach):x1 + reach]
    if crop.size == 0:
        return 1.0, 0.0
    cov = T._coverage(crop)
    rows, cols = np.where(cov.max(axis=1) > 0.35)[0], np.where(cov.max(axis=0) > 0.25)[0]
    if not rows.size or not cols.size:
        return 1.0, 0.0
    cov = cov[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    H, W = cov.shape
    xs = np.arange(W)

    def centre(a, b):
        part = cov[a:b]
        return float((part.sum(axis=0) * xs).sum() / max(part.sum(), 1e-6))

    top = centre(0, max(1, int(H * 0.4)))
    mid = centre(int(H * 0.4), max(int(H * 0.4) + 1, int(H * 0.75)))
    return W / H, (top - mid) / W


def refine(T, rgb: np.ndarray, items: list, doc_texts=()) -> list:
    """Re-decide each reading using vocabulary, the receipt's arithmetic and number format.

    `T` is the text_redraw module.  Items are updated in place: "text",
    "confirmed", "confidence" and an "evidence" note ("vocabulary",
    "arithmetic", "format", "recogniser").
    """
    rec = T.strong_recogniser()
    if rec is None or not items:
        return items
    all_items = items
    vocab = load_vocabulary()
    phrases = document_phrases(doc_texts)
    vocab["site"] = list(dict.fromkeys(vocab.get("site", []) + phrases))
    everything = list(dict.fromkeys(v for values in vocab.values() for v in values))
    scorers = {}

    def scorer(it):
        key = id(it)
        if key not in scorers:
            underline = it.get("underline")
            scorers[key] = Scorer(rgb, it["box"], rec, T._contrast, T._ENGINE_LOCK,
                                  bottom=underline[1] if underline else None)
        return scorers[key]

    for it in items:
        it["role"] = column_role(it.get("column", ""))
        it.setdefault("evidence", "recogniser")

    # Bank-detail tables (IBAN / Account Number / Account Title) are label and
    # value pairs, not rate rows: they keep the plain reading.
    tables = {}
    for it in items:
        tables.setdefault(it.get("table"), []).append(it)
    skip_tables = {t for t, its in tables.items()
                   if any(re.search(r"iban|account|bank", it.get("column", ""), re.I) for it in its)}
    items = [it for it in items if it.get("table") not in skip_tables]

    # One small glyph split into two stacked pieces ("2" read as "2" over "4")
    # becomes one value again.
    cells = {}
    for it in items:
        cells.setdefault((it.get("table"), tuple(it.get("row") or ()), it.get("column")), []).append(it)
    for group in cells.values():
        if len(group) != 2 or group[0]["kind"] == "text":
            continue
        a, b = sorted(group, key=lambda g: g["box"][1])
        ha, hb = a["box"][3] - a["box"][1], b["box"][3] - b["box"][1]
        gap = b["box"][1] - a["box"][3]
        if gap <= 1.5 and ha + hb + gap <= 12 and max(ha, hb) <= 5:
            a["box"] = [min(a["box"][0], b["box"][0]), a["box"][1], max(a["box"][2], b["box"][2]), b["box"][3]]
            if b.get("underline"):
                a["underline"] = b["underline"]
            b["drop"] = True
    items = [it for it in items if not it.get("drop")]

    _split_fused_underlines(T, rgb, items)

    # -- lines the detector missed ("mm" under "Khaka 0.5") -----------------
    by_cell = {}
    for it in items:
        if it.get("cell") and it["kind"] != "number" and it.get("serial") is None:
            by_cell.setdefault(it["cell"], []).append(it)
    for cell, siblings in by_cell.items():
        cx0, cy0, cx1, cy1 = cell
        crop = rgb[cy0:cy1, cx0:cx1]
        if crop.size == 0:
            continue
        for a, b in T._ink_line_bands(crop):
            top, bottom = cy0 + a, cy0 + b
            covered = any(min(bottom, o["box"][3]) - max(top, o["box"][1])
                          > 0.4 * min(bottom - top, max(1.0, o["box"][3] - o["box"][1])) for o in siblings)
            if covered or bottom - top < 4:
                continue
            ink, _ = T._ink_mask(crop[a:b])
            ys, xs = np.where(ink)
            if xs.size == 0:
                continue
            m = T._measure(rgb, [cx0 + xs.min(), top + ys.min(), cx0 + xs.max() + 1, top + ys.max() + 1])
            if m is None:
                continue
            m.pop("word_gaps", None)
            new = {k: v for k, v in siblings[0].items()
                   if k in ("column", "kind", "font", "group", "align", "row", "table", "cell", "headless", "role")}
            new.update(m)
            new.update(text="", ocr_text="", confidence=0.0, confirmed=False, top_vote="", evidence="recogniser")
            items.append(new)
            all_items.append(new)
            siblings.append(new)

    # -- words: categories, materials, site names --------------------------
    for it in items:
        role = it["role"]
        if it.get("serial") is not None or role in NUMBER_ROLES or it["kind"] == "number":
            continue
        _widen_to_ink(T, rgb, it)
        sc = scorer(it)
        reading = it["text"] or ""
        candidates = {reading} if reading else set()
        charset = T.CHARSETS["letter" if role == "category" else "text"]
        candidates |= {c for c in sc.proposals(charset, per_crop=4) if len(c) <= 40}
        pool = vocab.get(role) or everything
        if role == "site":
            pool = pool + vocab.get("material", [])[:0]
        known = set(pool) | set(phrases)
        for v in pool:
            if len(v) <= 1 or (reading and similar(v, reading) >= 0.5) or any(similar(v, c) >= 0.6 for c in candidates):
                candidates.add(v)
        if role == "category" and len(reading) <= 2:
            # A lone category glyph: its shape decides between the narrow,
            # right-leaning "/" (often blurred into an "L" by its underline)
            # and the wide letters. The recogniser alone confuses them.
            ratio, slant = glyph_shape(T, rgb, it["box"])
            singles = [v for v in pool if len(v) == 1]
            if "/" in singles and ratio <= 0.62 and slant >= 0.1:
                if reading != "/":
                    it["text"] = it["ocr_text"] = it["top_vote"] = "/"
                it["confirmed"] = True
                it["evidence"] = "shape"
                it["confidence"] = 0.95
                it["match"] = round(_pixel_match(T, rgb, it, "/"), 3)
                continue
            wide = [v for v in singles if v.isalpha()]
            if ratio >= 0.62 and wide:
                candidates = set(wide)
            else:
                candidates |= set(singles)
        candidates = {c for c in candidates if c.strip()}
        # A line ending in "-" continues on the next line ("Crush 8-12,10-19,19-" / "25,25-38 mm");
        # at this size that hyphen is easily read as a full stop.
        below = [o for o in items if o is not it and o.get("cell") == it.get("cell")
                 and o["box"][1] > it["box"][3] - 1]
        if below:
            hyphenated = {c[:-1] + "-" for c in candidates if c.endswith(".")}
            candidates |= hyphenated
            known |= hyphenated
        if not candidates:
            continue
        scores = {}
        for c in candidates:
            ctc = sc.score(c)
            if ctc <= NEG_INF / 2:
                continue
            bonus = 0.0
            if c in phrases and role in ("site", ""):
                bonus = TITLE_BONUS            # the receipt's own title names the site
            elif c in known:
                bonus = LEXICON_BONUS * (2.0 if role == "category" else 1.0)
            scores[c] = ctc + TEXT_WEIGHT * _pixel_match(T, rgb, it, c) + bonus
        if not scores:
            continue
        if below:      # "19." and "19-" before a continuation line are the same reading
            for c in [c for c in scores if c.endswith(".")]:
                h = c[:-1] + "-"
                scores[h] = max(scores.get(h, NEG_INF), scores.pop(c))
        post = posterior(scores)
        best = max(scores, key=scores.get)
        match = _pixel_match(T, rgb, it, best)
        if best != reading:
            it["text"] = it["ocr_text"] = best
            it["top_vote"] = best
            it["confirmed"] = False
        it["candidates"] = sorted(scores, key=scores.get, reverse=True)[:4]
        it["confidence"] = round(max(float(post[best]), 0.0), 3)
        it["match"] = round(match, 3)
        if best in known and post[best] >= 0.55 and match >= 0.4:
            it["confirmed"] = True
            it["evidence"] = "vocabulary"
        elif post[best] >= 0.9 and match >= 0.55 and len(best) >= 2:
            it["confirmed"] = True
            it["evidence"] = "recogniser"

    # -- numbers: candidates from the recogniser ---------------------------
    numbers = [it for it in items if it.get("serial") is None and (it["role"] in NUMBER_ROLES or it["kind"] == "number")]

    # An underline split off as its own "value" (a 2-3 px sliver under a
    # number) is not a value: drop it.
    groups = {}
    for it in numbers:
        groups.setdefault((it.get("table"), tuple(it.get("row") or ()), it.get("column")), []).append(it)
    for group in groups.values():
        if len(group) < 2:
            continue
        tallest = max(g["box"][3] - g["box"][1] for g in group)
        for g in group:
            if g["box"][3] - g["box"][1] < 0.5 * tallest:
                g["drop"] = True
    numbers = [it for it in numbers if not it.get("drop")]

    combined_cache = {}

    def num_score(it, text):
        key = (id(it), text)
        if key not in combined_cache:
            ctc = scorer(it).score(text)
            if ctc <= NEG_INF / 2:
                combined_cache[key] = NEG_INF
            else:
                combined_cache[key] = ctc + NUMBER_WEIGHT * T.verify_reading(rgb, {"box": it["box"], "text": text})
        return combined_cache[key]

    by_column = {}
    for it in numbers:
        by_column.setdefault((it.get("table"), it.get("column")), []).append(it)
    for column_items in by_column.values():
        texts = [it["text"] for it in column_items if it["text"]]
        decimal = sum("." in t or "," in t for t in texts) * 2 >= max(1, len(texts)) and \
            column_items[0]["role"] in ("rate", "total", "cft", "")
        for it in column_items:
            it["decimal"] = decimal
            sc = scorer(it)
            cands = set(sc.proposals(DIGITS + ".,"))
            for t in (it["text"], it.get("top_vote", "")):
                if t:
                    cands.add(t)
            cands = {v for c in cands for v in number_variants(c, decimal)}
            cands = {c for c in cands if as_number(c) is not None}
            scores = {c: num_score(it, c) for c in cands}
            scores = {c: v for c, v in scores.items() if v > NEG_INF / 2}
            if not scores:
                it["num_candidates"] = []
                continue
            if decimal:      # the column's own format: "164.00", not "16400"
                for c in list(scores):
                    if "." not in c:
                        scores[c] -= 2.5
            ranked = sorted(scores, key=scores.get, reverse=True)[:6]
            it["num_candidates"] = [(c, scores[c]) for c in ranked]
            post = posterior({c: scores[c] for c in ranked})
            # "166.00" and "166" are one value: pool their probability
            value_post = {}
            for c, pr in post.items():
                value_post[as_number(c)] = value_post.get(as_number(c), 0.0) + pr
            post = {c: value_post[as_number(c)] for c in post}
            it["posterior"] = post
            best = ranked[0]
            match = T.verify_reading(rgb, {"box": it["box"], "text": best})
            if best != it["text"]:
                same_digits = best.replace(".", "") == it["text"].replace(".", "").replace(",", "")
                it["text"] = it["ocr_text"] = best
                it["top_vote"] = best
                it["confirmed"] = False
                it["evidence"] = "format" if same_digits else "recogniser"
            it["confidence"] = round(float(post[best]), 3)
            it["match"] = round(float(match), 3)
            if (post[best] >= 0.85 and match >= 0.5) or (post[best] >= 0.8 and match >= 0.6):
                it["confirmed"] = True

    # -- arithmetic inside each row ----------------------------------------
    rows = {}
    for it in numbers:
        row = rows.setdefault((it.get("table"), tuple(it.get("row") or ())), {})
        if it["role"] in row:          # two values for one role: do not guess which is which
            row["_clash"] = True
        row[it["role"]] = it
    for row in rows.values():
        if row.get("_clash"):
            continue
        if all(r in row for r in ("material_rate", "carriage", "service", "rate")):
            _solve_row(rgb, T, [row["material_rate"], row["carriage"], row["service"]], row["rate"],
                       sum, num_score, additive=True)
        elif all(r in row for r in ("cft", "rate", "total")):
            _solve_row(rgb, T, [row["cft"], row["rate"]], row["total"],
                       lambda vals: vals[0] * vals[1], num_score, additive=False)

    _consistent_column_underlines(T, rgb, items)
    _consistent_column_ink(items)
    return [it for it in all_items if not it.get("drop")]


def _split_fused_underlines(T, rgb, items) -> None:
    """Separate an underline that blurred into its value ("76" over its red bar).

    Plain values on the same row (Sr No, Rate/FT, site) give the text height;
    an underlined value measuring taller than that by a pixel or more, with a
    bar in the extra rows, is split into glyphs and underline, so the redraw
    keeps the underline instead of stretching the digits over it.
    """
    rows = {}
    for it in items:
        rows.setdefault((it.get("table"), tuple(it.get("row") or ())), []).append(it)
    for row in rows.values():
        plain = [it["box"][3] - it["box"][1] for it in row
                 if it.get("role") in ("serial", "rate", "site", "material", "") and not it.get("underline")
                 and it.get("serial") is not None or (it.get("role") == "rate" and not it.get("underline"))]
        if len(plain) < 1:
            continue
        height = float(np.median(plain))
        for it in row:
            if it.get("underline") or it.get("role") not in ("category", "material_rate", "carriage", "service"):
                continue
            x0, y0, x1, y1 = it["box"]
            if y1 - y0 < height + 0.9 or len(it.get("text", "")) > 4:
                continue
            cut = y0 + round(height)
            bar = rgb[int(cut):int(round(y1)) + 1, int(x0):int(round(x1)) + 1]
            if bar.size == 0:
                continue
            ink, _ = T._ink_mask(bar)
            if not ink.size or ink.mean(axis=1).max() < 0.6:
                continue
            it["box"] = [x0, y0, x1, float(cut)]
            it["underline"] = [x0 - 0.5, float(cut), x1 + 0.5, y1]


def _consistent_column_underlines(T, rgb, items) -> None:
    """Values of an underlined column keep their underline even where it blurred away.

    When most values of a column (in every table of the screenshot) are
    underlined, a value of that column whose own bar is too faint to measure
    gets one in the same position and thickness - provided some ink is there.
    """
    by_role = {}
    for it in items:
        if it.get("role") in ("category", "material_rate", "carriage", "service") and it.get("text"):
            by_role.setdefault(it["role"], []).append(it)
    for group in by_role.values():
        lined = [it for it in group if it.get("underline")]
        if len(lined) < 2 or len(lined) * 5 < len(group) * 3:
            continue
        gaps = [it["underline"][1] - it["box"][3] for it in lined]
        thick = [it["underline"][3] - it["underline"][1] for it in lined]
        gap, th = float(np.median(gaps)), max(1.0, float(np.median(thick)))
        for it in group:
            if it.get("underline"):
                continue
            x0, y0, x1, y1 = it["box"]
            top = y1 + gap
            band = rgb[int(top):int(round(top + th)) + 1, int(x0):int(round(x1)) + 1]
            if band.size == 0:
                continue
            ink, _ = T._ink_mask(band)
            if not ink.size or ink.mean() < 0.05:
                continue
            it["underline"] = [x0 - 0.5, top, x1 + 0.5, top + th]


def _consistent_column_ink(items) -> None:
    """A tiny glyph ("/", "2") measures washed-out ink; give it its column's colour.

    Category and charge columns are printed in red. A 3-pixel-wide slash
    measures as pale grey, which would be redrawn black.
    """
    def saturation(ink):
        return max(ink) - min(ink) if ink else 0

    columns = {}
    for it in items:
        if it.get("ink"):
            columns.setdefault(column_role(it.get("column", "")) or it.get("column"), []).append(it)
    for group in columns.values():
        coloured = [it["ink"] for it in group if saturation(it["ink"]) >= 50]
        if len(coloured) * 2 < len(group) or not coloured:
            continue
        median = [int(np.median([c[i] for c in coloured])) for i in range(3)]
        for it in group:
            if saturation(it["ink"]) < 50:          # pale or grey-looking in a red column
                it["ink"] = median


NUMBER_WEIGHT = 12.0


def _format_like(value: float, decimal: bool) -> str:
    if decimal:
        return f"{value:.2f}"
    return str(int(round(value))) if abs(value - round(value)) < 1e-6 else f"{value:g}"


def _solve_row(rgb, T, parts, result, combine, num_score, additive: bool) -> None:
    """Choose the values of a row that agree with the receipt's own arithmetic.

    First the recogniser's own candidates are combined.  For a sum row
    (Material Rate + Carriage + Service Charges = Rate/FT), when no
    combination adds up, the one or two least certain parts are searched over
    every value that makes the sum right and has the same number of digits,
    and each is scored against the pixels.  The best combination must clearly
    beat the next one, so an ambiguous row is left for the user to check.
    """
    members = parts + [result]
    lists = [it.get("num_candidates") or [] for it in members]
    if any(not lst for lst in lists):
        return

    def fits(values):
        return abs(combine(values[:-1]) - values[-1]) <= 0.005 * max(1.0, abs(values[-1]) / 100)

    options = []         # (total score, [texts])
    for combo in itertools.product(*[lst[:5] for lst in lists]):
        values = [as_number(c) for c, _ in combo]
        if any(v is None for v in values) or not fits(values):
            continue
        options.append((sum(s for _, s in combo), [c for c, _ in combo]))

    if additive:
        tops = [lst[0][0] for lst in lists]
        unsure = sorted(range(len(parts)), key=lambda i: (members[i].get("posterior") or {}).get(tops[i], 0.0))
        for r_text, _ in lists[-1][:3]:
            target = as_number(r_text)
            if target is None:
                continue
            for k in (1, 2):
                for chosen in itertools.combinations(unsure[:3], k):
                    fixed = [as_number(tops[i]) for i in range(len(parts)) if i not in chosen]
                    if any(v is None for v in fixed):
                        continue
                    rest = target - sum(fixed)
                    if rest < 0:
                        continue
                    lengths = []
                    for i in chosen:
                        seen = {len(c.split(".")[0]) for c, _ in lists[i][:3]}
                        lengths.append({n + d for n in seen for d in (-1, 0, 1) if 1 <= n + d <= 4})
                    if k == 1:
                        combos = [(rest,)]
                    else:
                        combos = [(v, rest - v) for v in range(int(rest) + 1)]
                    for values in combos:
                        texts = list(tops[:len(parts)])
                        ok = True
                        for i, v, lens in zip(chosen, values, lengths):
                            if v < 0 or abs(v - round(v)) > 1e-6:
                                ok = False
                                break
                            text = _format_like(v, members[i].get("decimal", False))
                            if len(text.split(".")[0]) not in lens:
                                ok = False
                                break
                            texts[i] = text
                        if not ok:
                            continue
                        texts.append(r_text)
                        score = sum(num_score(m, t) for m, t in zip(members, texts))
                        options.append((score, texts))

    def distrust():
        # The current readings break the receipt's own arithmetic: something
        # is misread, so only very sure readings stay confirmed.
        current = [as_number(it["text"]) for it in members]
        if all(v is not None for v in current) and fits(current):
            return
        for it in members:
            if it.get("confirmed") and (it["confidence"] < 0.98 or it.get("match", 0) < 0.75):
                it["confirmed"] = False

    if not options:
        distrust()
        return

    best = {}
    for score, texts in options:
        key = tuple(texts)
        best[key] = max(best.get(key, NEG_INF), score)
    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    (texts, score) = ranked[0]
    if os.environ.get("VALUE_CONTEXT_DEBUG"):
        print("row options:", [(list(t), round(v, 2)) for t, v in ranked[:3]])
    if len(ranked) > 1 and score - ranked[1][1] < 1.0:
        distrust()
        return                                  # two answers fit about equally well: ask the user
    for it, text, lst in zip(members, texts, lists):
        if num_score(it, text) < lst[0][1] - 12 or \
                T.verify_reading(rgb, {"box": it["box"], "text": text}) < 0.3:
            distrust()
            return                              # far less likely than the top reading: do not force it
    for it, text in zip(members, texts):
        changed = it["text"] != text
        it["text"] = it["ocr_text"] = it["top_vote"] = text
        it["confirmed"] = True
        if changed or it.get("evidence") != "recogniser":
            it["evidence"] = "arithmetic"
        it["match"] = round(float(T.verify_reading(rgb, {"box": it["box"], "text": text})), 3)


def canonical_headers(T, rgb, columns, band, texts) -> list[str]:
    """Column titles mapped onto the known titles; unread ones are read again.

    Coloured titles ("Category" in red) are sometimes missed by the detector,
    which would leave that column without a role.
    """
    vocab = DEFAULT_VOCABULARY["header"]
    out = []
    rec = T.strong_recogniser()
    _, y0, _, y1 = band
    for (xa, xb), text in zip(columns, texts):
        if re.search(r"iban|account|bank|title", text or "", re.I):
            out.append(text)
            continue
        best = max(vocab, key=lambda v: similar(v, text)) if text else None
        if text and similar(best, text) >= 0.7:
            out.append(best)
            continue
        if text or rec is None:
            out.append(text)
            continue
        cell = rgb[y0 + 2:y1 - 2, xa + 3:xb - 3]
        ink, _ = T._ink_mask(cell) if cell.size else (np.zeros((0, 0), bool), 0)
        if not ink.any():
            out.append(text)
            continue
        ys, xs = np.where(ink)
        box = [xa + 3 + xs.min(), y0 + 2 + ys.min(), xa + 3 + xs.max() + 1, y0 + 2 + ys.max() + 1]
        sc = Scorer(rgb, box, rec, T._contrast, T._ENGINE_LOCK)
        scores = {v: sc.score(v) for v in vocab}
        post = posterior(scores)
        pick = max(scores, key=scores.get)
        out.append(pick if post[pick] >= 0.6 else text)
    return out
