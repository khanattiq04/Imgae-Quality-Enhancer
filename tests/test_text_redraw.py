import base64
import io
import unittest
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import text_redraw as tr
from api import index as api


def _font():
    try:
        return ImageFont.load_default(size=11)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def receipt_canvas():
    """White card with a ruled cell, a black value and a red underlined value."""
    img = Image.new("RGB", (120, 40), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle((2, 2, 117, 37), outline=(0, 0, 0))          # table rules
    d.line((60, 2, 60, 37), fill=(0, 0, 0))
    d.text((14, 12), "56", font=_font(), fill=(40, 40, 40))  # black value
    d.text((74, 12), "109", font=_font(), fill=(230, 20, 20))  # red value
    d.line((72, 25, 94, 25), fill=(230, 20, 20))               # its underline
    return img


class MeasureTests(unittest.TestCase):
    def test_box_is_tightened_and_underline_found(self):
        rgb = np.asarray(receipt_canvas())
        m = tr._measure(rgb, (68, 10, 97, 28))
        self.assertIsNotNone(m["underline"])
        x0, y0, x1, y1 = m["box"]
        self.assertGreaterEqual(x0, 73)
        self.assertLessEqual(y1, 25)                     # underline split off the text box
        self.assertGreater(m["ink"][0], m["ink"][1] + 80)  # red ink measured

    def test_table_rule_is_not_an_underline(self):
        img = Image.new("RGB", (80, 30), (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((22, 5), "Date", font=_font(), fill=(0, 0, 0))
        d.line((0, 18, 79, 18), fill=(0, 0, 0))          # box edge running past the text
        m = tr._measure(np.asarray(img), (20, 4, 46, 19))
        self.assertIsNone(m["underline"])


class SpacingTests(unittest.TestCase):
    def test_spaces_restored_at_word_gaps(self):
        self.assertEqual(tr.restore_spaces("RateFor", [0.55]), "Rate For")

    def test_digits_are_never_split(self):
        self.assertEqual(tr.restore_spaces("181.00", [0.2, 0.5]), "181.00")

    def test_existing_spaces_kept(self):
        self.assertEqual(tr.restore_spaces("Sr No:", [0.4]), "Sr No:")


class InkTests(unittest.TestCase):
    def test_washed_grey_becomes_black_and_pink_becomes_red(self):
        black = tr.solid_ink([120, 120, 120], [255, 255, 255])
        self.assertTrue(np.all(black < 40))
        red = tr.solid_ink([200, 110, 115], [255, 255, 255])
        self.assertGreater(red[0], 180)
        self.assertLess(red[1], 60)

    def test_dark_grey_value_ink_becomes_solid_black(self):
        np.testing.assert_array_equal(tr.solid_ink([40, 41, 38], [255, 255, 255]), [0, 0, 0])

    def test_table_values_are_slightly_heavier_than_regular(self):
        plain, _ = tr._glyph_mask("Chandarkey", "serif-regular", 48)
        medium, _ = tr._glyph_mask("Chandarkey", "serif-regular", 48, tr.VALUE_WEIGHT)
        bold, _ = tr._glyph_mask("Chandarkey", "serif", 48)
        self.assertGreater(medium.sum(), plain.sum())
        self.assertLess(medium.sum(), bold.sum())

    def test_dark_ink_kept(self):
        navy = tr.solid_ink([0, 15, 45], [90, 160, 220])
        np.testing.assert_array_equal(navy, [0, 15, 45])


class RedrawTests(unittest.TestCase):
    def setUp(self):
        self.scale = 3
        src = receipt_canvas()
        self.enhanced = src.resize((src.width * 3, src.height * 3), Image.Resampling.LANCZOS)
        rgb = np.asarray(src)
        first = tr._measure(rgb, (10, 8, 45, 26))
        second = tr._measure(rgb, (68, 8, 97, 28))
        self.items = [dict(first, text="56", font="serif", confirmed=True),
                      dict(second, text="109", font="serif", confirmed=False)]
        self.items[0].pop("word_gaps", None)
        self.items[1].pop("word_gaps", None)

    def test_only_confirmed_values_change_and_only_inside_their_box(self):
        out = np.asarray(tr.redraw_values(self.enhanced, self.scale, self.items)).astype(int)
        before = np.asarray(self.enhanced).astype(int)
        diff = np.abs(out - before).sum(axis=2) > 0
        ys, xs = np.where(diff)
        margin = 1.5 * self.scale + tr.HALO_RING * self.scale + 1   # erase margin + halo-cleaning ring
        bx0, by0, bx1, by1 = self.items[0]["box"]
        self.assertGreaterEqual(xs.min(), bx0 * 3 - margin)
        self.assertLessEqual(xs.max(), bx1 * 3 + margin)
        self.assertLessEqual(ys.max(), by1 * 3 + margin)
        self.assertFalse(diff[:, 60 * 3 + 3:].any(), "unconfirmed value was changed")

    def test_table_rules_survive(self):
        out = np.asarray(tr.redraw_values(self.enhanced, self.scale, self.items).convert("L"))
        self.assertLess(int(out[2 * 3:37 * 3, 2 * 3].mean()), 120)   # left rule still dark

    def test_text_is_drawn_dark_on_paper(self):
        out = np.asarray(tr.redraw_values(self.enhanced, self.scale, self.items).convert("L"))
        bx0, by0, bx1, by1 = [int(v * 3) for v in self.items[0]["box"]]
        cell = out[by0:by1, bx0:bx1]
        self.assertLess(int(cell.min()), 60)
        self.assertGreater(float((cell > 230).mean()), 0.4)          # mostly paper again

    def test_empty_or_unconfirmed_text_is_ignored(self):
        items = [dict(self.items[0], text="  "), dict(self.items[1])]
        out = tr.redraw_values(self.enhanced, self.scale, items)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(self.enhanced))


class ColouredBarTests(unittest.TestCase):
    def test_text_on_a_blue_bar_is_replaced_and_the_bar_stays_blue(self):
        img = Image.new("RGB", (100, 30), (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.rectangle((0, 8, 99, 22), fill=(90, 150, 210), outline=(10, 20, 40))   # blue strip with border
        d.text((30, 9), "Asad", font=_font(), fill=(5, 15, 40))
        m = tr._measure(np.asarray(img), (26, 7, 64, 23))
        m.pop("word_gaps", None)
        enhanced = img.resize((300, 90), Image.Resampling.LANCZOS)
        item = dict(m, text="Asad", font="sans", confirmed=True)
        out = np.asarray(tr.redraw_values(enhanced, 3, [item])).astype(int)
        bx0, by0, bx1, by1 = [int(v * 3) for v in item["box"]]
        cell = out[by0:by1, bx0:bx1]
        blue = (cell[..., 2] > cell[..., 0] + 60).mean()
        self.assertGreater(blue, 0.4)                   # background is still the blue bar
        self.assertLess(int(cell.sum(axis=2).min()), 200)   # new dark text drawn
        self.assertLess(int(out[8 * 3 - 1:8 * 3 + 3, 150].sum(axis=1).min()), 200)   # bar border kept


class LayoutTests(unittest.TestCase):
    def item(self, text, box, group="table", font="serif"):
        return {"text": text, "box": box, "group": group, "font": font, "confirmed": True}

    def test_table_values_share_one_size_despite_noisy_boxes(self):
        items = [self.item("167.00", [100, 50, 124, 57]), self.item("181.00", [100, 80, 124, 86]),
                 self.item("109", [60, 50, 72, 57]), self.item("Chandarkey", [10, 50, 56, 58])]
        sizes = {round(e["size"], 3) for e in tr._layout(items, 3)}
        self.assertEqual(len(sizes), 1, sizes)

    def test_text_is_not_stretched_and_fits_its_box(self):
        items = [self.item("Site measurement of truck", [10, 10, 130, 18], group=None)]
        e = tr._layout(items, 3)[0]
        left, _, right, _ = tr._ink_extent(e["text"], e["family"], e["size"])
        self.assertLessEqual(right - left, 120 * 3 * 1.08 + 3 + 1)

    def test_column_titles_share_one_size_no_larger_than_values(self):
        items = [self.item("167.00", [100, 50, 124, 57]), self.item("109", [60, 50, 72, 57]),
                 self.item("Sr No:", [0, 30, 20, 40], group="header", font="sans"),
                 self.item("Carriage", [30, 28, 60, 43], group="header", font="sans"),
                 self.item("Rate", [70, 32, 85, 39], group="header", font="sans")]
        laid = tr._layout(items, 3)
        titles = {round(e["size"], 3) for e in laid if e["item"]["group"] == "header"}
        value = max(e["size"] for e in laid if e["item"]["group"] == "table")
        self.assertEqual(len(titles), 1, titles)
        self.assertLessEqual(titles.pop(), value + 1e-6)

    def test_left_aligned_text_gets_padding_from_its_rule(self):
        img = Image.new("RGB", (200, 30), (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.line((5, 0, 5, 29), fill=(0, 0, 0))                       # box's left rule
        d.text((8, 9), "Site measurement", font=_font(), fill=(220, 20, 20))
        rgb = np.asarray(img)
        m = tr._measure(rgb, (7, 7, 110, 22))
        self.assertTrue(tr._rule_on_left(rgb, m["box"]))
        item = dict(m, text="Site measurement", font="serif", confirmed=True, align="left", group=None)
        item.pop("word_gaps", None)
        e = tr._layout([item], 3)[0]
        self.assertGreaterEqual(e["left"], m["box"][0] * 3 + 2 * 3)

    def test_unconfirmed_values_are_not_laid_out(self):
        items = [dict(self.item("56", [0, 0, 8, 7]), confirmed=False)]
        self.assertEqual(tr._layout(items, 3), [])


class TintedCellTests(unittest.TestCase):
    def test_pink_haze_behind_red_title_on_blue_cell_is_removed(self):
        img = Image.new("RGB", (90, 40), (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.rectangle((2, 2, 87, 37), fill=(190, 214, 235), outline=(20, 20, 20))   # light blue title cell
        d.text((20, 14), "Carriage", font=_font(), fill=(200, 30, 50))           # red title
        enhanced = img.resize((270, 120), Image.Resampling.LANCZOS)
        from PIL import ImageFilter
        enhanced = enhanced.filter(ImageFilter.GaussianBlur(4))                  # pink haze around it
        m = tr._measure(np.asarray(img), (18, 11, 70, 28))
        m.pop("word_gaps", None)
        item = dict(m, text="Carriage", font="sans", confirmed=True, group="header")
        out = np.asarray(tr.redraw_values(enhanced, 3, [item])).astype(int)
        bx0, by0, bx1, by1 = [int(v * 3) for v in item["box"]]
        ring = out[by0 - 9:by0 - 3, bx0:bx1].reshape(-1, 3)                     # just above the text
        blue = np.array([190, 214, 235])
        self.assertLess(float(np.abs(ring - blue).max()), 12, "pink haze left behind the title")


class CompareTests(unittest.TestCase):
    def test_changed_number_is_flagged_first(self):
        original = [{"text": "167.00", "box": [0, 0, 1, 1]}, {"text": "Chandarkey", "box": [0, 0, 1, 1]},
                    {"text": "Crush", "box": [0, 0, 1, 1]}]
        other = [{"text": "176.00", "box": [0, 0, 1, 1]}, {"text": "Chandarkey", "box": [0, 0, 1, 1]},
                 {"text": "Crash", "box": [0, 0, 1, 1]}]
        flags = tr.compare_readings(original, other)
        self.assertEqual([f["original"] for f in flags], ["167.00", "Crush"])
        self.assertTrue(flags[0]["has_digits"])
        self.assertEqual(flags[0]["other"], "176.00")


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = api.app.test_client()
        buf = io.BytesIO()
        receipt_canvas().save(buf, format="PNG")
        self.png = buf.getvalue()

    def upload(self, url, **extra):
        data = {"file": (io.BytesIO(self.png), "r.png"), **extra}
        return self.client.post(url, data=data, content_type="multipart/form-data")

    def test_detect_returns_items(self):
        fake = [{"id": 0, "text": "56", "box": [12, 14, 41, 23]}]
        with mock.patch.object(api.text_redraw, "ocr_available", return_value=True), \
                mock.patch.object(api.text_redraw, "detect_values", return_value=fake):
            res = self.upload("/api/ocr/detect")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["items"], fake)

    def test_detect_reports_missing_ocr(self):
        with mock.patch.object(api.text_redraw, "ocr_available", return_value=False):
            res = self.upload("/api/ocr/detect")
        self.assertEqual(res.status_code, 400)

    def test_redraw_endpoint(self):
        buf = io.BytesIO()
        receipt_canvas().resize((360, 120)).save(buf, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        items = [{"text": "56", "box": [12, 14, 41, 23], "ink": [40, 40, 40], "paper": [255, 255, 255],
                  "underline": None, "confirmed": True}]
        res = self.client.post("/api/ocr/redraw", json={"data_url": data_url, "scale": 3, "items": items})
        self.assertEqual(res.status_code, 200, res.get_json())
        self.assertEqual(res.get_json()["redrawn"], 1)

    def test_ai_redraw_needs_key_and_returns_check(self):
        res = self.upload("/api/ai/redraw")
        self.assertEqual(res.status_code, 400)
        self.assertIn("API key", res.get_json()["error"])
        with mock.patch.object(api.ai_redraw, "ai_redraw", return_value=self.png), \
                mock.patch.object(api.text_redraw, "ocr_available", return_value=False):
            res = self.upload("/api/ai/redraw", api_key="test")
        body = res.get_json()
        self.assertEqual(res.status_code, 200, body)
        self.assertFalse(body["checked"])
        self.assertTrue(body["data_url"].startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()


class AutoTableTests(unittest.TestCase):
    def test_column_kinds_from_titles(self):
        self.assertEqual(tr.column_kind("Sr No:"), "number")
        self.assertEqual(tr.column_kind("Material Rate"), "number")
        self.assertEqual(tr.column_kind("Carriage"), "number")
        self.assertEqual(tr.column_kind("Service Charges"), "number")
        self.assertEqual(tr.column_kind("Rate/FT"), "number")
        self.assertEqual(tr.column_kind("Category"), "letter")
        self.assertEqual(tr.column_kind("Site Name"), "text")
        self.assertEqual(tr.column_kind("Material"), "text")

    def test_auto_endpoint_redraws_only_table_values_and_returns_pdf(self):
        client = api.app.test_client()
        buf = io.BytesIO()
        receipt_canvas().save(buf, format="PNG")
        values = [{"text": "56", "box": [14, 14, 26, 22], "ink": [40, 40, 40], "paper": [255, 255, 255],
                   "underline": None, "font": "serif", "group": "table", "align": "centre",
                   "confirmed": True, "column": "Material Rate", "confidence": 0.9}]
        with mock.patch.object(api.text_redraw, "ocr_available", return_value=True), \
                mock.patch.object(api, "_receipt_table_grids", return_value=[{"columns": [], "rows": []}]), \
                mock.patch.object(api.text_redraw, "read_table_values", return_value=values):
            res = client.post("/api/ocr/auto", data={"file": (io.BytesIO(buf.getvalue()), "r.png"), "scale": "2"},
                              content_type="multipart/form-data")
        body = res.get_json()
        self.assertEqual(res.status_code, 200, body)
        self.assertTrue(body["data_url"].startswith("data:image/png;base64,"))
        self.assertEqual(len(body["items"]), 1)
        self.assertEqual([{k: r[k] for k in ("column", "text")} for r in body["redrawn"]],
                         [{"column": "Material Rate", "text": "56"}])
        self.assertEqual(body["skipped"], [])

    def test_auto_endpoint_without_table_sharpens_whole_image(self):
        client = api.app.test_client()
        buf = io.BytesIO()
        receipt_canvas().save(buf, format="PNG")
        with mock.patch.object(api.text_redraw, "ocr_available", return_value=True), \
                mock.patch.object(api, "_receipt_table_grids", return_value=[]):
            res = client.post("/api/ocr/auto", data={"file": (io.BytesIO(buf.getvalue()), "r.png"), "scale": "2"},
                              content_type="multipart/form-data")
        body = res.get_json()
        self.assertEqual(res.status_code, 200, body)
        self.assertIn("No printed receipt table", body["note"])
        self.assertEqual(body["items"], [])
        self.assertEqual(body["width"], receipt_canvas().width * 2)


class RobustnessTests(unittest.TestCase):
    def ruled_table(self, scale=1.0):
        w, h = int(600 * scale), int(300 * scale)
        img = Image.new("RGB", (w, h), (255, 255, 255))
        d = ImageDraw.Draw(img)
        xs = [int(v * scale) for v in (40, 110, 260, 380, 470, 560)]
        ys = [int(v * scale) for v in (60, 95, 160, 220)]
        for x in xs:
            d.line((x, ys[0], x, ys[-1]), fill=(30, 30, 30), width=max(1, int(2 * scale)))
        for y in ys:
            d.line((xs[0], y, xs[-1], y), fill=(30, 30, 30), width=max(1, int(2 * scale)))
        return img, xs, ys

    def test_table_found_at_any_size_without_blue_bar(self):
        for scale in (0.8, 1.0, 1.5):
            with self.subTest(scale=scale):
                img, xs, ys = self.ruled_table(scale)
                grids = tr.find_table_grids(img)
                self.assertEqual(len(grids), 1)
                self.assertEqual(len(grids[0]["columns"]), len(xs) - 1)
                self.assertEqual(len(grids[0]["rows"]), len(ys) - 1)

    def test_redrawn_underline_stays_thin(self):
        enhanced = Image.new("RGB", (300, 120), (255, 255, 255))
        item = {"text": "56", "box": [40, 10, 60, 24], "underline": [40, 26, 60, 30],   # blurred: 4 px thick
                "ink": [200, 30, 30], "paper": [255, 255, 255], "font": "serif-regular",
                "group": "table", "align": "centre", "confirmed": True}
        out = np.asarray(tr.redraw_values(enhanced, 3, [item]).convert("L"))
        rows = np.where((out[26 * 3 - 6:30 * 3 + 6, 40 * 3:60 * 3] < 200).mean(axis=1) > 0.6)[0]
        self.assertLessEqual(len(rows), int(round(tr.UNDERLINE_RATIO * 14 * 3)) + 1)

    def test_numbers_and_serials(self):
        self.assertEqual(tr._normalise_number("157,00"), "157.00")
        self.assertEqual(tr._normalise_number("1,250"), "1,250")
        self.assertTrue(tr._is_serial_column("Sr No:"))
        self.assertFalse(tr._is_serial_column("Site Name"))


class LineBandTests(unittest.TestCase):
    def test_two_lines_in_a_cell_with_underlines(self):
        img = Image.new("RGB", (60, 40), (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((8, 4), "Normal", font=_font(), fill=(200, 20, 20))
        d.line((8, 16, 44, 16), fill=(200, 20, 20))
        d.text((10, 20), "Black", font=_font(), fill=(200, 20, 20))
        d.line((10, 32, 40, 32), fill=(200, 20, 20))
        bands = tr._ink_line_bands(np.asarray(img))
        self.assertEqual(len(bands), 2)
        self.assertLess(bands[0][1], bands[1][1])

    def test_glyph_bar_is_not_a_second_line(self):
        img = Image.new("RGB", (30, 20), (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((10, 4), "2", font=_font(), fill=(0, 0, 0))
        d.line((9, 16, 18, 16), fill=(0, 0, 0))                 # underline under a lone "2"
        self.assertEqual(len(tr._ink_line_bands(np.asarray(img))), 1)


class StrongReaderTests(unittest.TestCase):
    def test_model_ships_with_the_app_and_reads_small_digits(self):
        self.assertIsNotNone(tr.strong_recogniser(), "models/PP-OCRv6_rec_small.onnx missing")
        img = Image.new("RGB", (60, 24), (255, 255, 255))
        ImageDraw.Draw(img).text((8, 4), "145", font=tr._font("serif-regular", 14), fill=(200, 20, 20))
        big = np.asarray(img.resize((240, 96), Image.Resampling.BICUBIC))
        text, conf = tr._constrained_decode(big, tr.CHARSETS["number"])
        self.assertEqual(text, "145")


class OnlineFallbackTests(unittest.TestCase):
    def test_without_ocr_the_button_still_returns_a_sharpened_image(self):
        client = api.app.test_client()
        buf = io.BytesIO()
        receipt_canvas().save(buf, format="PNG")
        with mock.patch.object(api.text_redraw, "ocr_available", return_value=False):
            res = client.post("/api/ocr/auto", data={"file": (io.BytesIO(buf.getvalue()), "r.png"), "scale": "2"},
                              content_type="multipart/form-data")
        body = res.get_json()
        self.assertEqual(res.status_code, 200, body)
        self.assertIn("desktop app", body["note"])
        self.assertEqual(body["items"], [])


class BestGuessTests(unittest.TestCase):
    """Pre-filled readings for values the tool was not sure of."""

    def item(self, column, text, confirmed=False, row=(0, 10), **extra):
        return dict(column=column, text=text, confirmed=confirmed, row=row, table=0, **extra)

    def test_text_snaps_to_known_name(self):
        it = self.item("Material", "Cruh 1/2-38")
        self.assertEqual(api._best_guess(it, [it], {"material": ["Crush 1/2+3/8"]}), "Crush 1/2+3/8")

    def test_row_sum_solves_rate(self):
        parts = [self.item("Material Rate", "56", True), self.item("Carriage", "109", True),
                 self.item("Service Charges", "2", True), self.item("Rate/FT", "167.80")]
        self.assertEqual(api._best_guess(parts[3], parts, {}), "167.00")

    def test_blurry_title_and_lookalike_digits(self):
        it = self.item("Ralesn", "16L80")
        self.assertEqual(api._best_guess(it, [it], {}), "161.80")

    def test_words_are_not_turned_into_numbers(self):
        it = self.item("Sr No:", "Notet")
        self.assertEqual(api._best_guess(it, [it], {}), "Notet")
