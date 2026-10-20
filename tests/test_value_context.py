import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import value_context as V  # noqa: E402


class CtcTests(unittest.TestCase):
    def setUp(self):
        # 4 classes: blank, "1", "2", "7"; frames spell "7" "7" blank "6"->"2"
        self.probs = np.array([
            [0.05, 0.05, 0.05, 0.85],
            [0.10, 0.05, 0.05, 0.80],
            [0.90, 0.03, 0.03, 0.04],
            [0.05, 0.10, 0.80, 0.05],
        ])

    def test_likely_reading_scores_higher(self):
        self.assertGreater(V.ctc_logp(self.probs, [3, 2]), V.ctc_logp(self.probs, [1, 2]))

    def test_beam_finds_the_best_path(self):
        best = V.ctc_beam(self.probs, np.array([0, 1, 2, 3]), width=4)[0][0]
        self.assertEqual(best, (3, 2))


class ContextTests(unittest.TestCase):
    def test_roles_from_titles(self):
        self.assertEqual(V.column_role("Material Rate"), "material_rate")
        self.assertEqual(V.column_role("Service Charges"), "service")
        self.assertEqual(V.column_role("Rate/FT"), "rate")
        self.assertEqual(V.column_role("Carriage"), "carriage")
        self.assertEqual(V.column_role("Category"), "category")
        self.assertEqual(V.column_role("Site Name"), "site")

    def test_site_name_taken_from_title(self):
        self.assertEqual(V.document_phrases(["RateForQilaKalarWalaSite Materlal"]), ["Qila Kalar Wala"])
        self.assertEqual(V.document_phrases(["Rate For AliPur Chowk_Sito Material"]), ["Ali Pur Chowk"])

    def test_number_format_follows_the_column(self):
        self.assertIn("164.00", V.number_variants("16400", True))
        self.assertIn("155.00", V.number_variants("155,00", True))

    def test_header_row_is_recognised(self):
        self.assertGreaterEqual(V.header_score(["Sr No:", "Site Name", "Material", "Rate/FT"]), 3)
        self.assertEqual(V.header_score(["1", "Shanawan Bridge", "148.00"]), 0)


class FakeT:
    @staticmethod
    def verify_reading(rgb, it):
        return 0.8


class ArithmeticTests(unittest.TestCase):
    def item(self, cands):
        return {"num_candidates": cands, "confirmed": False, "confidence": 0.5, "text": cands[0][0],
                "box": [0, 0, 1, 1], "posterior": {c: 0.5 for c, _ in cands}}

    def test_row_picks_values_that_add_up(self):
        mr = self.item([("11", 5.0), ("77", 4.0)])
        car = self.item([("26", 6.0), ("76", 5.5)])
        sc = self.item([("2", 8.0)])
        rate = self.item([("155.00", 8.0)])
        scores = {(id(mr), "11"): 5.0, (id(mr), "77"): 4.0, (id(car), "26"): 6.0, (id(car), "76"): 5.5,
                  (id(sc), "2"): 8.0, (id(rate), "155.00"): 8.0}
        V._solve_row(None, FakeT, [mr, car, sc], rate, sum,
                     lambda it, t: scores.get((id(it), t), -50.0), additive=True)
        self.assertEqual([mr["text"], car["text"], sc["text"]], ["77", "76", "2"])
        self.assertTrue(all(i["confirmed"] for i in (mr, car, sc, rate)))
        self.assertEqual(mr["evidence"], "arithmetic")

    def test_row_that_cannot_add_up_is_left_for_the_user(self):
        mr = self.item([("52", 5.0)])
        mr["confirmed"] = True
        car = self.item([("1", 6.0)])
        car["confirmed"] = True
        sc = self.item([("2", 8.0)])
        rate = self.item([("172.00", 8.0)])
        V._solve_row(None, FakeT, [mr, car, sc], rate, sum, lambda it, t: -50.0, additive=False)
        self.assertFalse(mr["confirmed"])
        self.assertFalse(car["confirmed"])


if __name__ == "__main__":
    unittest.main()


class UnderlineCropTests(unittest.TestCase):
    def test_underline_kept_out_of_the_reading(self):
        """A tiny "0" on its underline reads as "2" if the bar is in the crop."""
        import text_redraw as tr
        from PIL import Image, ImageDraw
        rec = tr.strong_recogniser()
        if rec is None:
            self.skipTest("recogniser model not available")
        big = Image.new("RGB", (80, 64), (255, 255, 255))
        d = ImageDraw.Draw(big)
        d.text((28, 6), "0", font=tr._font("serif", 40), fill=(200, 20, 20))
        ink = np.asarray(big.convert("L")) < 200
        ys, xs = np.where(ink)
        bar = int(ys.max()) + 3
        d.rectangle((xs.min() - 2, bar, xs.max() + 2, bar + 4), fill=(200, 20, 20))
        small = big.resize((20, 16), Image.Resampling.LANCZOS)       # 0 about 7 px tall, like the receipts
        rgb = np.asarray(small)
        box = [xs.min() / 4, ys.min() / 4, (xs.max() + 1) / 4, (ys.max() + 1) / 4]
        bottom = bar / 4
        with_bar = V.Scorer(rgb, box, rec, tr._contrast, tr._ENGINE_LOCK)
        without = V.Scorer(rgb, box, rec, tr._contrast, tr._ENGINE_LOCK, bottom=bottom)
        self.assertGreater(without.score("0") - without.score("2"), with_bar.score("0") - with_bar.score("2"))
        self.assertGreater(without.score("0"), without.score("2"))
