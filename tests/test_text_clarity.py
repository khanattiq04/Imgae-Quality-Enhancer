import ast
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from PIL import Image

from api import index as api
from text_clarity import (
    ClarityConfig,
    analyze_clarity,
    edge_profile,
    enhance_luma,
    estimate_blur_sigma,
    evaluate_clarity,
    make_config,
    rise_width,
    wiener_deblur,
)

FONT = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    ".": ["00", "00", "00", "00", "00", "11", "11"],
    ",": ["00", "00", "00", "00", "00", "11", "10"],
    " ": ["000"] * 7,
}
PAPER, INK = 238.0, 28.0


def render_text(text, gap=1, margin=6):
    """Crisp 1-pixel-per-font-pixel ink map (1 = ink): small printed receipt text."""
    glyphs = [np.array([[int(c) for c in row] for row in FONT[ch]], np.uint8) for ch in text]
    width = sum(g.shape[1] for g in glyphs) + gap * (len(glyphs) - 1)
    ink = np.zeros((7 + 2 * margin, width + 2 * margin), np.uint8)
    x = margin
    for g in glyphs:
        ink[margin:margin + 7, x:x + g.shape[1]] = g
        x += g.shape[1] + gap
    return ink


def ink_to_luma(ink, paper=PAPER, ink_level=INK):
    return np.where(ink > 0, ink_level, paper).astype(np.float32)


def blurred_source(text="0.86 5,498 3.60", sigma=1.2, noise=1.0, seed=3):
    ink = render_text(text)
    luma = cv2.GaussianBlur(ink_to_luma(ink), (0, 0), sigma)
    luma += np.random.default_rng(seed).normal(0, noise, luma.shape).astype(np.float32)
    return ink, np.clip(luma, 0, 255).astype(np.float32)


def upscale(luma, scale):
    if scale == 1:
        return luma
    return cv2.resize(luma, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)


def truth_at(ink, scale):
    return cv2.resize(ink_to_luma(ink), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


class ClarityConfigTests(unittest.TestCase):
    def test_profiles_are_separate_and_bw_binarizes(self):
        self.assertFalse(make_config("natural").binarize)
        self.assertFalse(make_config("crisp").binarize)
        self.assertTrue(make_config("bw").binarize)
        self.assertGreater(make_config("crisp").edge_refine, make_config("natural").edge_refine)

    def test_none_overrides_keep_profile_defaults(self):
        self.assertEqual(make_config("natural", deblur=None).deblur, make_config("natural").deblur)

    def test_safe_mode_caps_and_unsafe_mode_allows_more(self):
        safe = make_config("natural", deblur=1.0, edge_refine=1.0, threshold=1.0).resolved()
        self.assertLessEqual(safe.deblur, 0.5)
        self.assertLessEqual(safe.edge_refine, 0.6)
        self.assertLessEqual(safe.threshold, 0.5)
        unsafe = make_config("natural", deblur=1.0, safe_mode=False).resolved()
        self.assertEqual(unsafe.deblur, 1.0)

    def test_invalid_values_rejected(self):
        with self.assertRaises(ValueError):
            make_config("natural", deblur=1.5)
        with self.assertRaises(ValueError):
            make_config("nope")


class DeblurEstimationTests(unittest.TestCase):
    def test_sharp_text_is_not_deblurred_and_blur_is_ordered(self):
        sharp = analyze_clarity(ink_to_luma(render_text("0.86 5,498 3.60")))
        self.assertFalse(sharp["deblur_eligible"])
        estimates = []
        for sigma in (0.6, 1.0, 1.5):
            _, luma = blurred_source(sigma=sigma)
            estimates.append(estimate_blur_sigma(luma))
        self.assertTrue(all(e is not None for e in estimates))
        self.assertEqual(estimates, sorted(estimates))

    def test_irrecoverable_blur_warns_without_claiming_recovery(self):
        _, luma = blurred_source(sigma=3.0)
        report = analyze_clarity(luma)
        if report["blur_sigma"] is not None and not report["recoverable"]:
            self.assertIn("cannot restore", report["warning"])
            self.assertIn("OCR", report["warning"])

    def test_wiener_preserves_mean_and_is_finite(self):
        _, luma = blurred_source(sigma=1.2)
        out = wiener_deblur(luma, 1.0, 0.05)
        self.assertTrue(np.isfinite(out).all())
        self.assertAlmostEqual(float(out.mean()), float(luma.mean()), delta=2.0)


class ScaleQualityTests(unittest.TestCase):
    """Reject results that thicken, merge, close counters, or ring."""

    def run_scale(self, scale, profile="natural", sigma=0.6, **overrides):
        ink, source = blurred_source(sigma=sigma)
        base = upscale(source, scale)
        blur = estimate_blur_sigma(source)
        out = enhance_luma(base, scale, blur, make_config(profile, **overrides))
        return ink, base, out

    def assert_accepted(self, scale, profile="natural", **kw):
        ink, base, out = self.run_scale(scale, profile, **kw)
        truth = truth_at(ink, scale)
        metrics = evaluate_clarity(base, out, scale, truth)
        self.assertTrue(metrics["accepted"], f"{scale}x {profile}: {metrics}")
        self.assertLessEqual(metrics["overshoot"], 2.0)
        # Closer to the true glyphs than the plain upscale, never farther.
        self.assertLessEqual(np.abs(out - truth).mean(), np.abs(base - truth).mean() + 0.5)
        return base, out

    def test_natural_and_crisp_at_1x_2x_4x(self):
        for scale in (1, 2, 4):
            for profile in ("natural", "crisp"):
                with self.subTest(scale=scale, profile=profile):
                    self.assert_accepted(scale, profile)

    def test_edges_are_steeper_not_wider(self):
        for scale in (2, 4):
            with self.subTest(scale=scale):
                ink, base, out = self.run_scale(scale, "crisp")
                # Vertical stroke of the leading "0" at mid-height.
                row = (6 + 3) * scale
                x0, x1 = 4 * scale, 12 * scale
                before = rise_width(edge_profile(base, row, x0, x1))
                after = rise_width(edge_profile(out, row, x0, x1))
                self.assertLessEqual(after, before + 1e-3)

    def test_unsafe_settings_would_be_rejected_by_the_evaluator(self):
        # A deliberately fat result (ink boosted like the old pipeline) must fail.
        _, base, _ = self.run_scale(2)
        fat = np.where(base < 150, base * 0.2, base)
        self.assertFalse(evaluate_clarity(base, fat, 2)["accepted"])

    def test_binarized_print_keeps_counters_and_gaps(self):
        for scale in (1, 2, 4):
            with self.subTest(scale=scale):
                ink, base, out = self.run_scale(scale, "bw", sigma=0.8)
                self.assertTrue(set(np.unique(out)) <= {0.0, 255.0})
                metrics = evaluate_clarity(base, out, scale)
                self.assertGreaterEqual(metrics["counters_after"], metrics["counters_before"])
                self.assertGreaterEqual(metrics["components_after"], metrics["components_before"])
                self.assertLessEqual(metrics["thickness_ratio"], 1.25)

    def test_disabled_clarity_returns_input(self):
        _, base, _ = self.run_scale(2)
        out = enhance_luma(base, 2, 1.0, make_config("natural", enabled=False))
        np.testing.assert_allclose(out, base)

    def test_flat_paper_and_gradient_are_untouched(self):
        gradient = np.tile(np.linspace(150, 170, 120, dtype=np.float32), (60, 1))
        out = enhance_luma(gradient, 2, None, make_config("crisp"))
        self.assertLessEqual(float(np.abs(out - gradient).max()), 1.5)


class ReceiptPipelineTests(unittest.TestCase):
    def colored_receipt(self, scale=2):
        ink, source = blurred_source(sigma=1.2)
        h, w = source.shape
        rgb = np.zeros((h, w, 3), np.float32)
        paper = np.array([210, 235, 205], np.float32)       # tinted card background
        inkc = np.array([150, 20, 20], np.float32)          # red text
        alpha = (PAPER - source) / (PAPER - INK)
        rgb[:] = paper
        rgb = rgb + np.clip(alpha, 0, 1)[..., None] * (inkc - paper)
        return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))

    def test_patch_preserves_chroma_and_changes_only_luma(self):
        for scale in (1, 2, 4):
            with self.subTest(scale=scale):
                source = self.colored_receipt()
                patch = source.resize((source.width * scale, source.height * scale), Image.Resampling.LANCZOS)
                out = api._enhance_receipt_patch_uniform(patch, scale, make_config("crisp"), 1.0)
                a = cv2.cvtColor(np.asarray(patch), cv2.COLOR_RGB2YCrCb)
                b = cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2YCrCb)
                self.assertLessEqual(int(np.abs(a[..., 1:].astype(int) - b[..., 1:]).max()), 3)
                metrics = evaluate_clarity(a[..., 0], b[..., 0], scale)
                # YCrCb round trip and uint8 rounding add about 1-2 levels.
                self.assertLessEqual(metrics["thickness_ratio"], 1.06, metrics)
                self.assertLessEqual(metrics["overshoot"], 4.0, metrics)
                self.assertGreaterEqual(metrics["components_after"], metrics["components_before"])

    def test_only_detected_receipt_region_changes(self):
        scale = 2
        rng = np.random.default_rng(5)
        wallpaper = rng.integers(180, 200, (80, 160, 3)).astype(np.uint8)  # textured "wallpaper"
        card = self.colored_receipt()
        canvas = Image.fromarray(wallpaper)
        canvas.paste(card, (20, 20))
        resized = canvas.resize((canvas.width * scale, canvas.height * scale), Image.Resampling.LANCZOS)
        region = (20, 20, 20 + card.width, 20 + card.height)
        with mock.patch.object(api, "_find_all_receipt_regions", return_value=[region]):
            out = api._sharpen_receipt_values(resized.copy(), canvas, scale, make_config("natural"))
        diff = np.abs(np.asarray(out).astype(int) - np.asarray(resized).astype(int)).sum(axis=2)
        x0, y0, x1, y1 = [v * scale for v in region]
        outside = diff.copy()
        outside[y0:y1, x0:x1] = 0
        self.assertEqual(int(outside.sum()), 0, "pixels outside the receipt region changed")
        self.assertGreater(int(diff[y0:y1, x0:x1].sum()), 0)

    def test_enhance_image_modes_and_report(self):
        source = self.colored_receipt()
        for mode in ("receipt_print", "text_focused", "print_text", "general"):
            for scale in (1, 2, 4):
                with self.subTest(mode=mode, scale=scale):
                    result = api.enhance_image(source, scale_factor=scale, mode=mode)
                    self.assertEqual(result["width"], source.width * scale)
                    self.assertIn("clarity", result)
                    self.assertEqual(
                        result["clarity"]["profile"],
                        {"text_focused": "crisp", "print_text": "bw"}.get(mode, "natural"),
                    )

    def test_bw_mode_output_is_black_and_white_only(self):
        import base64
        import io

        result = api.enhance_image(self.colored_receipt(), scale_factor=2, mode="print_text")
        image = Image.open(io.BytesIO(base64.b64decode(result["data_url"].split(",", 1)[1])))
        self.assertTrue(set(np.unique(np.asarray(image.convert("L")))) <= {0, 255})

    def test_disabled_clarity_is_plain_upscale(self):
        source = self.colored_receipt()
        result = api.enhance_image(source, scale_factor=2, mode="receipt_print", clarity=make_config(enabled=False))
        self.assertFalse(result["clarity"]["enabled"])


class SourceStructureTests(unittest.TestCase):
    def test_no_unreachable_statements_in_api(self):
        """Every statement list must end at its first return/raise/continue/break."""
        tree = ast.parse(Path(api.__file__).read_text(encoding="utf-8"))
        problems = []
        for node in ast.walk(tree):
            for field in ("body", "orelse", "finalbody"):
                block = getattr(node, field, None)
                if not isinstance(block, list):
                    continue
                for index, stmt in enumerate(block[:-1]):
                    if isinstance(stmt, (ast.Return, ast.Raise, ast.Continue, ast.Break)):
                        problems.append(f"line {block[index + 1].lineno}")
        self.assertEqual(problems, [])

    def test_target_functions_end_with_single_return(self):
        tree = ast.parse(Path(api.__file__).read_text(encoding="utf-8"))
        for name in ("_enhance_receipt_patch_uniform", "_clarify_text_ink_selective"):
            function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
            returns = [n for n in ast.walk(function) if isinstance(n, ast.Return)]
            self.assertEqual(len(returns), 1, name)
            self.assertIs(function.body[-1], returns[0], name)


if __name__ == "__main__":
    unittest.main()
