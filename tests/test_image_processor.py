import io
import unittest

from PIL import Image

from image_processor import (
    EnhancementSettings,
    encode_jpeg,
    encode_png,
    load_image,
    process_image,
)


class ImageProcessorTests(unittest.TestCase):
    def setUp(self):
        self.source = Image.new("RGB", (20, 10), "white")
        for x in range(20):
            self.source.putpixel((x, 5), (0, 0, 0))
        self.original_bytes = self.source.tobytes()

    def test_scaling_dimensions(self):
        for scale in (1, 2, 3, 4, 6):
            output, _ = process_image(self.source, EnhancementSettings(scale=scale))
            self.assertEqual(output.size, (20 * scale, 10 * scale))

    def test_rgb_and_transparent_images(self):
        output, _ = process_image(self.source, EnhancementSettings(scale=2))
        self.assertEqual(output.mode, "RGB")
        transparent = Image.new("RGBA", (4, 3), (20, 40, 60, 80))
        output, _ = process_image(transparent, EnhancementSettings(scale=2))
        self.assertEqual(output.mode, "RGBA")
        self.assertEqual(output.getpixel((0, 0))[3], 80)

    def test_source_is_not_modified(self):
        process_image(self.source, EnhancementSettings(denoise=2, contrast=0.5, sharpen=1))
        self.assertEqual(self.source.tobytes(), self.original_bytes)

    def test_threshold_is_a_separate_variant(self):
        output, threshold = process_image(
            self.source, EnhancementSettings(text_focus=True, threshold_variant=True)
        )
        self.assertEqual(output.mode, "RGB")
        self.assertIsNotNone(threshold)
        self.assertEqual(threshold.size, output.size)

    def test_encoding_round_trips(self):
        png = encode_png(self.source)
        png_900_dpi = encode_png(self.source, dpi=900)
        jpeg = encode_jpeg(self.source)
        self.assertEqual(Image.open(io.BytesIO(png)).format, "PNG")
        self.assertEqual(Image.open(io.BytesIO(png_900_dpi)).format, "PNG")
        self.assertEqual(Image.open(io.BytesIO(jpeg)).format, "JPEG")

    def test_load_rejects_invalid_data(self):
        with self.assertRaises(ValueError):
            load_image(b"not an image")

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            process_image(self.source, EnhancementSettings(scale=0))
        with self.assertRaises(ValueError):
            process_image(self.source, EnhancementSettings(sharpen=3))


if __name__ == "__main__":
    unittest.main()
