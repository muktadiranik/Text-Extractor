import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from number_reader import (
    ConfirmedReading, HandwrittenReader, NO_NUMBER, OCRWord, READING_MAX_AGE,
    check_tesseract, draw_preview, find_phones, read_content,
)


APP = Path(__file__).resolve().parents[1] / "number_reader.py"
PEN_STROKES = {
    "0": [[(22, 4), (14, 3), (8, 7), (5, 18), (5, 31), (9, 41), (17, 43),
           (24, 38), (28, 28), (28, 14), (24, 7), (22, 4)]],
    "5": [[(10, 4), (24, 3), (29, 3)], [(9, 11), (7, 21), (16, 19), (24, 21),
           (29, 28), (28, 34), (24, 40), (16, 43), (6, 39)]],
    "7": [[(4, 7), (27, 5), (23, 14), (17, 23), (13, 34), (10, 43)]],
}


class ImageModeTests(unittest.TestCase):
    def run_cli(self, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(APP), *args],
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
        )

    def test_sample_images_with_tesseract(self) -> None:
        samples = [
            ("42", "42"),
            ("007", "007"),
            ("", NO_NUMBER),
            ("1", "1"),
            ("123456", "123456"),
            ("42   007", NO_NUMBER),
            ("1234567", "1234567"),
            ("A42", "A42"),
            ("01234-567890", "01234-567890"),
            ("01234567890", "01234567890"),
            ("001234567890123456789012345678", "001234567890123456789012345678"),
        ]
        with TemporaryDirectory() as directory:
            for index, (sample, expected) in enumerate(samples):
                with self.subTest(sample=sample):
                    path = Path(directory) / f"sample-{index}.png"
                    image_width = max(640, 54 * len(sample))
                    image = np.full((240, image_width, 3), 255, dtype=np.uint8)
                    if sample:
                        if len(sample) >= 11:
                            font, scale, stroke = cv2.FONT_HERSHEY_DUPLEX, 2, 4
                        else:
                            font, scale, stroke = cv2.FONT_HERSHEY_SIMPLEX, 3, 6
                        (width, _), _ = cv2.getTextSize(sample, font, scale, stroke)
                        origin = ((image_width - width) // 2, 150)
                        cv2.putText(image, sample, origin, font, scale, (0, 0, 0), stroke)
                    encoded, png = cv2.imencode(".png", image)
                    self.assertTrue(encoded)
                    path.write_bytes(png.tobytes())
                    result = self.run_cli("--image", str(path))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.strip(), expected)

    def test_unreadable_image(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "corrupt.png"
            path.write_bytes(b"invalid image data")
            result = self.run_cli("--image", str(path))
            self.assertEqual(result.returncode, 1)
            self.assertIn("Could not read image", result.stderr)

    def test_reads_printed_text_with_spaces_numbers_and_line_breaks(self) -> None:
        cases = [("HELLO WORLD", False), ("Room 42", False),
                 ("Meet at noon\nRoom 42", False), ("Open today", True)]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "text.png"
            for expected, inverted in cases:
                with self.subTest(text=expected, inverted=inverted):
                    image = np.full((168, 448, 3), 30 if inverted else 245, dtype=np.uint8)
                    color = (255, 255, 255) if inverted else (22, 22, 22)
                    for index, line in enumerate(expected.splitlines()):
                        origin = (36, 63 + 53 * index)
                        cv2.putText(image, line, origin, cv2.FONT_HERSHEY_DUPLEX, 1.05, color, 2, cv2.LINE_AA)
                    encoded, png = cv2.imencode(".png", image)
                    self.assertTrue(encoded)
                    path.write_bytes(png.tobytes())
                    result = self.run_cli("--image", str(path))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.strip(), expected)

    def test_reads_faint_block_letters(self) -> None:
        image = np.full((168, 448), 180, dtype=np.uint8)
        cv2.putText(image, "QWERT", (54, 105), cv2.FONT_HERSHEY_SIMPLEX, 2, 168, 3, cv2.LINE_AA)
        encoded, png = cv2.imencode(".png", image)
        self.assertTrue(encoded)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "faint-letters.png"
            path.write_bytes(png.tobytes())
            result = self.run_cli("--image", str(path))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "QWERT")

    def test_missing_tesseract(self) -> None:
        with TemporaryDirectory() as directory:
            missing_command = str(Path(directory) / "missing-tesseract")
            result = self.run_cli(env={**os.environ, "TESSERACT_CMD": missing_command})
            self.assertEqual(result.returncode, 1)
            self.assertIn("Install Tesseract OCR", result.stderr)

    def test_default_launch_reads_pen_strokes(self) -> None:
        sequences = ("0057", "000007", "0000007", "005707500570750057075")
        cases = [(sequence, sequence) for sequence in sequences] + [("", NO_NUMBER)]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "handwritten.png"
            for sample, expected in cases:
                with self.subTest(sample=sample):
                    image_width = max(448, 47 * len(sample) + 80)
                    image = np.full((168, image_width, 3), 65, dtype=np.uint8)
                    image[34:137, 35:-36] = 245
                    left = (image_width - 47 * len(sample)) // 2
                    for digit in sample:
                        for stroke in PEN_STROKES[digit]:
                            points = np.asarray(stroke, dtype=np.int32)
                            points += (left, 59)
                            cv2.polylines(image, [points], False, (24, 24, 24), 4, cv2.LINE_AA)
                        left += 47
                    encoded, png = cv2.imencode(".png", image)
                    self.assertTrue(encoded)
                    path.write_bytes(png.tobytes())
                    result = self.run_cli("--image", str(path))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.strip(), expected)

    def test_default_launch_reads_faint_pencil_on_unevenly_lit_paper(self) -> None:
        background = np.tile(np.linspace(195, 160, 448), (168, 1)).astype(np.uint8)
        mask = np.zeros_like(background)
        for index, digit in enumerate("007"):
            for stroke in PEN_STROKES[digit]:
                points = np.asarray(stroke, dtype=np.int32)
                points += (152 + 47 * index, 58)
                cv2.polylines(mask, [points], False, 255, 3, cv2.LINE_AA)
        faint_ink = np.rint(mask.astype(float) * (10 / 255)).astype(np.uint8)
        encoded, png = cv2.imencode(".png", background - faint_ink)
        self.assertTrue(encoded)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "pencil.png"
            path.write_bytes(png.tobytes())
            result = self.run_cli("--image", str(path))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "007")


class CameraCropTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        check_tesseract()
        cls.handwriting = HandwrittenReader()

    def camera_crop(self, text: str, scene: str) -> np.ndarray:
        height, width = 168, 448
        if scene == "shadow":
            gray = np.tile(np.linspace(130, 240, width), (height, 1)).astype(np.uint8)
            left, top, right, bottom = 0, 0, width, height
        else:
            gray = np.tile(np.linspace(30, 88, width), (height, 1)).astype(np.uint8)
            if scene == "small card":
                left, top, right, bottom = 164, 53, 286, 113
            else:
                left, top, right, bottom = 101, 29, 348, 145
                gray[:, :76] += 75
                gray[105:, 374:] += 88
                cv2.line(gray, (3, 0), (87, 166), 5, 7)
                cv2.circle(gray, (385, 61), 35, 110, -1)
            gray[top:bottom, left:right] = np.linspace(170, 244, right - left)

        if text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale, stroke = (0.72, 2) if scene == "small card" else (1.5, 3)
            (text_width, text_height), _ = cv2.getTextSize(text, font, scale, stroke)
            origin = ((left + right - text_width) // 2, (top + bottom + text_height) // 2)
            cv2.putText(gray, text, origin, font, scale, 28, stroke, cv2.LINE_AA)
        if scene == "shadow":
            gray[:, :205] = gray[:, :205] * 0.55

        gray = cv2.GaussianBlur(gray, (3, 3), 0.8)
        noise = np.random.default_rng(10).normal(0, 2.1, gray.shape)
        gray = np.clip(gray + noise, 0, 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def test_detects_print_with_card_edges_and_uneven_lighting(self) -> None:
        for scene, text in (("small card", "007"), ("clutter", "42"), ("shadow", "007")):
            with self.subTest(scene=scene):
                self.assertEqual(read_content(self.camera_crop(text, scene)), text)

    def test_rejects_blank_cards_and_invalid_groups(self) -> None:
        for text in ("", "42   07"):
            with self.subTest(text=text):
                self.assertIsNone(read_content(self.camera_crop(text, "clutter")))

    def test_detects_phone_number_among_screen_labels(self) -> None:
        number = "01234-567890"
        blue = np.linspace((220, 150, 95), (120, 53, 18), 168, dtype=np.uint8)
        image = np.repeat(blue[:, np.newaxis], 448, axis=1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.circle(image, (223, 0), 43, (250, 250, 250), -1)
        cv2.putText(image, number, (76, 100), font, 1.28, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(image, (384, 97), (396, 85), (235, 235, 235), 4, cv2.LINE_AA)
        cv2.putText(image, "Search the web", (139, 137), font, 0.65, (232, 232, 232), 2, cv2.LINE_AA)
        image = cv2.GaussianBlur(image, (3, 3), 0.7)
        self.assertEqual(read_content(image, handwriting=self.handwriting), number)


class ConfirmedReadingTests(unittest.TestCase):
    def test_confirms_text_when_ocr_changes_whitespace(self) -> None:
        reading = ConfirmedReading()
        reading.update("Meet at noon Room 42", 1.0)
        self.assertIsNone(reading.result(1.0))
        reading.update("Meet at noon\nRoom 42", 1.3)
        self.assertEqual(reading.result(1.3), "Meet at noon\nRoom 42")

    def test_confirms_consecutive_matches_and_clears_uncertain_results(self) -> None:
        reading = ConfirmedReading()
        reading.update("007", 1.0)
        self.assertIsNone(reading.result(1.0))
        reading.update("007", 1.3)
        self.assertEqual(reading.result(1.3), "007")
        reading.update("42", 1.6)
        self.assertIsNone(reading.result(1.6))
        reading.update("42", 1.9)
        self.assertEqual(reading.result(1.9), "42")
        reading.update(None, 2.2)
        self.assertIsNone(reading.result(2.2))
        reading.update("42", 2.5)
        self.assertIsNone(reading.result(2.5))

    def test_expires_when_ocr_stops_returning_fresh_samples(self) -> None:
        reading = ConfirmedReading()
        reading.update("42", 1.0)
        reading.update("42", 1.5)
        self.assertIsNone(reading.result(1.5 + READING_MAX_AGE))
        reading.update("42", 2.0 + READING_MAX_AGE)
        self.assertIsNone(reading.result(2.0 + READING_MAX_AGE))


class PhoneRecognitionTests(unittest.TestCase):
    def test_recovers_a_hyphen_that_ocr_puts_in_a_separate_token(self) -> None:
        line = (1, 1, 1)
        words = [OCRWord("01234", 93, line), OCRWord("-", 20, line), OCRWord("567890", 91, line)]
        self.assertEqual(find_phones(words), [("01234-567890", 91)])

    def test_does_not_join_digits_from_different_lines(self) -> None:
        words = [OCRWord("01234-", 96, (1, 1, 1)), OCRWord("567890", 96, (1, 1, 2))]
        self.assertEqual(find_phones(words), [])


class PreviewTests(unittest.TestCase):
    def test_wraps_long_text_inside_the_preview(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        box = (96, 144, 544, 312)
        content = "\n".join(["Please wait here until the next available person calls you."] * 12)
        with patch.object(cv2, "putText", wraps=cv2.putText) as put_text:
            draw_preview(frame, box, content)

        calls = [call.args for call in put_text.call_args_list]
        self.assertGreater(len(calls), 1)
        self.assertTrue(calls[-1][1].endswith("..."))
        for _, line, (x, y), font, scale, _, thickness, _ in calls:
            (width, height), baseline = cv2.getTextSize(line, font, scale, thickness)
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x + width, frame.shape[1])
            self.assertGreater(y - height, box[3])
            self.assertLess(y + baseline, frame.shape[0])
