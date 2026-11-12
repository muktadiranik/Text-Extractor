"""Read printed text, phone numbers, and handwritten digits from a camera or image."""

import argparse
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from itertools import groupby
import os
from pathlib import Path
import re
import shutil
import sys
import textwrap
from time import monotonic
from typing import Literal

import cv2
import numpy as np
import pytesseract


NO_NUMBER = "No number detected"
WINDOW_TITLE = "Camera Number Reader"
OCR_INTERVAL = 0.25
OCR_TIMEOUT = 1.8
READING_MAX_AGE = 2.0
MIN_CONFIDENCE = 65
MIN_TEXT_CONFIDENCE = 80
MIN_SINGLE_LETTER_CONFIDENCE = 95
DIGITS = re.compile(r"[0-9]+")
PHONE_NUMBER = re.compile(r"([0-9]{5})\s*[-‐‑–—−]\s*([0-9]{6})")
SAMPLE_PATH = Path(__file__).with_name("ocr-sample.png")
HANDWRITING_MODEL = Path(__file__).with_name("models") / "mnist-8.onnx"
HANDWRITING_MIN_CONFIDENCE = 0.90
PREVIEW_PUNCTUATION = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "..."})


class ReaderError(Exception):
    """An error with a message suitable for the command line."""


def check_tesseract() -> None:
    command = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if not command and sys.platform == "win32":
        program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        default_path = program_files / "Tesseract-OCR" / "tesseract.exe"
        if default_path.is_file():
            command = str(default_path)

    pytesseract.pytesseract.tesseract_cmd = command or "tesseract"
    try:
        pytesseract.get_tesseract_version()
        languages = pytesseract.get_languages()
    except (pytesseract.TesseractNotFoundError, OSError) as exc:
        raise ReaderError(
            "Could not start Tesseract. Install Tesseract OCR and add it to PATH, "
            "or set TESSERACT_CMD to its executable. See README.md."
        ) from exc

    if "eng" not in languages:
        raise ReaderError(
            "Tesseract's English language data is missing. Install eng.traineddata."
        )


def normalize_ink(ink: np.ndarray) -> np.ndarray | None:
    if not 24 <= cv2.countNonZero(ink) < ink.size / 2:
        return None

    x, y, width, height = cv2.boundingRect(ink)
    if height < 8 or width < 2:
        return None

    cropped = cv2.bitwise_not(ink[y : y + height, x : x + width])
    # Limit upscaling without shrinking a long row to fit a fixed width.
    scale = min(80 / height, max(1, 960 / width))
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(
        cropped,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=interpolation,
    )
    return cv2.copyMakeBorder(resized, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)


def crop_centered_card(gray: np.ndarray, bright: np.ndarray) -> np.ndarray:
    height, width = gray.shape
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, card_width, card_height = cv2.boundingRect(contour)
        right, bottom = x + card_width, y + card_height
        if not (0 < x <= width // 2 < right < width and 0 < y <= height // 2 < bottom < height):
            continue
        area = card_width * card_height
        if area >= 0.07 * width * height and cv2.contourArea(contour) >= 0.82 * area:
            return gray[y:bottom, x:right]
    return gray


def dark_ink_mask(
    gray: np.ndarray, bright: np.ndarray, *, faint: Literal["digits", "text"] | None = None
) -> np.ndarray:
    region = crop_centered_card(gray, bright)
    height, width = region.shape
    min_height, min_area = 1, 6
    if faint == "text":
        # Keep short horizontal strokes that may be separate from the rest of a letter.
        kernel_size = min(31, max(7, height // 18)) | 1
        min_area = 16
    elif faint == "digits":
        kernel_size = min(35, max(9, height // 11)) | 1
        min_height = max(8, height // 10)
    else:
        kernel_size = min(45, max(11, height // 6)) | 1
    kernel_shape = cv2.MORPH_ELLIPSE if faint else cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
    # Black-hat isolates dark strokes against the local background, even in shadow.
    contrast = cv2.morphologyEx(region, cv2.MORPH_BLACKHAT, kernel)
    if faint:
        threshold = 6
    else:
        threshold, _ = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        threshold = max(14, threshold)
    _, ink = cv2.threshold(contrast, threshold, 255, cv2.THRESH_BINARY)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    x, y, component_width, component_height, area = stats.T
    interior = (x > 0) & (y > 0) & (x + component_width < width) & (y + component_height < height)
    # Discard isolated noise and partial objects entering at the edge of the crop.
    keep = (area >= min_area) & (component_height >= min_height) & interior
    keep[0] = False
    return keep[labels].astype(np.uint8) * 255


def prepare_phone_line(gray: np.ndarray) -> np.ndarray:
    height = gray.shape[0]
    band = gray[height // 4 : max(height // 4 + 1, 3 * height // 4)]
    kernel_size = min(41, max(11, height // 12)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # Bring out light text on the display and leave the UI above and below the line out.
    contrast = cv2.morphologyEx(band, cv2.MORPH_TOPHAT, kernel)
    return cv2.bitwise_not(cv2.normalize(contrast, None, 0, 255, cv2.NORM_MINMAX))


@dataclass(frozen=True)
class OCRWord:
    text: str
    confidence: float
    line: tuple[int, int, int]


def ocr_words(
    image: np.ndarray | None, page_segmentation: int, deadline: float
) -> list[OCRWord]:
    remaining = deadline - monotonic()
    if image is None or remaining <= 0:
        return []

    try:
        data = pytesseract.image_to_data(
            image,
            lang="eng",
            config=f"--psm {page_segmentation}",
            output_type=pytesseract.Output.DICT,
            timeout=remaining,
        )
    except RuntimeError as exc:
        if str(exc) == "Tesseract process timeout":
            return []
        raise
    return [
        OCRWord(word.strip(), float(confidence), (block, paragraph, line))
        for word, confidence, block, paragraph, line in zip(
            data["text"], data["conf"], data["block_num"], data["par_num"], data["line_num"]
        )
        if word.strip()
    ]


def format_number(word: str) -> str | None:
    if DIGITS.fullmatch(word):
        return word
    phone = PHONE_NUMBER.fullmatch(word)
    if phone:
        return "-".join(phone.groups())
    return None


def parse_single_number(words: list[OCRWord]) -> str | None:
    if len(words) == 1 and words[0].confidence >= MIN_CONFIDENCE:
        return format_number(words[0].text)
    return None


def find_phones(words: list[OCRWord]) -> list[tuple[str, float]]:
    phones = []
    for _, line_words in groupby(words, key=lambda word: word.line):
        line = list(line_words)
        for start in range(len(line)):
            # Tesseract may emit the hyphen as its own token on the same line.
            for end in range(start + 1, min(len(line), start + 3) + 1):
                parts = line[start:end]
                phone = PHONE_NUMBER.fullmatch(" ".join(word.text for word in parts))
                if phone:
                    confidence = min(
                        word.confidence for word in parts if any(char.isdigit() for char in word.text)
                    )
                    phones.append(("-".join(phone.groups()), confidence))
    return phones


def parse_text(words: list[OCRWord]) -> str | None:
    meaningful = [word for word in words if any(char.isalnum() for char in word.text)]
    if not any(char.isalpha() for word in meaningful for char in word.text):
        return None

    single_character = len(meaningful) == 1 and sum(char.isalnum() for char in meaningful[0].text) == 1
    threshold = MIN_SINGLE_LETTER_CONFIDENCE if single_character else MIN_TEXT_CONFIDENCE
    confidences = [word.confidence for word in meaningful]
    if min(confidences) < 50 or sum(confidences) / len(confidences) < threshold:
        return None

    lines = groupby(words, key=lambda word: word.line)
    return "\n".join(" ".join(word.text for word in line) for _, line in lines)


def agreed_printed_text(words: list[OCRWord], alternate_words: list[OCRWord]) -> str | None:
    if len(words) != 1 or len(alternate_words) != 1:
        return None
    # The digit model can mistake letters for digits when both OCR passes see the same word.
    if words[0].text == alternate_words[0].text:
        return parse_text(words) if parse_text(alternate_words) is not None else None
    return None


def read_content(image: np.ndarray, *, handwriting: "HandwrittenReader | None" = None) -> str | None:
    deadline = monotonic() + OCR_TIMEOUT
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    low, high, _, _ = cv2.minMaxLoc(smooth)
    if high - low <= 6:
        return None

    _, bright = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    border = np.concatenate((bright[0], bright[-1], bright[:, 0], bright[:, -1]))
    light_background = np.median(border) >= 128
    ink = cv2.bitwise_not(bright) if light_background else bright
    try:
        if high - low < 15:
            faint_digits = dark_ink_mask(smooth, bright, faint="digits")
            handwritten = handwriting.read_ink(faint_digits) if handwriting else None
            if handwritten is not None:
                return handwritten
            faint_text = normalize_ink(dark_ink_mask(smooth, bright, faint="text"))
            text = parse_text(ocr_words(faint_text, 6, deadline))
            return text or parse_text(ocr_words(normalize_ink(faint_digits), 6, deadline))

        prepared = normalize_ink(ink)
        words = ocr_words(prepared, 8, deadline)
        number = parse_single_number(words)
        if number is not None:
            return number
        dark_ink = dark_ink_mask(smooth, bright)
        alternate_words = ocr_words(normalize_ink(dark_ink), 8, deadline)
        number = parse_single_number(alternate_words)
        if number is not None:
            return number

        agreed_text = agreed_printed_text(words, alternate_words)
        handwritten = None
        if handwriting and agreed_text is None:
            handwritten = handwriting.read_ink(dark_ink)
            if handwritten is None:
                handwritten = handwriting.read_ink(dark_ink_mask(smooth, bright, faint="digits"))
            if handwritten is not None and light_background:
                return handwritten

        text = None
        # Search the whole box before focusing on the expected line of a phone display.
        ocr_passes = [(gray, 11, True), (prepared, 11, True), (prepare_phone_line(smooth), 6, False)]
        if agreed_text is None and handwritten is None:
            faint_pass = (normalize_ink(dark_ink_mask(smooth, bright, faint="text")), 6, True)
            # Uneven lighting can make the thresholded border look darker than the paper.
            if light_background or np.median(smooth) >= 128:
                ocr_passes.insert(1, faint_pass)
            else:
                ocr_passes.append(faint_pass)

        for image_variant, mode, read_text in ocr_passes:
            recognized_words = ocr_words(image_variant, mode, deadline)
            phones = find_phones(recognized_words)
            if len(phones) > 1:
                return None
            if phones and phones[0][1] >= MIN_CONFIDENCE:
                return phones[0][0]
            if mode == 11 and number is None:
                number = parse_single_number(recognized_words)
            if read_text and text is None:
                text = parse_text(recognized_words)
                if text is not None and light_background:
                    return number or text
        return number or text or agreed_text or handwritten or parse_text(alternate_words) or parse_text(words)
    except pytesseract.TesseractError as exc:
        raise ReaderError(f"Tesseract could not read the image: {exc}") from exc


def digit_input(ink: np.ndarray) -> np.ndarray:
    height, width = ink.shape
    scale = 20 / max(height, width)
    resized = cv2.resize(
        ink,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    # MNIST digits fit inside a 20-pixel square and are centered by their ink mass.
    moments = cv2.moments(resized)
    center_x = moments["m10"] / moments["m00"]
    center_y = moments["m01"] / moments["m00"]
    height, width = resized.shape
    left = max(0, min(28 - width, round(13.5 - center_x)))
    top = max(0, min(28 - height, round(13.5 - center_y)))
    canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas[top : top + height, left : left + width] = resized
    return cv2.dnn.blobFromImage(canvas, scalefactor=1 / 255)


def digit_boxes(ink: np.ndarray) -> list[tuple[int, int, int, int]]:
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    components = sorted(
        (int(x), int(y), int(x + width), int(y + height))
        for x, y, width, height, area in stats[1:]
        if height >= 3 and area >= 12
    )
    boxes = []
    for left, top, right, bottom in components:
        if boxes and left < boxes[-1][2]:
            # A handwritten digit can have disconnected strokes in the same column.
            prev_left, prev_top, prev_right, prev_bottom = boxes[-1]
            vertical_gap = max(top - prev_bottom, prev_top - bottom)
            if vertical_gap > 0.25 * max(bottom - top, prev_bottom - prev_top):
                return []
            boxes[-1] = (prev_left, min(top, prev_top), max(right, prev_right), max(bottom, prev_bottom))
        else:
            boxes.append((left, top, right, bottom))
    return boxes


class HandwrittenReader:
    def __init__(self) -> None:
        try:
            model = np.frombuffer(HANDWRITING_MODEL.read_bytes(), dtype=np.uint8)
            self.net = cv2.dnn.readNetFromONNX(model)
        except (OSError, cv2.error) as exc:
            raise ReaderError(f"Could not load the handwriting model at '{HANDWRITING_MODEL}'.") from exc

    def read_ink(self, ink: np.ndarray) -> str | None:
        boxes = digit_boxes(ink)
        if not boxes:
            return None

        max_height = max(bottom - top for _, top, _, bottom in boxes)
        min_height = max(8, 0.55 * max_height)
        if any(
            bottom - top < min_height or right - left > 1.25 * (bottom - top)
            for left, top, right, bottom in boxes
        ):
            return None
        for (_, top, right, bottom), (next_left, next_top, _, next_bottom) in zip(boxes, boxes[1:]):
            overlap = min(bottom, next_bottom) - max(top, next_top)
            shorter_digit = min(bottom - top, next_bottom - next_top)
            if next_left - right > max_height or overlap < 0.5 * shorter_digit:
                return None

        digits = []
        for left, top, right, bottom in boxes:
            self.net.setInput(digit_input(ink[top:bottom, left:right]))
            scores = self.net.forward().reshape(10)
            digit = int(np.argmax(scores))
            probability = 1 / float(np.exp(scores - scores[digit]).sum())
            if probability < HANDWRITING_MIN_CONFIDENCE:
                return None
            digits.append(str(digit))
        return "".join(digits)


@dataclass
class ConfirmedReading:
    candidate: str | None = None
    confirmed: str | None = None
    sampled_at: float = float("-inf")

    def update(self, content: str | None, sampled_at: float) -> None:
        # The candidate comes from the previous sample. Require two fresh matches.
        recent = sampled_at - self.sampled_at < READING_MAX_AGE
        matches = (
            content is not None and self.candidate is not None
            and content.split() == self.candidate.split()
        )
        self.confirmed = content if recent and matches else None
        self.candidate = content
        self.sampled_at = sampled_at

    def result(self, now: float) -> str | None:
        if now - self.sampled_at < READING_MAX_AGE:
            return self.confirmed
        return None


def draw_preview(
    frame: np.ndarray, box: tuple[int, int, int, int], content: str | None
) -> None:
    left, top, right, bottom = box
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 220, 255), 2)
    label = content if content is not None else NO_NUMBER
    font = cv2.FONT_HERSHEY_SIMPLEX
    height, width = frame.shape[:2]
    scale = min(width / 640, height / 480)
    if content is not None and format_number(content) is None:
        scale *= 0.8
        label = label.translate(PREVIEW_PUNCTUATION)
    thickness = max(1, round(scale * 2))
    padding = max(2, round(6 * scale))
    (char_width, text_height), baseline = cv2.getTextSize("W", font, scale, thickness)
    line_width = max(1, (width - 4 * padding) // char_width)
    first_line_y = bottom + text_height + 2 * padding
    line_height = text_height + baseline + padding
    available_lines = max(1, 1 + (height - first_line_y - baseline - padding) // line_height)
    lines = [segment for line in label.splitlines() for segment in textwrap.wrap(line, width=line_width)]
    if len(lines) > available_lines:
        lines = lines[:available_lines]
        shortened = lines[-1][: max(0, line_width - 3)].rstrip()
        lines[-1] = (shortened + "...")[:line_width]

    widths = [cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines]
    max_width = max(widths)
    cv2.rectangle(
        frame,
        ((width - max_width) // 2 - padding, first_line_y - text_height - padding),
        ((width + max_width) // 2 + padding, first_line_y + (len(lines) - 1) * line_height + baseline + padding),
        (0, 0, 0),
        -1,
    )
    for index, (line, pixel_width) in enumerate(zip(lines, widths)):
        origin = ((width - pixel_width) // 2, first_line_y + index * line_height)
        cv2.putText(frame, line, origin, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def save_sample(crop: np.ndarray) -> None:
    try:
        ok, encoded = cv2.imencode(".png", crop)
        if not ok:
            raise OSError("PNG encoding failed")
        SAMPLE_PATH.write_bytes(encoded.tobytes())
        print(f"Saved the area inside the yellow box to {SAMPLE_PATH}", flush=True)
    except (OSError, cv2.error) as exc:
        print(f"Could not save the OCR sample: {exc}", file=sys.stderr, flush=True)


def run_camera(camera_index: int, read_sample: Callable[[np.ndarray], str | None] = read_content) -> None:
    capture = cv2.VideoCapture(camera_index)
    window_opened = False
    camera_help = "Check the connection and camera permissions, or try another --camera-index."
    worker = ThreadPoolExecutor(max_workers=1)
    try:
        if not capture.isOpened():
            raise ReaderError(f"Could not open camera {camera_index}. {camera_help}")
        has_display = os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY")
        if sys.platform.startswith("linux") and not has_display:
            raise ReaderError("The camera preview needs a graphical desktop session.")

        reading = ConfirmedReading()
        pending = None
        sampled_at = float("-inf")
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        window_opened = True

        while True:
            ok, frame = capture.read()
            if not ok or frame is None or not frame.size:
                raise ReaderError(f"Could not read a frame from camera {camera_index}. {camera_help}")

            height, width = frame.shape[:2]
            left, right = int(width * 0.15), int(width * 0.85)
            top, bottom = int(height * 0.30), int(height * 0.65)
            now = monotonic()
            if pending is not None and pending.done():
                reading.update(pending.result(), sampled_at)
                pending = None
            if pending is None and now - sampled_at >= OCR_INTERVAL:
                sampled_at = now
                pending = worker.submit(read_sample, frame[top:bottom, left:right].copy())

            preview = frame.copy()
            draw_preview(preview, (left, top, right, bottom), reading.result(now))
            cv2.imshow(WINDOW_TITLE, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S")):
                save_sample(frame[top:bottom, left:right])
            try:
                if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                # Some OpenCV GUI backends raise after the user closes the window.
                break
    finally:
        capture.release()
        if window_opened:
            cv2.destroyAllWindows()
        worker.shutdown(wait=True, cancel_futures=True)


def load_image(path: Path) -> np.ndarray:
    try:
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    except (OSError, cv2.error) as exc:
        raise ReaderError(f"Could not read image '{path}': {exc}") from exc
    if image is None:
        raise ReaderError(f"Could not read image '{path}'. Use a valid image file such as PNG or JPEG.")
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--camera-index", type=int, default=0, help="webcam index (default: 0)")
    mode.add_argument("--image", type=Path, help="read text or numbers from a still image")
    args = parser.parse_args()
    if args.camera_index < 0:
        parser.error("--camera-index must be zero or greater")

    try:
        image = load_image(args.image) if args.image is not None else None
        check_tesseract()
        reader = partial(read_content, handwriting=HandwrittenReader())
        if image is None:
            run_camera(args.camera_index, reader)
        else:
            print(reader(image) or NO_NUMBER)
    except (ReaderError, cv2.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
