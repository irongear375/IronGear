# ======================================================================
# Sprint 2 — EasyOCR Laterality Extraction Module
# ======================================================================
# Project:  WristAssist AI | Team: Iron Gear
# Ticket:   SCRUM-32
# Location: Sprint2/utils/ocr.py
#
# Purpose:
#   YOLO detects a "text" class bounding box on the X-ray → this module
#   crops that region, runs EasyOCR to read the text content, and parses
#   the result against an allowlist to extract wrist laterality (L/R).
#
# Pipeline:
#   YOLO text bbox → crop image region → EasyOCR readtext → parse
#   allowlist → return ("Left" | "Right" | "Unknown", raw_text_list)
#
# Usage:
#   from Sprint2.utils.ocr import LateralityExtractor
#   extractor = LateralityExtractor()
#   laterality, raw_text = extractor.extract(image_np, box_norm, img_h, img_w)
# ======================================================================

import numpy as np

# EasyOCR is imported lazily on first use to avoid slowing down imports.
# First-ever call downloads ~100 MB weights — run before demo sessions.
_reader = None


def _get_reader():
    """Lazy-initialise the EasyOCR reader (English, CPU-only).

    EasyOCR runs on CPU because:
    - We're reading 1–2 characters from a small crop (typically <100×100 px)
    - GPU inference overhead is not justified for this workload
    - Keeps GPU VRAM free for YOLO inference

    Returns:
        easyocr.Reader instance
    """
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        print("✅ EasyOCR reader initialised (CPU mode)")
    return _reader


# ──────────────────────────────────────────────────────────────
# Allowlist for laterality tokens
# ──────────────────────────────────────────────────────────────
# Only these tokens are accepted as valid laterality indicators.
# Everything else goes to raw_text only; laterality stays "Unknown".
# This prevents misreads like 'I' → 'L' or 'P' → 'R'.

_LEFT_TOKENS  = {"L", "LEFT"}
_RIGHT_TOKENS = {"R", "RIGHT"}
_ALLOWLIST    = _LEFT_TOKENS | _RIGHT_TOKENS


def extract_laterality(image_np, box_norm, img_h, img_w, padding=8):
    """Extract wrist laterality from a YOLO-detected text bounding box.

    Args:
        image_np: Full X-ray image as a numpy array (H, W) or (H, W, C).
                  Can be uint8 or uint16 (EasyOCR handles both).
        box_norm: Normalised bounding box [x1, y1, x2, y2] where each
                  value is in [0.0, 1.0] relative to image dimensions.
                  This is the YOLO output format for text class detections.
        img_h:    Image height in pixels (used to denormalise box coords).
        img_w:    Image width in pixels (used to denormalise box coords).
        padding:  Extra pixels added around the crop on each side.
                  Default 8px gives EasyOCR more context for edge characters.

    Returns:
        tuple: (laterality: str, raw_text: list[str])
            - laterality: "Left", "Right", or "Unknown"
            - raw_text: list of all strings EasyOCR detected in the crop
              (useful for debugging and logging in the clinical report)
    """
    # ── Step 1: Denormalise bounding box coordinates ──
    # YOLO returns normalised [0,1] coords; convert to pixel coords.
    # Add padding to give EasyOCR context around the character.
    x1 = max(0, int(box_norm[0] * img_w) - padding)
    y1 = max(0, int(box_norm[1] * img_h) - padding)
    x2 = min(img_w, int(box_norm[2] * img_w) + padding)
    y2 = min(img_h, int(box_norm[3] * img_h) + padding)

    # ── Step 2: Crop the detected text region ──
    crop = image_np[y1:y2, x1:x2]

    # Guard: if crop is too small or empty, return Unknown
    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
        return "Unknown", []

    # ── Step 3: Run EasyOCR on the cropped region ──
    reader = _get_reader()
    # detail=0 returns just the text strings (no bounding boxes or confidence)
    result = reader.readtext(crop, detail=0)

    # Clean OCR output: strip whitespace, convert to strings
    raw_text = [str(t).strip() for t in result if str(t).strip()]

    # ── Step 4: Parse against allowlist ──
    # Check each detected token against the laterality allowlist.
    # First match wins (there should only be one letter: L or R).
    laterality = "Unknown"
    for token in raw_text:
        token_upper = token.upper()
        if token_upper in _ALLOWLIST:
            if token_upper in _LEFT_TOKENS:
                laterality = "Left"
            else:
                laterality = "Right"
            break  # First valid match is definitive

    return laterality, raw_text


class LateralityExtractor:
    """Convenience wrapper for stateful EasyOCR laterality extraction.

    Initialises the EasyOCR reader once and provides a clean extract() method
    that can be called repeatedly without re-initialisation overhead.

    Usage:
        extractor = LateralityExtractor()
        laterality, raw = extractor.extract(image, box, h, w)
    """

    def __init__(self):
        """Pre-initialise the EasyOCR reader on construction."""
        self.reader = _get_reader()

    def extract(self, image_np, box_norm, img_h, img_w, padding=8):
        """Extract laterality — delegates to module-level function."""
        return extract_laterality(image_np, box_norm, img_h, img_w, padding)

    def extract_from_detections(self, image_np, detections, text_class_id=4):
        """Extract laterality from YOLO detection results.

        Finds the highest-confidence text class detection, crops it,
        runs EasyOCR, and returns laterality.

        Args:
            image_np:       Full image as numpy array
            detections:     List of dicts with keys: class_id, confidence,
                            box (normalised [x1, y1, x2, y2])
            text_class_id:  Class ID for text (default 4)

        Returns:
            tuple: (laterality: str, raw_text: list[str])
        """
        img_h, img_w = image_np.shape[:2]

        # Find text class detections, sorted by confidence descending
        text_dets = [
            d for d in detections
            if d.get("class_id") == text_class_id
        ]
        text_dets.sort(key=lambda d: d.get("confidence", 0), reverse=True)

        if not text_dets:
            return "Unknown", []

        # Use the highest-confidence text detection
        best = text_dets[0]
        return self.extract(image_np, best["box"], img_h, img_w)


# ──────────────────────────────────────────────────────────────
# Self-test (run this file directly to verify)
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running ocr.py self-test...")

    # Create a simple test image with an "L" character
    # (In real usage, this would be a cropped X-ray region)
    test_img = np.zeros((100, 100), dtype=np.uint8)
    test_img[20:80, 30:50] = 255  # Vertical bar of "L"
    test_img[60:80, 30:70] = 255  # Horizontal bar of "L"

    extractor = LateralityExtractor()
    lat, raw = extractor.extract(test_img, [0.0, 0.0, 1.0, 1.0], 100, 100)
    print(f"  Laterality: {lat}")
    print(f"  Raw text:   {raw}")
    print("✅ Self-test complete")
    