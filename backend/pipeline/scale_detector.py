"""Detect scale/measurement reference from floorplan text and images."""
import re
import numpy as np

def parse_scale_text(text: str) -> dict | None:
    if not text:
        return None
    px_match = re.search(r"1\s*px\s*=\s*([\d.]+)\s*m", text, re.IGNORECASE)
    if px_match:
        meters_per_px = float(px_match.group(1))
        if meters_per_px > 0:
            return {"px_per_meter": 1.0 / meters_per_px, "meters_per_px": meters_per_px, "format": "px_to_meter"}
    ratio_match = re.search(r"(?:scale\s*)?1\s*:\s*(\d+)", text, re.IGNORECASE)
    if ratio_match:
        ratio = int(ratio_match.group(1))
        if ratio > 0:
            return {"scale_ratio": ratio, "format": "ratio"}
    return None

def detect_scale_from_image(image: np.ndarray) -> dict | None:
    try:
        import pytesseract
        from PIL import Image
        pil_image = Image.fromarray(image)
        ocr_text = pytesseract.image_to_string(pil_image)
        return parse_scale_text(ocr_text)
    except ImportError:
        return None
    except Exception:
        return None

def detect_scale(text: str = "", image: np.ndarray | None = None, manual_px_per_meter: float | None = None) -> dict | None:
    if manual_px_per_meter is not None:
        return {"px_per_meter": manual_px_per_meter, "source": "manual"}
    parsed = parse_scale_text(text)
    if parsed and "px_per_meter" in parsed:
        return {"px_per_meter": parsed["px_per_meter"], "source": "auto"}
    if parsed and "scale_ratio" in parsed:
        return {"scale_ratio": parsed["scale_ratio"], "source": "auto"}
    if image is not None:
        ocr_parsed = detect_scale_from_image(image)
        if ocr_parsed and "px_per_meter" in ocr_parsed:
            return {"px_per_meter": ocr_parsed["px_per_meter"], "source": "auto_ocr"}
    return None
