"""Extract raster floorplan images from PDF files using PyMuPDF."""
import fitz
import numpy as np
from pathlib import Path

def extract_floorplan(pdf_path: str, page_num: int = 0) -> dict:
    """Extract the floorplan image and metadata from a PDF page."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    doc = fitz.open(str(path))
    if page_num < 0 or page_num >= len(doc):
        raise ValueError(f"Page {page_num} out of range (0-{len(doc)-1})")
    page = doc[page_num]
    images = page.get_images(full=True)
    if images:
        xref = images[0][0]
        pix = fitz.Pixmap(doc, xref)
        if pix.n != 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    else:
        mat = fitz.Matrix(3.0, 3.0)
        pix = page.get_pixmap(matrix=mat)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    text = page.get_text()
    rect = page.rect
    result = {
        "image": img_array.copy(),
        "page_width": rect.width,
        "page_height": rect.height,
        "text": text,
        "page_count": len(doc),
        "image_width": img_array.shape[1],
        "image_height": img_array.shape[0],
    }
    doc.close()
    return result
