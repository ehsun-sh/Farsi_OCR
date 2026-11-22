"""Turns images and PDF files into page images (uint8 RGB numpy arrays)."""
import pathlib

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tif', '.tiff'}
PDF_EXTS = {'.pdf'}
MAX_IMAGE_WIDTH = 1600


def is_supported(path):
    return pathlib.Path(path).suffix.lower() in IMAGE_EXTS | PDF_EXTS


def parse_pages(spec, count):
    """'1-3, 5' -> [1, 2, 3, 5] (1-based, clipped to count). Empty -> all pages."""
    spec = (spec or '').strip()
    if not spec:
        return list(range(1, count + 1))
    out = []
    for part in spec.replace('،', ',').split(','):
        part = part.strip().translate(str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789'))
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            a = int(a) if a.strip() else 1
            b = int(b) if b.strip() else count
            out.extend(range(max(1, a), min(count, b) + 1))
        else:
            n = int(part)
            if 1 <= n <= count:
                out.append(n)
    return sorted(dict.fromkeys(out))


def _to_rgb(image):
    image = ImageOps.exif_transpose(image)
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGBA')
        background = Image.new('RGBA', image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    return image.convert('RGB')


def image_to_rgb(image):
    image = _to_rgb(image)
    if image.width > MAX_IMAGE_WIDTH:
        h = round(image.height * MAX_IMAGE_WIDTH / image.width)
        image = image.resize((MAX_IMAGE_WIDTH, h), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8).copy()


class Document:
    """A file to OCR. pages() yields (page_number, rgb) lazily."""

    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.name = self.path.name
        self.is_pdf = self.path.suffix.lower() in PDF_EXTS
        if self.is_pdf:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(str(self.path))
            self.page_count = len(pdf)
            pdf.close()
        else:
            with Image.open(self.path) as im:
                self.page_count = getattr(im, 'n_frames', 1) if self.path.suffix.lower() in ('.tif', '.tiff') else 1

    def page_numbers(self, spec=''):
        return parse_pages(spec, self.page_count) if self.is_pdf or self.page_count > 1 else [1]

    def render(self, number, pdf_width=1240):
        if self.is_pdf:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(str(self.path))
            try:
                page = pdf[number - 1]
                scale = pdf_width / page.get_width()
                image = page.render(scale=scale, fill_color=(255, 255, 255, 255)).to_pil()
                page.close()
            finally:
                pdf.close()
            return np.asarray(image.convert('RGB'), dtype=np.uint8).copy()
        with Image.open(self.path) as im:
            if self.page_count > 1:
                im.seek(number - 1)
            return image_to_rgb(im)
