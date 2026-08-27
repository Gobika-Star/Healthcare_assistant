import cv2
import numpy as np
from PIL import Image
import pymupdf  # PyMuPDF

class ImagePreprocessor:
    @staticmethod
    def read_file_as_cv2(file_bytes: bytes, filename: str) -> np.ndarray:
        """Converts uploaded file bytes (PDF or Image) into a CV2 BGR Image Matrix."""
        if filename.lower().endswith(".pdf"):
            # Load PDF from memory and render first page to image
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            page = doc.load_page(0)
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        else:
            # Decode standard image formats (PNG, JPG, JPEG)
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not decode image from file '{filename}'. The file may be corrupted or in an unsupported format.")
            return img

    @staticmethod
    def enhance_for_ocr(img: np.ndarray) -> np.ndarray:
        """Applies OpenCV transformations to optimize text legibility while keeping 3 channels for PaddleOCR."""
        if img is None or img.size == 0:
            raise ValueError("Input image for OCR enhancement is empty or None.")
        
        # Ensure 3-channel format initially
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 1. Grayscale Conversion
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Contrast Stretching (CLAHE) for sharp text edges
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        
        # 3. Convert back to 3-channel BGR (PaddleOCR requires shape [H, W, 3])
        enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        return enhanced_bgr