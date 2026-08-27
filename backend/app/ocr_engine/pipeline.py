import os
# Disable PIR API and oneDNN/MKLDNN on CPU to prevent ConvertPirAttribute2RuntimeAttribute errors on Windows
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_onednn"] = "0"

import re
import spacy
from paddleocr import PaddleOCR
from app.ocr_engine.preprocessor import ImagePreprocessor

# Load scisPaCy model safely
try:
    nlp_ner = spacy.load("en_ner_bc5cdr_md")
except Exception:
    nlp_ner = None

# Singleton OCR instance to avoid reloading models on every request
_ocr_engine = None

def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = PaddleOCR(use_angle_cls=False, lang='en', enable_mkldnn=False)
    return _ocr_engine

class OCRPipeline:
    @staticmethod
    def process_document(file_bytes: bytes, filename: str) -> dict:
        # 1. Preprocess Image
        raw_cv_img = ImagePreprocessor.read_file_as_cv2(file_bytes, filename)
        processed_img = ImagePreprocessor.enhance_for_ocr(raw_cv_img)

        # 2. Run PaddleOCR
        ocr = get_ocr_engine()
        ocr_results = ocr.ocr(processed_img)
        
        extracted_lines = []

        # 3. Fail-Safe Extraction across PaddleOCR versions
        if ocr_results:
            for block in ocr_results:
                if block is None:
                    continue
                
                # Check for PaddleOCR 3.x / Paddlex Dictionary Output
                if isinstance(block, dict):
                    rec_texts = block.get("rec_texts", [])
                    rec_scores = block.get("rec_scores", [])
                    for text, score in zip(rec_texts, rec_scores):
                        if (score is None or score > 0.4) and text and str(text).strip():
                            extracted_lines.append(str(text).strip())

                # Check for standard List Output Structure
                elif isinstance(block, (list, tuple)):
                    for line in block:
                        if line is None:
                            continue
                        if isinstance(line, (list, tuple)) and len(line) >= 2:
                            text_info = line[1]
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 1:
                                text = str(text_info[0]).strip()
                                confidence = text_info[1] if len(text_info) > 1 and isinstance(text_info[1], (int, float)) else 1.0
                                if confidence > 0.4 and text:
                                    extracted_lines.append(text)
                            elif isinstance(text_info, str) and text_info.strip():
                                extracted_lines.append(text_info.strip())
                        elif isinstance(line, str) and line.strip():
                            extracted_lines.append(line.strip())

        full_text = " ".join(extracted_lines)

        # 4. Medical Entity Extraction (scisPaCy)
        medications, diseases = [], []
        if nlp_ner and full_text.strip():
            doc = nlp_ner(full_text)
            for ent in doc.ents:
                if ent.label_ == "CHEMICAL":
                    medications.append(ent.text)
                elif ent.label_ == "DISEASE":
                    diseases.append(ent.text)

        # 5. Regex Patterns for Dosages and Instructions
        dosage_pattern = r'(\d+\s*(?:mg|g|ml|tablet|tablets|capsule|capsules))'
        frequency_pattern = r'\b(once|twice|thrice|\d+\s*times? a day|1-0-1|1-1-1|1-0-0|0-0-1|after food|before food)\b'
        
        dosages = re.findall(dosage_pattern, full_text, re.IGNORECASE)
        frequencies = re.findall(frequency_pattern, full_text, re.IGNORECASE)

        return {
            "raw_text": full_text,
            "extracted_data": {
                "medications": list(set(medications)),
                "diagnoses_or_symptoms": list(set(diseases)),
                "dosages": list(set(dosages)),
                "frequencies": list(set(frequencies))
            }
        }