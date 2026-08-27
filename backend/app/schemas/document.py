from pydantic import BaseModel
from typing import List, Optional

class ExtractedMedicalData(BaseModel):
    medications: List[str]
    diagnoses_or_symptoms: List[str]
    dosages: List[str]
    frequencies: List[str]

class DocumentAnalysisResponse(BaseModel):
    filename: str
    status: str
    raw_text: str
    extracted_medical_data: ExtractedMedicalData