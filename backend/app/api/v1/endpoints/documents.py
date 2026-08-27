from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.ocr_engine.pipeline import OCRPipeline
from app.schemas.document import DocumentAnalysisResponse, ExtractedMedicalData

router = APIRouter()

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}

@router.post(
    "/upload", 
    response_model=DocumentAnalysisResponse, 
    status_code=status.HTTP_200_OK,
    summary="Upload scanned medical document/prescription and receive extracted JSON data."
)
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename
    ext = filename.split(".")[-1].lower() if "." in filename else ""
    
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file format. Only {ALLOWED_EXTENSIONS} files are permitted."
        )

    try:
        # Read uploaded file bytes directly from memory
        file_bytes = await file.read()
        
        # Execute OCR and Medical NER Extraction
        extraction_result = OCRPipeline.process_document(file_bytes, filename)
        
        return DocumentAnalysisResponse(
            filename=filename,
            status="SUCCESS",
            raw_text=extraction_result["raw_text"],
            extracted_medical_data=ExtractedMedicalData(
                medications=extraction_result["extracted_data"]["medications"],
                diagnoses_or_symptoms=extraction_result["extracted_data"]["diagnoses_or_symptoms"],
                dosages=extraction_result["extracted_data"]["dosages"],
                frequencies=extraction_result["extracted_data"]["frequencies"]
            )
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred while processing the document: {str(e)}"
        )