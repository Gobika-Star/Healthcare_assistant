import os
# Disable PIR API and oneDNN/MKLDNN on CPU for PaddlePaddle inference stability on Windows
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_onednn"] = "0"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router

app = FastAPI(
    title="AI Powered Healthcare Communication Assistant",
    description="Backend API for Document Parsing, Simplification, and Voice Guidance",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API V1 routes
app.include_router(api_router, prefix="/api/v1")

# Route to list sample prescriptions for quick testing in UI
DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "raw_prescriptions"))

@app.get("/api/v1/samples")
def list_samples():
    if not os.path.exists(DATASET_DIR):
        return {"samples": []}
    files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:12]
    return {"samples": files}

@app.get("/api/v1/samples/{filename}")
def get_sample_file(filename: str):
    # Prevent directory traversal
    safe_name = os.path.basename(filename)
    file_path = os.path.join(DATASET_DIR, safe_name)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Sample prescription not found")

# Serve Web UI
INDEX_PATH = os.path.join(os.path.dirname(__file__), "index.html")

@app.get("/", response_class=FileResponse)
def serve_ui():
    if os.path.exists(INDEX_PATH):
        return FileResponse(INDEX_PATH)
    return JSONResponse(content={"status": "Online", "system": "AI Healthcare Communication Assistant API"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)