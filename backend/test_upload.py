import requests
import json
import os

# Server endpoint URL
URL = "http://127.0.0.1:8000/api/v1/documents/upload"

def test_document_upload(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    print(f"--> Sending file '{file_path}' to {URL}...")
    
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "image/jpeg")}
        response = requests.post(URL, files=files)

    print(f"Status Code: {response.status_code}\n")
    if response.status_code == 200:
        print("--- JSON Response from API ---")
        print(json.dumps(response.json(), indent=4))
    else:
        print("Error Response:", response.text)

if __name__ == "__main__":
    # Test with a sample image file from your datasets folder
    sample_image = "../datasets/raw_prescriptions/eka_000.jpg" 
    test_document_upload(sample_image)