import os
import requests
import pytest
from pathlib import Path

# Base URL for the API - can be configured via environment variable
API_URL = os.environ.get("API_URL", "http://localhost:8000")

def test_api_root():
    """Test that the API root endpoint returns the expected message."""
    response = requests.get(f"{API_URL}/")
    assert response.status_code == 200
    assert response.json() == {"message": "YOLOv8 Object Detection API"}

def test_object_detection():
    """Test object detection endpoint with a test image."""
    # Check if test image exists
    test_image_path = Path("tests/test.jpg")
    assert test_image_path.exists(), f"Test image not found at {test_image_path.absolute()}"
    
    # Prepare the file for upload
    files = {"file": ("test.jpg", open(test_image_path, "rb"), "image/jpeg")}
    
    # Send request to the detection endpoint
    response = requests.post(f"{API_URL}/detect/", files=files)
    
    # Check response
    assert response.status_code == 200, f"API returned error: {response.text}"
    
    # Verify that results key exists in response
    result = response.json()
    
    # Verify the structure of the response
    assert "confidence" in result, "Response does not contain 'confidence' key"
    assert "class_ids" in result, "Response does not contain 'class_ids' key"
    assert "coordinates" in result, "Response does not contain 'coordinates' key"
    
    # Check data types
    assert isinstance(result["confidence"], list), "Confidence should be a list"
    assert isinstance(result["class_ids"], list), "Class IDs should be a list"
    assert isinstance(result["coordinates"], list), "Coordinates should be a list"

if __name__ == "__main__":
    # This allows running the test directly with python
    test_api_root()
    test_object_detection()
    print("All tests passed!")
