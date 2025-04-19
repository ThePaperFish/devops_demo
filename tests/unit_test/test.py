# test_app.py
import sys
sys.path.append("..")  # Add the parent directory to the path

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
# from app import main
from app.main import app  # Adjust the import based on your actual app filename/module

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "YOLOv8 Object Detection API"}

@pytest.fixture
def mock_yolo_inference():
    # Patch the model call to return mocked results
    with patch("app.model") as mock_model:
        mock_result = MagicMock()
        mock_boxes = MagicMock()
        mock_boxes.conf.tolist.return_value = [0.85]
        mock_boxes.cls.tolist.return_value = [1]
        mock_boxes.xyxy.tolist.return_value = [[10, 20, 30, 40]]
        mock_result.__getitem__.return_value = mock_result
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]
        yield mock_model

def test_detect_endpoint(mock_yolo_inference):
    # Create a fake image file
    file_content = b"fake-image-bytes"
    files = {"file": ("test.jpg", file_content, "image/jpeg")}
    response = client.post("/detect/", files=files)

    assert response.status_code == 200
    json_data = response.json()
    assert "confidence" in json_data
    assert "class_ids" in json_data
    assert "coordinates" in json_data
    assert json_data["confidence"] == [0.85]
    assert json_data["class_ids"] == [1]
    assert json_data["coordinates"] == [[10, 20, 30, 40]]
