import logging
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize YOLOv8 model
model = YOLO("yolo11n.pt")

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "YOLOv8 Object Detection API"}

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    logger.info(f"Processing file: {file.filename}")
    contents = await file.read()
    with open("temp.jpg", "wb") as f:
        f.write(contents)
    
    # Perform inference
    results = model("temp.jpg")[0]
    boxes = results.boxes
    
    # Extract confidence scores, class IDs and coordinates
    conf = boxes.conf.tolist()
    cls = boxes.cls.tolist()
    coords = boxes.xyxy.tolist()
    
    return {
        "confidence": conf,
        "class_ids": cls,
        "coordinates": coords
    }