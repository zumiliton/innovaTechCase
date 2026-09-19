from io import BytesIO

from fastapi import APIRouter, File, UploadFile
from PIL import Image

from app.vision.classifier import ArduinoClassifier


router = APIRouter()

classifier = ArduinoClassifier()


@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()

    image = Image.open(BytesIO(contents))

    prediction = classifier.predict(image)

    print(
        f"[VISION] Board: {prediction['board']} "
        f"| Confidence: {prediction['confidence']:.4f}"
    )

    return {
        "filename": file.filename,
        "board": prediction["board"],
        "confidence": prediction["confidence"],
    }