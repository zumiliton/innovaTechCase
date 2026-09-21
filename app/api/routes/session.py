from io import BytesIO

from fastapi import APIRouter, File, UploadFile
from fastapi import HTTPException
from pydantic import BaseModel
from PIL import Image

from app.pipeline.session import ArduinoSession


router = APIRouter()

session = ArduinoSession()


class AskRequest(BaseModel):
    question: str


@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents))

    prediction = session.classify_image(image)

    print(
        f"[VISION] Board: {prediction['board']} "
        f"| Confidence: {prediction['confidence']:.4f}"
    )

    return {
        "filename": file.filename,
        "board": prediction["board"],
        "confidence": prediction["confidence"],
    }


@router.post("/ask")
async def ask_question(request: AskRequest):
    try:
        return session.ask(request.question)

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )