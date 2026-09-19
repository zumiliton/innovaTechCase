from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

from app.api.routes.health import router as health_router
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"


app = FastAPI(
    title="Arduino Multimodal Assistant",
    version="0.1.0",
)

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static",
)

app.include_router(health_router)


@app.get("/", include_in_schema=False)
async def frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "message": "Image received successfully",
    }