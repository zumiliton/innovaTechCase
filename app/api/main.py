from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.routes.session import router as session_router

app = FastAPI(
    title="Arduino Multimodal Assistant",
)


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR), 
    name="static",
)
 
 
# app.include_router(
#     predict_router,
#     prefix="/api",
# )  
# app.include_router(
#     ask_router,
#     prefix="/api"
# )

app.include_router(
    session_router,
    prefix = "/api"
)


@app.get("/")
async def root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/dev/reload-version")
async def reload_version():
    files = [
        FRONTEND_DIR / "index.html",
        FRONTEND_DIR / "script.js",
    ]

    return {
        "version": max(
            file.stat().st_mtime_ns
            for file in files
            if file.exists()
        )
    }

@app.get("/health")
async def health():
    return {"status": "ok"}
