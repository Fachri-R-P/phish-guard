from fastapi import FastAPI, HTTPException, status
from app.api.predict import router
from app.core.config import get_settings
from contextlib import asynccontextmanager
from app.ml.inference import load_model, model_loaded

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    print("Model dimuat")
    yield

app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)

@app.get("/health", tags=["Health"])
async def health_check():
    if not model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded"
        )

    return {"status": "ok", "model_loaded": True, "version": settings.app_version}

app.include_router(router, prefix=settings.api_v1_prefix)





    