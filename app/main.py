from fastapi import FastAPI 
from app.api.predict import router
from app.core.config import get_settings

settings = get_settings()
app = FastAPI(title=settings.app_name, version=settings.app_version)

app.include_router(router, prefix=settings.api_v1_prefix)