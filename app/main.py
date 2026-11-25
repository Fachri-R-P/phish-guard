from fastapi import FastAPI 
from app.api.predict import router

app = FastAPI(title="PishGuard API")

app.include_router(router)