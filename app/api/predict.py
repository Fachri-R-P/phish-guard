from fastapi import APIRouter 
from app.schemas.prediction import PredictResponse, PredictRequest
from app.services.prediction_service import run_prediction

router = APIRouter(tags=["prediction"])

@router.post("/predict", response_model=PredictResponse)
def get_prediction(payload: PredictRequest):
    return run_prediction(payload)
