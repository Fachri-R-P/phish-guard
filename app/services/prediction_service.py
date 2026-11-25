from app.schemas.prediction import PredictRequest, PredictResponse
from app.ml.inference import predict

def run_prediction(payload: PredictRequest) -> PredictResponse:
    result = predict(
        subject=payload.subject, body=payload.body,
    )

    return PredictResponse(
        **result
    )