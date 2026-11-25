from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    subject: str | None = None
    body: str = Field(min_length=1)
    sender: str | None = None

class PredictResponse(BaseModel):
    label: int
    label_name: str
    score: float = Field(ge=0, le=1)