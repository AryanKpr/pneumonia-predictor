from pydantic import BaseModel
from datetime import datetime

class PredictionResponse(BaseModel):
    id: int
    result: str
    confidence: float
    gradcam_b64: str
    created_at: datetime

    class Config:
        from_attributes = True

class StatsResponse(BaseModel):
    total: int
    pneumonia_count: int
    normal_count: int
    pneumonia_pct: float
    avg_confidence: float