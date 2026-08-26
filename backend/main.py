import hashlib
import io
from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func
from PIL import Image

from db import init_db, get_db, Prediction
from model import load_model, predict
from gradcam import generate_gradcam
from schemas import PredictionResponse, StatsResponse

app = FastAPI(title="Pneumonia Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model, device = load_model()

@app.on_event("startup")
def startup():
    init_db()

@app.post("/predict", response_model=PredictionResponse)
async def predict_xray(file: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = await file.read()
    image    = Image.open(io.BytesIO(contents))
    result, confidence = predict(image, model, device)
    gradcam_b64        = generate_gradcam(image, model, device)
    image_hash         = hashlib.sha256(contents).hexdigest()

    record = Prediction(
        result=result,
        confidence=confidence,
        gradcam_b64=gradcam_b64,
        image_hash=image_hash,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

@app.get("/predictions", response_model=list[PredictionResponse])
def get_predictions(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    return db.query(Prediction).order_by(Prediction.created_at.desc()).offset(skip).limit(limit).all()

@app.get("/stats", response_model=StatsResponse)
def get_stats(db: Session = Depends(get_db)):
    total     = db.query(Prediction).count()
    pneumonia = db.query(Prediction).filter(Prediction.result == "PNEUMONIA").count()
    normal    = total - pneumonia
    avg_conf  = db.query(func.avg(Prediction.confidence)).scalar() or 0.0
    return StatsResponse(
        total=total,
        pneumonia_count=pneumonia,
        normal_count=normal,
        pneumonia_pct=round(pneumonia / total * 100, 1) if total else 0.0,
        avg_confidence=round(avg_conf, 2),
    )

@app.get("/health")
def health():
    return {"status": "ok"}