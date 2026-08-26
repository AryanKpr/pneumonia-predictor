# 🫁 Pneumonia Detector

An AI-powered full stack web app that analyzes chest X-rays and predicts pneumonia with Grad-CAM heatmap visualization showing exactly what the model focused on.

🔗 **[Live Demo](https://pneumonia-predictor-kappa.vercel.app/)**

---

## Features
- Upload a chest X-ray and get an instant NORMAL / PNEUMONIA prediction
- Grad-CAM heatmap overlay showing which regions influenced the prediction
- Prediction history stored in Postgres — every scan is saved
- Stats dashboard showing total scans, detection rate, and avg confidence

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js, Tailwind CSS |
| Backend | FastAPI, Python |
| ML Model | PyTorch CNN |
| Explainability | Grad-CAM |
| Database | PostgreSQL, SQLAlchemy |
| Model Hosting | Hugging Face |
| Deployment | Vercel (frontend), Render (backend) |

---

## Architecture

User → Next.js frontend → FastAPI backend → PyTorch CNN → Grad-CAM
↓

PostgreSQL

---

## Model
- Custom CNN trained on the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset
- ~5,800 images across NORMAL and PNEUMONIA classes
- 95%+ accuracy, 97.8% sensitivity
- Weights hosted on [Hugging Face](https://huggingface.co/AryanKpr/pneumonia-predictor)

---
