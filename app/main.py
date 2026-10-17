"""FastAPI service that serves the serialized wine classification pipeline."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import PredictionResponse, WineFeatures

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "wine_classifier.joblib"
METADATA_PATH = ROOT / "artifacts" / "model_metadata.json"

@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="Wine Classifier Prediction API",
    description=(
        "Educational API that predicts one of the three wine cultivar categories from 13 numeric features. "
        "It is a demonstration project, not a wine-quality, safety, or authenticity assessment."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

model: Any | None = None
metadata: dict[str, Any] = {}
load_error: str | None = None


def load_artifacts() -> None:
    """Load model and metadata once when the API process starts."""
    global model, metadata, load_error
    try:
        if not MODEL_PATH.exists() or not METADATA_PATH.exists():
            raise FileNotFoundError("Model artifacts are missing. Run `python -m ml.train` from the project root first.")
        model = joblib.load(MODEL_PATH)
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        load_error = None
    except Exception as exc:  # exposed safely by /health; details are logged by server in development
        model = None
        metadata = {}
        load_error = str(exc)


@app.get("/", tags=["Service"])
def root() -> dict[str, str]:
    return {
        "message": "Wine Classifier API is running.",
        "docs": "/docs",
        "health": "/health",
        "prediction_endpoint": "POST /predict",
    }


@app.get("/health", tags=["Service"])
def health() -> dict[str, str]:
    if model is None:
        return {"status": "not_ready", "detail": load_error or "Model is not loaded"}
    return {"status": "ok", "model": "loaded"}


@app.get("/model-info", tags=["Model"])
def model_info() -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail=load_error or "Model is not loaded")
    return {
        "project_name": metadata.get("project_name"),
        "selected_model": metadata.get("best_model"),
        "feature_count": metadata.get("n_features"),
        "features": metadata.get("feature_names", []),
        "classes": metadata.get("class_names", []),
        "training_rows": metadata.get("training_rows"),
        "test_rows": metadata.get("test_rows"),
        "note": "Dataset categories only; the classes do not represent a quality grade.",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(payload: WineFeatures) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail=load_error or "Model is not loaded")
    try:
        features = metadata["feature_names"]
        values = payload.model_dump()
        api_name_map = {"od280/od315_of_diluted_wines": "od280_od315_of_diluted_wines"}
        row = pd.DataFrame([[values[api_name_map.get(name, name)] for name in features]], columns=features, dtype=float)
        prediction = int(model.predict(row)[0])
        class_names = metadata["class_names"]
        if hasattr(model, "predict_proba"):
            raw_probs = model.predict_proba(row)[0]
            probabilities = {class_names[int(label)]: float(prob) for label, prob in zip(model.classes_, raw_probs)}
        else:
            probabilities = {class_names[prediction]: 1.0}
        return PredictionResponse(
            predicted_class=prediction,
            class_label=class_names[prediction],
            probabilities=probabilities,
            model_name=metadata.get("best_model", "unknown"),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Unable to process provided features: {exc}") from exc
