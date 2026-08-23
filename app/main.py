from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.schemas import PredictionRequest, PredictionResponse
import app.preprocess
import joblib
import os
import json

vectorizer = None
model = None
THRESHOLD = None
SPAM_CLASS_INDEX = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Executes once when Uvicorn boots up. Loads the Scikit-Learn artifacts
    from disk into memory so inference is instantaneous.
    """
    global vectorizer, model, THRESHOLD, SPAM_CLASS_INDEX

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vectorizer_path = os.path.join(base_dir, "artifacts", "models", "tfidf_vectorizer.joblib")
    model_path = os.path.join(base_dir, "artifacts", "models", "logistic_regression_model.joblib")
    threshold_path = os.path.join(base_dir, "artifacts", "models", "threshold.json")

    if not os.path.exists(vectorizer_path) or not os.path.exists(model_path) or not os.path.exists(threshold_path):
        raise RuntimeError(
            "Machine learning artifacts not found in the /models directory. "
            "Please run `python train.py` to generate them before starting the server."
        )

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)

    with open(threshold_path) as f:
        threshold_data = json.load(f)
        THRESHOLD = threshold_data["threshold"]
        SPAM_CLASS_INDEX = threshold_data["spam_class_index"]

    print("CORAL Engine: Scikit-Learn artifacts loaded into memory.")
    print(f"CORAL Engine: Using threshold={THRESHOLD}, spam_class_index={SPAM_CLASS_INDEX}")

    yield
    vectorizer = None
    model = None
    THRESHOLD = None
    SPAM_CLASS_INDEX = None


app = FastAPI(
    title="CORAL Spam Detection API",
    description="Real-time text classification microservice for spam detection.",
    version="3.0.0",
    lifespan=lifespan,
)

@app.get("/health")
def health_check():
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Machine learning model is not loaded.")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def classify_text(payload: PredictionRequest):
    """
    Receives text, vectorizes it, and returns a unified spam score
    (0 = confidently ham, 1 = confidently spam) plus a label derived
    from the current decision threshold.
    """
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Machine learning model is not loaded.")

    try:
        text_matrix = vectorizer.transform([payload.text])

        # 2. Get P(spam) directly - this IS the unified spam score
        probability_array = model.predict_proba(text_matrix)[0]
        spam_score = float(probability_array[SPAM_CLASS_INDEX])

        # 3. Derive the label using the tuned threshold
        predicted_label = "spam" if spam_score >= THRESHOLD else "ham"

        return PredictionResponse(
            prediction=predicted_label,
            spam_score=round(spam_score, 4),
            threshold_used=THRESHOLD,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")