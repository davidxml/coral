from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.schemas import PredictionRequest, PredictionResponse
import joblib
import os
import json

vectorizer = None
model = None
THRESHOLD = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Executes once when Uvicorn boots up. Loads the Scikit-Learn artifacts
    from disk into memory so inference is instantaneous.
    """
    global vectorizer, model, THRESHOLD

    # Resolve paths relative to where Uvicorn is executed
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vectorizer_path = os.path.join(base_dir, "artifacts", "models", "tfidf_vectorizer.joblib")
    model_path = os.path.join(base_dir, "artifacts", "models", "naive_bayes_model.joblib")
    threshold_path = os.path.join(base_dir, "artifacts", "models", "threshold.json")
                                   
    if not os.path.exists(vectorizer_path) or not os.path.exists(model_path) or not os.path.exists(threshold_path): 
        raise RuntimeError(
            "Machine learning artifacts not found in the /models directory. "
            "Please run `python train.py` to generate them before starting the server."
        )

    with open(threshold_path) as f:
        threshold = json.load(f)["Threshold"]

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    THRESHOLD = threshold 
    print("CORAL Engine: Scikit-Learn artifacts loaded into memory.")

    yield 
    
    vectorizer = None
    model = None
    THRESHOLD = None

app = FastAPI(
    title="CORAL",
    description="Real-time text classification microservice for spam detection.",
    version="2.0.0",
    lifespan = lifespan,
)


@app.post("/predict", response_model=PredictionResponse)
def classify_text(payload: PredictionRequest):
    """
    Receives text, vectorizes it, and returns the Naive Bayes probability prediction.
    """
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Machine learning model is not loaded.")

    try:
        # 1. Transform the raw English text into the TF-IDF statistical matrix
        # (Must pass as a list because the vectorizer expects an iterable of documents)
        text_matrix = vectorizer.transform([payload.text])

        # 2. Extract the categorical prediction ('ham' or 'spam')
    
        # 3. Extract the raw statistical probability (confidence score)
        # predict_proba returns a 2D array: [[prob_class_0, prob_class_1]]
        probability_array = model.predict_proba(text_matrix)[0]

        # Scikit-Learn orders classes alphabetically: 'ham' is index 0, 'spam' is index 1.
        # We grab the probability corresponding to the predicted label.
        predicted_label = "spam" if probability_array[1] >= THRESHOLD else "ham"
        confidence_idx = 1 if predicted_label == 'spam' else 0
        confidence_score = float(probability_array[confidence_idx])

        # 4. Return the exact JSON schema defined in the README
        return PredictionResponse(
            prediction=predicted_label,
            confidence_score=round(confidence_score, 4)
        )

    except Exception as e:
        # If the math engine crashes, return a clean 500 error, not a stack trace to the user.
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")  
