import contextlib
import tensorflow as tf

from fastapi import FastAPI, Request, HTTPException
from tensorflow import keras

from schemas import MessagePayload

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the state variable safely
    app.state.ml_model = None
    
    try:
        # Passes the string path directly
        app.state.ml_model = keras.models.load_model('../training/spam_dense_model.keras')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")

    yield 
    # Clean up memory on shutdown
    app.state.ml_model = None

app = FastAPI(
    title='Spam Classifier',
    description='A Micro-service for text classification into "ham" or "spam"',
    version='1.0.0',
    lifespan=lifespan
)

@app.post('/predict')
async def classify(features: MessagePayload, request: Request):
    # Safely extract the model from the app state
    model = request.app.state.ml_model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model engine is unavailable.")
    
    # Extract the string from the instance, not the class
    raw_text = features.message

    # Wrap the text in a list to satisfy TensorFlow's batch requirement
    # Use verbose=0 to prevent terminal spam on every request
    tensor_input = tf.constant([raw_text])
    prediction_array = model.predict(tensor_input, verbose=0)
    
    # Extract the actual float value from the nested matrix output
    probability = float(prediction_array[0][0])
    label = "spam" if probability > 0.5 else "ham"

    # Return a clean JSON structure
    return {
        "text": raw_text,
        "label": label,
        "probability": probability
    }