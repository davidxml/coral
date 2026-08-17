import contextlib

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from tensorflow import keras

class MessagePayload(BaseModel):
    message: str      # The client is exxpected to send a single message

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):

    global ml_model

    try:
        with open('../training/spam_dense_model.keras', 'rb') as file:
            ml_model = keras.model.load_model(file)

    except FileNotFoundError:
        print('Model file not present at location')

    yield 

app = FastAPI(
    title = 'Spam Classifier',
    description = 'A Micro-service for text classification into "ham" or "Spam" ',
    version = '1.0.0',
    lifespan = lifespan
)
@app.post('/predict', response_model=str)
def classify(Features: MessagePayload):
    if ml_model is None:
        return {'error': 'model not found '}
    
    data =  MessagePayload.message

    prediction = ml_model.predict()