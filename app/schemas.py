from pydantic import BaseModel,Field, field_validator

class PredictionRequest(BaseModel):
    text: str = Field(..., example="Congratulations! You've won a free $1,000 gift card. Click here.")

    @field_validator("text")
    @classmethod
    def validate_text(cls, value):
        if not value.strip():
            raise ValueError("Text must contain readable characters.")
        return value
    
class PredictionResponse(BaseModel):
    prediction: str
    confidence_score: float

