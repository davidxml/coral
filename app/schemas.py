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
    prediction: str = Field(
        ..., 
        description="The classification label, either 'spam' or 'ham'."
    )
    spam_score: float = Field(
        ..., 
        description="The statistical probability (0.0 to 1.0) that the message is spam."
    )
    threshold_used: float = Field(
        ...,
        description="The optimized probability threshold used to determine the final prediction."
    )

