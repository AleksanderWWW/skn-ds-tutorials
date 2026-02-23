from fastapi.responses import FileResponse
import joblib
import pandas as pd

from fastapi import FastAPI

from pydantic import BaseModel


app = FastAPI(title="SKN Data Science Model Server")

model = joblib.load("model.joblib")


class PredictionRequest(BaseModel):
    bmi: float

class PredictionResponse(BaseModel):
    risk_score: float


@app.post("/predict")
async def predict(to_predict: PredictionRequest) -> PredictionResponse:
    """Estimate 1-year diabetes progression based on BMI"""
    input_data = pd.DataFrame([[to_predict.bmi]], columns=['bmi'])
    
    prediction = model.predict(input_data)
    
    return PredictionResponse(risk_score=float(prediction[0]))


@app.get("/")
async def read_index() -> FileResponse:
    return FileResponse('index.html')
