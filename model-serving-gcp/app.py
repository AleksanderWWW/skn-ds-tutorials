from fastapi.responses import FileResponse
import joblib
import pandas as pd

from fastapi import FastAPI

from pydantic import BaseModel


app = FastAPI(title="SKN Data Science Model Server")

model = joblib.load("model.joblib")


class PredictionRequest(BaseModel):
    bmi: float
    mars_retro: bool

class PredictionResponse(BaseModel):
    risk_score: float


@app.post("/predict")
async def predict(to_predict: PredictionRequest) -> PredictionResponse:
    """Estimate 1-year diabetes progression based on BMI and planetary alignment"""
    input_data = pd.DataFrame([[to_predict.bmi]], columns=['bmi'])

    prediction = model.predict(input_data)
    score = float(prediction[0])

    if to_predict.mars_retro:
        score = score * 2.0

    return PredictionResponse(risk_score=score)


@app.get("/")
async def read_index() -> FileResponse:
    return FileResponse('index.html')
