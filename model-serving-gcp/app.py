from fastapi import FastAPI

from pydantic import BaseModel


app = FastAPI(title="SKN Data Science Model Server")


class PredictionRequest(BaseModel):
    data: int


class PredictionResponse(BaseModel):
    result: float


@app.post("/predict")
async def predict(pred_data: PredictionRequest) -> PredictionResponse:
    return PredictionResponse(result=pred_data.data * 0.1 + 7)
