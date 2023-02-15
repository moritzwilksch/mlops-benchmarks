from fastapi import FastAPI
from inference.run_bm import pyfun_joblib, pyfun_onnx
from pydantic import BaseModel

app = FastAPI()

# run with:
# uvicorn --log-level warning --app-dir src/restapi api:app


class InputData(BaseModel):
    total_bill: float
    sex: str
    smoker: str
    day: str
    time: str
    size: int


@app.post("/predict-joblib")
async def predict_joblib(d: InputData) -> dict:
    return {"result": pyfun_joblib(d.__dict__).item()}


@app.post("/predict-onnx")
async def predict_onnx(d: InputData) -> dict:
    return {"result": pyfun_onnx(d.__dict__).item()}
