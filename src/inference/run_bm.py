# %%

import json
import time
from collections.abc import Callable

import joblib
import numpy as np
import onnxruntime as rt
import pandas as pd
import requests
from loguru import logger
from sklearn.pipeline import Pipeline

logger.add(sink="logs/logs.txt")

model: Pipeline = joblib.load("artifacts/model.joblib")
data_point = {
    "total_bill": 16.99,
    "sex": "Female",
    "smoker": "No",
    "day": "Sun",
    "time": "Dinner",
    "size": 2,
}

so = rt.SessionOptions()
so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

onnx_sess = rt.InferenceSession("artifacts/model.onnx", sess_options=so)
input_names = [ii.name for ii in onnx_sess.get_inputs()]
label_name = onnx_sess.get_outputs()[0].name


# %%
def pyfun_joblib(d: dict) -> np.ndarray:
    df = pd.DataFrame({k: [v] for k, v in d.items()})
    return model.predict(df)


def pyfun_onnx(d: dict) -> np.ndarray:
    return onnx_sess.run(
        [label_name],
        {name: np.array([d.get(name)]).reshape(1, -1) for name in input_names},
    )[0]


def restapi_joblib(d: dict) -> requests.Response:
    return requests.post(
        "http://localhost:8000/predict-joblib",
        json.dumps(d),
        headers={"ContentType": "application/json"},
        timeout=1,
    )


def restapi_onnx(d: dict) -> requests.Response:
    return requests.post(
        "http://localhost:8000/predict-onnx",
        json.dumps(d),
        headers={"ContentType": "application/json"},
        timeout=1,
    )


def run(func: Callable, n: int = 10_000) -> float:
    start = time.perf_counter()

    for _i in range(n):
        _ = func(data_point)

    end = time.perf_counter()

    return end - start


# %%
if __name__ == "__main__":
    USE_FUNCTION = restapi_onnx

    time_taken = run(USE_FUNCTION)
    logger.info(f"{USE_FUNCTION.__name__} took {time_taken:.4f} seconds.")
