# %%
import joblib
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    DoubleTensorType,
    Int64TensorType,
    StringTensorType,
)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# %%

df = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
)
X = df.drop("tip", axis=1)
y = df["tip"]

# %%
model = Pipeline(
    [
        (
            "ct",
            ColumnTransformer(
                [("oe", OrdinalEncoder(), ["sex", "smoker", "day", "time"])],
            ),
        ),
        (
            "model",
            HistGradientBoostingRegressor(),
        ),
    ],
)

# %%
model.fit(X, y)

# %%

# %%
joblib.dump(model, "artifacts/model.joblib")

# %%

initial_type = [
    ("total_bill", DoubleTensorType([None, 1])),
    ("sex", StringTensorType([None, 1])),
    ("smoker", StringTensorType([None, 1])),
    ("day", StringTensorType([None, 1])),
    ("time", StringTensorType([None, 1])),
    ("size", Int64TensorType([None, 1])),
]
onx = convert_sklearn(model, initial_types=initial_type)
with open("artifacts/model.onnx", "wb") as f:
    f.write(onx.SerializeToString())

# %%
