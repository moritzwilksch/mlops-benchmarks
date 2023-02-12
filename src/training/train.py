# %%
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

# %%

df = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
)
X = df.drop("tip", axis=1)
y = df["tip"]

# %%
model = Pipeline(
    [
        (
            "ct",
            ColumnTransformer(
                [("oe", OrdinalEncoder(), ["sex", "smoker", "day", "time"])]
            ),
        ),
        (
            "model",
            HistGradientBoostingRegressor(),
        ),
    ]
)

# %%
model.fit(X, y)

# %%
print({model.score(X, y)})
