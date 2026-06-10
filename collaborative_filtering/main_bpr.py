import os
import gc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import ParameterGrid

from common import encode_column, train_test_split, train_and_evaluate

OUTPUT_CSV = "ml1m_bpr_final.csv"

df = pd.read_csv("Movielens-1M.csv")
df.columns = ["user_id", "item_id", "rating", "timestamp"]

df, _ = encode_column(df, col="item_id", new_col="item_id")

train, test = train_test_split(df, quantile=0.9)

train_csr = csr_matrix(
    (train["rating"], (train["user_id"], train["item_id"]))
)

PARAM_GRID = {
    "factors": [16, 32, 64, 128, 256],
    "regularization": [0.001, 0.01, 0.1],
    "learning_rate": [1e-3, 3e-3, 1e-2, 3e-2],
}
grid = list(ParameterGrid(PARAM_GRID))
print(f"Всего моделей: {len(grid)}")

results = []
for i, params in enumerate(grid):
    try:
        res = train_and_evaluate(
            model_type="bpr",
            train_csr=train_csr,
            test=test,
            params=params,
            N=10,
            batch_size=1000,
            n_test_users=500,
        )
        results.append(res)
    except Exception as e:
        print(f"Error with parameters {params}: {e}")

    if (i + 1) % 10 == 0:
        print(f"Progress: {i + 1}/{len(grid)}")
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    gc.collect()

df_res = pd.DataFrame(results)
df_res.to_csv(OUTPUT_CSV, index=False)
print(f"Results saved ib {OUTPUT_CSV}")
