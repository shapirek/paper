import os
import sys
import gc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import ParameterGrid

from common import encode_column, train_test_split, train_and_evaluate

OUTPUT_CSV = "ml1m_als_final.csv"
N_TEST_USERS = 1500

df = pd.read_csv("Movielens-1M.csv")
df.columns = ["user_id", "item_id", "rating", "timestamp"]

df, _ = encode_column(df, col="item_id", new_col="item_id")

train, test = train_test_split(df, quantile=0.9)

train_csr = csr_matrix(
    (train["rating"], (train["user_id"], train["item_id"]))
)

PARAM_GRID = {
    "factors": [16, 32, 64, 128, 256],
    "regularization": [0.01, 0.1, 1],
    "alpha": [0.01, 0.1, 0.5, 1.0],
}
grid = list(ParameterGrid(PARAM_GRID))
print(f"Number of models: {len(grid)}")

results = []
for i, params in enumerate(grid):
    try:
        res = train_and_evaluate(
            model_type="als",
            train_csr=train_csr,
            test=test,
            params=params,
            N=10,
            batch_size=1000,
            n_test_users=N_TEST_USERS,
        )
        results.append(res)
    except Exception as e:
        print(f"Error with parameters {params}: {e}")

    if (i + 1) % 10 == 0:
        print(f"Прогресс: {i + 1}/{len(grid)}")
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    gc.collect()

df_res = pd.DataFrame(results)
df_res.to_csv(OUTPUT_CSV, index=False)
print(f"Results saved in {OUTPUT_CSV}")
