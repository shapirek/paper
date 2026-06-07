import os
import sys
import logging
import glob
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ptls.data_preprocessing import PandasDataPreprocessor

from run_exp_age import create_truncated_params_grid, run_grid_search

OUT_FOLDER = "results"
LOGS_DIR = "logs"
CHECKPOINTS_PATH = "checkpoints"
CACHE_DIR = "cache"

for d in [OUT_FOLDER, LOGS_DIR, CHECKPOINTS_PATH, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

OUT_PREFIX = os.path.join(OUT_FOLDER, "out")

logger = logging.getLogger("age_weighted")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(LOGS_DIR, "experiment.log"))
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.handlers.clear()
logger.addHandler(fh)
logger.info("Start of the experiment")

transactions = pd.read_csv(
    "https://huggingface.co/datasets/dllllb/age-group-prediction/resolve/main/transactions_train.csv.gz?download=true",
    compression="gzip"
)
targets = pd.read_csv(
    "https://huggingface.co/datasets/dllllb/age-group-prediction/resolve/main/train_target.csv?download=true"
)

small_group_in = transactions["small_group"].nunique()
trans_date_in = transactions["trans_date"].nunique()

preprocessor = PandasDataPreprocessor(
    col_id="client_id",
    col_event_time="trans_date",
    event_time_transformation="none",
    cols_category=["small_group"],
    cols_numerical=["amount_rur"],
    return_records=False,
)
transactions = preprocessor.fit_transform(transactions)

train_df, test_df = train_test_split(transactions, test_size=0.1, random_state=42)
train_df, valid_df = train_test_split(train_df, test_size=0.1, random_state=42)

train_dict = train_df.reset_index(drop=True).to_dict("records")
valid_dict = valid_df.reset_index(drop=True).to_dict("records")
test_dict = test_df.reset_index(drop=True).to_dict("records")

fixed_params = {
    "batch_size": 128,
    "learning_rate": 0.001,
    "split_count": 5,
    "cnt_min": 10,
    "cnt_max": 80,
    "embedding_dim": 64,
    "category_embedding_dim": 16,
    "hidden_size": 128,
    "small_group_in": small_group_in,
    "trans_date_in": trans_date_in,
    "num_epochs": 35,
    "loss": "ContrastiveLoss",
    "rnn_encoder_type": "gru",
}

variable_params = {
    "loss": ["ContrastiveLoss", "VicregLoss", "BarlowTwinsLoss"],
    "learning_rate": [0.0003, 0.0005, 0.001, 0.003, 0.005],
    "hidden_size": [128, 192, 256],
    "category_embedding_dim": [16, 24, 32],
}

all_grids = create_truncated_params_grid(fixed_params, variable_params)
print(f"Trained models: {len(all_grids)}")

run_grid_search(
    all_hyperparameter_grids=all_grids,
    sample_fractions=[0.5, 1.0],
    train_data_in=train_dict,
    valid_data_in=valid_dict,
    test_data_in=test_dict,
    targets=targets,
    checkpoints_path=CHECKPOINTS_PATH,
    cache_dir=CACHE_DIR,
    logger=logger,
    col_id="client_id",
    target_col="bins",
    out_prefix=OUT_PREFIX,
    verbose=0,
    n_samples=4,
    downstream_type="catboost",
    devices=[0],
    resume=True,
)

print(Training's over. Look for the results in ", OUT_FOLDER)
