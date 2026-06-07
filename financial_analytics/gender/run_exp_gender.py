import os, gc, torch, shutil, pandas as pd
from time import time
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, Any, List
from run_models_gender import ModelKeeper
from run_metrics_gender import evaluate_one_emb

def embeddings_cache_path(checkpoints_path: str, hyperparams: Dict[str, Any]) -> str:
    key = (f"{hyperparams['loss']}_{hyperparams['rnn_encoder_type']}"
           f"_bs{hyperparams['batch_size']}_lr{hyperparams['learning_rate']}"
           f"_hid{hyperparams['hidden_size']}_emb{hyperparams['embedding_dim']}"
           f"_cat{hyperparams['category_embedding_dim']}_split{hyperparams['split_count']}"
           f"_cnt{hyperparams['cnt_min']}-{hyperparams['cnt_max']}")
    filename = f"cached_embs_{key}.pkl"
    return os.path.join(checkpoints_path, filename)

def clear_checkpoints_dir(checkpoints_path: str) -> None:
    if not os.path.exists(checkpoints_path): return
    for name in os.listdir(checkpoints_path):
        path = os.path.join(checkpoints_path, name)
        try:
            if os.path.isfile(path): os.remove(path)
            elif os.path.isdir(path): shutil.rmtree(path)
        except Exception as e:
            print(f"{path}: {e}")

def create_truncated_params_grid(fixed_params, variable_params):
    grids = []
    for param_name, values in variable_params.items():
        for value in values:
            grid = {**fixed_params, param_name: value}
            grids.append((param_name, grid))
    return grids

def run_grid_search(all_hyperparameter_grids, sample_fractions, train_data_in, valid_data_in,
                    test_data_in, targets, checkpoints_path, cache_dir, logger,
                    col_id="customer_id", target_col="gender", out_prefix=None,
                    verbose=0, n_samples=10, downstream_type="catboost", devices=0):
    start_time = time()
    all_embeddings = []
    for hyperparam_name, hyperparams in tqdm(all_hyperparameter_grids, desc="Grid search"):
        logger.info(f"{hyperparam_name}: {hyperparams}")
        cache_file = embeddings_cache_path(cache_dir, hyperparams)
        if os.path.exists(cache_file + ".gz"):
            logger.info(f"Cache found: {cache_file}")
            embs = pd.read_pickle(cache_file + ".gz", compression="gzip")
            all_embeddings.extend(embs)
            continue
        torch.cuda.empty_cache()
        gc.collect()
        model_keeper = ModelKeeper()
        model_keeper.create_datasets(
            train_data_in=train_data_in, valid_data_in=valid_data_in,
            hyperparams=hyperparams, col_id=col_id
        )
        model_keeper.train_model(
            hyperparams=hyperparams, checkpoints_path=checkpoints_path, devices=devices
        )
        embs = model_keeper.calc_embs_from_trained(test_data_in)
        all_embeddings.extend(embs)
        try:
            pd.to_pickle(embs, cache_file + ".gz", compression="gzip")
            logger.info(f"Saved to {cache_file}")
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
        clear_checkpoints_dir(checkpoints_path)
    logger.info(f"Grid search completed in {round(time() - start_time, 2)} sec.")
    eval_many_embs(
        embs_list=all_embeddings, targets=targets, col_id=col_id, target_col=target_col,
        out_prefix=out_prefix, sample_fractions=sample_fractions, verbose=verbose,
        n_samples=n_samples, downstream_type=downstream_type
    )

def eval_many_embs(embs_list, targets, col_id="customer_id", target_col="gender",
                   out_prefix=None, sample_fractions=(1/20,), verbose=0,
                   n_samples=10, downstream_type="catboost"):
    results = defaultdict(list)
    for curr_emb in tqdm(embs_list, desc="Evaluating embeddings"):
        res = evaluate_one_emb(
            curr_emb["emb"], targets, sample_fractions=sample_fractions,
            col_id=col_id, target_col=target_col, verbose=verbose,
            n_samples=n_samples, downstream_type=downstream_type
        )
        for metrics in res:
            sample_frac = metrics["sample_fraction"]
            metrics_flat = {k: v for k, v in metrics.items() if k != "sample_fraction"}
            result_row = {**curr_emb["info"], **metrics_flat}
            results[sample_frac].append(result_row)
    for sample_frac, result_list in results.items():
        df = pd.DataFrame(result_list)
        output_csv = f"{out_prefix}_{sample_frac:.3f}".rstrip("0").rstrip(".") + ".csv"
        df.to_csv(output_csv, index=False)
        print(f"Saved: {output_csv}")
