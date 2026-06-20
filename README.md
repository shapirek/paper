# Boundary-Aware Unsupervised Embedding Evaluation via Neighborhood-Overlap Weighting

Official implementation of the CIKM 2026 paper  
**“Boundary-Aware Unsupervised Embedding Evaluation via Neighborhood-Overlap Weighting”**.

## Overview

This repository provides the complete code to reproduce the experiments from the paper.  
The proposed **boundary‑aware distance rescaling** contracts large inter‑cluster gaps while preserving local geometry.  
For each pair of embedded points we compute the Jaccard distance between their \(k\)-NN neighborhoods and apply an additive‑negative contraction to the original Euclidean distance:

```bibtex
\begin{equation*}
d^w(x,y) = d(x,y)\,\bigl(1 - \alpha\,C(x,y)^\beta\bigr),\qquad
C(x,y)=1-\frac{|\mathcal N_k(x)\cap\mathcal N_k(y)|}{|\mathcal N_k(x)\cup\mathcal N_k(y)|}.
\end{equation*}
```

The rescaled distance matrix is then evaluated by three intrinsic metric families:  
- **Persistence** (\(H_0\)) – topological summary  
- **RankMe** – spectral effective rank  
- **SelfCluster** – clustering‑based quality score  

The code covers two application domains:

- **Financial analytics** – Age group and Gender prediction from banking transactions (target‑agnostic setting)
- **Collaborative filtering** – MovieLens‑1M embeddings trained with implicit ALS and Bayesian Personalized Ranking (BPR)

All scripts run hyperparameter grid searches, compute original and boundary‑weighted intrinsic metrics, measure downstream performance, and output CSV results ready for analysis.

## Main Entry Points

### Financial Analytics (Age & Gender Prediction)

```
python main_age.py      # Age prediction experiment
python main_gender.py   # Gender prediction experiment
```

These scripts will:
- Automatically download the datasets from Hugging Face.
- Preprocess transaction sequences (label encoding, train/valid/test split).
- Train CoLES (contrastive event‑sequence) models with various hyperparameters over 35 epochs.
- Compute original and weighted Persistence, RankMe, and SelfCluster.
- Evaluate downstream ROC AUC using CatBoost.
- Save per‑configuration results in `results_age/` and `results_gender/`.

### Collaborative Filtering (MovieLens‑1M)

```
python main_als.py      # iALS experiment (60 models, 1500 test users)
python main_bpr.py      # BPR experiment  (60 models, 500 test users)
```

These scripts:
- Expect `Movielens-1M.csv` in the repository root (see **Data**).
- Perform time‑based train/test split (90/10).
- Iterate over a grid of hyperparameters for ALS (`factors`, `regularization`, `alpha`) or BPR (`factors`, `regularization`, `learning_rate`).
- Extract user and item embeddings, compute recommendation metrics (NDCG@10, Recall@10, HitRate@10, MAP@10).
- Compute original and weighted intrinsic metrics on the embeddings.
- Save results in `ml1m_als_final.csv` and `ml1m_bpr_final.csv`.

## Repository Structure
```
├── main_age.py

├── main_gender.py

├── main_als.py

├── main_bpr.py

├── run_models_age.py # CoLES model, training, inference (Age)

├── run_exp_age.py # Grid search, caching, orchestration (Age)

├── run_metrics_age.py # Intrinsic metrics + downstream eval (Age)

├── run_models_gender.py # CoLES model, training, inference (Gender)

├── run_exp_gender.py # Grid search, caching, orchestration (Gender)

├── run_metrics_gender.py # Intrinsic metrics + downstream eval (Gender)

├── compute_metrics.py # Standalone intrinsic metrics (CF)

├── common.py # CF utilities (data split, ALS/BPR training, ranking metrics)

├── requirements.txt

└── README.md
```

Key modules:
- `run_models_*.py` – model definition, training, and inference for GRU + CoLES framework.
- `run_exp_*.py` – grid search loop, embedding caching, checkpoint management.
- `run_metrics_*.py` – all implementations of intrinsic metrics (Persistence, RankMe, SelfCluster) with boundary weighting, plus downstream evaluation.
- `compute_metrics.py` – the same intrinsic metrics, used by collaborative filtering scripts.
- `common.py` – data loading, label encoding, time‑based split, ALS/BPR training routines, and recommendation metric computation.

## Dependencies

Install the required packages with:

```
pip install -r requirements.txt
```

Main dependencies:
- `pytorch-lightning`
- `pytorch-lifestream`
- `implicit`
- `catboost`
- `scikit-learn`
- `scipy`
- `pandas`
- `numpy`
- `tqdm` (for progress bars)

GPU acceleration is recommended for the financial grid searches; CPU‑only execution is possible but slower.

## Data

**Age & Gender prediction**  
Datasets are downloaded automatically when running `main_age.py` or `main_gender.py` (from Hugging Face). No manual download is needed.

**MovieLens‑1M**  
Place the file `Movielens-1M.csv` in the repository root.  
The file should contain four columns without header: `user_id`, `item_id`, `rating`, `timestamp`.  
You can obtain it from the [MovieLens website](https://grouplens.org/datasets/movielens/1m/).

## Results

Processed experiment outputs are saved in CSV format:
```
| Experiment           | Output File(s)                |
|----------------------|-------------------------------|
| Age prediction       | `results_age/out_*.csv`       |
| Gender prediction    | `results_gender/out_*.csv`    |
| iALS                 | `ml1m_als_final.csv`          |
| BPR                  | `ml1m_bpr_final.csv`          |
```

Each CSV row corresponds to one trained configuration (or one checkpoint) and includes:
- Hyperparameter values
- Downstream performance (ROC AUC for financial; NDCG, Recall, HitRate, MAP for CF)
- Original and boundary‑weighted intrinsic metric values (Persistence, RankMe, SelfCluster, and their weighted variants)

These files can be directly used to compute Spearman correlations between intrinsic metrics and downstream quality, exactly as reported in the paper.

## Citation

If you use this code or the proposed method, please cite:

```bibtex
@inproceedings{boundaryaware2026,
  title     = {Boundary-Aware Unsupervised Embedding Evaluation via
               Neighborhood-Overlap Weighting},
  author    = {Anonymous Author(s)},
  booktitle = {Proceedings of the 35th ACM International Conference on
               Information and Knowledge Management (CIKM)},
  year      = {2026},
  note      = {To appear}
}

