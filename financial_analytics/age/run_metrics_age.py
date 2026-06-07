import gc, sys, numpy as np, pandas as pd, catboost
from time import time
from typing import Dict, Any, List, Optional, Tuple
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage
from scipy.linalg import eigvalsh as scipy_eigvalsh
from sklearn.utils import resample

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def local_dim_mle(pts, k=10):
    N = pts.shape[0]
    k = min(k, N-1) if N > 1 else 1
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(pts)
    dist, _ = nbrs.kneighbors(pts)
    r_k = dist[:, -1]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = (r_k[:, None] + 1e-12) / (dist[:, 1:-1] + 1e-12)
        log_ratios = np.log(ratio)
        d_mle = 1.0 / (np.mean(log_ratios, axis=1) + 1e-12)
    d_mle[np.isinf(d_mle)] = np.nan
    return d_mle

def compute_scalar_0d_persistence(condensed_dist, d_std_orig=None, norm_quantile=0.9):
    if len(condensed_dist) == 0:
        return 0.0
    finite = np.isfinite(condensed_dist)
    if not finite.all():
        max_fin = np.max(condensed_dist[finite]) if finite.any() else 1.0
        condensed_dist = np.where(finite, condensed_dist, max_fin)
    Z = linkage(condensed_dist, method='single')
    lifetimes = Z[:, 2]
    total = np.sum(lifetimes)
    if d_std_orig is not None:
        norm_val = np.quantile(d_std_orig, norm_quantile)
    else:
        norm_val = np.max(condensed_dist)
    if norm_val < 1e-12 or np.isnan(total) or np.isnan(norm_val):
        return 0.0
    return total / norm_val

def apply_weight(wt, C, alpha=None, lam=None, gamma=None, normalize=True):
    eps = 1e-12
    C_safe = np.maximum(C, 0.0)
    if wt == 'add_pos':
        omega = 1.0 + alpha * C_safe
    elif wt == 'add_neg':
        omega = np.maximum(eps, 1.0 - alpha * C_safe)
    elif wt == 'inv_pos':
        omega = 1.0 + alpha / np.maximum(C_safe, eps)
    elif wt == 'inv_neg':
        omega = np.maximum(eps, 1.0 - alpha / np.maximum(C_safe, eps))
    elif wt == 'exp':
        omega = lam * np.exp(np.clip(C_safe, 0, 80) ** gamma)
    else:
        omega = np.ones_like(C_safe)
    omega = np.clip(omega, 0, 1e10)
    if normalize:
        mean_omega = np.mean(omega)
        if mean_omega > 1e-15:
            omega /= mean_omega
    return omega

def _compute_rankme(X, epsilon=1e-12):
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    p = s / (np.sum(s) + epsilon) + epsilon
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy)

def _compute_self_clustering(embeddings, G=None):
    if G is None:
        eps = 1e-12
        norm_emb = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + eps)
        G = norm_emb @ norm_emb.T
    n, d = embeddings.shape
    G_sq = G ** 2
    actual = np.sum(G_sq) - n
    expected = n * (n - 1) / d
    return (actual - expected) / (n * n - expected)

def _add_weighted_metrics(results, wname, wt, embeddings, alpha=None, lam=None, gamma=None,
                          d_std=None, omega=None, G=None, norm_quantile=0.9):
    if alpha is not None:
        param_str = f'alpha{alpha}'
    else:
        param_str = f'lam{lam}_gam{gamma}'
    base_key = f'{wname}_{wt}_{param_str}'

    # persistence
    wdist = d_std * omega
    if np.isnan(wdist).any() or np.ptp(wdist[np.isfinite(wdist)]) < 1e-12 or np.max(wdist[np.isfinite(wdist)]) < 1e-12:
        pers = compute_scalar_0d_persistence(d_std, d_std_orig=d_std, norm_quantile=norm_quantile)
    else:
        pers = compute_scalar_0d_persistence(wdist, d_std_orig=d_std, norm_quantile=norm_quantile)
    results[f'persistence_{base_key}'] = 0.0 if np.isnan(pers) else pers

    # rankme
    omega_sq = squareform(omega)
    omega_sq = np.nan_to_num(omega_sq, nan=0.0, posinf=1e4, neginf=0.0)
    G_w = G * omega_sq
    G_w = (G_w + G_w.T) / 2
    eigvals = scipy_eigvalsh(G_w)
    eigvals = np.maximum(eigvals, 1e-12)
    sum_eig = np.sum(eigvals)
    if sum_eig > 1e-12:
        p = eigvals / sum_eig
        rankme_val = np.exp(-np.sum(p * np.log(p + 1e-12)))
    else:
        rankme_val = 1.0
    results[f'rankme_{base_key}'] = 1.0 if np.isnan(rankme_val) else rankme_val

    # self_clustering
    G_sq = G ** 2
    actual = np.sum(G_sq * omega_sq) - np.sum(omega_sq.diagonal())
    n = G.shape[0]
    d = embeddings.shape[1] if len(embeddings.shape) > 1 else 1
    expected = n * (n - 1) / d if d > 0 else 1.0
    denom = n * n - expected
    if abs(denom) > 1e-12:
        sc_val = (actual - expected) / denom
    else:
        sc_val = 0.0
    results[f'self_clustering_{base_key}'] = 0.0 if np.isnan(sc_val) else sc_val


def all_weighted_metrics(embeddings, K=10,
                         alphas=[0.5,1.0,1.5,2.0],
                         lambdas=[0.5,1.0,1.5,2.0],
                         gammas=[0.5,1.0],
                         weight_types=None,
                         norm_quantile=0.9):
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1e4, neginf=-1e4)
    if weight_types is None:
        weight_types = ['add_pos','add_neg','inv_pos','inv_neg','exp']
    N, D = embeddings.shape
    if N < 2:
        return {'persistence_std': 0.0, 'rankme': 1.0, 'self_clustering_std': 0.0}
    eps = 1e-12

    d_std = pdist(embeddings).astype(np.float32)
    finite_mask = np.isfinite(d_std)
    if not finite_mask.all():
        max_fin = np.max(d_std[finite_mask]) if finite_mask.any() else 1.0
        d_std = np.where(finite_mask, d_std, max_fin)

    if N >= D:
        try:
            kde = gaussian_kde(embeddings.T)
            rho = kde(embeddings.T)
            rho = np.nan_to_num(rho, nan=1e-12, posinf=1e12, neginf=1e-12)
            rho = np.clip(rho, 1e-12, None) + 1e-12
        except np.linalg.LinAlgError:
            rho = np.full(N, 1e-12)
    else:
        rho = np.full(N, 1e-12)
    log_rho = np.log(rho)

    dim_est = local_dim_mle(embeddings, K)
    if np.isnan(dim_est).all():
        mean_dim = 1e-6
    else:
        mean_dim = np.nanmean(dim_est)
    dim_est = np.nan_to_num(dim_est, nan=mean_dim)
    nbrs = NearestNeighbors(n_neighbors=K).fit(embeddings)
    _, idx = nbrs.kneighbors(embeddings)
    psi = np.zeros(N)
    for i in range(N):
        neigh = dim_est[idx[i]]
        psi[i] = np.mean(np.abs(dim_est[i] - neigh)) / (np.mean(neigh) + 1e-6)

    q_low, q_high = np.quantile(d_std, [0.1, 0.3])
    q_low = max(q_low, eps)
    q_high = max(q_high, q_low + eps)
    tree = NearestNeighbors(radius=q_high).fit(embeddings)
    c_low = tree.radius_neighbors(embeddings, radius=q_low, return_distance=False)
    c_high = tree.radius_neighbors(embeddings, radius=q_high, return_distance=False)
    v_rate = np.array([np.log(max(len(c_high[i]),1) / max(len(c_low[i]),1))
                       / np.log(q_high/q_low) for i in range(N)])
    v_rate = np.nan_to_num(v_rate, posinf=0.0, neginf=0.0)

    neighbor_sets = [set(idx[i]) for i in range(N)]

    tri_i, tri_j = np.triu_indices(N, k=1)
    C_density = np.abs(log_rho[tri_i] - log_rho[tri_j])
    C_dimension = np.maximum(psi[tri_i], psi[tri_j])
    C_volume = np.abs(v_rate[tri_i] - v_rate[tri_j])
    C_overlap = np.empty(len(tri_i), dtype=np.float32)
    for k, (i, j) in enumerate(zip(tri_i, tri_j)):
        inter = len(neighbor_sets[i] & neighbor_sets[j])
        union = len(neighbor_sets[i] | neighbor_sets[j])
        C_overlap[k] = 1.0 - (inter / union) if union > 0 else 1.0

    norm_emb = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + eps)
    G = norm_emb @ norm_emb.T

    results = {}
    results['persistence_std'] = compute_scalar_0d_persistence(d_std, d_std_orig=d_std, norm_quantile=norm_quantile)
    results['rankme'] = _compute_rankme(embeddings)
    results['self_clustering_std'] = _compute_self_clustering(embeddings, G)

    C_dict = {'density': C_density, 'dimension': C_dimension,
              'volume': C_volume, 'overlap': C_overlap}

    for wname, C in C_dict.items():
        for wt in weight_types:
            if wt in ['add_pos','add_neg','inv_pos','inv_neg']:
                for alpha in alphas:
                    omega = apply_weight(wt, C, alpha, None, None)
                    _add_weighted_metrics(results, wname, wt, embeddings, alpha=alpha,
                                          d_std=d_std, omega=omega, G=G, norm_quantile=norm_quantile)
            else:  # exp
                for lam in lambdas:
                    for gamma in gammas:
                        omega = apply_weight(wt, C, None, lam, gamma)
                        _add_weighted_metrics(results, wname, wt, embeddings, lam=lam, gamma=gamma,
                                              d_std=d_std, omega=omega, G=G, norm_quantile=norm_quantile)

    for wname in ['density','dimension','volume','overlap']:
        if wname == 'density':       v = rho
        elif wname == 'dimension':   v = np.exp(-dim_est)
        elif wname == 'volume':      v = np.abs(v_rate) + eps
        elif wname == 'overlap':     continue
        v = v / np.sum(v) * N
        Wsqrt = np.sqrt(v)
        Xw = embeddings * Wsqrt[:, np.newaxis]
        results[f'rankme_{wname}'] = _compute_rankme(Xw)

    return results

def compute_metrics(embeddings, selected_metrics=None, n_samples=10, sample_fraction=1/20, verbose=0):
    N = embeddings.shape[0]
    if N < 2:
        return {}
    if sample_fraction >= 1.0 or n_samples <= 1:
        res = all_weighted_metrics(embeddings)
        return {'metric_' + k: (0.0 if np.isnan(v) else v) for k, v in res.items()}
    sample_size = max(2, int(sample_fraction * N))
    metrics_accum = {}
    valid_samples = 0
    for i in range(n_samples):
        sample = resample(embeddings, n_samples=sample_size, replace=False, random_state=42 + i)
        if sample.shape[0] < 2:
            continue
        res = all_weighted_metrics(sample)
        for k, v in res.items():
            metrics_accum.setdefault(k, []).append(v)
        valid_samples += 1
    if valid_samples == 0:
        return {}
    result = {}
    for k, vlist in metrics_accum.items():
        arr = np.array(vlist, dtype=np.float64)
        mean_val = np.nanmean(arr)
        result['metric_' + k] = 0.0 if np.isnan(mean_val) else mean_val
    return result

def eval_downstream(inf_test_embeddings, targets, col_id="client_id", target_col="bins", downstream_type="catboost"):
    targets_df = targets.set_index(col_id)
    merged = inf_test_embeddings.merge(targets_df, how="inner", on=col_id).set_index(col_id)
    X = merged.drop(columns=[target_col]).values
    y = merged[target_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    if downstream_type == "catboost":
        model = catboost.CatBoostClassifier(iterations=150, loss_function='MultiClass', random_seed=42, verbose=0)
    else:
        raise ValueError(f"Unknown downstream_type: {downstream_type}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return accuracy_score(y_test, y_pred), roc_auc_score(y_test, y_proba, multi_class="ovo", average="macro"), X_train, X_test

def evaluate_one_emb(inf_test_embeddings, targets, selected_metrics=None, sample_fractions=(1/20,),
                     col_id="client_id", target_col="bins", verbose=0, n_samples=10,
                     downstream_type="catboost"):
    embeddings_np = inf_test_embeddings.drop(columns=[col_id]).to_numpy(dtype=np.float32)
    accuracy, auc, X_train, _ = eval_downstream(inf_test_embeddings, targets, col_id, target_col, downstream_type)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e4, neginf=-1e4)
    results = []
    for fraction in sample_fractions:
        metrics = compute_metrics(X_train, selected_metrics, n_samples, fraction, verbose)
        metrics.update({"accuracy": accuracy, "roc_auc": auc, "sample_fraction": fraction})
        results.append(metrics)
    return results
