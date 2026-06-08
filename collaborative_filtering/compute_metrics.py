import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage
from collections import defaultdict

def local_dim_mle(pts, k=10):
    N = pts.shape[0]
    k = min(k, N-1) if N > 1 else 1
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(pts)
    dist, _ = nbrs.kneighbors(pts)
    r_k = dist[:, -1]
    with np.errstate(divide='ignore', invalid='ignore'):
        log_ratios = np.log(r_k[:, None] / dist[:, 1:-1])
        d_mle = 1.0 / np.mean(log_ratios, axis=1)
    d_mle[np.isinf(d_mle)] = np.nan
    return d_mle

def compute_scalar_0d_persistence(condensed_dist, d_std_orig=None, norm_quantile=0.9):
    finite = np.isfinite(condensed_dist)
    if not finite.all():
        max_fin = np.max(condensed_dist[finite]) if finite.any() else 1.0
        condensed_dist = np.where(finite, condensed_dist, max_fin)
    Z = linkage(condensed_dist, method='single')
    lifetimes = Z[:, 2]
    total = np.sum(lifetimes)
    if d_std_orig is not None:
        norm_value = np.quantile(d_std_orig, norm_quantile)
    else:
        norm_value = np.max(condensed_dist)
    if norm_value == 0:
        return 0.0
    return total / norm_value

def rankme(tensor, s=None, epsilon=1e-12):
    if s is None:
        s = np.linalg.svd(tensor, compute_uv=False)
    p = s / (np.sum(s) + epsilon) + epsilon
    return np.exp(-np.sum(p * np.log(p)))

def self_clustering(tensor, epsilon=1e-12):
    tensor = tensor + epsilon
    tensor /= np.linalg.norm(tensor, axis=1, keepdims=True)
    n, d = tensor.shape
    expected = n + n * (n - 1) / d
    actual = np.sum(np.square(tensor @ tensor.T))
    return (actual - expected) / (n * n - expected)

def apply_weight(wt, C, alpha, lam, gamma, normalize=False):
    eps = 1e-12
    if wt == 'add_pos':
        omega = 1.0 + alpha * C
    elif wt == 'add_neg':
        omega = np.maximum(eps, 1.0 - alpha * C)
    elif wt == 'inv_pos':
        omega = 1.0 + alpha / np.maximum(C, eps)
    elif wt == 'inv_neg':
        omega = np.maximum(eps, 1.0 - alpha / np.maximum(C, eps))
    elif wt == 'exp':
        omega = lam * np.exp(np.clip(C, -80, 80) ** gamma)
    else:
        omega = np.ones_like(C)
    omega = np.clip(omega, 0, 1e10)
    if normalize:
        mean_omega = np.mean(omega)
        if mean_omega > 1e-8:
            omega /= mean_omega
    return omega

def all_weighted_metrics(embeddings, u=None, s=None, K=10,
                         alphas=[0.5,1.0,1.5,2.0],
                         lambdas=[0.5,1.0,1.5,2.0],
                         gammas=[0.5,1.0],
                         weight_types=None,
                         normalize_weights=False,
                         norm_quantile=0.9):
    if weight_types is None:
        weight_types = ['add_pos','add_neg','inv_pos','inv_neg','exp']
    N, D = embeddings.shape

    d_std = pdist(embeddings).astype(np.float32)
    finite_mask = np.isfinite(d_std)
    if not finite_mask.all():
        max_fin = np.max(d_std[finite_mask]) if finite_mask.any() else 1.0
        d_std = np.where(finite_mask, d_std, max_fin)

    if N >= D:
        try:
            kde = gaussian_kde(embeddings.T)
            rho = kde(embeddings.T) + 1e-12
        except np.linalg.LinAlgError:
            rho = np.ones(N) * 1e-12
    else:
        rho = np.ones(N) * 1e-12
    log_rho = np.log(rho)

    dim_est = local_dim_mle(embeddings, K)
    dim_est = np.nan_to_num(dim_est, nan=np.nanmean(dim_est))
    nbrs = NearestNeighbors(n_neighbors=K).fit(embeddings)
    _, idx = nbrs.kneighbors(embeddings)
    psi = np.zeros(N)
    for i in range(N):
        neigh = dim_est[idx[i]]
        psi[i] = np.mean(np.abs(dim_est[i] - neigh)) / (np.mean(neigh) + 1e-6)

    eps1, eps2 = np.quantile(d_std, [0.1, 0.3])
    eps1, eps2 = max(eps1,1e-8), max(eps2,eps1+1e-8)
    tree = NearestNeighbors(radius=eps2).fit(embeddings)
    c1 = tree.radius_neighbors(embeddings, radius=eps1, return_distance=False)
    c2 = tree.radius_neighbors(embeddings, radius=eps2, return_distance=False)
    v_rate = np.array([np.log(len(c2[i]) / max(1, len(c1[i]))) / np.log(eps2/eps1)
                       for i in range(N)])
    v_rate = np.nan_to_num(v_rate, posinf=0.0, neginf=0.0)
    neighbor_sets = [set(idx[i]) for i in range(N)]

    M = N * (N - 1) // 2
    C_density = np.empty(M, dtype=np.float32)
    C_dimension = np.empty(M, dtype=np.float32)
    C_volume = np.empty(M, dtype=np.float32)
    C_overlap = np.empty(M, dtype=np.float32)
    idx_cond = 0
    for i in range(N):
        row_log_rho = log_rho[i]
        row_psi = psi[i]
        row_vrate = v_rate[i]
        row_set = neighbor_sets[i]
        for j in range(i+1, N):
            C_density[idx_cond] = np.abs(row_log_rho - log_rho[j])
            C_dimension[idx_cond] = max(row_psi, psi[j])
            C_volume[idx_cond] = np.abs(row_vrate - v_rate[j])
            inter = len(row_set & neighbor_sets[j])
            union = len(row_set | neighbor_sets[j])
            C_overlap[idx_cond] = 1.0 - (inter / union) if union > 0 else 1.0
            idx_cond += 1

    eps = 1e-12
    norm_emb = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + eps)
    G = norm_emb @ norm_emb.T

    results = {}
    results['persistence_std'] = compute_scalar_0d_persistence(d_std, d_std_orig=d_std, norm_quantile=norm_quantile)
    results['rankme'] = rankme(embeddings, s=s)
    results['self_clustering_std'] = self_clustering(embeddings)

    weight_names = ['density','dimension','volume','overlap']
    C_dict = {'density': C_density, 'dimension': C_dimension,
              'volume': C_volume, 'overlap': C_overlap}

    for wname in weight_names:
        C = C_dict[wname]
        for wt in weight_types:
            if wt in ['add_pos','add_neg','inv_pos','inv_neg']:
                for alpha in alphas:
                    omega = apply_weight(wt, C, alpha, None, None, normalize_weights)
                    wdist = d_std * omega
                    pers = compute_scalar_0d_persistence(wdist, d_std_orig=d_std, norm_quantile=norm_quantile)
                    results[f'persistence_{wname}_{wt}_alpha{alpha}'] = pers

                    omega_sq = squareform(omega)
                    G_sq = G ** 2
                    actual = np.sum(G_sq * omega_sq)
                    n, d = embeddings.shape
                    expected = n + n * (n - 1) / d
                    sc_val = (actual - expected) / (n * n - expected)
                    results[f'self_clustering_{wname}_{wt}_alpha{alpha}'] = sc_val

                    G_w = G * omega_sq
                    G_w = (G_w + G_w.T) / 2
                    eigvals = np.linalg.eigvalsh(G_w)
                    eigvals = np.maximum(eigvals, 1e-12)
                    p = eigvals / np.sum(eigvals)
                    rk = np.exp(-np.sum(p * np.log(p + 1e-12)))
                    results[f'rankme_{wname}_{wt}_alpha{alpha}'] = rk
            else:
                for lam in lambdas:
                    for gamma in gammas:
                        omega = apply_weight(wt, C, None, lam, gamma, normalize_weights)
                        wdist = d_std * omega
                        pers = compute_scalar_0d_persistence(wdist, d_std_orig=d_std, norm_quantile=norm_quantile)
                        results[f'persistence_{wname}_{wt}_lam{lam}_gam{gamma}'] = pers

                        omega_sq = squareform(omega)
                        G_sq = G ** 2
                        actual = np.sum(G_sq * omega_sq)
                        n, d = embeddings.shape
                        expected = n + n * (n - 1) / d
                        sc_val = (actual - expected) / (n * n - expected)
                        results[f'self_clustering_{wname}_{wt}_lam{lam}_gam{gamma}'] = sc_val

                        G_w = G * omega_sq
                        G_w = (G_w + G_w.T) / 2
                        eigvals = np.linalg.eigvalsh(G_w)
                        eigvals = np.maximum(eigvals, 1e-12)
                        p = eigvals / np.sum(eigvals)
                        rk = np.exp(-np.sum(p * np.log(p + 1e-12)))
                        results[f'rankme_{wname}_{wt}_lam{lam}_gam{gamma}'] = rk

    for wname in weight_names:
        if wname == 'density':
            v = rho
        elif wname == 'dimension':
            v = np.exp(-dim_est)
        elif wname == 'volume':
            v = np.abs(v_rate) + 1e-6
        elif wname == 'overlap':
            v = np.ones(N)
        else:
            v = np.ones(N)
        v = v / np.sum(v) * N
        Wsqrt = np.sqrt(v)
        Xw = embeddings * Wsqrt[:, np.newaxis]
        results[f'rankme_{wname}'] = rankme(Xw)

    return results

def compute_metrics(embeddings_np, selected_metrics=None,
                    n_samples=10, sample_fraction=1/20, verbose=0):
    res = all_weighted_metrics(embeddings_np)
    return {'metric_'+k: v for k, v in res.items()}
