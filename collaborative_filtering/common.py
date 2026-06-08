import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from sklearn.preprocessing import LabelEncoder
from compute_metrics import compute_metrics

def recall_at_k(recommended, relevant, k):
    if len(relevant) == 0: return 0.0
    return len(set(recommended[:k]) & set(relevant)) / len(relevant)

def ndcg_at_k(recommended, relevant, k):
    dcg = sum(1.0 / np.log2(i+2) for i, item in enumerate(recommended[:k]) if item in relevant)
    idcg = sum(1.0 / np.log2(i+2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0

def hit_rate_at_k(recommended, relevant, k):
    return 1.0 if len(set(recommended[:k]) & set(relevant)) > 0 else 0.0

def map_at_k(recommended, relevant, k):
    ap, hits = 0.0, 0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            ap += hits / (i+1)
    return ap / min(len(relevant), k) if relevant else 0.0

def encode_column(df, col, new_col=None, encoder=None):
    if new_col is None: new_col = col
    if encoder is None:
        encoder = LabelEncoder()
        df[new_col] = encoder.fit_transform(df[col])
        return df, encoder
    else:
        mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        df[new_col] = df[col].map(mapping)
        return df

def train_test_split(df, quantile=0.9):
    timeline = np.quantile(df.timestamp, quantile)
    train = df[df.timestamp < timeline]
    test = df[df.timestamp >= timeline]
    test = test[test.user_id.isin(train.user_id.unique())]
    test = test[test.item_id.isin(train.item_id.unique())]
    return train, test

def predict(model, user_ids, train_csr, N=10, batch_size=1000):
    recs = []
    for start in range(0, len(user_ids), batch_size):
        batch = user_ids[start:start+batch_size]
        item_ids, _ = model.recommend(batch, train_csr[batch], N=N,
                                      filter_already_liked_items=True,
                                      recalculate_user=False)
        recs.append(item_ids)
    recs = np.concatenate(recs)
    return pd.DataFrame({'user_id': user_ids, 'item_id': list(recs)})

def train_and_evaluate(model_type, train_csr, test, params, N=10, batch_size=1000,
                       n_test_users=None):
    if model_type == 'als':
        model = AlternatingLeastSquares(iterations=15, **params)
    elif model_type == 'bpr':
        model = BayesianPersonalizedRanking(iterations=50, **params)
    else:
        raise ValueError('Unknown model type')
    model.fit(train_csr)

    test_users_all = test.user_id.unique()
    if n_test_users and n_test_users < len(test_users_all):
        rng = np.random.RandomState(42)
        test_users = rng.choice(test_users_all, size=n_test_users, replace=False)
    else:
        test_users = test_users_all

    recs = predict(model, test_users, train_csr, N=N, batch_size=batch_size)
    recs = recs.explode('item_id')
    recs['rating'] = recs.groupby('user_id').cumcount(ascending=False)

    recall_sum = ndcg_sum = hit_sum = map_sum = 0.0
    n_users = 0
    for uid in test_users:
        user_test = test[test.user_id == uid]
        if len(user_test) == 0: continue
        user_recs = recs[recs.user_id == uid]['item_id'].tolist()
        relevant = user_test['item_id'].tolist()
        recall_sum += recall_at_k(user_recs, relevant, N)
        ndcg_sum += ndcg_at_k(user_recs, relevant, N)
        hit_sum += hit_rate_at_k(user_recs, relevant, N)
        map_sum += map_at_k(user_recs, relevant, N)
        n_users += 1

    rec_metrics = {
        'Recall': recall_sum / n_users if n_users else 0.0,
        'NDCG': ndcg_sum / n_users if n_users else 0.0,
        'HitRate': hit_sum / n_users if n_users else 0.0,
        'MAP': map_sum / n_users if n_users else 0.0,
    }

    def to_numpy(arr):
        if hasattr(arr, 'to_numpy'):
            return arr.to_numpy()
        return np.asarray(arr)

    user_factors_np = to_numpy(model.user_factors)
    item_factors_np = to_numpy(model.item_factors)

    user_embeddings = user_factors_np[test_users]
    item_embeddings = item_factors_np

    user_metrics = compute_metrics(user_embeddings, n_samples=1, sample_fraction=1, verbose=0)
    item_metrics = compute_metrics(item_embeddings, n_samples=1, sample_fraction=1, verbose=0)

    user_metrics = {'user_' + '_'.join(k.split('_')[1:]) : v
                    for k, v in user_metrics.items() if 'metric' in k}
    item_metrics = {'item_' + '_'.join(k.split('_')[1:]) : v
                    for k, v in item_metrics.items() if 'metric' in k}

    return {**params, **rec_metrics, **user_metrics, **item_metrics}
