import pandas as pd 
import numpy as np 

# ----------------------------
# 1) PERIOD (n_lines) ESTIMATION
# ----------------------------

def _acf(y, max_lag): # ->basicamente isso aqui calcula a galera com maior correlação e deixa em uma só coluna -> uantas vezes esse é o utilizado??
    """Biased ACF up to max_lag (lag 0..max_lag). NaNs are linearly interpolated first."""
    y = pd.Series(y).interpolate(limit_direction="both").to_numpy()
    y = y - np.nanmean(y)
    n = len(y)  
    acf_vals = np.empty(max_lag + 1)
    denom = np.dot(y, y) + 1e-12
    for lag in range(max_lag + 1):
        acf_vals[lag] = np.dot(y[: n - lag], y[lag:]) / denom
    return acf_vals

def estimate_period(y,min_period,max_period):  
    """
    Pick period (n_lines) automatically using ACF peak
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if max_period is None:
        max_period = max(7, min(n // 4, 1000))  # sensible cap

    if n < min_period * 2:
        # too short — just return something small
        return max(min_period, min(n, 8))

    acf_vals = _acf(y, max_period)
    # Ignore lag 0; pick the best lag in [min_period, max_period]
    candidate_lags = np.arange(min_period, max_period + 1)
    best_lag = candidate_lags[np.argmax(acf_vals[min_period: max_period + 1])]
    return int(best_lag)

# ---------------------------------
# 2) FOLDING (PERIODIC TEMPORAL MATRIX)
# ---------------------------------

def fold_series_to_matrix(y, period): # basicamente isso cria a matriz temporal 
    """
    Fold 1D series into a (n_blocks, period) matrix. 
    Pads the last block with NaN if needed.
    Returns (M, original_len).
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    n_blocks = int(np.ceil(n / period)) #n de coluans
    pad_len = n_blocks * period - n
    if pad_len > 0:
        y = np.concatenate([y, np.full(pad_len, np.nan)])
    M = y.reshape(n_blocks, period)
    return M, n

def unfold_matrix_to_series(M, original_len): # volta pro formato original
    """Inverse of fold: row-wise flatten and trim to original length."""
    y = M.reshape(-1)
    return y[:original_len]

# ----------------------------
# 3) SVD + RANK SELECTION
# ----------------------------

def svd_rank(M_filled, energy=0.9): # faz o svd na tora e escolhe o r com a soma dos valore sisnuglares so deus sabe pq
    """
    Compute SVD and choose rank r by cumulative explained 'energy' (sum of singular values).
    """
    U, s, Vt = np.linalg.svd(M_filled, full_matrices=False)
    cum = np.cumsum(s) / (np.sum(s) + 1e-12)
    r = int(np.searchsorted(cum, energy) + 1)
    r = max(1, min(r, min(M_filled.shape)))
    return U, s, Vt, r

# -------------------------------------------
# 4) KNN IMPUTE USING ROW EMBEDDINGS FROM SVD
# -------------------------------------------

def _warm_start_fill(M): # coloca mediana em tudo e vapo só pra começar
    """Column-wise median fill as a stable warm start."""
    M_filled = M.copy()
    col_medians = np.nanmedian(M_filled, axis=0)
    # If an entire column is NaN, fallback to global median
    if np.any(np.isnan(col_medians)):
        global_med = np.nanmedian(M_filled)
        col_medians = np.where(np.isnan(col_medians), global_med, col_medians)
    inds = np.where(np.isnan(M_filled))
    M_filled[inds] = np.take(col_medians, inds[1])
    return M_filled

def knn_in_latent(M, k=5, energy=0.9, allow_future=True):
    """
    Impute NaNs in M by KNN in SVD latent space (rows ≈ cycles/weeks). que porra eh latent space
    For each missing cell (i,j), find k nearest rows to row i in latent space
    among those with M[row, j] observed (and optionally row < i).
    """
    M_filled0 = _warm_start_fill(M)
    U, s, Vt, r = svd_rank(M_filled0, energy=energy)
    # Row embeddings (T x r): U_r * S_r
    Z = U[:, :r] * s[:r]  # broadcasting: each column of U scaled by s

    M_imp = M.copy()
    T, P = M.shape
    eps = 1e-8

    # Precompute which rows have each column observed
    observed_mask = ~np.isnan(M)
    for i in range(T):
        # indices (columns) that are missing in row i
        miss_cols = np.where(~observed_mask[i])[0]
        if len(miss_cols) == 0:
            continue
        zi = Z[i]

        # Candidate rows for neighbors (global, filtered below per col)
        if allow_future:
            candidate_rows_global = np.arange(T)
        else:
            candidate_rows_global = np.arange(0, i)  # only past

        if len(candidate_rows_global) == 0:
            # If we can't use past (i==0), allow future just for this row:
            candidate_rows_global = np.arange(T)

        # Distances in latent space to all candidates
        Zc = Z[candidate_rows_global]
        dists = np.linalg.norm(Zc - zi[None, :], axis=1)
        dists = dists + eps  # avoid zero

        for j in miss_cols:
            # keep only candidates that have this column observed
            obs_rows = candidate_rows_global[observed_mask[candidate_rows_global, j]]
            if len(obs_rows) == 0:
                # fall back to warm-start value if nothing observed
                M_imp[i, j] = M_filled0[i, j]
                continue

            # distances for those rows
            d = np.linalg.norm(Z[obs_rows] - zi[None, :], axis=1) + eps
            # k nearest
            if len(d) > k:
                idx = np.argpartition(d, k)[:k]
                nn_rows = obs_rows[idx]
                d = d[idx]
            else:
                nn_rows = obs_rows

            w = 1.0 / d  # inverse-distance weights
            vals = M[nn_rows, j]
            # safety: if still NaN (shouldn't happen), drop them
            ok = ~np.isnan(vals)
            if not np.any(ok):
                M_imp[i, j] = M_filled0[i, j]
            else:
                vals = vals[ok]
                w = w[ok]
                M_imp[i, j] = np.sum(w * vals) / np.sum(w)

    return M_imp

# ----------------------------
# 5) MAIN PIPELINE
# ----------------------------

def impute_throughput_svd_knn(df, col="throughput_bps", min_period=4, max_period=None, energy=0.9, k=5, allow_future=True):
    """
    Full pipeline:
    1) auto period detection -> n_lines (period)
    2) fold into (n_blocks, period) matrix
    3) SVD -> latent row embeddings
    4) KNN in latent space to impute NaNs
    5) unfold back to 1D series
    Returns (imputed_series, diagnostics).
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in df.")

    y = df[col].to_numpy(dtype=float)
    original_index = df.index

    # Detect period (n_lines)
    period = estimate_period(y, min_period=min_period, max_period=max_period)

    # Fold
    M, orig_len = fold_series_to_matrix(y, period=period)

    # Impute in latent KNN space
    M_imp = knn_in_latent(M, k=k, energy=energy, allow_future=allow_future)

    # Unfold
    y_imp = unfold_matrix_to_series(M_imp, original_len=orig_len)

    # print(y_imp)

    diagnostics = {
        "period_estimated": period,
        "matrix_shape": M.shape,
        "rank_energy_target": energy,
        "k_neighbors": k,
        "allow_future": allow_future,
    }
    # return pd.Series(y_imp, index=original_index, name=f"{col}_imputed"), diagnostics
    df_imp = df.copy()
    df_imp[col] = y_imp  # Replace the column with imputed values
    
    return df_imp  # Now returns full DataFrame with same structure