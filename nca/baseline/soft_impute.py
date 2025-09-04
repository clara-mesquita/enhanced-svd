import numpy as np
import pandas as pd
from fancyimpute import SoftImpute
from sklearn.preprocessing import StandardScaler

def impute_softimpute(df_missing, max_iters=200, convergence_threshold=1e-5, random_state=42, shrinkage_value=None):
    """
    Imputa NaNs em 'throughput_bps' usando SoftImpute (SVD com thresholding),
    preservando valores observados e preenchendo só os NaNs.
    """
    from fancyimpute import SoftImpute

    df_imp = df_missing.copy()
    y = df_imp["throughput_bps"].astype(float).to_numpy()
    mask_missing = np.isnan(y)

    Y = y.reshape(-1, 1)
    imputer = SoftImpute(max_iters=max_iters,
                         convergence_threshold=convergence_threshold,
                         shrinkage_value=shrinkage_value,
                         verbose=False)
    Y_hat = imputer.fit_transform(Y).reshape(-1)

    y_out = y.copy()
    y_out[mask_missing] = Y_hat[mask_missing]
    df_imp["throughput_bps"] = y_out
    return df_imp