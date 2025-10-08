import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy import signal
from sklearn.impute import KNNImputer
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Try to import fancyimpute, provide fallback if not available
try:
    from fancyimpute import SoftImpute
    SOFTIMPUTE_AVAILABLE = True
except ImportError:
    SOFTIMPUTE_AVAILABLE = False
    print("Warning: fancyimpute not available. SoftImpute method will be skipped.")
    print("Install with: pip install fancyimpute")

# Configuration
MISSING_RATES_FOLDER = "missing_rates_datasets"
LONGEST_INTERVAL_FOLDER = "longest_interval_datasets"
OUTPUT_FOLDER = "imputation_results"
FIGURES_FOLDER = os.path.join(OUTPUT_FOLDER, "figures")
REPORT_FILE = os.path.join(OUTPUT_FOLDER, "imputation_report.txt")
TIMESTAMP_COL = "Data"
VALUE_COL = "Vazao"
MISSING_SENTINEL = -1
RANDOM_SEED = 42

def setup_folders():
    """Create necessary output folders"""
    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)
    Path(FIGURES_FOLDER).mkdir(exist_ok=True)

# ============================================================================
# FFT Period Estimation
# ============================================================================

def estimate_period_fft(y, min_period=4, max_period=None):
    """Estimate period using FFT spectral analysis"""
    y = np.asarray(y, dtype=float)
    n = len(y)
    
    if max_period is None:
        max_period = max(7, min(n // 4, 1000))
    
    if n < min_period * 2:
        return max(min_period, min(n, 8))
    
    y_series = pd.Series(y)
    y_filled = y_series.interpolate(limit_direction="both").ffill().bfill().to_numpy()
    
    if np.std(y_filled) < 1e-10:
        return min_period
    
    y_detrended = signal.detrend(y_filled)
    y_normalized = (y_detrended - np.mean(y_detrended)) / (np.std(y_detrended) + 1e-10)
    
    fft_vals = np.fft.rfft(y_normalized)
    power_spectrum = np.abs(fft_vals) ** 2
    frequencies = np.fft.rfftfreq(n)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        periods = 1.0 / frequencies
        periods[0] = np.inf
    
    valid_mask = (periods >= min_period) & (periods <= max_period)
    
    if not np.any(valid_mask):
        return min_period
    
    valid_power = power_spectrum[valid_mask]
    valid_periods = periods[valid_mask]
    best_period = valid_periods[np.argmax(valid_power)]
    
    return int(np.round(best_period))

# ============================================================================
# Matrix Construction and SVD Utilities
# ============================================================================

def fold_series_to_matrix(y, period):
    """Fold 1D series into (n_blocks, period) matrix"""
    y = np.asarray(y, dtype=float)
    n = len(y)
    n_blocks = int(np.ceil(n / period))
    pad_len = n_blocks * period - n
    
    if pad_len > 0:
        y = np.concatenate([y, np.full(pad_len, np.nan)])
    
    M = y.reshape(n_blocks, period)
    return M, n

def unfold_matrix_to_series(M, original_len):
    """Unfold matrix back to series"""
    y = M.reshape(-1)
    return y[:original_len]

def build_hankel_matrix(y, window_size):
    """Build Hankel matrix from time series"""
    y = np.asarray(y, dtype=float)
    n = len(y)
    
    if window_size > n:
        window_size = n
    
    K = n - window_size + 1
    L = window_size
    H = np.zeros((K, L))
    
    for i in range(K):
        H[i, :] = y[i:i+L]
    
    return H

def hankel_to_series(H, method='diagonal_average'):
    """Reconstruct series from Hankel matrix"""
    K, L = H.shape
    n = K + L - 1
    
    if method == 'diagonal_average':
        y = np.zeros(n)
        counts = np.zeros(n)
        
        for i in range(K):
            for j in range(L):
                idx = i + j
                y[idx] += H[i, j]
                counts[idx] += 1
        
        y = y / counts
        
    elif method == 'first_row_col':
        y = np.concatenate([H[0, :], H[1:, -1]])
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return y

def svd_rank(M_filled, energy=0.9):
    """Compute SVD and choose rank by cumulative energy"""
    U, s, Vt = np.linalg.svd(M_filled, full_matrices=False)
    
    total_energy = np.sum(s)
    if total_energy < 1e-12:
        return U, s, Vt, 1
    
    cum_energy = np.cumsum(s) / total_energy
    r = int(np.searchsorted(cum_energy, energy) + 1)
    r = max(1, min(r, min(M_filled.shape)))
    
    return U, s, Vt, r

def _initial_fill(M):
    """Column-wise median fill for initial SVD"""
    M_filled = M.copy()
    col_medians = np.nanmedian(M_filled, axis=0)
    global_median = np.nanmedian(M_filled)
    
    if np.isnan(global_median):
        global_median = 0.0
    
    col_medians = np.where(np.isnan(col_medians), global_median, col_medians)
    nan_indices = np.where(np.isnan(M_filled))
    M_filled[nan_indices] = np.take(col_medians, nan_indices[1])
    
    return M_filled

def knn_in_latent(M, k=5, energy=0.9, allow_future=True):
    """Impute missing values using KNN in SVD latent space"""
    M_filled = _initial_fill(M)
    U, s, Vt, r = svd_rank(M_filled, energy=energy)
    Z = U[:, :r] * s[:r]
    
    M_imputed = M.copy()
    T, P = M.shape
    observed_mask = ~np.isnan(M)
    
    for i in range(T):
        missing_cols = np.where(~observed_mask[i])[0]
        if len(missing_cols) == 0:
            continue
        
        zi = Z[i]
        candidates = np.arange(T) if allow_future else np.arange(0, i)
        
        if len(candidates) == 0:
            candidates = np.arange(T)
        
        for j in missing_cols:
            valid_candidates = candidates[observed_mask[candidates, j]]
            
            if len(valid_candidates) == 0:
                continue
            
            distances = np.linalg.norm(Z[valid_candidates] - zi[None, :], axis=1) + 1e-8
            
            if len(distances) > k:
                nearest_indices = np.argpartition(distances, k)[:k]
                neighbor_rows = valid_candidates[nearest_indices]
                neighbor_distances = distances[nearest_indices]
            else:
                neighbor_rows = valid_candidates
                neighbor_distances = distances
            
            weights = 1.0 / neighbor_distances
            values = M[neighbor_rows, j]
            valid_mask = ~np.isnan(values)
            
            values = values[valid_mask]
            weights = weights[valid_mask]
            
            if len(values) > 0:
                M_imputed[i, j] = np.sum(weights * values) / np.sum(weights)
    
    return M_imputed

# ============================================================================
# Imputation Methods - Original
# ============================================================================

def impute_svd_knn(df, col=VALUE_COL, min_period=4, max_period=None, 
                   energy=0.9, k=5, allow_future=True, use_hankel=False):
    """SVD-KNN imputation with optional Hankel matrix"""
    y = df[col].to_numpy(dtype=float)
    period = estimate_period_fft(y, min_period=min_period, max_period=max_period)
    
    if use_hankel:
        M = build_hankel_matrix(y, window_size=period)
        M_imputed = knn_in_latent(M, k=k, energy=energy, allow_future=allow_future)
        y_imputed = hankel_to_series(M_imputed, method='diagonal_average')
    else:
        M, orig_len = fold_series_to_matrix(y, period=period)
        M_imputed = knn_in_latent(M, k=k, energy=energy, allow_future=allow_future)
        y_imputed = unfold_matrix_to_series(M_imputed, original_len=orig_len)
    
    df_imputed = df.copy()
    df_imputed[col] = y_imputed
    return df_imputed

def impute_linear(df, col=VALUE_COL):
    """Linear interpolation"""
    df_imputed = df.copy()
    df_imputed[col] = df_imputed[col].interpolate(method="linear", limit_direction="both")
    return df_imputed

def impute_knn_sklearn(df, k=5, col=VALUE_COL):
    """sklearn KNN imputer"""
    df_imputed = df.copy()
    imputer = KNNImputer(n_neighbors=k, weights="uniform")
    imputed_values = imputer.fit_transform(df_imputed[[col]])
    df_imputed[col] = imputed_values[:, 0]
    return df_imputed

def impute_spline(df, col=VALUE_COL, order=3):
    """Spline interpolation"""
    df_imputed = df.copy()
    df_imputed[col] = df_imputed[col].interpolate(method='spline', order=order, limit_direction='both')
    return df_imputed

def impute_mean(df, col=VALUE_COL):
    """Mean imputation"""
    df_imputed = df.copy()
    mean_value = df_imputed[col].mean()
    df_imputed[col] = df_imputed[col].fillna(mean_value)
    return df_imputed

def impute_locf(df, col=VALUE_COL):
    """Last Observation Carried Forward"""
    df_imputed = df.copy()
    df_imputed[col] = df_imputed[col].ffill()
    return df_imputed

# ============================================================================
# Imputation Methods - NEW ADDITIONS
# ============================================================================

def impute_kalman(df, col=VALUE_COL, min_period=4, max_period=None):
    """Kalman Filter imputation using UnobservedComponents"""
    df_imputed = df.copy()
    y = df_imputed[col].copy()
    
    # Estimate seasonal period
    y_temp = y.interpolate(limit_direction="both").ffill().bfill()
    period = estimate_period_fft(y_temp.values, min_period=min_period, max_period=max_period)
    period = max(2, min(period, len(y) // 3))
    
    try:
        # Fit UnobservedComponents model with trend and seasonality
        mod = UnobservedComponents(
            y, 
            level="local linear trend", 
            seasonal=period,
            irregular=True
        )
        res = mod.fit(disp=False, maxiter=100)
        
        # Use Kalman smoother estimates
        df_imputed[col] = y.fillna(res.fittedvalues)
        
    except Exception as e:
        # Fallback to simpler model if fitting fails
        try:
            mod = UnobservedComponents(y, level="local level", irregular=True)
            res = mod.fit(disp=False, maxiter=50)
            df_imputed[col] = y.fillna(res.fittedvalues)
        except:
            # Final fallback to linear interpolation
            df_imputed[col] = y.interpolate(limit_direction="both")
    
    return df_imputed

def impute_arima(df, col=VALUE_COL, order=(1,1,1)):
    """ARIMA imputation"""
    df_imputed = df.copy()
    y = df_imputed[col].copy()
    
    # Initial fill for ARIMA fitting
    y_filled = y.interpolate(limit_direction="both").ffill().bfill()
    
    try:
        # Fit ARIMA model
        model = ARIMA(y_filled, order=order)
        fit = model.fit()
        
        # Get fitted values
        fitted = fit.fittedvalues
        
        # Fill missing values with fitted values
        missing_mask = y.isna()
        df_imputed.loc[missing_mask, col] = fitted[missing_mask]
        
    except Exception as e:
        # Fallback to linear interpolation
        df_imputed[col] = y.interpolate(limit_direction="both")
    
    return df_imputed

def impute_holtwinters(df, col=VALUE_COL, min_period=4, max_period=None):
    """Holt-Winters Exponential Smoothing imputation"""
    df_imputed = df.copy()
    y = df_imputed[col].copy()
    
    # Estimate seasonal period
    y_temp = y.interpolate(limit_direction="both").ffill().bfill()
    period = estimate_period_fft(y_temp.values, min_period=min_period, max_period=max_period)
    period = max(2, min(period, len(y) // 3))
    
    # Need at least 2 full seasons
    if len(y_temp) < 2 * period:
        df_imputed[col] = y.interpolate(limit_direction="both")
        return df_imputed
    
    try:
        # Fit Holt-Winters model
        model = ExponentialSmoothing(
            y_temp, 
            trend="add", 
            seasonal="add", 
            seasonal_periods=period
        )
        fit = model.fit(optimized=True)
        
        # Fill missing values with fitted values
        missing_mask = y.isna()
        df_imputed.loc[missing_mask, col] = fit.fittedvalues[missing_mask]
        
    except Exception as e:
        # Fallback to additive trend only
        try:
            model = ExponentialSmoothing(y_temp, trend="add", seasonal=None)
            fit = model.fit(optimized=True)
            missing_mask = y.isna()
            df_imputed.loc[missing_mask, col] = fit.fittedvalues[missing_mask]
        except:
            # Final fallback
            df_imputed[col] = y.interpolate(limit_direction="both")
    
    return df_imputed

def impute_softimpute(df, col=VALUE_COL, max_rank=5):
    """SoftImpute matrix completion"""
    if not SOFTIMPUTE_AVAILABLE:
        raise ImportError("fancyimpute not available. Install with: pip install fancyimpute")
    
    df_imputed = df.copy()
    y = df_imputed[col].values.reshape(-1, 1).astype(float)
    
    try:
        # Apply SoftImpute
        imputer = SoftImpute(max_rank=max_rank, verbose=False)
        y_imputed = imputer.fit_transform(y)
        df_imputed[col] = y_imputed.ravel()
        
    except Exception as e:
        # Fallback to linear interpolation
        df_imputed[col] = df_imputed[col].interpolate(limit_direction="both")
    
    return df_imputed

def impute_iterativesvd(df, col=VALUE_COL, rank=5):
    """IterativeSVD matrix completion (similar to SoftImpute)"""
    if not SOFTIMPUTE_AVAILABLE:
        raise ImportError("fancyimpute not available. Install with: pip install fancyimpute")
    
    try:
        from fancyimpute import IterativeSVD
    except ImportError:
        # Use SoftImpute as fallback
        return impute_softimpute(df, col=col, max_rank=rank)
    
    df_imputed = df.copy()
    y = df_imputed[col].values.reshape(-1, 1).astype(float)
    
    try:
        # Apply IterativeSVD
        imputer = IterativeSVD(rank=rank, verbose=False)
        y_imputed = imputer.fit_transform(y)
        df_imputed[col] = y_imputed.ravel()
        
    except Exception as e:
        # Fallback to linear interpolation
        df_imputed[col] = df_imputed[col].interpolate(limit_direction="both")
    
    return df_imputed

# ============================================================================
# Evaluation Metrics
# ============================================================================

def rmse(a, b):
    """Root Mean Squared Error"""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.nanmean(diff**2)))

def mae(a, b):
    """Mean Absolute Error"""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.nanmean(np.abs(diff)))

def evaluate_imputation(df_original, df_imputed, missing_mask):
    """Calculate RMSE and MAE for imputed values"""
    orig_vals = df_original[VALUE_COL].to_numpy()[missing_mask]
    imp_vals = df_imputed[VALUE_COL].to_numpy()[missing_mask]
    
    return {
        'rmse': rmse(orig_vals, imp_vals),
        'mae': mae(orig_vals, imp_vals),
        'n_points': int(missing_mask.sum())
    }

# ============================================================================
# Visualization
# ============================================================================

def visualize_imputation(df_original, df_missing, df_imputed, method_name, 
                        dataset_name, missing_rate, save_path):
    """Visualize original, missing, and imputed time series"""
    fig, ax = plt.subplots(figsize=(14, 5))
    
    missing_mask = (df_missing[VALUE_COL] == MISSING_SENTINEL) | df_missing[VALUE_COL].isna()
    
    ax.plot(df_original.index, df_original[VALUE_COL], 'b-', 
            label='Original', alpha=0.7, linewidth=1.5)
    ax.plot(df_imputed.index, df_imputed[VALUE_COL], 'r--', 
            label=f'Imputed ({method_name})', alpha=0.7, linewidth=1.2)
    ax.scatter(df_original.index[missing_mask], df_original[VALUE_COL][missing_mask], 
               c='orange', s=30, label='Missing Points', zorder=5, alpha=0.6)
    
    ax.set_title(f'{dataset_name} - {method_name} - Missing Rate: {missing_rate*100:.0f}%')
    ax.set_xlabel('Time Index')
    ax.set_ylabel(VALUE_COL)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# ============================================================================
# Main Processing Pipeline
# ============================================================================

def process_all_datasets():
    """Process all datasets with all imputation methods"""
    setup_folders()
    
    missing_files = [f for f in os.listdir(MISSING_RATES_FOLDER) if f.endswith('.csv')]
    
    if not missing_files:
        print(f"No CSV files found in {MISSING_RATES_FOLDER}")
        return

    # Define all imputation methods
    imputation_methods = {
        'SVD_KNN': lambda df: impute_svd_knn(df, k=10, use_hankel=False),
        'SVD_KNN_Hankel': lambda df: impute_svd_knn(df, k=10, use_hankel=True),
        'Kalman': impute_kalman,
        'ARIMA': impute_arima,
        'HoltWinters': impute_holtwinters,
        'Linear': impute_linear,
        'KNN_Sklearn': lambda df: impute_knn_sklearn(df, k=5),
        'Spline': impute_spline,
        'Mean': impute_mean,
        'LOCF': impute_locf
    }
    
    # Add SoftImpute and IterativeSVD if available
    # if SOFTIMPUTE_AVAILABLE:
    #     imputation_methods['SoftImpute'] = lambda df: impute_softimpute(df, max_rank=5)
    #     imputation_methods['IterativeSVD'] = lambda df: impute_iterativesvd(df, rank=5)
    
    all_results = []
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as report:
        report.write("="*80 + "\n")
        report.write("IMPUTATION EVALUATION REPORT\n")
        report.write("="*80 + "\n\n")
        
        for missing_file in sorted(missing_files):
            base_name = missing_file.replace('_missing_10.csv', '') \
                                   .replace('_missing_20.csv', '') \
                                   .replace('_missing_30.csv', '') \
                                   .replace('_missing_40.csv', '')
            
            missing_rate_str = missing_file.split('_missing_')[-1].replace('.csv', '')
            missing_rate = int(missing_rate_str) / 100.0
            
            original_file = f"{base_name}_longest.csv"
            original_path = os.path.join(LONGEST_INTERVAL_FOLDER, original_file)
            missing_path = os.path.join(MISSING_RATES_FOLDER, missing_file)
            
            if not os.path.exists(original_path):
                report.write(f"WARNING: Original file not found for {missing_file}\n\n")
                continue
            
            report.write("\n" + "="*80 + "\n")
            report.write(f"Dataset: {base_name}\n")
            report.write(f"Missing Rate: {missing_rate*100:.0f}%\n")
            report.write("="*80 + "\n\n")
            
            try:
                df_original = pd.read_csv(original_path)
                df_missing = pd.read_csv(missing_path)

                df_missing.replace(MISSING_SENTINEL, np.nan, inplace=True)
                
                df_missing_processed = df_missing.copy()
                df_missing_processed[VALUE_COL] = df_missing_processed[VALUE_COL].replace(MISSING_SENTINEL, np.nan)
                
                missing_mask = (df_missing[VALUE_COL] == MISSING_SENTINEL) | df_missing[VALUE_COL].isna()
                
                report.write(f"Original shape: {df_original.shape}\n")
                report.write(f"Missing points: {missing_mask.sum()} ({missing_mask.sum()/len(df_missing)*100:.1f}%)\n\n")
                
                report.write("Method Performance:\n")
                report.write("-" * 60 + "\n")
                report.write(f"{'Method':<20} {'RMSE':>12} {'MAE':>12} {'Points':>10}\n")
                report.write("-" * 60 + "\n")
                
                for method_name, impute_func in imputation_methods.items():
                    try:
                        df_imputed = impute_func(df_missing_processed)
                        metrics = evaluate_imputation(df_original, df_imputed, missing_mask)
                        
                        report.write(f"{method_name:<20} {metrics['rmse']:>12.4f} "
                                   f"{metrics['mae']:>12.4f} {metrics['n_points']:>10}\n")
                        
                        all_results.append({
                            'dataset': base_name,
                            'missing_rate': missing_rate,
                            'method': method_name,
                            'rmse': metrics['rmse'],
                            'mae': metrics['mae'],
                            'n_points': metrics['n_points']
                        })
                        
                        fig_name = f"{base_name}_mr{int(missing_rate*100)}_{method_name}.png"
                        fig_path = os.path.join(FIGURES_FOLDER, fig_name)
                        visualize_imputation(df_original, df_missing, df_imputed, 
                                           method_name, base_name, missing_rate, fig_path)
                        
                    except Exception as e:
                        report.write(f"{method_name:<20} ERROR: {str(e)}\n")
                        print(f"Error in {method_name} for {missing_file}: {str(e)}")
                
                report.write("\n")
                
            except Exception as e:
                report.write(f"ERROR processing file: {str(e)}\n\n")
                print(f"Error processing {missing_file}: {str(e)}")
                continue
        
        results_df = pd.DataFrame(all_results)
        
        if len(results_df) > 0:
            report.write("\n" + "="*80 + "\n")
            report.write("SUMMARY STATISTICS\n")
            report.write("="*80 + "\n\n")
            
            for missing_rate in sorted(results_df['missing_rate'].unique()):
                report.write(f"\nMissing Rate: {missing_rate*100:.0f}%\n")
                report.write("-" * 60 + "\n")
                
                rate_data = results_df[results_df['missing_rate'] == missing_rate]
                summary = rate_data.groupby('method').agg({
                    'rmse': ['mean', 'std'],
                    'mae': ['mean', 'std']
                }).round(4)
                
                report.write(summary.to_string())
                report.write("\n\n")
                
                best_rmse = rate_data.loc[rate_data['rmse'].idxmin()]
                best_mae = rate_data.loc[rate_data['mae'].idxmin()]
                
                report.write(f"Best RMSE: {best_rmse['method']} ({best_rmse['rmse']:.4f})\n")
                report.write(f"Best MAE: {best_mae['method']} ({best_mae['mae']:.4f})\n")
            
            csv_path = os.path.join(OUTPUT_FOLDER, "detailed_results.csv")
            results_df.to_csv(csv_path, index=False)
            report.write(f"\n\nDetailed results saved to: {csv_path}\n")
    
    print(f"Processing complete!")
    print(f"Report saved to: {REPORT_FILE}")
    print(f"Figures saved to: {FIGURES_FOLDER}")
    print(f"Detailed results saved to: {os.path.join(OUTPUT_FOLDER, 'detailed_results.csv')}")

if __name__ == "__main__":
    process_all_datasets()