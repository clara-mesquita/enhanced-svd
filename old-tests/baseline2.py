import numpy as np
import pandas as pd
from fancyimpute import SoftImpute, IterativeSVD
from sklearn.impute import KNNImputer
import os


def introduce_missing(data, missing_rate, random_state=42):
    """Introduz valores faltantes em um array numpy."""
    rng = np.random.default_rng(random_state)
    n_missing = int(missing_rate * data.size)
    missing_indices = rng.choice(data.size, n_missing, replace=False)
    data_missing = data.copy()
    nan_rows = missing_indices // data.shape[1]
    nan_cols = missing_indices % data.shape[1]
    data_missing[nan_rows, nan_cols] = np.nan
    nan_mask = np.isnan(data_missing)
    return data_missing, nan_mask


def calculate_rmse_nrmse(original, imputed, nan_mask, amplitude):
    """Calcula RMSE e NRMSE entre os valores originais e imputados."""
    rmse = np.sqrt(np.mean((original[nan_mask] / 1e6 - imputed[nan_mask] / 1e6) ** 2))
    nrmse = rmse / amplitude if amplitude != 0 else np.nan
    return rmse, nrmse


def create_temporal_matrices(data, line_counts):
    """Cria matrizes temporais com diferentes números de linhas."""
    matrices = {}
    for n_lines in line_counts:
        columns_quantity = len(data) // n_lines
        if columns_quantity == 0:
            continue
        temporal_matrix = data[: columns_quantity * n_lines].reshape(columns_quantity, n_lines).T
        matrices[n_lines] = temporal_matrix
    return matrices


def main():
    dataset_ids = [1, 2, 3, 4]
    dataset_paths = [
        f"institution_subnets/agg_1_hour/{i}.csv"
        for i in dataset_ids
    ]

    # Parâmetros para os métodos
    shrinkage_values = [0.01, 0.05, 0.1, 0.2, 0.5]  # Soft Impute
    ranks = [1, 2, 4, 8, 12, 16]  # Iterative SVD
    line_counts = [24, 120, 168, 336, 504, 672]  # Matriz temporal 
    missing_rates = [0.1, 0.2, 0.3, 0.4] 
    knn_neighbors_grid = [2, 4, 8, 12, 16]  # KNN

    for dataset_id, dataset_path in zip(dataset_ids, dataset_paths):
        # Carrega e prepara os dados originais
        df = pd.read_csv(dataset_path)
        df["throughput"] = df["n_bytes"] / 3600 * 8  # bps
        throughput_vector = df["throughput"].values
        amplitude = (throughput_vector.max() - throughput_vector.min()) / 1e6

        # Cria datasets com missing values para cada taxa
        missing_datasets = {}
        for miss_rate in missing_rates:
            # Cria versão com missing values (como array 1D)
            data_missing, nan_mask = introduce_missing(
                throughput_vector.reshape(-1, 1),  # Transforma em matriz 2D para a função
                miss_rate, 
                random_state=42
            )
            missing_datasets[miss_rate] = {
                'data': data_missing.flatten(),  # Volta para 1D
                'mask': nan_mask.flatten(),
                'amplitude': amplitude
            }

        # Para cada dataset com missing, aplica os métodos
        for miss_rate, missing_data in missing_datasets.items():
            results = []
            data_missing = missing_data['data']
            nan_mask = missing_data['mask']
            amplitude = missing_data['amplitude']
            
            # 1. KNN Imputer (aplicado diretamente nos dados 1D)
            best_knn_nrmse = np.inf
            best_knn_rmse = np.inf
            best_knn_k = None
            
            for k in knn_neighbors_grid:
                try:
                    knn = KNNImputer(n_neighbors=k)
                    X_filled_knn = knn.fit_transform(data_missing.reshape(-1, 1)).flatten()
                    rmse_knn, nrmse_knn = calculate_rmse_nrmse(
                        throughput_vector, X_filled_knn, nan_mask, amplitude
                    )
                    if nrmse_knn < best_knn_nrmse:
                        best_knn_nrmse = nrmse_knn
                        best_knn_rmse = rmse_knn
                        best_knn_k = k
                except Exception as e:
                    continue

            results.append({
                "method": "KNNImputer",
                "shrinkage_value": np.nan,
                "rank": best_knn_k,
                "n_lines": np.nan,  # Não aplicável
                "missing_rate": miss_rate,
                "rmse": best_knn_rmse,
                "nrmse": best_knn_nrmse,
                "dataset_id": dataset_id,
            })

            # 2. Linear Interpolation (aplicado diretamente nos dados 1D)
            df_missing = pd.DataFrame({'throughput': data_missing})
            df_interp = df_missing.interpolate(
                method="linear", limit_direction="both"
            )
            rmse, nrmse = calculate_rmse_nrmse(
                throughput_vector, df_interp['throughput'].values, nan_mask, amplitude
            )
            results.append({
                "method": "LinearInterpolation",
                "shrinkage_value": np.nan,
                "rank": np.nan,
                "n_lines": np.nan,  # Não aplicável
                "missing_rate": miss_rate,
                "rmse": rmse,
                "nrmse": nrmse,
                "dataset_id": dataset_id,
            })

            # 3. Métodos de matriz temporal (SoftImpute e IterativeSVD)
            # Cria diferentes matrizes temporais
            temporal_matrices = create_temporal_matrices(data_missing, line_counts)
            original_matrices = create_temporal_matrices(throughput_vector, line_counts)
            
            for n_lines, temp_matrix in temporal_matrices.items():
                original_matrix = original_matrices[n_lines]
                temp_nan_mask = np.isnan(temp_matrix)
                temp_amplitude = (original_matrix.max() - original_matrix.min()) / 1e6

                # SoftImpute
                best_soft_nrmse = np.inf
                best_soft_rmse = np.inf
                best_soft_shrink = None
                
                for shrinkage in shrinkage_values:
                    try:
                        imputer = SoftImpute(max_iters=1000, shrinkage_value=shrinkage)
                        X_filled_soft = imputer.fit_transform(temp_matrix.copy())
                        rmse, nrmse = calculate_rmse_nrmse(
                            original_matrix, X_filled_soft, temp_nan_mask, temp_amplitude
                        )
                        if nrmse < best_soft_nrmse:
                            best_soft_nrmse = nrmse
                            best_soft_rmse = rmse
                            best_soft_shrink = shrinkage
                    except Exception as e:
                        continue

                results.append({
                    "method": "SoftImpute",
                    "shrinkage_value": best_soft_shrink,
                    "rank": np.nan,
                    "n_lines": n_lines,
                    "missing_rate": miss_rate,
                    "rmse": best_soft_rmse,
                    "nrmse": best_soft_nrmse,
                    "dataset_id": dataset_id,
                })

                # IterativeSVD
                best_svd_nrmse = np.inf
                best_svd_rmse = np.inf
                best_svd_rank = None
                
                for rank in ranks:
                    try:
                        imputer = IterativeSVD(rank=rank, max_iters=1000)
                        X_filled_svd = imputer.fit_transform(temp_matrix.copy())
                        rmse, nrmse = calculate_rmse_nrmse(
                            original_matrix, X_filled_svd, temp_nan_mask, temp_amplitude
                        )
                        if nrmse < best_svd_nrmse:
                            best_svd_nrmse = nrmse
                            best_svd_rmse = rmse
                            best_svd_rank = rank
                    except Exception as e:
                        continue

                results.append({
                    "method": "IterativeSVD",
                    "shrinkage_value": np.nan,
                    "rank": best_svd_rank,
                    "n_lines": n_lines,
                    "missing_rate": miss_rate,
                    "rmse": best_svd_rmse,
                    "nrmse": best_svd_nrmse,
                    "dataset_id": dataset_id,
                })

            # Salva resultados para esta taxa de missing
            results_df = pd.DataFrame(results)
            out_name = f"results_dataset{dataset_id}_missing{int(miss_rate * 100)}.csv"
            results_df.to_csv(out_name, index=False)
            print(f"Resultados salvos em {out_name}")
            print(results_df.head())


if __name__ == "__main__":
    main()