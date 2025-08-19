import numpy as np
import pandas as pd
from fancyimpute import SoftImpute, IterativeSVD
from itertools import product
import os


def introduce_missing(data, missing_rate, random_state=None):
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
    rmse = np.sqrt(np.mean((original[nan_mask] / 1e6 - imputed[nan_mask] / 1e6) ** 2))
    nrmse = rmse / amplitude if amplitude != 0 else np.nan
    return rmse, nrmse


def main():
    # Lista de datasets (ajuste os paths conforme necessário)
    dataset_ids = [1, 2, 3, 4]
    dataset_paths = [
        f"institution_subnets/institution_subnets/agg_1_hour/{i}.csv"
        for i in dataset_ids
    ]

    shrinkage_values = [0.1]
    ranks = [4]
    line_counts = [24, 120, 168, 336, 504, 672]  # tamanhos de janela
    missing_rates = [0.1, 0.2, 0.3, 0.4]

    for dataset_id, dataset_path in zip(dataset_ids, dataset_paths):
        # Carregue seu dataset real
        df = pd.read_csv(dataset_path)
        df["throughput"] = df["n_bytes"] / 3600 * 8  # bps
        throughput_vector = df["throughput"].values

        for miss_rate in missing_rates:
            results = []
            for n_lines in line_counts:
                columns_quantity = len(throughput_vector) // n_lines
                if columns_quantity == 0:
                    continue  # ignora se não há dados suficientes

                # Matriz temporal original (shape: n_lines x columns_quantity)
                temporal_matrix = (
                    throughput_vector[: columns_quantity * n_lines]
                    .reshape(columns_quantity, n_lines)
                    .T
                )
                amplitude = (temporal_matrix.max() - temporal_matrix.min()) / 1e6

                # Introduzir missing (nova amostra a cada grid!)
                data_missing, nan_mask = introduce_missing(
                    temporal_matrix, miss_rate, random_state=42
                )

                # SoftImpute
                for shrinkage in shrinkage_values:
                    imputer = SoftImpute(max_iters=1000, shrinkage_value=shrinkage)
                    X_filled_soft = imputer.fit_transform(data_missing)
                    rmse, nrmse = calculate_rmse_nrmse(
                        temporal_matrix, X_filled_soft, nan_mask, amplitude
                    )
                    results.append(
                        {
                            "method": "SoftImpute",
                            "shrinkage_value": shrinkage,
                            "rank": np.nan,
                            "n_lines": n_lines,
                            "missing_rate": miss_rate,
                            "rmse": rmse,
                            "nrmse": nrmse,
                            "dataset_id": dataset_id,
                        }
                    )

                # IterativeSVD
                for rank in ranks:
                    imputer = IterativeSVD(rank=rank, max_iters=1000)
                    X_filled_svd = imputer.fit_transform(data_missing)
                    rmse, nrmse = calculate_rmse_nrmse(
                        temporal_matrix, X_filled_svd, nan_mask, amplitude
                    )
                    results.append(
                        {
                            "method": "IterativeSVD",
                            "shrinkage_value": np.nan,
                            "rank": rank,
                            "n_lines": n_lines,
                            "missing_rate": miss_rate,
                            "rmse": rmse,
                            "nrmse": nrmse,
                            "dataset_id": dataset_id,
                        }
                    )

            # Salvar resultados em CSV: um arquivo por missing_rate por dataset
            results_df = pd.DataFrame(results)
            out_name = f"results_dataset{dataset_id}_missing{int(miss_rate * 100)}.csv"
            results_df.to_csv(out_name, index=False)
            print(f"Resultados salvos em {out_name}")
            print(results_df.head(3))


if __name__ == "__main__":
    main()
