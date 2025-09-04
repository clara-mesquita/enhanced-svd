import os
import argparse
import pandas as pd
import numpy as np


def introduce_missing_data(df, missing_rate, seed=42):
    """Introduz missing data na coluna throughput_bps"""
    rng = np.random.default_rng(seed)
    df_missing = df.copy()
    mask = rng.random(len(df_missing)) < missing_rate
    df_missing.loc[mask, "throughput_bps"] = np.nan
    print(f"Introduced {missing_rate * 100:.0f}% missing data.")
    return df_missing


def main(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(input_dir, file))
            base_key = file.removesuffix(".throughput.csv")

            for rate in [0.1, 0.2, 0.3, 0.4]:
                df_missing = introduce_missing_data(df, missing_rate=rate, seed=42)
                rate_key = f"{int(rate * 100)}"
                output_file = f"{base_key}_missing_{rate_key}.csv"
                df_missing.to_csv(os.path.join(output_dir, output_file), index=False)
                print(f"Saved: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Introduz missing data em datasets CSV.")
    parser.add_argument("--input_dir", type=str, required=True, help="Diretório com arquivos de entrada (.csv)")
    parser.add_argument("--output_dir", type=str, required=True, help="Diretório para salvar os arquivos de saída")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
    print("Missing data introduction completed.")
    # Usage example:
    # python nca/create_missing_data.py --input_dir ./cesnet-institutions-throughput/institutions/agg_6_hours --output ./cesnet-institutions-throughput/institutions/agg_6_hours_missing