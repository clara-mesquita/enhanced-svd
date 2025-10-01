#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pathlib
import numpy as np
import pandas as pd


def load_and_impute(csv_path: str) -> pd.DataFrame:
    """
    Lê o CSV (colunas: Data, Intervalo, Vazao), converte -1 em NaN
    e faz interpolação (time se possível, senão linear) em Vazao.
    """
    df = pd.read_csv(csv_path)

    # Checagem básica
    expected = {"Data", "Intervalo", "Vazao"}
    missing_cols = expected - set(df.columns)
    if missing_cols:
        raise ValueError(f"Faltam colunas no CSV: {missing_cols}")

    # Garantir numérico e trocar sentinela -1 por NaN
    df = df.copy()
    df["Vazao"] = pd.to_numeric(df["Vazao"], errors="coerce")
    df.loc[df["Vazao"] == -1, "Vazao"] = np.nan

    # Tentar montar timestamp: muitos datasets BR são dia/mes/ano
    ts = pd.to_datetime(
        df["Data"].astype(str) + " " + df["Intervalo"].astype(str),
        errors="coerce",
        dayfirst=True,          # útil para formatos 31/12/2024
        infer_datetime_format=True,
    )
    df["__ts__"] = ts

    # Se todos válidos, interpolar no tempo; senão, fallback para linear
    if not df["__ts__"].isna().any():
        df = df.sort_values("__ts__").set_index("__ts__", drop=True)
        # Interpolação temporal bidirecional (preenche bordas também)
        df["Vazao"] = df["Vazao"].interpolate(method="time", limit_direction="both")
        df = df.reset_index(drop=True)
    else:
        # Fallback seguro: mantém ordem original e usa linear
        df["Vazao"] = df["Vazao"].interpolate(method="linear", limit_direction="both")

    # Limpeza
    df.drop(columns=["__ts__"], errors="ignore", inplace=True)
    return df


def make_mcar_versions(df: pd.DataFrame, seed: int = 42):
    """
    Cria versões com 10%, 20%, 30% e 40% MCAR em 'Vazao'.
    Retorna dict {percentual:int -> DataFrame}.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    y = pd.to_numeric(df["Vazao"], errors="coerce").to_numpy()

    versions = {}
    for frac in [0.10, 0.20, 0.30, 0.40]:
        k = int(round(frac * n))
        idx = rng.choice(n, size=k, replace=False)
        y_new = y.copy()
        y_new[idx] = np.nan
        df_mcar = df.copy()
        df_mcar["Vazao"] = y_new
        versions[int(frac * 100)] = df_mcar

    return versions


def main():
    ap = argparse.ArgumentParser(description="Imputar -1 e gerar versões MCAR (10/20/30/40%).")
    ap.add_argument("csv_path", help="Caminho do CSV (colunas: Data, Intervalo, Vazao)")
    ap.add_argument("--seed", type=int, default=42, help="Semente do sorteio MCAR (default: 42)")
    args = ap.parse_args()

    in_path = pathlib.Path(args.csv_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {in_path}")

    df_imp = load_and_impute(str(in_path))

    # Salva imputado
    out_dir = in_path.parent
    stem = in_path.stem
    imputed_path = out_dir / f"{stem}_imputed.csv"
    df_imp.to_csv(imputed_path, index=False)
    print(f"[OK] Imputado salvo em: {imputed_path}")

    # Versões MCAR
    versions = make_mcar_versions(df_imp, seed=args.seed)
    for pct, df_mcar in versions.items():
        out_path = out_dir / f"{stem}_mcar_{pct}.csv"
        df_mcar.to_csv(out_path, index=False)
        print(f"[OK] MCAR {pct}% salvo em: {out_path}")


if __name__ == "__main__":
    main()
