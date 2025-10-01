#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import re


# ---------------------------
# Helpers de parsing/normalização
# ---------------------------

def _try_parse_ts(series: pd.Series) -> pd.Series:
    """Tenta parsear datetime de forma tolerante (BR + US)."""
    # Primeiro tenta formatos mais comuns e rápidos; depois cai pro parser flexível
    fmts = [
        "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
        "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M",
        "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M",
    ]
    for fmt in fmts:
        ts = pd.to_datetime(series, format=fmt, errors="coerce")
        if ts.notna().any():
            # Se esse formato funcionou para a maioria, use-o
            ok = ts.notna().mean()
            if ok > 0.7:
                return ts
    # Fallback: parser flexível (lento, mas robusto)
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def _normalize_intervalo(s: pd.Series) -> pd.Series:
    """
    Normaliza 'Intervalo' para HH:MM:SS.
    Aceita entradas como '7:5', '07:5', '7:05:9', '07:05', '0705', '705', etc.
    """
    def norm_one(x: str) -> str:
        if x is None:
            return None
        x = str(x).strip()
        if not x:
            return None

        # Se vier algo como '705' ou '0705', tenta interpretar como HHMM
        if re.fullmatch(r"\d{3,4}", x):
            # HHMM
            if len(x) == 3:
                h, m = int(x[0]), int(x[1:])
            else:
                h, m = int(x[:2]), int(x[2:])
            return f"{h:02d}:{m:02d}:00"

        # Divide por ':' e zera à esquerda
        parts = x.split(":")
        try:
            parts = [int(p) for p in parts if p != ""]
        except ValueError:
            return None

        if len(parts) == 1:
            h = parts[0]
            m = 0
            s = 0
        elif len(parts) == 2:
            h, m = parts
            s = 0
        else:
            h, m, s = (parts + [0, 0, 0])[:3]

        # Corrige limites básicos
        h = max(0, min(23, h))
        m = max(0, min(59, m))
        s = max(0, min(59, s))
        return f"{h:02d}:{m:02d}:{s:02d}"

    return s.astype(str).map(norm_one)


def _ensure_columns(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Garante que existam colunas Data, Intervalo, Vazao.
    Aceita timestamp (gera Data/Intervalo) e value/valor como Vazao.
    """
    df = df.copy()

    # Resolve VAZAO
    if "Vazao" not in df.columns:
        for alt in ["value", "valor", "throughput", "bw"]:
            if alt in df.columns:
                df["Vazao"] = df[alt]
                break
    df["Vazao"] = pd.to_numeric(df.get("Vazao", pd.Series(index=df.index)), errors="coerce")

    # Se houver timestamp único, derive Data/Intervalo
    if "timestamp" in df.columns and ("Data" not in df.columns or "Intervalo" not in df.columns):
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
        # Se vier com timezone, remove
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_convert(None)
        df["Data"] = ts.dt.strftime("%Y-%m-%d")
        df["Intervalo"] = ts.dt.strftime("%H:%M:%S")

    # Agora garanta Data/Intervalo
    if "Data" not in df.columns or "Intervalo" not in df.columns:
        raise ValueError(f"[{name}] Não encontrei colunas Data/Intervalo nem timestamp para derivar.")

    # Normaliza Intervalo
    df["Intervalo"] = _normalize_intervalo(df["Intervalo"])
    # Monta string completa para parsing conjunto
    datetime_str = (df["Data"].astype(str).str.strip() + " " + df["Intervalo"].astype(str).str.strip())
    ts = _try_parse_ts(datetime_str)

    # Relaxe: se ainda houver NaT, tente parsear só Data e colar Intervalo literal
    missing = ts.isna()
    if missing.any():
        # Tenta mais um passo: parse Data puro e anexar Intervalo já normalizado
        d_only = pd.to_datetime(df.loc[missing, "Data"].astype(str), errors="coerce", dayfirst=True)
        fix = pd.to_datetime(d_only.dt.strftime("%Y-%m-%d") + " " + df.loc[missing, "Intervalo"].astype(str),
                             errors="coerce", format="%Y-%m-%d %H:%M:%S")
        ts.loc[missing] = fix

    df["__ts__"] = ts
    # Cria chave canônica 'YYYY-MM-DD HH:MM:SS'
    df["__key__"] = df["__ts__"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def _align_and_rmse(df_ref: pd.DataFrame, df_test: pd.DataFrame, ref_name: str, test_name: str) -> float:
    """Alinha por __key__ (ts canônico em segundo) e calcula RMSE."""
    # Diagnóstico antes do merge
    n_ref = df_ref["__key__"].notna().sum()
    n_test = df_test["__key__"].notna().sum()

    merged = df_ref[["__key__", "Vazao"]].merge(
        df_test[["__key__", "Vazao"]],
        on="__key__", how="inner", suffixes=("_ref", "_test"), validate="one_to_one"
    )

    inter = len(merged)
    print(f"[DIAG] {ref_name}: chaves válidas = {n_ref}")
    print(f"[DIAG] {test_name}: chaves válidas = {n_test}")
    print(f"[DIAG] Interseção de chaves = {inter}")

    if inter == 0:
        # Mostra alguns exemplos para depuração
        ref_keys = set(df_ref["__key__"].dropna().unique()[:10])
        test_keys = set(df_test["__key__"].dropna().unique()[:10])
        print("[HINT] Exemplos de chaves no REF:", sorted(ref_keys)[:5])
        print("[HINT] Exemplos de chaves no TEST:", sorted(test_keys)[:5])
        print("[HINT] Verifique se os segundos existem em ambos, se o fuso horário está igual, e se há zeros à esquerda.")
        return float("nan")

    y_true = merged["Vazao_ref"].to_numpy(dtype=float)
    y_pred = merged["Vazao_test"].to_numpy(dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


# ---------------------------
# Interpolação linear para MCAR
# ---------------------------

def interpolate_linear(df: pd.DataFrame) -> pd.DataFrame:
    """Interpola linearmente a Vazao usando eixo temporal quando possível; senão, linear simples."""
    out = df.copy()
    ts = out["__ts__"]
    out = out.copy()
    if ts.notna().all():
        out = out.set_index("__ts__", drop=False).sort_index()
        out["Vazao"] = pd.to_numeric(out["Vazao"], errors="coerce")
        out["Vazao"] = out["Vazao"].interpolate(method="time", limit_direction="both")
        out = out.reset_index(drop=True)
    else:
        out["Vazao"] = pd.to_numeric(out["Vazao"], errors="coerce")
        out["Vazao"] = out["Vazao"].interpolate(method="linear", limit_direction="both")
    return out


# ---------------------------
# CLI
# ---------------------------

def cmd_rmse_pair(ref_path: Path, test_path: Path):
    df_ref = _ensure_columns(pd.read_csv(ref_path), ref_path.name)
    df_test = _ensure_columns(pd.read_csv(test_path), test_path.name)
    score = _align_and_rmse(df_ref, df_test, ref_path.name, test_path.name)
    print(f"RMSE({test_path.name} vs {ref_path.name}) = {score:.6f}")


def cmd_rmse_impute(ref_path: Path, mcar_path: Path, save: bool):
    df_ref = _ensure_columns(pd.read_csv(ref_path), ref_path.name)
    df_mcar = _ensure_columns(pd.read_csv(mcar_path), mcar_path.name)

    df_mcar_imp = interpolate_linear(df_mcar)
    # Recalcula chave após possível reordenação
    df_mcar_imp["__key__"] = df_mcar_imp["__ts__"].dt.strftime("%Y-%m-%d %H:%M:%S")

    score = _align_and_rmse(df_ref, df_mcar_imp, ref_path.name, mcar_path.name + " (interp)")
    print(f"RMSE({mcar_path.name}_interp vs {ref_path.name}) = {score:.6f}")

    if save:
        out_path = mcar_path.with_name(mcar_path.stem + "_imputed_linear.csv")
        df_mcar_imp[["Data", "Intervalo", "Vazao"]].to_csv(out_path, index=False)
        print(f"[OK] MCAR interpolado salvo em: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="RMSE de Vazao entre CSVs + interpolação linear em MCAR.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("rmse-pair", help="RMSE entre resultado e baseline (alinhando por timestamp canônico).")
    p1.add_argument("--ref", required=True, help="CSV baseline (ex.: intervalo_logo_pt_imputed.csv)")
    p1.add_argument("--test", required=True, help="CSV de resultado (ex.: jisa_experiments/resultado2.csv)")

    p2 = sub.add_parser("rmse-impute", help="Interpola linearmente *mcar* e calcula RMSE vs baseline.")
    p2.add_argument("--ref", required=True, help="CSV baseline (ex.: intervalo_logo_pt_imputed.csv)")
    p2.add_argument("--mcar", required=True, help="CSV MCAR (ex.: intervalo_logo_pt_mcar_10.csv)")
    p2.add_argument("--save", action="store_true", help="Salvar *_imputed_linear.csv")

    args = ap.parse_args()

    if args.cmd == "rmse-pair":
        cmd_rmse_pair(Path(args.ref), Path(args.test))
    elif args.cmd == "rmse-impute":
        cmd_rmse_impute(Path(args.ref), Path(args.mcar), args.save)


if __name__ == "__main__":
    main()
