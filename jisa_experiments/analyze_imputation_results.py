
import argparse
import os
import math
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============
# I/O Utilities
# =============

def _ensure_dirs(base_out: str) -> Tuple[str, str]:
    plots_dir = os.path.join(base_out, "plots")
    tables_dir = os.path.join(base_out, "summaries")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    return plots_dir, tables_dir

def _read_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Standardize expected column names (case-insensitive match)
    rename_map = {}
    for col in df.columns:
        low = col.strip().lower()
        if low == "dataset":
            rename_map[col] = "dataset"
        elif low in ("missing_rate","missing","miss","missing%","missing_pct","missingrate"):
            rename_map[col] = "missing_rate"
        elif low in ("method","algo","algorithm"):
            rename_map[col] = "method"
        elif low == "rmse":
            rename_map[col] = "rmse"
        elif low == "mae":
            rename_map[col] = "mae"
        elif low in ("n_points","n","npoints","num_points","n_obs","nobs"):
            rename_map[col] = "n_points"
    if rename_map:
        df = df.rename(columns=rename_map)
    required = ["dataset","missing_rate","method","rmse","mae","n_points"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Found columns: {list(df.columns)}")

    # Coerce types
    df["missing_rate"] = pd.to_numeric(df["missing_rate"], errors="coerce")
    df["rmse"] = pd.to_numeric(df["rmse"], errors="coerce")
    df["mae"] = pd.to_numeric(df["mae"], errors="coerce")
    df["n_points"] = pd.to_numeric(df["n_points"], errors="coerce").astype("Int64")

    # Clean method/dataset strings (strip whitespace)
    df["method"] = df["method"].astype(str).str.strip()
    df["dataset"] = df["dataset"].astype(str).str.strip()

    # Drop rows with missing critical values
    df = df.dropna(subset=["missing_rate","method","rmse","mae"])
    return df

# =====================
# Summary Computations
# =====================

def compute_summaries(df: pd.DataFrame) -> dict:
    out = {}

    # Overall by method
    by_method = (df.groupby("method")[["rmse","mae"]]
                   .agg(["mean","median","std","count"])
                   .sort_values(("rmse","mean")))
    out["by_method"] = by_method

    # By method & missing_rate
    by_method_missing = (df.groupby(["method","missing_rate"])[["rmse","mae"]]
                           .agg(["mean","median","std","count"])
                           .sort_values(("rmse","mean"))
                           .reset_index())
    out["by_method_missing"] = by_method_missing

    # By missing_rate (averaging across methods/datasets)
    by_missing = (df.groupby("missing_rate")[["rmse","mae"]]
                    .agg(["mean","median","std","count"])
                    .sort_index())
    out["by_missing_rate"] = by_missing

    # Per (dataset, missing_rate): rank methods (lower is better)
    def _ranks(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        g["rmse_rank"] = g["rmse"].rank(method="min", ascending=True)
        g["mae_rank"] = g["mae"].rank(method="min", ascending=True)
        return g

    ranked = (df.groupby(["dataset","missing_rate"], as_index=False)
                .apply(_ranks)
                .reset_index(drop=True))
    out["ranked"] = ranked

    # Win rates by method (how often a method is best per matchup) 
    wins_rmse = (ranked.loc[ranked["rmse_rank"]==1]
                       .groupby("method").size()
                       .rename("rmse_wins"))
    wins_mae = (ranked.loc[ranked["mae_rank"]==1]
                      .groupby("method").size()
                      .rename("mae_wins"))
    matchups_count = ranked.groupby(["dataset","missing_rate"]).size().groupby(level=[0,1]).size().shape[0]

    win_table = pd.concat([wins_rmse, wins_mae], axis=1).fillna(0).astype(int)
    win_table["matchups_total"] = matchups_count
    win_table["rmse_win_rate"] = win_table["rmse_wins"] / win_table["matchups_total"]
    win_table["mae_win_rate"] = win_table["mae_wins"] / win_table["matchups_total"]
    win_table = win_table.sort_values(["rmse_win_rate","mae_win_rate"], ascending=False)
    out["win_table"] = win_table

    return out

# ==============
# Plotting Utils
# ==============

def plot_bar_avg(df: pd.DataFrame, metric: str, outdir: str):
    avg = df.groupby("method")[metric].mean().sort_values(ascending=True)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.barh(avg.index, avg.values)
    ax.set_xlabel(f"Average {metric.upper()} (lower is better)")
    ax.set_ylabel("Method")
    ax.set_title(f"Average {metric.upper()} by Method")
    fig.tight_layout()
    fp = os.path.join(outdir, f"bar_avg_{metric}.png")
    fig.savefig(fp, dpi=180)
    plt.close(fig)
    return fp

def plot_box_by_method(df: pd.DataFrame, metric: str, outdir: str):
    methods = list(df["method"].unique())
    data = [df.loc[df["method"]==m, metric].values for m in methods]
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.boxplot(data, labels=methods, vert=True, showmeans=True)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} Distribution by Method")
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fp = os.path.join(outdir, f"box_{metric}_by_method.png")
    fig.savefig(fp, dpi=180)
    plt.close(fig)
    return fp

def plot_lines_missing_rate(df: pd.DataFrame, metric: str, outdir: str):
    # Average across datasets for clarity
    pivot = (df.groupby(["method","missing_rate"])[metric].mean()
               .reset_index()
               .pivot(index="missing_rate", columns="method", values=metric)
               .sort_index())
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    for method in pivot.columns:
        ax.plot(pivot.index.values, pivot[method].values, marker="o", label=method)
    ax.set_xlabel("Missing rate")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} vs Missing Rate (averaged across datasets)")
    ax.legend(loc="best")
    fig.tight_layout()
    fp = os.path.join(outdir, f"lines_{metric}_vs_missing_rate.png")
    fig.savefig(fp, dpi=180)
    plt.close(fig)
    return fp

def plot_scatter_metric_vs_npoints(df: pd.DataFrame, metric: str, outdir: str):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    # Scatter, one series per method
    for method, g in df.groupby("method"):
        ax.scatter(g["n_points"].astype(float).values, g[metric].values, alpha=0.7, label=method)
    ax.set_xlabel("n_points")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} vs n_points")
    ax.legend(loc="best")
    ax.set_xscale("linear")
    fig.tight_layout()
    fp = os.path.join(outdir, f"scatter_{metric}_vs_npoints.png")
    fig.savefig(fp, dpi=180)
    plt.close(fig)
    return fp

def plot_heatmap_method_missing(df: pd.DataFrame, metric: str, outdir: str):
    # Heatmap of mean metric per (method, missing_rate)
    pv = (df.groupby(["method","missing_rate"])[metric].mean()
            .reset_index()
            .pivot(index="method", columns="missing_rate", values=metric))
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(pv.values, aspect="auto")
    ax.set_xticks(range(len(pv.columns)))
    ax.set_xticklabels(pv.columns.astype(str), rotation=0)
    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels(pv.index)
    ax.set_xlabel("Missing rate")
    ax.set_ylabel("Method")
    ax.set_title(f"Mean {metric.upper()} (Method x Missing rate)")
    # Add value annotations
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            val = pv.values[i, j]
            if not (isinstance(val, float) and math.isnan(val)):
                ax.text(j, i, f"{val:.2e}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fp = os.path.join(outdir, f"heatmap_{metric}_method_missing.png")
    fig.savefig(fp, dpi=180)
    plt.close(fig)
    return fp

# ===================
# Public Entry Points
# ===================

def analyze_results(csv_path: str, outdir: str = "./imputation_report") -> dict:
    plots_dir, tables_dir = _ensure_dirs(outdir)
    df = _read_and_clean(csv_path)
    summaries = compute_summaries(df)

    # Save summary tables
    summaries["by_method"].to_csv(os.path.join(tables_dir, "by_method.csv"))
    summaries["by_method_missing"].to_csv(os.path.join(tables_dir, "by_method_missing.csv"), index=False)
    summaries["by_missing_rate"].to_csv(os.path.join(tables_dir, "by_missing_rate.csv"))
    summaries["ranked"].to_csv(os.path.join(tables_dir, "ranked_per_matchup.csv"), index=False)
    summaries["win_table"].to_csv(os.path.join(tables_dir, "win_table.csv"))

    # Make plots
    generated = []
    for metric in ("rmse","mae"):
        generated.append(plot_bar_avg(df, metric, plots_dir))
        generated.append(plot_box_by_method(df, metric, plots_dir))
        generated.append(plot_lines_missing_rate(df, metric, plots_dir))
        generated.append(plot_scatter_metric_vs_npoints(df, metric, plots_dir))
        generated.append(plot_heatmap_method_missing(df, metric, plots_dir))

    return {"outdir": outdir, "plots": generated, "tables_dir": tables_dir}

def main():
    parser = argparse.ArgumentParser(description="Analyze imputation results CSV (methods x missing_rate x metrics).")
    parser.add_argument("--csv", required=True, help="Path to results CSV with columns: dataset,missing_rate,method,rmse,mae,n_points")
    parser.add_argument("--outdir", default="./imputation_report", help="Output directory for plots and tables")
    args = parser.parse_args()

    res = analyze_results(args.csv, args.outdir)

    print("Analysis complete.")
    print(f"Summaries saved to: {res['tables_dir']}")
    print("Generated plots:")
    for p in res["plots"]:
        print(" -", p)

if __name__ == "__main__":
    main()
