def impute_linear_interpolation(df_missing):
    df_imputed = df_missing.copy()
    df_imputed["throughput_bps"] = df_imputed["throughput_bps"].interpolate(
        method="linear", limit_direction="both"
    )
    return df_imputed