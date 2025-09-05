def impute_moving_average(df_missing, window=5):
    """
    Imputação usando média móvel - boa para dados com ruído
    """
    df_imp = df_missing.copy()
    y = df_imp["throughput_bps"].astype(float)
    
    # Primeira passada: interpolação linear
    y_temp = y.interpolate(method='linear')
    
    # Segunda passada: média móvel
    y_ma = y_temp.rolling(window=window, center=True, min_periods=1).mean()
    
    # Preservar valores originais não-NaN
    y_imputed = y.copy()
    missing_mask = y.isna()
    y_imputed[missing_mask] = y_ma[missing_mask]
    
    # Preencher extremidades se necessário
    y_imputed = y_imputed.fillna(method='ffill').fillna(method='bfill')
    
    df_imp["throughput_bps"] = y_imputed
    return df_imp
