def impute_moving_average(df_missing, window=5, center=True):
    """
    Imputa NaNs em 'throughput_bps' usando média móvel (SMA).
    
    Parâmetros:
      window: tamanho da janela (3–10 usualmente)
      center: True para janela centrada (vizinho passado e futuro)
    """
    df_imp = df_missing.copy()
    y = df_imp["throughput_bps"]
    
    # Primeiro, preenche os NaNs com um método simples para calcular a média móvel
    # Usamos ffill e bfill para garantir que não haja NaNs na entrada
    y_filled = y.ffill().bfill()
    
    # Se todos os valores forem NaN mesmo após o preenchimento, use um valor padrão
    if y_filled.isna().all():
        y_filled = y_filled.fillna(0)
    
    # Calcula a média móvel
    sma = y_filled.rolling(window=window, center=center, min_periods=1).mean()
    
    # Substitui apenas os pontos que eram NaN originalmente
    mask_missing = y.isna()
    df_imp.loc[mask_missing, "throughput_bps"] = sma[mask_missing].values
    
    return df_imp