import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents

def impute_kalman(
    df_missing,
    model="arima",
    # --- ARIMA/SARIMA params ---
    arima_order=(1, 1, 1),
    seasonal_order=(0, 0, 0, 0),
    # --- Estrutural params ---
    level="local level",         # opções comuns: "local level", "local linear trend"
    seasonal_period=None         # por ex., 24 p/ sazonalidade diária em dados horários
):
    """
    Imputa NaNs em 'throughput_bps' via Kalman smoothing.

    Parâmetros
    ----------
    df_missing : pd.DataFrame
        DataFrame com colunas 'time' (datetime string com timezone) e 'throughput_bps'.
    model : {'arima', 'structural'}
        Escolhe a abordagem:
        - 'arima': ajusta SARIMAX(ARIMA/SARIMA) e usa as previsões in-sample (Kalman).
        - 'structural': ajusta UnobservedComponents (nível/trend/seasonal) + Kalman.
    arima_order : tuple
        Ordem (p, d, q) do ARIMA.
    seasonal_order : tuple
        Ordem sazonal (P, D, Q, s) para SARIMA.
    level : str
        Componente de nível do modelo estrutural (ex.: 'local level', 'local linear trend').
    seasonal_period : int or None
        Período sazonal para o modelo estrutural (ex.: 24, 7*24, etc.). None = sem sazonalidade.

    Retorna
    -------
    df_imputed : pd.DataFrame
        Cópia de df_missing com 'throughput_bps' imputado nos pontos NaN.
    """
    df_imp = df_missing.copy()

    # Garante dtype datetime (assumido correto e com tz, sem tratamento de erros)
    t = pd.to_datetime(df_imp["time"])
    y = df_imp["throughput_bps"].astype(float)

    # Máscara de faltantes
    miss_mask = y.isna()

    if model == "arima":
        # SARIMAX lida com NaNs na endógena e usa Kalman internamente
        mod = SARIMAX(
            y,
            order=arima_order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res = mod.fit(disp=False)

        # Previsão in-sample (predicted_mean) já incorpora filtro/smoother de Kalman
        y_hat = res.get_prediction().predicted_mean

    elif model == "structural":
        # Modelo estrutural: nível / tendência / sazonal (state-space) + Kalman
        # Ex.: level='local level' ou 'local linear trend'
        #     seasonal_period define sazonalidade (ex.: 24 p/ hora, 7*24 p/ semanal, etc.)
        ucm = UnobservedComponents(
            y,
            level=level,
            seasonal=seasonal_period  # None => sem sazonal
        )
        res = ucm.fit(disp=False)

        # predicted_mean é a série observável estimada (alisada) pelo modelo
        y_hat = res.get_prediction().predicted_mean

    else:
        raise ValueError("model must be 'arima' or 'structural'")

    # Imputa apenas onde havia NaN, preservando os valores observados
    y_imp = y.copy()
    y_imp[miss_mask] = y_hat[miss_mask].values

    df_imp["throughput_bps"] = y_imp.values
    return df_imp
