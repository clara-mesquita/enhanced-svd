from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

def impute_arima(df_missing, order=(1,1,1)):
    """
    Imputação usando modelo ARIMA para séries temporais
    """
    df_imp = df_missing.copy()
    
    if df_imp["throughput_bps"].isna().sum() == 0:
        return df_imp
    
    # Identificar blocos de valores missing
    series = df_imp["throughput_bps"].copy()
    missing_mask = series.isna()
    
    # Se não há índice temporal, usar índice numérico
    if not isinstance(df_imp.index, pd.DatetimeIndex):
        # Criar uma cópia temporária com índice numérico para ARIMA
        temp_series = series.reset_index(drop=True)
    else:
        temp_series = series.copy()
    
    # Preencher inicialmente com interpolação linear para treinar o modelo
    temp_filled = temp_series.interpolate(method='linear')
    
    try:
        # Ajustar modelo ARIMA
        model = ARIMA(temp_filled, order=order)
        model_fit = model.fit()
        
        # Fazer previsão para todos os pontos
        predictions = model_fit.predict(start=0, end=len(temp_series)-1)
        
        # Substituir apenas os valores missing
        if not isinstance(df_imp.index, pd.DatetimeIndex):
            missing_indices = temp_series.index[missing_mask].tolist()
        else:
            missing_indices = series.index[missing_mask]
        
        df_imp.loc[missing_indices, "throughput_bps"] = predictions[missing_mask]
        
    except Exception as e:
        print(f"Erro no ARIMA: {e}")
        # Fallback para interpolação linear
        df_imp["throughput_bps"] = df_imp["throughput_bps"].interpolate(method='linear')
    
    return df_imp