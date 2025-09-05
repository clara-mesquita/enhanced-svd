import numpy as np
import pandas as pd
from scipy import stats, interpolate
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')

def impute_seasonal_decompose(df_missing, period=24):
    """
    Imputação considerando sazonalidade - útil para dados com padrões cíclicos
    """
    df_imp = df_missing.copy()
    y = df_imp["throughput_bps"].astype(float)

    # Primeira passada: interpolação para ter série completa
    y_temp = y.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
    
    # Decomposição sazonal simples
    # Tendência usando média móvel
    trend = y_temp.rolling(window=period, center=True, min_periods=1).mean()
    
    # Sazonalidade
    detrended = y_temp - trend
    seasonal_pattern = detrended.groupby(np.arange(len(detrended)) % period).transform('mean')
    
    # Resíduo
    residual = detrended - seasonal_pattern
    
    # Reconstruir série para valores faltantes
    y_imputed = y.copy()
    missing_mask = y.isna()
    
    reconstructed = trend + seasonal_pattern + residual.rolling(3, center=True, min_periods=1).mean().fillna(0)
    y_imputed[missing_mask] = reconstructed[missing_mask]
    
    df_imp["throughput_bps"] = y_imputed
        
    
    return df_imp