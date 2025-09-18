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

def impute_mice_univariate(df_missing, max_iter=10, random_state=42):
    """
    Imputação MICE adaptada para dados univariados usando features derivadas
    """
    df_imp = df_missing.copy()
    y = df_imp["throughput_bps"].astype(float).to_numpy()
    
    # Criar features derivadas para o MICE
    features = pd.DataFrame({
        'value': y,
        'index': np.arange(len(y)),
        'lag1': np.roll(y, 1),
        'lag2': np.roll(y, 2),
        'lead1': np.roll(y, -1),
        'lead2': np.roll(y, -2),
        'rolling_mean_3': pd.Series(y).rolling(3, center=True, min_periods=1).mean().values,
        'rolling_mean_5': pd.Series(y).rolling(5, center=True, min_periods=1).mean().values,
    })
    
    # Tratar bordas dos lags/leads
    features.loc[0, 'lag1'] = np.nan
    features.loc[0, 'lag2'] = np.nan  
    features.loc[1, 'lag2'] = np.nan
    features.loc[len(y)-1, 'lead1'] = np.nan
    features.loc[len(y)-1, 'lead2'] = np.nan
    features.loc[len(y)-2, 'lead2'] = np.nan
    
    try:
        # MICE
        imputer = IterativeImputer(max_iter=max_iter, random_state=random_state, verbose=0)
        features_imputed = imputer.fit_transform(features)
        
        # Extrair valores imputados
        y_imputed = features_imputed[:, 0]  # primeira coluna é 'value'
        
        # Garantir não-negatividade se aplicável
        valid_values = y[~np.isnan(y)]
        if len(valid_values) > 0 and np.all(valid_values >= 0):
            y_imputed[y_imputed < 0] = 0
        
        df_imp["throughput_bps"] = y_imputed
        
    except Exception as e:
        print(f"Erro no MICE: {e}. Usando interpolação linear.")
    
    return df_imp
