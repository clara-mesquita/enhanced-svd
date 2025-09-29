from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

def impute_pca(df_missing, n_components=1):
    """
    Imputação usando PCA (Principal Component Analysis)
    """
    df_imp = df_missing.copy()
    
    if df_imp["throughput_bps"].isna().sum() == 0:
        return df_imp
    
    # Primeiro, imputar valores missing com média para poder aplicar PCA
    initial_imputer = SimpleImputer(strategy='mean')
    values_initial = initial_imputer.fit_transform(df_imp[["throughput_bps"]])
    
    # Aplicar PCA
    pca = PCA(n_components=n_components)
    pca_values = pca.fit_transform(values_initial)
    
    # Reconstruir os dados
    reconstructed = pca.inverse_transform(pca_values)
    
    # Substituir apenas os valores missing
    mask_missing = df_imp["throughput_bps"].isna()
    df_imp.loc[mask_missing, "throughput_bps"] = reconstructed[mask_missing, 0]
    
    return df_imp