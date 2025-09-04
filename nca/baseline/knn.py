from sklearn.impute import KNNImputer

def impute_knn_imputer(df_missing, k=5):
    df_imp = df_missing.copy()

    imputer = KNNImputer(n_neighbors=k, weights="uniform")
    imputed_values = imputer.fit_transform(df_imp[["throughput_bps"]])

    df_imp["throughput_bps"] = imputed_values[:, 0]
    return df_imp