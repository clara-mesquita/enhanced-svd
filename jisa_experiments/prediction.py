import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit
import json
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURAÇÕES - AJUSTE AQUI
# ============================================
INPUT_FOLDER = './imputation_results_original'
OUTPUT_MODEL_FOLDER = './modelo_salvo'
OUTPUT_JSON = 'evaluation_rmse_mae.json'
TRAIN_TEST_SPLIT = 0.8
LOOK_BACK = 3
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 5
N_SPLITS = 3  # Reduzido para 3 para acelerar grid search

# Grid Search Parameters
GRID_SEARCH_ENABLED = True
PARAM_GRID = {
    'gru_units': [32, 64, 128],
    'learning_rate': [0.001, 0.0001],
    'look_back': [3, 5, 7],
    'dropout_rate': [0.0, 0.2]
}
# ============================================

def create_dataset(X, look_back):
    """Cria dataset com janela temporal para séries temporais"""
    Xs, ys = [], []
    for i in range(len(X) - look_back):
        Xs.append(X[i:i+look_back])
        ys.append(X[i+look_back])
    return np.array(Xs), np.array(ys)

def create_gru_model(units, train_shape, learning_rate, dropout_rate=0.0):
    """Cria modelo GRU com opção de dropout"""
    model = Sequential()
    model.add(GRU(units=units, return_sequences=True, input_shape=[train_shape[1], train_shape[2]]))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    model.add(GRU(units=units))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(
        loss=MeanSquaredError(), 
        optimizer=Adam(learning_rate=learning_rate), 
        metrics=[RootMeanSquaredError()]
    )
    return model

def fit_model_with_cross_validation(model, xtrain, ytrain, patience, epochs, batch_size, n_splits):
    """Treina modelo com validação cruzada temporal"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    val_losses = []
    
    for fold_idx, (train_index, val_index) in enumerate(tscv.split(xtrain)):
        print(f"  Fold {fold_idx + 1}/{n_splits}")
        x_train_fold, x_val_fold = xtrain[train_index], xtrain[val_index]
        y_train_fold, y_val_fold = ytrain[train_index], ytrain[val_index]
        
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=patience, 
            restore_best_weights=True,
            verbose=0
        )
        
        history = model.fit(
            x_train_fold, y_train_fold, 
            epochs=epochs, 
            validation_data=(x_val_fold, y_val_fold), 
            batch_size=batch_size, 
            callbacks=[early_stop], 
            verbose=0
        )
        
        val_losses.append(min(history.history['val_loss']))
    
    mean_val_loss = np.mean(val_losses)
    return mean_val_loss

def grid_search(train_scaled, param_grid, epochs, batch_size, patience, n_splits):
    """Realiza grid search para encontrar melhores hiperparâmetros"""
    print("\n" + "="*60)
    print("INICIANDO GRID SEARCH")
    print("="*60)
    
    # Gera todas as combinações de parâmetros
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                         for v in product(*param_grid.values())]
    
    best_score = float('inf')
    best_params = None
    results = []
    
    for idx, params in enumerate(param_combinations):
        print(f"\nTestando combinação {idx + 1}/{len(param_combinations)}: {params}")
        
        try:
            # Cria dataset com look_back específico
            X_train, y_train = create_dataset(train_scaled, params['look_back'])
            
            # Cria modelo com parâmetros atuais
            model = create_gru_model(
                units=params['gru_units'],
                train_shape=X_train.shape,
                learning_rate=params['learning_rate'],
                dropout_rate=params['dropout_rate']
            )
            
            # Treina e avalia
            val_loss = fit_model_with_cross_validation(
                model, X_train, y_train, patience, epochs, batch_size, n_splits
            )
            
            print(f"  Validation Loss: {val_loss:.6f}")
            
            results.append({
                'params': params,
                'val_loss': val_loss
            })
            
            # Atualiza melhores parâmetros
            if val_loss < best_score:
                best_score = val_loss
                best_params = params
                print(f"  *** Novo melhor resultado! ***")
            
            # Limpa memória
            del model
            tf.keras.backend.clear_session()
            
        except Exception as e:
            print(f"  Erro: {e}")
            results.append({
                'params': params,
                'error': str(e)
            })
    
    print("\n" + "="*60)
    print("GRID SEARCH CONCLUÍDO")
    print(f"Melhores parâmetros: {best_params}")
    print(f"Melhor validation loss: {best_score:.6f}")
    print("="*60 + "\n")
    
    return best_params, results

def save_model(model, directory, filename):
    """Salva modelo treinado"""
    if directory:
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, f'{filename}_GRU.keras')
        model.save(file_path)
        print(f"Modelo salvo em: '{file_path}'")

def predict_and_evaluate(model, xtest, ytest, scaler):
    """Faz predições e calcula métricas"""
    predictions = model.predict(xtest, verbose=0)
    predictions_inv = scaler.inverse_transform(predictions)
    ytest_inv = scaler.inverse_transform(ytest)
    
    errors = predictions_inv - ytest_inv
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'predictions': predictions_inv.flatten().tolist()
    }

def process_file(filepath, filename, output_model_dir, use_grid_search):
    """Processa um arquivo CSV individual"""
    print(f"\n{'='*60}")
    print(f"Processando: {filename}")
    print('='*60)
    
    # Carrega dados sem definir index
    df = pd.read_csv(filepath)
    
    print(f"Colunas detectadas: {list(df.columns)}")
    
    # Detecta coluna de data/timestamp e remove (não é necessária para o modelo)
    date_cols = [col for col in df.columns if col.lower() in ['data', 'date', 'timestamp', 'datetime']]
    if date_cols:
        print(f"Coluna de data encontrada: '{date_cols[0]}' (será ignorada)")
        # Não define como index, apenas ignora
    
    # Remove coluna '0' se existir
    if '0' in df.columns:
        df.drop(columns=['0'], inplace=True)
    
    # Detecta coluna de valores (Vazao, Throughput, etc)
    value_cols = [col for col in df.columns if col.lower() in ['vazao', 'throughput', 'value', 'valor', 'flow']]
    if value_cols:
        target_col = value_cols[0]
    else:
        # Usa primeira coluna numérica disponível (excluindo datas)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove colunas de data se foram detectadas como numéricas
        numeric_cols = [col for col in numeric_cols if col.lower() not in ['data', 'date', 'timestamp', 'datetime']]
        if len(numeric_cols) == 0:
            raise ValueError(f"Nenhuma coluna numérica encontrada em {filename}")
        target_col = numeric_cols[0]
    
    print(f"Coluna de valores detectada: '{target_col}'")
    print(f"Total de registros: {len(df)}")
    
    # Remove valores NaN ou inválidos
    df_clean = df[[target_col]].dropna()
    print(f"Registros após limpeza: {len(df_clean)}")
    
    if len(df_clean) < 50:
        raise ValueError(f"Dados insuficientes após limpeza: {len(df_clean)} registros")
    
    # Split treino/teste
    split_idx = int(len(df_clean) * TRAIN_TEST_SPLIT)
    train_data = df_clean[:split_idx][target_col].values.reshape(-1, 1)
    test_data = df_clean[split_idx:][target_col].values.reshape(-1, 1)
    
    print(f"Tamanho treino: {len(train_data)}, Tamanho teste: {len(test_data)}")
    
    # Normalização
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    # Grid Search ou parâmetros padrão
    if use_grid_search:
        best_params, grid_results = grid_search(
            train_scaled, PARAM_GRID, EPOCHS, BATCH_SIZE, PATIENCE, N_SPLITS
        )
    else:
        best_params = {
            'gru_units': 64,
            'learning_rate': 0.0001,
            'look_back': LOOK_BACK,
            'dropout_rate': 0.0
        }
        grid_results = None
    
    # Cria datasets finais com melhores parâmetros
    X_train, y_train = create_dataset(train_scaled, best_params['look_back'])
    X_test, y_test = create_dataset(test_scaled, best_params['look_back'])
    
    print(f"\nTreinando modelo final com melhores parâmetros...")
    print(f"Parâmetros: {best_params}")
    
    # Treina modelo final
    final_model = create_gru_model(
        units=best_params['gru_units'],
        train_shape=X_train.shape,
        learning_rate=best_params['learning_rate'],
        dropout_rate=best_params['dropout_rate']
    )
    
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=PATIENCE, 
        restore_best_weights=True,
        verbose=1
    )
    
    # Usa último split para validação
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    train_idx, val_idx = list(tscv.split(X_train))[-1]
    
    history = final_model.fit(
        X_train[train_idx], y_train[train_idx],
        epochs=EPOCHS,
        validation_data=(X_train[val_idx], y_train[val_idx]),
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Salva modelo
    if output_model_dir:
        save_model(final_model, output_model_dir, filename.replace('.csv', ''))
    
    # Avalia no conjunto de teste
    results = predict_and_evaluate(final_model, X_test, y_test, scaler)
    results['best_params'] = best_params
    if grid_results:
        results['grid_search_results'] = grid_results
    
    print(f"\nResultados no conjunto de teste:")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"MAE: {results['mae']:.4f}")
    
    # Limpa memória
    del final_model
    tf.keras.backend.clear_session()
    
    return results

def gru_prediction(source_dir, output_model_dir=None, output_json='evaluation_rmse_mae.json', 
                   use_grid_search=True):
    """Função principal para processar todos os arquivos"""
    evaluation = {}
    
    print(f"\n{'='*70}")
    print(f"INÍCIO DO PROCESSAMENTO")
    print(f"Pasta de entrada: {source_dir}")
    print(f"Grid Search: {'ATIVADO' if use_grid_search else 'DESATIVADO'}")
    print(f"{'='*70}\n")
    
    # Processa cada arquivo CSV
    csv_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append((root, file))
    
    print(f"Encontrados {len(csv_files)} arquivos CSV\n")
    
    for idx, (root, file) in enumerate(csv_files, 1):
        filepath = os.path.join(root, file)
        print(f"\n[{idx}/{len(csv_files)}]")
        
        try:
            results = process_file(filepath, file, output_model_dir, use_grid_search)
            evaluation[file] = results
        except Exception as e:
            print(f"ERRO ao processar {file}: {e}")
            evaluation[file] = {'error': str(e)}
    
    # Salva resultados em JSON
    with open(output_json, 'w') as f:
        json.dump(evaluation, f, indent=4)
    
    print(f"\n{'='*70}")
    print(f"PROCESSAMENTO CONCLUÍDO")
    print(f"Resultados salvos em: {output_json}")
    print(f"{'='*70}\n")
    
    # Resumo
    print("\nRESUMO DOS RESULTADOS:")
    print("="*70)
    successful = 0
    total_rmse = 0
    total_mae = 0
    
    for file, metrics in evaluation.items():
        if 'error' in metrics:
            print(f"❌ {file}: ERRO - {metrics['error']}")
        else:
            successful += 1
            total_rmse += metrics['rmse']
            total_mae += metrics['mae']
            print(f"✓ {file}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
    
    if successful > 0:
        print(f"\n{'='*70}")
        print(f"Média RMSE: {total_rmse/successful:.4f}")
        print(f"Média MAE: {total_mae/successful:.4f}")
        print(f"Taxa de sucesso: {successful}/{len(evaluation)}")
        print(f"{'='*70}\n")
    
    return evaluation

if __name__ == "__main__":
    results = gru_prediction(
        source_dir=INPUT_FOLDER,
        output_model_dir=OUTPUT_MODEL_FOLDER,
        output_json=OUTPUT_JSON,
        use_grid_search=GRID_SEARCH_ENABLED
    )