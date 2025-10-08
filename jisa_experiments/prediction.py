import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit
import json

# ============================================
# CONFIGURAÇÕES - AJUSTE AQUI
# ============================================
INPUT_FOLDER = './imputation_results_original'  # Pasta onde estão os arquivos CSV
OUTPUT_MODEL_FOLDER = './modelo_salvo'  # Pasta onde salvar os modelos (deixe None para não salvar)
OUTPUT_JSON = 'evaluation_rmse_mae.json'  # Nome do arquivo JSON de avaliação
TRAIN_TEST_SPLIT = 0.8  # Proporção treino/teste (80% treino, 20% teste)
LOOK_BACK = 3  # Janela de tempo para criar o dataset
GRU_UNITS = 64  # Número de unidades na camada GRU
LEARNING_RATE = 0.0001
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 5  # Early stopping patience
N_SPLITS = 4  # Número de splits para validação cruzada
# ============================================

def create_dataset(X, look_back=LOOK_BACK):
    Xs, ys = [], []
    for i in range(len(X) - look_back):
        Xs.append(X[i:i+look_back])
        ys.append(X[i+look_back])
    return np.array(Xs), np.array(ys)

def create_gru(units, train, learning_rate):
    model = Sequential()
    model.add(GRU(units=units, return_sequences=True, input_shape=[train.shape[1], train.shape[2]]))
    model.add(GRU(units=units))
    model.add(Dense(1))
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])
    return model

def fit_model_with_cross_validation(model, xtrain, ytrain, patience, epochs, batch_size, n_splits):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    histories = []
    for train_index, val_index in tscv.split(xtrain):
        x_train_fold, x_val_fold = xtrain[train_index], xtrain[val_index]
        y_train_fold, y_val_fold = ytrain[train_index], ytrain[val_index]
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        history = model.fit(x_train_fold, y_train_fold, epochs=epochs, validation_data=(x_val_fold, y_val_fold), batch_size=batch_size, callbacks=[early_stop], verbose=1)
        histories.append(history)
    return histories

def calculate_mean_history(histories):
    mean_history = {'loss': [], 'val_loss': []}
    for fold_history in histories:
        for key in mean_history.keys():
            mean_history[key].append(fold_history.history[key])
    for key, values in mean_history.items():
        max_len = max(len(val) for val in values)
        for i in range(len(values)):
            if len(values[i]) < max_len:
                values[i] += [values[i][-1]] * (max_len - len(values[i]))
    for key, values in mean_history.items():
        mean_history[key] = [sum(vals) / len(vals) for vals in zip(*values)]
    return mean_history

def save_model(model, directory, filename):
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, f'{filename} - GRU.keras')
    model.save(file_path)
    print(f"Model saved in '{file_path}'")

def prediction(model, xtest, myscaler):
    prediction = model.predict(xtest)
    return myscaler.inverse_transform(prediction)

def evaluate_prediction(predictions, actual):
    errors = predictions - actual
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()
    print(f'RMSE: {rmse:.4f}, MAE: {mae:.4f}')
    return {'rmse': float(rmse), 'mae': float(mae)}

def gru_prediction(source_dir, output_model_dir=None, output_json='evaluation_rmse_mae.json'):
    evaluation = {}
    print(f"\n{'='*60}")
    print(f"Starting GRU predictions for files in: {source_dir}")
    print(f"{'='*60}\n")
    
    for pasta_raiz, _, arquivos in os.walk(source_dir):
        for arquivo in arquivos:
            if arquivo.endswith('.csv'):
                caminho_arquivo = os.path.join(pasta_raiz, arquivo)
                print(f"\nProcessing: {arquivo}")
                print("-" * 60)
                
                try:
                    df = pd.read_csv(caminho_arquivo, index_col='Timestamp')
                    if '0' in df.columns:
                        df.drop(columns=['0'], inplace=True)
                    
                    tamanho = int(len(df.index) * TRAIN_TEST_SPLIT)
                    train_data = df[:tamanho]['Throughput'].values.reshape(-1, 1)
                    test_data = df[tamanho:]['Throughput'].values.reshape(-1, 1)
                    
                    scaler = MinMaxScaler().fit(train_data)
                    train_scaled = scaler.transform(train_data)
                    test_scaled = scaler.transform(test_data)
                    
                    X_train, y_train = create_dataset(train_scaled)
                    X_test, y_test = create_dataset(test_scaled)
                    
                    model_gru = create_gru(GRU_UNITS, X_train, LEARNING_RATE)
                    prev_history_gru = fit_model_with_cross_validation(
                        model_gru, X_train, y_train, PATIENCE, EPOCHS, BATCH_SIZE, N_SPLITS
                    )
                    history_gru = calculate_mean_history(prev_history_gru)
                    
                    # Salvar modelo se pasta foi especificada
                    if output_model_dir:
                        save_model(model_gru, output_model_dir, arquivo.replace('.csv', ''))
                    
                    y_test = scaler.inverse_transform(y_test)
                    prediction_gru = prediction(model_gru, X_test, scaler)
                    evaluation[arquivo] = evaluate_prediction(prediction_gru, y_test)
                    
                except Exception as e:
                    print(f"Error processing {arquivo}: {e}")
                    evaluation[arquivo] = {'error': str(e)}

    # Salvar resultados em JSON
    with open(output_json, 'w') as f:
        json.dump(evaluation, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"Evaluation results saved to: {output_json}")
    print(f"{'='*60}\n")
    
    return evaluation

if __name__ == "__main__":
    # Executar predições com as configurações definidas no início do arquivo
    results = gru_prediction(
        source_dir=INPUT_FOLDER,
        output_model_dir=OUTPUT_MODEL_FOLDER,
        output_json=OUTPUT_JSON
    )
    
    print("\nSummary of Results:")
    print("=" * 60)
    for file, metrics in results.items():
        if 'error' in metrics:
            print(f"{file}: ERROR - {metrics['error']}")
        else:
            print(f"{file}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")