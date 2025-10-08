# import pandas as pd
# import numpy as np
# import os
# from pathlib import Path

# # Configuration 
# INPUT_FOLDER = "geant-datasets"
# AGGREGATED_FOLDER = "aggregated_datasets"
# LONGEST_INTERVAL_FOLDER = "longest_interval_datasets"
# MISSING_RATES_FOLDER = "missing_rates_datasets"
# REPORT_FILE = "initial_preprocessing_report.txt"
# TIMESTAMP_COL = "Data"
# VALUE_COL = "Vazao"
# MISSING_RATES = [0.1, 0.2, 0.3, 0.4]
# RANDOM_SEED = 42

# def setup_folders():
#     """Create necessary folders if they don't exist"""
#     for folder in [AGGREGATED_FOLDER, LONGEST_INTERVAL_FOLDER, MISSING_RATES_FOLDER]:
#         Path(folder).mkdir(exist_ok=True)

# def agg_by_interval(df, time_column=TIMESTAMP_COL, value_column=VALUE_COL, hours=4):
#     """
#     Aggregate values of a time series in fixed N-hour blocks.
#     Keeps NaN in intervals without measurements.

#     """
#     df = df.copy()
#     df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
#     df = df.set_index(time_column).sort_index() # important to sort by time

#     df_resampled = df[value_column].resample(f"{hours}h").mean()

#     df_resampled = df_resampled.asfreq(f"{hours}h")

#     return df_resampled

# def find_longest_interval(vazao: pd.Series, max_failures=2, max_missing_percentage=None, allow_consecutive_failures=False):
#     """
#     Find longest interval (in number of points) containing at most:
#     - a specific number of failures, OR
#     - a maximum percentage of missing data
    
#     Falhas são valores NaN ou -1.
    
#     Parâmetros:
#     - vazao: pd.Series com os dados de vazão
#     - max_failures: número máximo de falhas permitidas (padrão: 2)
#     - max_missing_percentage: porcentagem máxima de dados faltantes (0-1). Se especificado, sobrescreve max_failures
#     - allow_consecutive_failures: se True, permite falhas consecutivas (padrão: False)
    
#     Retorna:
#         (idx_inicio, idx_fim, tamanho, total_falhas, porcentagem_falhas)
#     """
#     v = vazao.copy()
#     v = v.replace(-1, np.nan)  # trata -1 como falha
#     is_fail = v.isna().to_numpy()
#     n = len(is_fail)

#     melhor_tam = 0
#     melhor_inicio = 0
#     melhor_fim = 0
#     melhor_falhas = 0

#     for i in range(n):
#         falhas = 0
#         consecutivas = False
#         last_was_fail = False

#         for j in range(i, n):
#             if is_fail[j]:
#                 falhas += 1
                
#                 # Verifica se houve falha consecutiva (se não permitido)
#                 if not allow_consecutive_failures and last_was_fail:
#                     consecutivas = True
#                 last_was_fail = True
#             else:
#                 last_was_fail = False

#             # Calcula porcentagem atual de falhas
#             current_size = j - i + 1
#             current_missing_pct = falhas / current_size if current_size > 0 else 0
            
#             # Critério de parada baseado no tipo de limite
#             if max_missing_percentage is not None:
#                 # Usa porcentagem máxima como critério
#                 if current_missing_pct > max_missing_percentage or (not allow_consecutive_failures and consecutivas):
#                     break
#             else:
#                 # Usa número máximo de falhas como critério
#                 if falhas > max_failures or (not allow_consecutive_failures and consecutivas):
#                     break

#             # Atualiza melhor intervalo encontrado
#             if current_size > melhor_tam:
#                 melhor_tam = current_size
#                 melhor_inicio = i
#                 melhor_fim = j
#                 melhor_falhas = falhas

#     porcentagem_falhas = melhor_falhas / melhor_tam if melhor_tam > 0 else 0
    
#     return melhor_inicio, melhor_fim, melhor_tam, melhor_falhas, porcentagem_falhas

# # Função de compatibilidade para manter o comportamento original
# def longest_interval_with_max_2_failures(vazao: pd.Series):
#     """
#     Versão original mantida para compatibilidade.
#     Encontra o intervalo mais longo com no máximo 2 falhas não consecutivas.
#     """
#     inicio, fim, tamanho, falhas, pct = find_longest_interval(
#         vazao, 
#         max_failures=2, 
#         max_missing_percentage=None, 
#         allow_consecutive_failures=False
#     )
#     return inicio, fim, tamanho

# def analyze_and_aggregate_file(file_path, report_handle):
#     """Process a single file through steps 1 and 2"""
#     # Step 1: Read and print dataset info
#     df = pd.read_csv(file_path)
#     filename = os.path.basename(file_path)
    
#     report_handle.write(f"\n{'='*80}\n")
#     report_handle.write(f"PROCESSING FILE: {filename}\n")
#     report_handle.write(f"{'='*80}\n")
#     report_handle.write(f"Dataset shape: {df.shape}\n")
#     report_handle.write(f"First 10 rows:\n{df.head(10)}\n\n")
    
#     # Aggregation analysis
#     report_handle.write("AGGREGATION ANALYSIS:\n")
    
#     # 4-hour aggregation
#     serie_4h = agg_by_interval(df, TIMESTAMP_COL, VALUE_COL, horas=4)
#     missing_4h = serie_4h.isna().sum()
#     report_handle.write(f"4-hour aggregation:\n")
#     report_handle.write(f"  Size: {len(serie_4h)} points\n")
#     report_handle.write(f"  Missing values: {missing_4h}\n")
#     report_handle.write(f"  First 10 values:\n{serie_4h.head(10)}\n\n")
    
#     # 6-hour aggregation
#     serie_6h = agg_by_interval(df, TIMESTAMP_COL, VALUE_COL, horas=6)
#     missing_6h = serie_6h.isna().sum()
#     report_handle.write(f"6-hour aggregation:\n")
#     report_handle.write(f"  Size: {len(serie_6h)} points\n")
#     report_handle.write(f"  Missing values: {missing_6h}\n")
#     report_handle.write(f"  First 10 values:\n{serie_6h.head(10)}\n\n")
    
#     # Save aggregated datasets
#     base_name = os.path.splitext(filename)[0]
#     serie_4h.to_csv(os.path.join(AGGREGATED_FOLDER, f"{base_name}_4h.csv"), header=True)
#     serie_6h.to_csv(os.path.join(AGGREGATED_FOLDER, f"{base_name}_6h.csv"), header=True)
    
#     return df, serie_4h, serie_6h

# def find_longest_interval_and_add_missing_rates(df, filename, report_handle):
#     """Process steps 2 and 3 for a file"""
#     # Step 2: Find longest interval with max 2 failures
#     vazao_series = pd.Series(df[VALUE_COL])
    
#     report_handle.write(f"LONGEST INTERVAL ANALYSIS for {filename}:\n")
#     report_handle.write(f"Original data series:\n{vazao_series}\n\n")
    
#     inicio, fim, tamanho = intervalo_mais_longo_ate2_falhas(vazao_series)
#     report_handle.write(f"Longest interval: {tamanho} points (indices {inicio}–{fim})\n")
    
#     # Extract longest interval
#     longest_interval = df.iloc[inicio:fim+1].copy()
    
#     # Save longest interval
#     base_name = os.path.splitext(filename)[0]
#     longest_interval_path = os.path.join(LONGEST_INTERVAL_FOLDER, f"{base_name}_longest.csv")
#     longest_interval.to_csv(longest_interval_path, index=False)
#     report_handle.write(f"Longest interval saved to: {longest_interval_path}\n\n")
    
#     # Step 3: Add missing rates
#     report_handle.write("MISSING RATES GENERATION:\n")
#     missing_longest = longest_interval.copy()
    
#     for missing_rate in MISSING_RATES:
#         missing_idx = missing_longest.sample(
#             frac=missing_rate,
#             random_state=RANDOM_SEED
#         ).index
#         missing_df = missing_longest.copy()
#         missing_df.loc[missing_idx, VALUE_COL] = -1
        
#         # Save missing rate dataset
#         missing_path = os.path.join(MISSING_RATES_FOLDER, f"{base_name}_missing_{int(missing_rate*100)}.csv")
#         missing_df.to_csv(missing_path, index=False)
#         report_handle.write(f"Missing rate {missing_rate:.1%} saved to: {missing_path}\n")
        
#     report_handle.write(f"First 20 rows of {missing_rate:.1%} missing rate:\n{missing_df.head(20)}\n\n")
    
#     return longest_interval_path

# def process_entire_folder():
#     """Main function to process all CSV files in the input folder"""
#     setup_folders()
    
#     # Get all CSV files in input folder
#     csv_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.csv')]
    
#     if not csv_files:
#         print(f"No CSV files found in {INPUT_FOLDER}")
#         return
    
#     with open(REPORT_FILE, 'w', encoding='utf-8') as report:
#         report.write("INITIAL PREPROCESSING REPORT\n")
#         report.write("="*50 + "\n\n")
        
#         for csv_file in csv_files:
#             file_path = os.path.join(INPUT_FOLDER, csv_file)
            
#             try:
#                 # Steps 1 and 2
#                 df, serie_4h, serie_6h = analyze_and_aggregate_file(file_path, report)
                
#                 # Steps 2 (continued) and 3
#                 find_longest_interval_and_add_missing_rates(df, csv_file, report)
                
#             except Exception as e:
#                 report.write(f"ERROR processing {csv_file}: {str(e)}\n\n")
#                 continue
    
#     print(f"Processing complete! Report saved to {REPORT_FILE}")
#     print(f"Aggregated datasets saved to: {AGGREGATED_FOLDER}")
#     print(f"Longest interval datasets saved to: {LONGEST_INTERVAL_FOLDER}")
#     print(f"Missing rates datasets saved to: {MISSING_RATES_FOLDER}")

# # Functions that can be imported and used in other scripts
# def process_single_file(file_path, output_prefix=None):
#     """Process a single file and return results (for use in other scripts)"""
#     df = pd.read_csv(file_path)
    
#     # Aggregation
#     serie_4h = agg_by_interval(df, TIMESTAMP_COL, VALUE_COL, horas=4)
#     serie_6h = agg_by_interval(df, TIMESTAMP_COL, VALUE_COL, horas=6)

#     # Longest interval
#     inicio, fim, tamanho = intervalo_mais_longo_ate2_falhas(pd.Series(df[VALUE_COL]))
#     longest_interval = df.iloc[inicio:fim+1].copy()
    
#     # Missing rates
#     missing_dfs = {}
#     for missing_rate in MISSING_RATES:
#         missing_idx = longest_interval.sample(
#             frac=missing_rate,
#             random_state=RANDOM_SEED
#         ).index
#         missing_df = longest_interval.copy()
#         missing_df.loc[missing_idx, VALUE_COL] = -1
#         missing_dfs[missing_rate] = missing_df
    
#     results = {
#         'original': df,
#         'aggregated_4h': serie_4h,
#         'aggregated_6h': serie_6h,
#         'longest_interval': longest_interval,
#         'longest_interval_info': (inicio, fim, tamanho),
#         'missing_dfs': missing_dfs
#     }
    
#     return results

# def save_processing_results(results, base_filename):
#     """Save results from process_single_file to appropriate folders"""
#     setup_folders()
    
#     # Save aggregated
#     results['aggregated_4h'].to_csv(os.path.join(AGGREGATED_FOLDER, f"{base_filename}_4h.csv"), header=True)
#     results['aggregated_6h'].to_csv(os.path.join(AGGREGATED_FOLDER, f"{base_filename}_6h.csv"), header=True)
    
#     # Save longest interval
#     longest_path = os.path.join(LONGEST_INTERVAL_FOLDER, f"{base_filename}_longest.csv")
#     results['longest_interval'].to_csv(longest_path, index=False)
    
#     # Save missing rates
#     for missing_rate, missing_df in results['missing_dfs'].items():
#         missing_path = os.path.join(MISSING_RATES_FOLDER, f"{base_filename}_missing_{int(missing_rate*100)}.csv")
#         missing_df.to_csv(missing_path, index=False)

# if __name__ == "__main__":
#     # Run the entire pipeline
#     process_entire_folder()

### O código acima ainda precisa de consertos. Mantendo como backup para caso o abaixo não faça o que eu quero ###

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Configuration 
INPUT_FOLDER = "geant-datasets"
AGGREGATED_FOLDER = "aggregated_datasets"
LONGEST_INTERVAL_FOLDER = "longest_interval_datasets"
MISSING_RATES_FOLDER = "missing_rates_datasets"
REPORT_FILE = "initial_preprocessing_report.txt"
TIMESTAMP_COL = "Data"
VALUE_COL = "Vazao"
MISSING_RATES = [0.1, 0.2, 0.3, 0.4]
RANDOM_SEED = 42

def setup_folders():
    """Create necessary folders if they don't exist"""
    for folder in [AGGREGATED_FOLDER, LONGEST_INTERVAL_FOLDER, MISSING_RATES_FOLDER]:
        Path(folder).mkdir(exist_ok=True)

def aggregate_by_interval(df, time_column=TIMESTAMP_COL, value_column=VALUE_COL, hours=4):
    """
    Aggregate time series values in fixed N-hour blocks.
    Keeps NaN for intervals without measurements.
    """
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
    df = df.set_index(time_column).sort_index()
    
    df_resampled = df[value_column].resample(f"{hours}h").mean()
    df_resampled = df_resampled.asfreq(f"{hours}h")
    
    return df_resampled

def find_longest_interval(flow: pd.Series, max_failures=2, max_missing_percentage=None, 
                          allow_consecutive_failures=False):
    """
    Find longest interval containing at most:
    - a specific number of failures, OR
    - a maximum percentage of missing data
    
    Failures are NaN or -1 values.
    
    Parameters:
    - flow: pd.Series with flow data
    - max_failures: maximum number of failures allowed (default: 2)
    - max_missing_percentage: maximum percentage of missing data (0-1). Overrides max_failures if specified
    - allow_consecutive_failures: if True, allows consecutive failures (default: False)
    
    Returns:
        (start_idx, end_idx, size, total_failures, failure_percentage)
    """
    v = flow.copy().replace(-1, np.nan)
    is_fail = v.isna().to_numpy()
    n = len(is_fail)

    best_size = 0
    best_start = 0
    best_end = 0
    best_failures = 0

    for i in range(n):
        failures = 0
        has_consecutive = False
        last_was_fail = False

        for j in range(i, n):
            if is_fail[j]:
                failures += 1
                if not allow_consecutive_failures and last_was_fail:
                    has_consecutive = True
                last_was_fail = True
            else:
                last_was_fail = False

            current_size = j - i + 1
            current_missing_pct = failures / current_size if current_size > 0 else 0
            
            if max_missing_percentage is not None:
                if current_missing_pct > max_missing_percentage or (not allow_consecutive_failures and has_consecutive):
                    break
            else:
                if failures > max_failures or (not allow_consecutive_failures and has_consecutive):
                    break

            if current_size > best_size:
                best_size = current_size
                best_start = i
                best_end = j
                best_failures = failures

    failure_percentage = best_failures / best_size if best_size > 0 else 0
    
    return best_start, best_end, best_size, best_failures, failure_percentage

def longest_interval_max_2_failures(flow: pd.Series):
    """
    Find longest interval with at most 2 non-consecutive failures.
    Compatibility wrapper for original behavior.
    """
    start, end, size, _, _ = find_longest_interval(
        flow, 
        max_failures=2, 
        max_missing_percentage=None, 
        allow_consecutive_failures=False
    )
    return start, end, size

def analyze_and_aggregate_file(file_path, report_handle):
    """Process a single file through aggregation analysis"""
    df = pd.read_csv(file_path)
    filename = os.path.basename(file_path)
    
    report_handle.write(f"\n{'='*80}\n")
    report_handle.write(f"PROCESSING FILE: {filename}\n")
    report_handle.write(f"{'='*80}\n")
    report_handle.write(f"Dataset shape: {df.shape}\n")
    report_handle.write(f"First 10 rows:\n{df.head(10)}\n\n")
    
    report_handle.write("AGGREGATION ANALYSIS:\n")
    
    series_4h = aggregate_by_interval(df, TIMESTAMP_COL, VALUE_COL, hours=4)
    missing_4h = series_4h.isna().sum()
    report_handle.write(f"4-hour aggregation:\n")
    report_handle.write(f"  Size: {len(series_4h)} points\n")
    report_handle.write(f"  Missing values: {missing_4h}\n")
    report_handle.write(f"  First 10 values:\n{series_4h.head(10)}\n\n")
    
    series_6h = aggregate_by_interval(df, TIMESTAMP_COL, VALUE_COL, hours=6)
    missing_6h = series_6h.isna().sum()
    report_handle.write(f"6-hour aggregation:\n")
    report_handle.write(f"  Size: {len(series_6h)} points\n")
    report_handle.write(f"  Missing values: {missing_6h}\n")
    report_handle.write(f"  First 10 values:\n{series_6h.head(10)}\n\n")
    
    base_name = os.path.splitext(filename)[0]
    series_4h.to_csv(os.path.join(AGGREGATED_FOLDER, f"{base_name}_4h.csv"), header=True)
    series_6h.to_csv(os.path.join(AGGREGATED_FOLDER, f"{base_name}_6h.csv"), header=True)
    
    return df, series_4h, series_6h

def find_longest_interval_and_add_missing_rates(df, filename, report_handle):
    """Find longest interval and generate datasets with missing rates"""
    flow_series = pd.Series(df[VALUE_COL])
    
    report_handle.write(f"LONGEST INTERVAL ANALYSIS for {filename}:\n")
    report_handle.write(f"Original data series:\n{flow_series}\n\n")
    
    start, end, size = longest_interval_max_2_failures(flow_series)
    report_handle.write(f"Longest interval: {size} points (indices {start}–{end})\n")
    
    longest_interval = df.iloc[start:end+1].copy()
    
    base_name = os.path.splitext(filename)[0]
    longest_interval_path = os.path.join(LONGEST_INTERVAL_FOLDER, f"{base_name}_longest.csv")
    longest_interval.to_csv(longest_interval_path, index=False)
    report_handle.write(f"Longest interval saved to: {longest_interval_path}\n\n")
    
    report_handle.write("MISSING RATES GENERATION:\n")
    
    for missing_rate in MISSING_RATES:
        missing_idx = longest_interval.sample(
            frac=missing_rate,
            random_state=RANDOM_SEED
        ).index
        missing_df = longest_interval.copy()
        missing_df.loc[missing_idx, VALUE_COL] = -1
        
        missing_path = os.path.join(MISSING_RATES_FOLDER, f"{base_name}_missing_{int(missing_rate*100)}.csv")
        missing_df.to_csv(missing_path, index=False)
        report_handle.write(f"Missing rate {missing_rate:.1%} saved to: {missing_path}\n")
        
    report_handle.write(f"First 20 rows of {missing_rate:.1%} missing rate:\n{missing_df.head(20)}\n\n")
    
    return longest_interval_path

def process_entire_folder():
    """Main function to process all CSV files in the input folder"""
    setup_folders()
    
    csv_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {INPUT_FOLDER}")
        return
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as report:
        report.write("INITIAL PREPROCESSING REPORT\n")
        report.write("="*50 + "\n\n")
        
        for csv_file in csv_files:
            file_path = os.path.join(INPUT_FOLDER, csv_file)
            
            try:
                df, series_4h, series_6h = analyze_and_aggregate_file(file_path, report)
                find_longest_interval_and_add_missing_rates(df, csv_file, report)
                
            except Exception as e:
                report.write(f"ERROR processing {csv_file}: {str(e)}\n\n")
                continue
    
    print(f"Processing complete! Report saved to {REPORT_FILE}")
    print(f"Aggregated datasets saved to: {AGGREGATED_FOLDER}")
    print(f"Longest interval datasets saved to: {LONGEST_INTERVAL_FOLDER}")
    print(f"Missing rates datasets saved to: {MISSING_RATES_FOLDER}")

def process_single_file(file_path):
    """Process a single file and return results (for use in other scripts)"""
    df = pd.read_csv(file_path)
    
    series_4h = aggregate_by_interval(df, TIMESTAMP_COL, VALUE_COL, hours=4)
    series_6h = aggregate_by_interval(df, TIMESTAMP_COL, VALUE_COL, hours=6)

    start, end, size = longest_interval_max_2_failures(pd.Series(df[VALUE_COL]))
    longest_interval = df.iloc[start:end+1].copy()
    
    missing_dfs = {}
    for missing_rate in MISSING_RATES:
        missing_idx = longest_interval.sample(
            frac=missing_rate,
            random_state=RANDOM_SEED
        ).index
        missing_df = longest_interval.copy()
        missing_df.loc[missing_idx, VALUE_COL] = -1
        missing_dfs[missing_rate] = missing_df
    
    return {
        'original': df,
        'aggregated_4h': series_4h,
        'aggregated_6h': series_6h,
        'longest_interval': longest_interval,
        'longest_interval_info': (start, end, size),
        'missing_dfs': missing_dfs
    }

def save_processing_results(results, base_filename):
    """Save results from process_single_file to appropriate folders"""
    setup_folders()
    
    results['aggregated_4h'].to_csv(os.path.join(AGGREGATED_FOLDER, f"{base_filename}_4h.csv"), header=True)
    results['aggregated_6h'].to_csv(os.path.join(AGGREGATED_FOLDER, f"{base_filename}_6h.csv"), header=True)
    
    longest_path = os.path.join(LONGEST_INTERVAL_FOLDER, f"{base_filename}_longest.csv")
    results['longest_interval'].to_csv(longest_path, index=False)
    
    for missing_rate, missing_df in results['missing_dfs'].items():
        missing_path = os.path.join(MISSING_RATES_FOLDER, f"{base_filename}_missing_{int(missing_rate*100)}.csv")
        missing_df.to_csv(missing_path, index=False)

if __name__ == "__main__":
    process_entire_folder()