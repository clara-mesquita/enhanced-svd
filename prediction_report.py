import json
import os
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


class ImputationAnalyzer:
    def __init__(
        self,
        json_file,
        original_pref_folder="geant-datasets",
        aggregated_folder="aggregated_6h_datasets",
        value_cols=("Vazao", "throughput_bps", "value", "n_bytes", "bytes"),
        timestamp_cols=("Data", "timestamp", "time", "Datetime", "date", "Date")
    ):
        """
        Initialize analyzer with JSON file containing prediction results.

        Parameters
        ----------
        json_file : str
            JSON com resultados (rmse, mae, predictions) por arquivo.
        original_pref_folder : str
            Pasta prioritária que contém as séries originais em granularidade nativa.
        aggregated_folder : str
            Pasta fallback que contém séries agregadas em 6h (quando disponível).
        value_cols : tuple[str]
            Nomes de colunas possíveis para o valor.
        timestamp_cols : tuple[str]
            Nomes de colunas possíveis para o timestamp.
        """
        self.json_file = json_file
        self.original_pref_folder = Path(original_pref_folder)
        self.aggregated_folder = Path(aggregated_folder)
        self.value_cols = value_cols
        self.timestamp_cols = timestamp_cols

        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.df = None
        self.parse_data()

    # -------------------------
    # Parsing do JSON / filename
    # -------------------------
    def parse_data(self):
        """Parse JSON data and extract file information."""
        records = []

        for filename, metrics in self.data.items():
            # Extrai método a partir do padrão "_6h_"
            parts = filename.replace('_imputed.csv', '').split('_6h_')
            if len(parts) == 2:
                base_name = parts[0]
                imputation_method = parts[1]
            else:
                base_name = filename
                imputation_method = 'Unknown'

            # Extrai data
            date_match = re.search(r'(\d{2}-\d{2}-\d{4})', base_name)
            date = date_match.group(1) if date_match else 'Unknown'

            # Extrai rota (perfsonar-AAA to BBB DD-MM-YYYY)
            route_match = re.search(r'perfsonar-(.+?) to (.+?) \d{2}-\d{2}-\d{4}', base_name)
            if route_match:
                source = route_match.group(1)
                destination = route_match.group(2)
                route = f"{source} -> {destination}"
            else:
                # fallback mais simples
                route = base_name.replace(f'_{date}', '').replace('esmond data ', '')

            records.append({
                'filename': filename,
                'route': route,
                'date': date,
                'imputation_method': imputation_method,
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'predictions': metrics['predictions'],
                'n_predictions': len(metrics['predictions'])
            })

        self.df = pd.DataFrame(records)

    # -------------------------
    # Helpers de dados originais
    # -------------------------
    @staticmethod
    def _slugify(s: str) -> str:
        s = s.lower()
        s = re.sub(r'\s+', ' ', s)
        s = s.replace(' -> ', ' ')
        return re.sub(r'[^a-z0-9._\- ]+', '', s).strip()

    def _candidate_paths(self, folder: Path, source: str, destination: str, date: str):
        """Gera candidatos de caminhos para busca no folder."""
        if not folder.exists():
            return []

        src = self._slugify(source)
        dst = self._slugify(destination)
        d = date  # já no formato DD-MM-YYYY

        # Estratégias:
        # 1) Arquivos que contenham source, destination e data no nome
        # 2) Relaxado: contenham source e destination
        patterns = [
            (True, True, True),
            (True, True, False),
        ]
        candidates = []
        for p in folder.rglob("*.csv"):
            name = self._slugify(p.name)
            has_src = src in name
            has_dst = dst in name
            has_date = d in name
            for need_src, need_dst, need_date in patterns:
                if (not need_src or has_src) and (not need_dst or has_dst) and (not need_date or has_date):
                    candidates.append(p)
                    break
        return candidates

    def _read_series(self, csv_path: Path, aggregate_to_6h=True):
        """Lê um CSV, detecta colunas de tempo/valor e retorna uma série (pd.Series)."""
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            warnings.warn(f"Falha lendo {csv_path}: {e}")
            return None

        # Encontra colunas
        ts_col = None
        for c in self.timestamp_cols:
            if c in df.columns:
                ts_col = c
                break

        val_col = None
        for c in self.value_cols:
            if c in df.columns:
                val_col = c
                break

        if val_col is None:
            # tenta inferir uma coluna numérica
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if num_cols:
                val_col = num_cols[0]
            else:
                warnings.warn(f"Não encontrei coluna numérica de valor em {csv_path}")
                return None

        s = df[val_col].astype(float)

        # Se tiver timestamp válido, indexa e ordena
        if ts_col is not None:
            try:
                ts = pd.to_datetime(df[ts_col], dayfirst=True, errors="coerce")
                valid = ts.notna()
                s = pd.Series(s[valid].values, index=ts[valid])
                s = s.sort_index()
            except Exception:
                # mantém sem índice datetime
                s = pd.Series(s.values)

        # Agrega para 6H (média) se tiver datetime index e pedir
        if aggregate_to_6h and isinstance(s.index, pd.DatetimeIndex):
            s = s.resample("6H").mean()

        # Remove NaNs internos no original para o cômputo do DTW
        s = s.dropna()

        return s

    def _get_reference_series(self, route: str, date: str, aggregate_to_6h=True):
        """
        Tenta achar a série original primeiro em geant-datasets; se não, em aggregated_6h_datasets.
        Se encontrar na geant, agrega para 6H quando aggregate_to_6h=True.
        """
        # Decompõe rota
        if "->" in route:
            source, destination = [x.strip() for x in route.split("->")]
        else:
            # fallback bruto: usa a rota como 'source', sem destino
            source, destination = route, ""

        # 1) geant-datasets (preferência)
        cand = self._candidate_paths(self.original_pref_folder, source, destination, date)
        for p in cand:
            s = self._read_series(p, aggregate_to_6h=aggregate_to_6h)
            if s is not None and len(s) > 0:
                return s, str(p), "geant-datasets"

        # 2) aggregated_6h_datasets (fallback, já deve estar em 6h)
        cand = self._candidate_paths(self.aggregated_folder, source, destination, date)
        for p in cand:
            s = self._read_series(p, aggregate_to_6h=False)  # já é 6H
            if s is not None and len(s) > 0:
                return s, str(p), "aggregated_6h_datasets"

        return None, None, None

    # -------------------------
    # Rankings e resumos
    # -------------------------
    def find_best_imputation_per_route(self):
        """Find the best imputation method for each route based on RMSE and MAE."""
        best_by_rmse = self.df.loc[self.df.groupby('route')['rmse'].idxmin()]
        best_by_mae = self.df.loc[self.df.groupby('route')['mae'].idxmin()]

        summary = pd.DataFrame({
            'route': best_by_rmse['route'].values,
            'best_imputation_rmse': best_by_rmse['imputation_method'].values,
            'rmse_value': best_by_rmse['rmse'].values,
            'best_imputation_mae': best_by_mae['imputation_method'].values,
            'mae_value': best_by_mae['mae'].values
        })
        return summary

    def imputation_method_ranking(self):
        """Rank imputation methods by average performance."""
        ranking = self.df.groupby('imputation_method').agg({
            'rmse': ['mean', 'std', 'min', 'max'],
            'mae': ['mean', 'std', 'min', 'max']
        }).round(2)
        ranking = ranking.sort_values(('rmse', 'mean'))
        return ranking

    def calculate_relative_performance(self):
        """Calculate relative performance of each imputation method per route."""
        results = []
        for route in self.df['route'].unique():
            route_data = self.df[self.df['route'] == route].copy()
            if len(route_data) > 1:
                min_rmse = route_data['rmse'].min()
                min_mae = route_data['mae'].min()
                route_data['rmse_relative'] = (route_data['rmse'] / min_rmse - 1) * 100
                route_data['mae_relative'] = (route_data['mae'] / min_mae - 1) * 100
                results.append(route_data)
        if results:
            return pd.concat(results, ignore_index=True)
        return self.df

    # -------------------------
    # NOVO: DTW Previsão vs Original
    # -------------------------
    def calculate_dtw_vs_original(self, prefer_geant=True):
        """
        Calcula distâncias DTW entre as previsões (por arquivo/método) e a série original.
        - Se prefer_geant=True: tenta geant-datasets (agregando para 6h na leitura).
        - Caso não encontre, cai para aggregated_6h_datasets.
        Retorna DataFrame com distâncias e metadados.
        """
        results = []
        # agrega para 6h quando for comparar com geant (para ficar compatível com as previsões 6h)
        aggregate_to_6h = True

        for idx, row in self.df.iterrows():
            route = row['route']
            date = row['date']
            method = row['imputation_method']
            preds = np.array(row['predictions'], dtype=float)
            n = len(preds)

            if n == 0 or (not np.isfinite(preds).all()):
                continue

            ref_series, ref_path, ref_src = self._get_reference_series(
                route, date, aggregate_to_6h=aggregate_to_6h
            )

            if ref_series is None or len(ref_series) == 0:
                # nada encontrado
                results.append({
                    'filename': row['filename'],
                    'route': route,
                    'date': date,
                    'imputation_method': method,
                    'ref_found': False,
                    'ref_path': None,
                    'ref_source': None,
                    'dtw_distance': np.nan,
                    'normalized_dtw': np.nan,
                    'sequence_length': n
                })
                continue

            # escolhe o último segmento do original com mesmo tamanho das previsões
            ref_values = ref_series.values.astype(float)
            if len(ref_values) >= n:
                ref_segment = ref_values[-n:]
            else:
                # se original menor, corta as previsões para o comprimento do original
                preds = preds[-len(ref_values):]
                ref_segment = ref_values

            # DTW (escala 1D -> reshape para (t,1) por segurança)
            try:
                dist, _ = fastdtw(preds.reshape(-1, 1), ref_segment.reshape(-1, 1), dist=euclidean)
                norm = dist / len(preds) if len(preds) > 0 else np.nan
            except Exception as e:
                warnings.warn(f"DTW falhou em {route} ({method}) vs original: {e}")
                dist, norm = np.nan, np.nan

            results.append({
                'filename': row['filename'],
                'route': route,
                'date': date,
                'imputation_method': method,
                'ref_found': True,
                'ref_path': ref_path,
                'ref_source': ref_src,
                'dtw_distance': dist,
                'normalized_dtw': norm,
                'sequence_length': len(preds)
            })

        return pd.DataFrame(results)

    def summarize_dtw_vs_original_by_method(self, dtw_df):
        """
        Média do DTW normalizado por método (quanto menor, mais fiel ao original).
        """
        valid = dtw_df.dropna(subset=['normalized_dtw'])
        if valid.empty:
            return pd.DataFrame()
        return (valid.groupby('imputation_method')['normalized_dtw']
                .agg(['count', 'mean', 'std', 'min', 'max'])
                .sort_values('mean')
                .round(4))

    def plot_dtw_vs_original(self, save_path='dtw_vs_original.png'):
        """
        Visualizações para DTW (previsão vs original):
        - Heatmap (método x média DTW normalizado)
        - Distribuição
        - Top métodos mais fiéis (menor DTW)
        - Correlação DTW vs RMSE
        """
        dtw_df = self.calculate_dtw_vs_original(prefer_geant=True)
        if dtw_df.empty or dtw_df['normalized_dtw'].dropna().empty:
            print("Sem dados suficientes para DTW vs original.")
            return None

        summary = self.summarize_dtw_vs_original_by_method(dtw_df)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1) "Heatmap" 1D (coluna única: média por método)
        sns.heatmap(summary[['mean']], annot=True, fmt='.4f', cmap='YlGnBu',
                    cbar_kws={'label': 'Avg Normalized DTW'}, ax=axes[0, 0])
        axes[0, 0].set_title('Fidelidade ao Original por Método (menor = melhor)')

        # 2) Distribuição global
        axes[0, 1].hist(dtw_df['normalized_dtw'].dropna().values, bins=30,
                        edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Normalized DTW (predição vs original)')
        axes[0, 1].set_ylabel('Frequência')
        axes[0, 1].set_title('Distribuição de DTW Normalizado (Predição vs Original)')
        axes[0, 1].grid(axis='y', alpha=0.3)

        # 3) Top-N métodos mais fiéis ao original
        top = summary.sort_values('mean').head(10)
        axes[1, 0].barh(top.index, top['mean'].values)
        axes[1, 0].invert_yaxis()
        axes[1, 0].set_xlabel('Avg Normalized DTW')
        axes[1, 0].set_title('Top 10 Métodos Mais Fiéis ao Original (menor DTW)')
        axes[1, 0].grid(axis='x', alpha=0.3)

        # 4) Correlação DTW vs RMSE (por arquivo/método)
        merged = pd.merge(
            dtw_df[['filename', 'normalized_dtw']],
            self.df[['filename', 'rmse']],
            on='filename',
            how='inner'
        ).dropna()

        if not merged.empty:
            axes[1, 1].scatter(merged['normalized_dtw'], merged['rmse'], alpha=0.6, s=50)
            axes[1, 1].set_xlabel('Normalized DTW (vs original)')
            axes[1, 1].set_ylabel('RMSE (vs ground truth)')
            axes[1, 1].set_title('DTW (vs original) x RMSE')
            axes[1, 1].grid(alpha=0.3)

            if len(merged) > 2:
                corr = np.corrcoef(merged['normalized_dtw'], merged['rmse'])[0, 1]
                axes[1, 1].text(0.05, 0.95, f'Correlação: {corr:.3f}',
                                transform=axes[1, 1].transAxes,
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                                verticalalignment='top')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"DTW vs original salvo em {save_path}")
        return fig

    # -------------------------
    # (Original) DTW entre métodos
    # -------------------------
    def calculate_dtw_distances(self):
        """DTW entre previsões de métodos diferentes (mesma rota)."""
        dtw_results = []

        for route in self.df['route'].unique():
            route_data = self.df[self.df['route'] == route]
            if len(route_data) < 2:
                continue

            methods = route_data['imputation_method'].values
            predictions_list = route_data['predictions'].values

            for i in range(len(methods)):
                for j in range(i + 1, len(methods)):
                    pred1 = np.array(predictions_list[i], dtype=float)
                    pred2 = np.array(predictions_list[j], dtype=float)

                    try:
                        distance, path = fastdtw(pred1.reshape(-1, 1), pred2.reshape(-1, 1), dist=euclidean)
                        normalized_distance = distance / len(pred1)
                        dtw_results.append({
                            'route': route,
                            'method1': methods[i],
                            'method2': methods[j],
                            'dtw_distance': distance,
                            'normalized_dtw': normalized_distance,
                            'sequence_length': len(pred1)
                        })
                    except Exception as e:
                        print(f"Warning: DTW calculation failed for {route}, {methods[i]} vs {methods[j]}: {e}")

        return pd.DataFrame(dtw_results)

    def analyze_dtw_similarity(self):
        """Matriz média de DTW entre métodos (menor = mais parecido entre si)."""
        dtw_df = self.calculate_dtw_distances()
        if dtw_df.empty:
            return None

        methods = sorted(list(set(dtw_df['method1'].unique()) | set(dtw_df['method2'].unique())))
        n_methods = len(methods)
        similarity_matrix = np.zeros((n_methods, n_methods))
        count_matrix = np.zeros((n_methods, n_methods))

        for _, row in dtw_df.iterrows():
            i = methods.index(row['method1'])
            j = methods.index(row['method2'])
            similarity_matrix[i, j] += row['normalized_dtw']
            similarity_matrix[j, i] += row['normalized_dtw']
            count_matrix[i, j] += 1
            count_matrix[j, i] += 1

        with np.errstate(divide='ignore', invalid='ignore'):
            similarity_matrix = np.where(count_matrix > 0, similarity_matrix / count_matrix, 0)

        return pd.DataFrame(similarity_matrix, index=methods, columns=methods), dtw_df

    def plot_dtw_analysis(self, save_path='dtw_analysis_between_methods.png'):
        """Visualizações do DTW entre métodos (mantido como no seu script)."""
        similarity_matrix, dtw_df = self.analyze_dtw_similarity()
        if similarity_matrix is None or dtw_df.empty:
            print("Not enough data for DTW analysis")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='YlOrRd_r',
                    ax=axes[0, 0], cbar_kws={'label': 'Avg Normalized DTW Distance'})
        axes[0, 0].set_title('DTW Similarity Between Imputation Methods\n(Lower = More Similar)')
        axes[0, 0].set_xlabel('Imputation Method')
        axes[0, 0].set_ylabel('Imputation Method')

        axes[0, 1].hist(dtw_df['normalized_dtw'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Normalized DTW Distance')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of DTW Distances Between Methods')
        axes[0, 1].grid(axis='y', alpha=0.3)

        dtw_summary = dtw_df.groupby(['method1', 'method2'])['normalized_dtw'].mean().sort_values()
        top_pairs = dtw_summary.head(10)
        pair_labels = [f"{idx[0]}\nvs\n{idx[1]}" for idx in top_pairs.index]

        axes[1, 0].barh(range(len(top_pairs)), top_pairs.values)
        axes[1, 0].set_yticks(range(len(top_pairs)))
        axes[1, 0].set_yticklabels(pair_labels, fontsize=8)
        axes[1, 0].set_xlabel('Average Normalized DTW Distance')
        axes[1, 0].set_title('Top 10 Most Similar Method Pairs (Lowest DTW)')
        axes[1, 0].grid(axis='x', alpha=0.3)

        # Correlação DTW x RMSE entre métodos (diferença)
        rmse_diffs = []
        for _, row in dtw_df.iterrows():
            route_data = self.df[self.df['route'] == row['route']]
            rmse1 = route_data[route_data['imputation_method'] == row['method1']]['rmse'].values
            rmse2 = route_data[route_data['imputation_method'] == row['method2']]['rmse'].values
            if len(rmse1) > 0 and len(rmse2) > 0:
                rmse_diffs.append(abs(rmse1[0] - rmse2[0]))
            else:
                rmse_diffs.append(np.nan)

        dtw_df = dtw_df.copy()
        dtw_df['rmse_diff'] = rmse_diffs
        dtw_clean = dtw_df.dropna(subset=['rmse_diff'])

        if not dtw_clean.empty:
            axes[1, 1].scatter(dtw_clean['normalized_dtw'], dtw_clean['rmse_diff'], alpha=0.6, s=50)
            axes[1, 1].set_xlabel('Normalized DTW Distance')
            axes[1, 1].set_ylabel('RMSE Difference')
            axes[1, 1].set_title('DTW Distance vs RMSE Difference\n(Correlation Analysis)')
            axes[1, 1].grid(alpha=0.3)

            if len(dtw_clean) > 2:
                corr = np.corrcoef(dtw_clean['normalized_dtw'], dtw_clean['rmse_diff'])[0, 1]
                axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}',
                                transform=axes[1, 1].transAxes,
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                                verticalalignment='top')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot (entre métodos) salvo em {save_path}")
        return fig

    # -------------------------
    # Comparações de desempenho (seu original)
    # -------------------------
    def plot_performance_comparison(self, save_path='imputation_comparison.png'):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        avg_rmse = self.df.groupby('imputation_method')['rmse'].mean().sort_values()
        axes[0, 0].barh(avg_rmse.index, avg_rmse.values)
        axes[0, 0].set_xlabel('Average RMSE')
        axes[0, 0].set_title('Average RMSE by Imputation Method')
        axes[0, 0].grid(axis='x', alpha=0.3)
        for i, v in enumerate(avg_rmse.values):
            axes[0, 0].text(v, i, f' {v:.0f}', va='center', fontsize=8)

        avg_mae = self.df.groupby('imputation_method')['mae'].mean().sort_values()
        axes[0, 1].barh(avg_mae.index, avg_mae.values)
        axes[0, 1].set_xlabel('Average MAE')
        axes[0, 1].set_title('Average MAE by Imputation Method')
        axes[0, 1].grid(axis='x', alpha=0.3)
        for i, v in enumerate(avg_mae.values):
            axes[0, 1].text(v, i, f' {v:.0f}', va='center', fontsize=8)

        methods = self.df['imputation_method'].unique()
        axes[1, 0].boxplot([self.df[self.df['imputation_method'] == m]['rmse'].values
                            for m in methods], labels=methods)
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].set_title('RMSE Distribution by Imputation Method')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(axis='y', alpha=0.3)

        best_counts = self.df.loc[self.df.groupby('route')['rmse'].idxmin()]['imputation_method'].value_counts()
        axes[1, 1].pie(best_counts.values, labels=best_counts.index, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Win Rate: Best Method by Route (RMSE)')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot salvo em {save_path}")
        return fig

    # -------------------------
    # Relatórios
    # -------------------------
    def generate_report(self, output_file='imputation_analysis_report.txt'):
        """Relatório geral + DTW entre métodos (mantido)."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("IMPUTATION METHOD IMPACT ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("1. OVERALL STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total files analyzed: {len(self.df)}\n")
            f.write(f"Number of routes: {self.df['route'].nunique()}\n")
            f.write(f"Imputation methods tested: {', '.join(self.df['imputation_method'].unique())}\n\n")

            f.write("2. IMPUTATION METHOD RANKING (by average RMSE)\n")
            f.write("-" * 80 + "\n")
            ranking = self.imputation_method_ranking()
            f.write(ranking.to_string() + "\n\n")

            f.write("3. BEST IMPUTATION METHOD PER ROUTE\n")
            f.write("-" * 80 + "\n")
            best_per_route = self.find_best_imputation_per_route()
            f.write(best_per_route.to_string(index=False) + "\n\n")

            f.write("4. WIN RATE ANALYSIS\n")
            f.write("-" * 80 + "\n")
            win_rate_rmse = self.df.loc[self.df.groupby('route')['rmse'].idxmin()]['imputation_method'].value_counts()
            win_rate_mae = self.df.loc[self.df.groupby('route')['mae'].idxmin()]['imputation_method'].value_counts()

            f.write("Based on RMSE:\n")
            for method, count in win_rate_rmse.items():
                pct = (count / len(best_per_route)) * 100
                f.write(f"  {method}: {count} routes ({pct:.1f}%)\n")

            f.write("\nBased on MAE:\n")
            for method, count in win_rate_mae.items():
                pct = (count / len(best_per_route)) * 100
                f.write(f"  {method}: {count} routes ({pct:.1f}%)\n")

            f.write("\n5. DTW SIMILARITY ANALYSIS (Between Methods)\n")
            f.write("-" * 80 + "\n")
            result = self.analyze_dtw_similarity()
            if result is not None:
                similarity_matrix, dtw_df = result
                f.write("Average DTW distances between imputation methods:\n")
                f.write("(Lower values indicate more similar prediction patterns)\n\n")
                f.write(similarity_matrix.to_string() + "\n\n")

                dtw_summary = dtw_df.groupby(['method1', 'method2'])['normalized_dtw'].mean().sort_values()
                f.write("Top 5 Most Similar Method Pairs:\n")
                for i, (pair, dist) in enumerate(dtw_summary.head(5).items(), 1):
                    f.write(f"  {i}. {pair[0]} <-> {pair[1]}: {dist:.2f}\n")

                f.write("\nTop 5 Most Different Method Pairs:\n")
                for i, (pair, dist) in enumerate(dtw_summary.tail(5).items(), 1):
                    f.write(f"  {i}. {pair[0]} <-> {pair[1]}: {dist:.2f}\n")

                f.write(f"\nDTW Stats (between methods):\n")
                f.write(f"  Mean normalized: {dtw_df['normalized_dtw'].mean():.2f}\n")
                f.write(f"  Median normalized: {dtw_df['normalized_dtw'].median():.2f}\n")
                f.write(f"  Std normalized: {dtw_df['normalized_dtw'].std():.2f}\n")
                f.write(f"  Min normalized: {dtw_df['normalized_dtw'].min():.2f}\n")
                f.write(f"  Max normalized: {dtw_df['normalized_dtw'].max():.2f}\n")
            else:
                f.write("Not enough data for DTW analysis.\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"Report saved to {output_file}")

    def generate_report_dtw_vs_original(self, output_file='imputation_dtw_vs_original_report.txt'):
        """Relatório específico do DTW (previsão vs original)."""
        dtw_df = self.calculate_dtw_vs_original(prefer_geant=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DTW (PREDICTION vs ORIGINAL) REPORT\n")
            f.write("=" * 80 + "\n\n")

            found_pct = 100.0 * dtw_df['ref_found'].mean() if not dtw_df.empty else 0.0
            f.write(f"Arquivos com referência encontrada: {found_pct:.1f}%\n")
            if not dtw_df.empty and dtw_df['ref_found'].any():
                f.write(f"Média DTW normalizado (global): {dtw_df['normalized_dtw'].mean():.4f}\n")
                f.write(f"Mediana DTW normalizado (global): {dtw_df['normalized_dtw'].median():.4f}\n")
                f.write(f"Desvio-padrão DTW normalizado: {dtw_df['normalized_dtw'].std():.4f}\n\n")

            # Por método
            summary = self.summarize_dtw_vs_original_by_method(dtw_df)
            if not summary.empty:
                f.write("Resumo por método (quanto menor o 'mean', mais fiel ao original):\n")
                f.write(summary.to_string() + "\n\n")

            # Exemplos com pior/melhor fidelidade
            valid = dtw_df.dropna(subset=['normalized_dtw']).copy()
            if not valid.empty:
                f.write("Top 5 melhores (menor DTW normalizado):\n")
                for i, r in enumerate(valid.sort_values('normalized_dtw').head(5).itertuples(), 1):
                    f.write(f"  {i}. {r.imputation_method} | {r.route} | {r.date} | DTW={r.normalized_dtw:.4f}\n")

                f.write("\nTop 5 piores (maior DTW normalizado):\n")
                for i, r in enumerate(valid.sort_values('normalized_dtw', ascending=False).head(5).itertuples(), 1):
                    f.write(f"  {i}. {r.imputation_method} | {r.route} | {r.date} | DTW={r.normalized_dtw:.4f}\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"Report (DTW vs original) salvo em {output_file}")

    # -------------------------
    # Export
    # -------------------------
    def export_detailed_results(self, output_file='detailed_results.csv'):
        export_df = self.df.drop('predictions', axis=1)
        export_df.to_csv(output_file, index=False)
        print(f"Detailed results exported to {output_file}")


# -------------------------
# Execução principal
# -------------------------
if __name__ == "__main__":
    print("Starting Imputation Impact Analysis...")
    analyzer = ImputationAnalyzer(
        json_file='evaluation_rmse_mae_arima.json',
        original_pref_folder='geant-datasets',         # preferência 1
        aggregated_folder='aggregated_6h_datasets'     # fallback
    )

    # Relatório geral (mantido)
    print("\nGenerating analysis report (between methods)...")
    analyzer.generate_report("imputation_analysis_report_dtw_between_methods_arima.txt")

    # Visualizações de desempenho e DTW (entre métodos)
    print("\nCreating visualizations (performance + between-methods DTW)...")
    analyzer.plot_performance_comparison("imputation_comparison_dtw_arima.png")
    analyzer.plot_dtw_analysis("dtw_analysis_between_methods_arima.png")

    # NOVO: DTW vs Original (prefer geant, agrega para 6H se necessário)
    print("\nCreating DTW vs Original visualizations...")
    analyzer.plot_dtw_vs_original("dtw_vs_original_arima.png")
    analyzer.generate_report_dtw_vs_original("imputation_dtw_vs_original_report_arima.txt")

    # Export detalhado
    print("\nExporting detailed results...")
    analyzer.export_detailed_results()

    # Resumo rápido no console
    print("\n" + "=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)
    ranking = analyzer.imputation_method_ranking()
    print("\nTop 3 Imputation Methods (by avg RMSE):")
    print(ranking[('rmse', 'mean')].head(3).to_string())
    print("\n" + "=" * 80)
    print("Analysis complete! Check the output files for detailed results.")
