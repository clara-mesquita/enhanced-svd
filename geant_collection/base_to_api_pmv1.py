import os
import time
from datetime import date, datetime, timedelta
import requests
import json
from urllib3.exceptions import InsecureRequestWarning

# Otimizações e configurações
# Removida a linha `disable_warnings(InsecureRequestWarning)` para garantir a segurança.
# A requisição padrão do 'requests' já vem com o 'verify=True', 
# o que valida o certificado SSL e protege contra ataques "man-in-the-middle".
# O URL base foi fixado para o arquivo público do GÉANT PMP.
base_url = "https://pmp-archive.geant.org/esmond/perfsonar/archive/?"
today = date.today()

# Configurações dinâmicas
# A faixa de tempo pode ser facilmente alterada aqui (em segundos).
# 15552000 segundos = 6 meses
TIME_RANGE = 15552000
FOLDER = f"./pmp_throughput_data/{today.strftime('%Y-%m-%d')}"
# O nome do tipo de evento para throughput.
EVENT_TYPE = "throughput"

# Função para fazer a requisição HTTP com tentativas limitadas e backoff
def make_request_with_retries(url, params=None, max_retries=5, initial_wait=5):
    """
    Tenta fazer uma requisição GET com um número limitado de tentativas e backoff exponencial.
    Isso torna o script mais robusto contra falhas temporárias de rede ou no servidor.
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, verify=True)
            response.raise_for_status()  # Lança um erro para status de erro (4xx ou 5xx)
            return response
        except requests.exceptions.RequestException as e:
            print(f"Erro na tentativa {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                wait_time = initial_wait * (2 ** attempt)
                print(f"Tentando novamente em {wait_time} segundos...")
                time.sleep(wait_time)
            else:
                print(f"Todas as {max_retries} tentativas falharam. Abortando.")
                raise # Re-lança o erro após a última tentativa

# Função principal para coletar os dados
def collect_all_throughput_data():
    """
    Busca automaticamente todos os testes de throughput e coleta os dados.
    """
    # 1. Encontrar todos os testes de throughput disponíveis.
    print(f"Buscando todos os testes de '{EVENT_TYPE}' disponíveis...")
    metadata_params = {"pscheduler-test-type": EVENT_TYPE}
    
    try:
        metadata_response = make_request_with_retries(base_url, metadata_params)
        measurements = metadata_response.json()
        print(f"Encontrados {len(measurements)} testes de throughput.")
        
    except requests.exceptions.RequestException as e:
        print(f"Não foi possível obter a lista de medições. Verifique o URL ou a conexão: {e}")
        return

    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)
        print(f"Pasta de saída criada: {FOLDER}")
        
    # 2. Iterar sobre cada teste e coletar os dados.
    processed_count = 0
    for measurement in measurements:
        source = measurement.get('source')
        destination = measurement.get('destination')
        
        if not source or not destination:
            continue

        base_uri = None
        for event in measurement.get('event-types',):
            if event.get('event-type') == EVENT_TYPE:
                base_uri = event.get('base-uri')
                break

        if not base_uri:
            print(f"Skipping measurement from {source} to {destination}, no '{EVENT_TYPE}' URI found.")
            continue
        
        file_name = f"{source}_to_{destination}_{EVENT_TYPE}_{today.strftime('%Y-%m-%d')}.csv"
        file_path = os.path.join(FOLDER, file_name)
        
        print(f"Processando teste de {source} para {destination}...")

        # 3. Coletar os dados de série temporal para o teste específico.
        data_url = "https://pmp-archive.geant.org" + base_uri
        data_params = {"time-range": TIME_RANGE}

        try:
            data_response = make_request_with_retries(data_url, data_params)
            time_series_data = data_response.json()

            # 4. Escrever os dados no arquivo CSV.
            with open(file_path, "w", newline='') as f:
                f.write("timestamp,data,vazao_bps\n")
                for entry in time_series_data:
                    ts = entry.get('ts')
                    val = entry.get('val')
                    if ts is not None and val is not None:
                        # O PerfSONAR armazena o valor de vazão em bits por segundo (bps)
                        f.write(f"{ts},{datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')},{val}\n")
            print(f"Dados salvos com sucesso em {file_path}")
            processed_count += 1
            
        except requests.exceptions.RequestException as e:
            print(f"Erro ao obter dados para {source} para {destination}: {e}")
            
    print(f"\nColeta de dados concluída. {processed_count} arquivos gerados.")

if __name__ == "__main__":
    collect_all_throughput_data()