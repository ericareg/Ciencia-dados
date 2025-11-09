from dataCleaningFuncs import *
import pandas as pd
import argparse

# --------------------
SIMAM_PATH = 'org_dados/simam_data/'
INPE_PATH = 'org_dados/risco_fogo/'
OUTPUT_PATH = 'new_data_merge/cleaned_data/'

#   Define argumentos de linha de comando.
#   --years: lista de anos a serem processados
parser = argparse.ArgumentParser(description='Processa dados de arquivos CSV.')
parser.add_argument('--years', nargs='+', type=int, help='Lista de anos para processar', required=True)
args = parser.parse_args()
anos = args.years

for ano in anos:
    risco_fogo = pd.read_csv(f"{INPE_PATH}{ano}.csv", sep=",", compression=None)
    simam_data = pd.read_csv(f"{SIMAM_PATH}{ano}.csv", sep=",", compression=None)

    merged, diag, col_rf = join_riscofogo_into_simam_6h(
        risco_fogo, 
        simam_data, 
        estado_filtro="MATO GROSSO", 
        agg="mean"  # ou "max"
    )
    
    useful_cols = [
        "datahora", "slot_6h", "uf_nome", "municipio_nome",
        "co_ppb", "no2_ppb", 
        "o3_ppb", "pm25_ugm3", "so2_ugm3", 
        "precipitacao_mmdia", "temperatura_c", 
        "umidade_relativa_percentual", "vento_direcao_grau", 
        "vento_velocidade_ms", "rf_riscofogo_6h_mean", "latitude", "longitude"
    ]
    merged = merged[useful_cols]

    merged.to_csv(f"{OUTPUT_PATH}merged_data_{ano}.csv", index=False)
    diag.to_csv(f"{OUTPUT_PATH}data_diagnostics_{ano}.csv", index=False)

    col_rf = "rf_riscofogo_6h_mean"
    print("Cobertura global (%):", (merged[col_rf].notna().mean() * 100).round(2))
    

"""
Exemplo de uso:
python new_data_merge/dataCleaning.py --years 2016
"""