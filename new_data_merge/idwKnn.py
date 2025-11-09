from idwKnnFuncs import *
import pandas as pd
import argparse

# --------------------
INPUT_PATH = 'new_data_merge/cleaned_data/'
OUTPUT_PATH = 'new_data_merge/spatial_filled_data/'

#   Define argumentos de linha de comando.
#   --years: lista de anos a serem processados
parser = argparse.ArgumentParser(description='Processa dados de arquivos CSV.')
parser.add_argument('--years', nargs='+', type=int, help='Lista de anos para processar', required=True)
args = parser.parse_args()
anos = args.years

for ano in anos:
    merged = pd.read_csv(f"{INPUT_PATH}merged_data_{ano}.csv", parse_dates=["datahora"])
    risk_col = "rf_riscofogo_6h_mean"

    filled_df, log_df = iterative_spatial_fill_until(
        df=merged,
        risk_col=risk_col,
        slot_col="datahora",
        lat_col="latitude",
        lon_col="longitude",
        target_coverage=0.80  # 80%
    )

    filled_df.to_csv(f"{OUTPUT_PATH}merged_data_spatial_iter_{ano}.csv", index=False)
    print(log_df)