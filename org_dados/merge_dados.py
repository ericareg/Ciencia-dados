import pandas as pd
import numpy as np
import unicodedata
import argparse

"""
Define argumentos de linha de comando.
Exemplo de uso:
python org_dados/merge_dados.py --years 2019 2020 2021
"""
parser = argparse.ArgumentParser(description='Processa dados de arquivos CSV.')
parser.add_argument('--years', nargs='+', type=int, help='Lista de anos para processar', required=True)
args = parser.parse_args()

def norm_str(s: pd.Series) -> pd.Series:
    """
    Normaliza strings: maiúsculas, sem acentos, sem espaços extras.
    Utilizado para remover inconsistências em nomes de estados e municípios.
    Exemplo: "  São Paulo " -> "SAO PAULO"
    """
    s = s.astype(str).str.strip().str.upper()
    s = s.apply(lambda x: ''.join(ch for ch in unicodedata.normalize('NFKD', x) if not unicodedata.combining(ch)))
    s = s.str.replace(r'\s+', ' ', regex=True)
    return s

# ---------------------------
anos = args.years

for ano in anos:
    """
    Processa dados de risco de fogo e dados do SIMAM para um ano específico.
    Realiza as seguintes etapas:
    1. Carrega os dados de risco de fogo do arquivo CSV correspondente ao ano.
    2. Converte a coluna "DataHora" para o tipo datetime e "RiscoFogo" para numérico.
    3. Agrupa os dados em janelas de 6 horas, calculando a média do risco de fogo para cada estado e município.
    4. Normaliza os nomes dos estados e municípios para facilitar a junção.
    5. Carrega os dados do SIMAM do arquivo CSV correspondente ao ano.
    6. Normaliza os nomes dos estados e municípios no conjunto de dados do SIMAM.
    7. Realiza a junção dos dois conjuntos de dados com base na data/hora, estado e município.
    8. Remove colunas desnecessárias e linhas com valores ausentes no risco de fogo.
    9. Salva o conjunto de dados resultante em um novo arquivo CSV.
    10. Repete o processo para cada ano na lista fornecida.
    """
    df_risco = pd.read_csv(f"risco_fogo/{ano}.csv")

    df_risco["DataHora"]   = pd.to_datetime(df_risco["DataHora"], format="%Y/%m/%d %H:%M:%S", errors="coerce")
    df_risco["RiscoFogo"]  = pd.to_numeric(df_risco["RiscoFogo"], errors="coerce")
    df_risco               = df_risco.dropna(subset=["DataHora"])

    grouper_6h = pd.Grouper(key="DataHora", freq="6h", label="right", closed="left")

    agg = (
        df_risco.groupby([grouper_6h, "Estado", "Municipio"])["RiscoFogo"]
        .mean()
        .reset_index()
        .rename(columns={"DataHora": "JanelaFim", "RiscoFogo": "RiscoFogoMedia"})
    )

    agg["Estado_n"]    = norm_str(agg["Estado"])
    agg["Municipio_n"] = norm_str(agg["Municipio"])

    agg_merge = agg[["JanelaFim", "Estado_n", "Municipio_n", "RiscoFogoMedia"]]

    # ---------------------------
    df_simam = pd.read_csv(f"simam_data/dados_{ano}.csv")
    df_simam = df_simam[df_simam["uf_nome"].isin(["MATO GROSSO"])]

    df_simam["datahora"] = pd.to_datetime(df_simam["datahora"], errors="coerce")

    # normalizar chaves
    df_simam["uf_nome_n"]           = norm_str(df_simam["uf_nome"])
    df_simam["municipio_nome_n"]    = norm_str(df_simam["municipio_nome"])

    # ---------------------------
    merged = (
        df_simam.merge(
            agg_merge,
            left_on = ["datahora", "uf_nome_n", "municipio_nome_n"],
            right_on= ["JanelaFim", "Estado_n", "Municipio_n"],
            how="left"
        )
    )

    merged = merged.drop(columns=["JanelaFim", "Estado_n", "Municipio_n", "uf_nome_n", "municipio_nome_n"])
    merged = merged.sort_values(["datahora", "uf_nome", "municipio_nome"]).reset_index(drop=True)
    merged = merged.dropna(subset=["RiscoFogoMedia"])
    
    merged.to_csv(f"proc_data/saida_merged_{ano}.csv", index=False)
