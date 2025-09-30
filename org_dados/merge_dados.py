import pandas as pd
import numpy as np
import unicodedata
import argparse

parser = argparse.ArgumentParser(description='Processa dados de arquivos CSV.')
parser.add_argument('--years', nargs='+', type=int, help='Lista de anos para processar', required=True)
parser.add_argument('-D', '--describe', action='store_true', help='Exibir estatísticas descritivas dos dados')
args = parser.parse_args()


'''
Processa arquivos CSV para filtrar dados específicos e salvar os resultados.
'''
import pandas as pd
# ---------------------------
# 1) Função de normalização
# ---------------------------
def norm_str(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.upper()
    # remove acentos
    s = s.apply(lambda x: ''.join(ch for ch in unicodedata.normalize('NFKD', x) if not unicodedata.combining(ch)))
    # espaços múltiplos -> único
    s = s.str.replace(r'\s+', ' ', regex=True)
    return s


anos = args.years

for ano in anos:
    # ---------------------------
    # 2) Agregar csv2 por janelas de 6h
    # ---------------------------
    df2 = pd.read_csv(f"risco_fogo/{ano}.csv")

    df2["DataHora"]   = pd.to_datetime(df2["DataHora"], format="%Y/%m/%d %H:%M:%S", errors="coerce")
    df2["RiscoFogo"]  = pd.to_numeric(df2["RiscoFogo"], errors="coerce")
    df2               = df2.dropna(subset=["DataHora"])

    # janela 6h ancorada em 00:00, fechada à esquerda, label no fim da janela (06:00, 12:00, 18:00, 00:00 do dia seguinte)
    grouper_6h = pd.Grouper(key="DataHora", freq="6h", label="right", closed="left")

    agg = (
        df2.groupby([grouper_6h, "Estado", "Municipio"])["RiscoFogo"]
        .mean()
        .reset_index()
        .rename(columns={"DataHora": "JanelaFim", "RiscoFogo": "RiscoFogoMedia"})
    )

    # normalizar chaves textuais para casar com o outro CSV
    agg["Estado_n"]    = norm_str(agg["Estado"])
    agg["Municipio_n"] = norm_str(agg["Municipio"])

    # manter somente o necessário para o merge
    agg_merge = agg[["JanelaFim", "Estado_n", "Municipio_n", "RiscoFogoMedia"]]

    # ---------------------------
    # 3) Carregar CSV "completo" (6/6h, todos municípios)
    # ---------------------------
    met = pd.read_csv(f"raw_data/dados_{ano}.csv")
    met = met[met["uf_nome"].isin(["MATO GROSSO"])]

    met["datahora"] = pd.to_datetime(met["datahora"], errors="coerce")

    # normalizar chaves
    met["uf_nome_n"]       = norm_str(met["uf_nome"])
    met["municipio_nome_n"] = norm_str(met["municipio_nome"])

    # ---------------------------
    # 4) MERGE (LEFT)
    # ---------------------------
    merged = (
        met.merge(
            agg_merge,
            left_on = ["datahora", "uf_nome_n", "municipio_nome_n"],
            right_on= ["JanelaFim", "Estado_n", "Municipio_n"],
            how="left"
        )
    )

    # opcional: limpar colunas auxiliares
    merged = merged.drop(columns=["JanelaFim", "Estado_n", "Municipio_n", "uf_nome_n", "municipio_nome_n"])

    # ordenar por dia/hora
    merged = merged.sort_values(["datahora", "uf_nome", "municipio_nome"]).reset_index(drop=True)

    # Exportar
    merged = merged.dropna(subset=["RiscoFogoMedia"])
    merged.to_csv(f"proc_data/saida_merged_{ano}.csv", index=False)
