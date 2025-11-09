import pandas as pd
import unicodedata
import re

def _normalize_key(s: str) -> str:
    if pd.isna(s): return s
    s = unicodedata.normalize("NFKD", str(s)).encode("ASCII","ignore").decode("ASCII")
    s = s.upper()
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _strip_accents_upper(s: str) -> str:
    if pd.isna(s):
        return s
    s = str(s).strip().upper()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = " ".join(s.split())
    return s

def _ensure_datetime(df, col):
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def join_riscofogo_into_simam_6h(
    risco_fogo: pd.DataFrame,
    simam_data: pd.DataFrame,
    estado_filtro: str = "MATO GROSSO",
    agg: str = "mean"  # "mean", "max" ou "median"
):
    # 1) DateTime
    rf = _ensure_datetime(risco_fogo, "DataHora")
    sm = _ensure_datetime(simam_data, "datahora")

    # 2) Normalização de chaves
    rf["estado_norm"] = rf["Estado"].map(_strip_accents_upper)
    rf["municipio_norm"] = rf["Municipio"].map(_strip_accents_upper)

    sm["estado_norm"] = sm["uf_nome"].map(_strip_accents_upper)
    sm["municipio_norm"] = sm["municipio_nome"].map(_strip_accents_upper)

    estado_alvo = _strip_accents_upper(estado_filtro)

    # 3) Filtro Mato Grosso + limpeza mínima
    rf = rf.loc[rf["estado_norm"] == estado_alvo].dropna(subset=["municipio_norm","DataHora"]).copy()
    sm = sm.loc[sm["estado_norm"] == estado_alvo].dropna(subset=["municipio_norm","datahora"]).copy()

    # 4) Slots de 6h
    # simam_data costuma já estar em 6h certinho; ainda assim, garantimos:
    sm["slot_6h"] = sm["datahora"].dt.floor("6h")
    rf["slot_6h"] = rf["DataHora"].dt.floor("6h")

    # 5) Agregação do RiscoFogo por município+slot
    agg_funcs = {"mean": "mean", "max": "max", "median": "median"}
    if agg not in agg_funcs:
        raise ValueError("agg deve ser 'mean', 'max' ou 'median'.")
    rf_agg = (rf
              .groupby(["municipio_norm","slot_6h"], as_index=False)
              .agg(rf_riscofogo=( "RiscoFogo", agg_funcs[agg] ))
             )
    # renomeia coluna final para deixar claro o método
    rf_agg = rf_agg.rename(columns={"rf_riscofogo": f"rf_riscofogo_6h_{agg}"})

    # 6) Merge determinístico (mantém TODO simam_data)
    merged = sm.merge(
        rf_agg,
        how="left",
        left_on=["municipio_norm","slot_6h"],
        right_on=["municipio_norm","slot_6h"]
    )

    # 7) Diagnóstico simples
    col_rf = f"rf_riscofogo_6h_{agg}"
    cobertura_global = merged[col_rf].notna().mean() * 100.0
    diag = (merged
            .groupby(["estado_norm","municipio_norm"], as_index=False)
            .agg(
                linhas=("datahora","count"),
                com_rf=(col_rf, lambda s: int(s.notna().sum()))
            ))
    diag["cobertura_%"] = 100 * diag["com_rf"] / diag["linhas"]

    # 8) Organização opcional de colunas
    front = ["datahora","uf_nome","municipio_nome","slot_6h", col_rf]
    merged = merged[front + [c for c in merged.columns if c not in front]]

    return merged, diag, col_rf