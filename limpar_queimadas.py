#!/usr/bin/env python3
# limpar_queimadas.py — Limpeza e redução de bases do INPE (SISAM) focando MT/TO/PA.
#
# Uso:
#   python limpar_queimadas.py --in caminho/arquivo.csv --out saida.parquet --ufs MT TO PA
#
# Requisitos: pandas, pyarrow
#
import argparse
import io
import os
import re
import sys
import zipfile
from typing import Iterable, List

import pandas as pd
from pandas.api.types import is_string_dtype
import unicodedata


def strip_accents(s: str) -> str:
    if not isinstance(s, str):
        return s
    return ''.join(ch for ch in unicodedata.normalize('NFKD', s) if not unicodedata.combining(ch))


def fix_mojibake_series(ser: pd.Series) -> pd.Series:
    def _fix_one(x):
        if not isinstance(x, str):
            return x
        if 'Ã' in x or 'Â' in x or '¢' in x:
            try:
                return x.encode('latin1', errors='ignore').decode('utf-8', errors='ignore')
            except Exception:
                return x
        return x
    return ser.apply(_fix_one)


def coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if is_string_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()
            df[col] = fix_mojibake_series(df[col])
    return df


def choose_useful_columns(cols: List[str]) -> List[str]:
    preferred = [
        'datahora', 'data', 'dt', 'dt_foco',
        'uf_sigla', 'uf', 'uf_nome', 'estado',
        'municipio', 'municipio_nome',
        'latitude', 'longitude',
        'bioma', 'satelite', 'risco_fogo', 'frp'
    ]
    found = [c for c in preferred if c in cols]
    return list(dict.fromkeys(found)) or list(cols)


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for c in [c for c in df.columns if c.lower() in {'datahora','data','dt','dt_foco'}]:
        try:
            df[c] = pd.to_datetime(df[c], errors='coerce', dayfirst=True)
        except Exception:
            pass
    return df


def normalize_uf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if 'uf_sigla' in df.columns:
        df['uf_sigla_norm'] = df['uf_sigla'].str.upper().str.strip()
    elif 'uf' in df.columns:
        df['uf_sigla_norm'] = df['uf'].str.upper().str.strip()
    else:
        df['uf_sigla_norm'] = pd.NA

    name_col = None
    for cand in ['uf_nome','estado']:
        if cand in df.columns:
            name_col = cand
            break
    if name_col is not None:
        df['uf_nome_norm'] = df[name_col].astype(str).str.upper().map(strip_accents).str.strip()
    else:
        df['uf_nome_norm'] = pd.NA
    return df


def filter_states(df: pd.DataFrame, states: Iterable[str]) -> pd.DataFrame:
    states = {s.upper() for s in states}
    df = normalize_uf_columns(df)

    mask = pd.Series(False, index=df.index)
    if 'uf_sigla_norm' in df.columns:
        mask = mask | df['uf_sigla_norm'].isin(states)

    if 'uf_nome_norm' in df.columns:
        mapping = {'MT':'MATO GROSSO','TO':'TOCANTINS','PA':'PARA'}
        target_names = set(mapping.get(s, s) for s in states)
        mask = mask | df['uf_nome_norm'].isin(target_names)

    return df.loc[mask]


def read_weird_csv_like(df_singlecol: pd.DataFrame) -> pd.DataFrame:
    s = df_singlecol.iloc[:,0].astype(str)
    header = s.iloc[0]
    data = s.iloc[1:]
    cols = [c.strip() for c in header.split(',')]
    rows = data.str.split(',', expand=True)
    rows.columns = cols
    return rows


def smart_read(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(('.xlsx','.xls')):
        df = pd.read_excel(path, header=None)
        if df.shape[1] == 1:
            df = read_weird_csv_like(df)
        else:
            # tenta promover a primeira linha para header se fizer sentido
            if df.columns.astype(str).str.contains('Unnamed').all():
                first_row = df.iloc[0].astype(str).tolist()
                df.columns = first_row
                df = df.iloc[1:].reset_index(drop=True)
        return df

    if lower.endswith('.zip'):
        with zipfile.ZipFile(path) as zf:
            name = next((n for n in zf.namelist() if n.lower().endswith('.csv')), None)
            if name is None:
                raise ValueError('ZIP sem CSV.')
            with zf.open(name) as fh:
                try:
                    df = pd.read_csv(fh, sep=None, engine='python', encoding='utf-8', on_bad_lines='skip')
                except Exception:
                    fh.seek(0)
                    df = pd.read_csv(fh, sep=None, engine='python', encoding='latin1', on_bad_lines='skip')
                if df.shape[1] == 1:
                    fh.seek(0)
                    text = io.TextIOWrapper(fh, encoding='utf-8', errors='ignore')
                    lines = text.read().splitlines()
                    df = pd.DataFrame(lines, columns=['_raw'])
                    df = read_weird_csv_like(df)
                return df

    # CSV
    try:
        df = pd.read_csv(path, sep=None, engine='python', encoding='utf-8', on_bad_lines='skip')
    except Exception:
        df = pd.read_csv(path, sep=None, engine='python', encoding='latin1', on_bad_lines='skip')

    if df.shape[1] == 1:
        s = df.iloc[:,0].astype(str)
        header = s.iloc[0]
        data = s.iloc[1:]
        cols = [c.strip() for c in header.split(',')]
        rows = data.str.split(',', expand=True)
        rows.columns = cols
        df = rows

    return df


def process_file(input_path: str, output_path: str, states: Iterable[str]) -> None:
    df = smart_read(input_path)
    # padroniza lowercase para seleção case-insensitive
    df.columns = [str(c).strip() for c in df.columns]
    df = coerce_columns(df)
    lower_map = {c: c.lower() for c in df.columns}
    df.rename(columns=lower_map, inplace=True)

    useful_cols = choose_useful_columns(list(df.columns))
    df = df[useful_cols] if useful_cols else df

    df = parse_dates(df)
    df = filter_states(df, states)

    for col in ['latitude','longitude','frp','risco_fogo']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')

    for cand in ['datahora','data','dt','dt_foco']:
        if cand in df.columns:
            df = df.sort_values(by=cand, ascending=True)
            break

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    ext = os.path.splitext(output_path)[1].lower()
    if ext in {'.parquet', '.pq'}:
        df.to_parquet(output_path, index=False)
    elif ext == '.csv':
        df.to_csv(output_path, index=False)
    elif ext == '.gz':
        df.to_csv(output_path, index=False, compression='gzip')
    else:
        df.to_parquet(output_path, index=False)

    print(f"[ok] Linhas após filtro: {len(df):,}. Colunas: {list(df.columns)}")


def main():
    parser = argparse.ArgumentParser(description='Limpeza e redução de dados de queimadas (INPE/SISAM).')
    parser.add_argument('--in', dest='input_path', required=True, help='Arquivo de entrada (CSV/XLSX/ZIP).')
    parser.add_argument('--out', dest='output_path', required=True, help='Arquivo de saída (Parquet/CSV/GZ).')
    parser.add_argument('--ufs', nargs='+', default=['MT','TO','PA'], help='Siglas das UFs para filtrar.')
    args = parser.parse_args()
    process_file(args.input_path, args.output_path, args.ufs)


if __name__ == '__main__':
    main()
