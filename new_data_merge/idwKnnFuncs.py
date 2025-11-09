import pandas as pd
import numpy as np

EARTH_RADIUS_KM = 6371.0088

def _haversine_km(lat1, lon1, lat2, lon2):
    """Distância Haversine (km) vetorizada. Entradas em radianos."""
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(h))

def spatial_impute_riscofogo_idw(
    df: pd.DataFrame,
    risk_col: str,                 # ex.: "rf_riscofogo_6h_max"
    slot_col: str = "datahora",    # use sua coluna de horário
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    k: int = 5,
    max_radius_km: float = 150.0,
    power: float = 2.0,
    min_neighbors: int = 2,
    max_slot_shift: int = 0,       # 0 = mesmo slot; 1 = ±1 slot como fallback
    eps: float = 1e-6,
    out_suffix: str = "_spatial"
) -> pd.DataFrame:

    req = [risk_col, slot_col, lat_col, lon_col]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Coluna obrigatória ausente: {c}")

    # Cópia e normalização básica
    data = df.copy().reset_index(drop=True)  # <- garante posições 0..n-1
    data[slot_col] = pd.to_datetime(data[slot_col], errors="coerce")
    data[lat_col]  = pd.to_numeric(data[lat_col], errors="coerce")
    data[lon_col]  = pd.to_numeric(data[lon_col], errors="coerce")

    # Máscaras válidas
    keys_ok = data[[lat_col, lon_col, slot_col]].notna().all(axis=1)
    m_obs = data[risk_col].notna() & keys_ok
    m_nan = data[risk_col].isna() & keys_ok

    # Coordenadas em radianos (numpy)
    lat_rad_all = np.deg2rad(data[lat_col].to_numpy(dtype=float))
    lon_rad_all = np.deg2rad(data[lon_col].to_numpy(dtype=float))

    # Índices de posições por slot (observados e faltantes)
    slots_obs = data.loc[m_obs, slot_col].unique()
    obs_by_slot = {s: np.where(m_obs & (data[slot_col] == s))[0] for s in slots_obs}

    slots_miss = data.loc[m_nan, slot_col].unique()
    miss_by_slot = {s: np.where(m_nan & (data[slot_col] == s))[0] for s in slots_miss}

    # Helper para slots vizinhos (±6h)
    def _neighbor_slots(s, shift):
        if shift == 0:
            return [s]
        delta = pd.Timedelta(hours=6)
        return [s - i*delta for i in range(1, shift+1)] + [s + i*delta for i in range(1, shift+1)]

    out_col = f"{risk_col}{out_suffix}"
    data[out_col] = np.nan
    data["impute_neighbors"] = np.nan
    data["impute_dist_km"]   = np.nan

    # Vetor de valores observados (numpy) para acesso rápido
    risk_vals_np = data[risk_col].to_numpy(dtype=float)

    for s, miss_pos in miss_by_slot.items():
        if miss_pos.size == 0:
            continue

        # Candidatos: mesmo slot primeiro
        cand_pos_list = []
        if s in obs_by_slot:
            cand_pos_list.extend(obs_by_slot[s].tolist())

        # Fallback temporal (±6h) se habilitado e poucos vizinhos
        if (len(cand_pos_list) < min_neighbors) and (max_slot_shift > 0):
            for sh in range(1, max_slot_shift + 1):
                for sn in _neighbor_slots(s, sh):
                    if sn in obs_by_slot:
                        cand_pos_list.extend(obs_by_slot[sn].tolist())
                if len(cand_pos_list) >= min_neighbors:
                    break

        if not cand_pos_list:
            continue

        cand_pos = np.array(cand_pos_list, dtype=int)
        cand_lat = lat_rad_all[cand_pos]  # (n,)
        cand_lon = lon_rad_all[cand_pos]  # (n,)
        cand_val = risk_vals_np[cand_pos] # (n,)

        # Coordenadas dos faltantes (m,1) — agora com numpy, ok usar [:, None]
        miss_lat = lat_rad_all[miss_pos][:, None]  # (m,1)
        miss_lon = lon_rad_all[miss_pos][:, None]  # (m,1)

        # Distâncias (m x n)
        d_km = _haversine_km(miss_lat, miss_lon, cand_lat[None, :], cand_lon[None, :])
        within = d_km <= max_radius_km

        for i, pos in enumerate(miss_pos):
            mask_row = within[i]
            if not mask_row.any():
                continue

            d_row = d_km[i, mask_row]
            v_row = cand_val[mask_row]

            # Ordena por distância e limita a k
            ord_idx = np.argsort(d_row)
            take = ord_idx[:min(k, ord_idx.size)]
            d_sel = d_row[take]
            v_sel = v_row[take]

            if d_sel.size < min_neighbors:
                continue

            # Pesos IDW
            w = 1.0 / np.power(d_sel + eps, power)
            w = w / w.sum()

            imputed = float(np.sum(w * v_sel))
            data.at[pos, out_col] = imputed
            data.at[pos, "impute_neighbors"] = int(d_sel.size)
            data.at[pos, "impute_dist_km"]   = float(np.average(d_sel, weights=w))

    # Coluna "filled" que preserva observados e usa spatial para os NaN
    filled_col = f"{risk_col}_filled"
    data[filled_col] = data[risk_col].copy()
    mask_fill = data[filled_col].isna() & data[out_col].notna()
    data.loc[mask_fill, filled_col] = data.loc[mask_fill, out_col]

    # Diagnóstico simples
    antes = df[risk_col].notna().mean() * 100.0
    depois = data[filled_col].notna().mean() * 100.0
    print(f"Cobertura {risk_col}: antes={antes:.2f}% | depois={depois:.2f}% (+{depois-antes:.2f} p.p.)")

    return data

def iterative_spatial_fill_until(
    df: pd.DataFrame,
    risk_col: str,                 # ex.: "rf_riscofogo_6h_max" ou "rf_riscofogo_6h_mean"
    slot_col: str = "datahora",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    target_coverage: float = 0.80, # 80%
    passes: list[dict] | None = None
):
    """
    Roda múltiplos passes de IDW-kNN até alcançar target_coverage (ou esgotar os passes).
    Requer a função spatial_impute_riscofogo_idw já definida/importada no escopo.
    Retorna (df_final, log_df).
    """

    # Plano padrão: vai ficando mais permissivo a cada passo
    if passes is None:
        passes = [
            # k, raio, potência, viz mínimos, fallback temporal (±slots*6h)
            {"k": 5,  "max_radius_km": 150, "power": 2.0, "min_neighbors": 3, "max_slot_shift": 0},
            {"k": 8,  "max_radius_km": 200, "power": 2.0, "min_neighbors": 3, "max_slot_shift": 0},
            {"k": 8,  "max_radius_km": 250, "power": 1.8, "min_neighbors": 3, "max_slot_shift": 0},
            {"k": 10, "max_radius_km": 300, "power": 1.6, "min_neighbors": 2, "max_slot_shift": 0},
            {"k": 10, "max_radius_km": 300, "power": 1.6, "min_neighbors": 2, "max_slot_shift": 1},
            {"k": 12, "max_radius_km": 400, "power": 1.5, "min_neighbors": 2, "max_slot_shift": 1},
        ]

    work = df.copy()
    work[slot_col] = pd.to_datetime(work[slot_col], errors="coerce")
    total = len(work)
    base_cov = work[risk_col].notna().mean()
    logs = []

    for i, params in enumerate(passes, start=1):
        # roda um passe (gera col *_filled e métricas de imputação para esse passe)
        out = spatial_impute_riscofogo_idw(
            df=work,
            risk_col=risk_col,
            slot_col=slot_col,
            lat_col=lat_col,
            lon_col=lon_col,
            out_suffix=f"_spatial_p{i}",
            **params
        )

        filled_col = f"{risk_col}_filled"   # criado pela função
        out_col    = f"{risk_col}_spatial_p{i}"  # criado pela função (imputado neste passe)

        # quantos foram imputados neste passe (eram NaN e viraram não-NaN)?
        was_nan   = work[risk_col].isna()
        now_filled = out[filled_col].notna()
        newly_imputed_mask = was_nan & now_filled
        newly_imputed = int(newly_imputed_mask.sum())

        # adota o preenchido deste passe como novo estado do risk_col
        work[risk_col] = out[filled_col]

        # traz metadados deste passe (opcional para auditoria)
        work[f"impute_neighbors_p{i}"] = out.get("impute_neighbors", np.nan)
        work[f"impute_dist_km_p{i}"]   = out.get("impute_dist_km", np.nan)
        work[out_col]                  = out.get(out_col, np.nan)

        cov = work[risk_col].notna().mean()
        logs.append({
            "pass": i,
            **params,
            "imputed_this_pass": newly_imputed,
            "coverage_after": float(cov),
        })

        # critério de parada
        if cov >= target_coverage:
            break

        # proteção contra estagnação: se este passe não imputou nada, segue para o próximo; se zerar 2x, encerra
        if i >= 2 and logs[-1]["imputed_this_pass"] == 0 and logs[-2]["imputed_this_pass"] == 0:
            break

    # renomeia a saída final para não confundir com a coluna original
    work[f"{risk_col}_filled_final"] = work[risk_col]

    log_df = pd.DataFrame(logs)
    print(f"Cobertura inicial: {base_cov*100:.2f}% | Cobertura final: {work[f'{risk_col}_filled_final'].notna().mean()*100:.2f}%")
    return work, log_df