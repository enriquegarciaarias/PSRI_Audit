"""
exist/meme_reliability.py

EXPLORATORY DIAGNOSTIC, NOT for the paper as confirmatory.

Diagnóstico de la DESALINEACIÓN POBLACIONAL de EXIST MEDIDO con los propios
datos del PSRI, sin depender de las etiquetas. La hipótesis central de EXIST
(cadena de causa común: meme ambiguo -> reacción fisiológica más ruidosa en
quien lo ve Y más desacuerdo en quien lo juzga) exige que el agregado de
fisiología POR MEME tenga señal reproducible propia: si el agregado es solo
ruido promediado sobre los pocos sujetos que vieron cada meme, la correlación
con el desacuerdo anotador es imposible, exista o no la cadena. Medir esa
fiabilidad del agregado es medir la desconexión de diseño con datos.

El PSRI es un método, no un índice suelto: aplicar el método obliga a auditar
los pre-requisitos de su claim. Aquí la unidad de análisis es el meme, un
agregado sobre pocos sujetos, y el pre-requisito es que ese agregado tenga
señal reproducible propia; si es solo ruido promediado, la correlación con el
desacuerdo es imposible por diseño. Es la auditoría de agregación, análoga a
la verificación de latencia de sensores que exige S_coher en K-EmoCon: cada
dimensión obliga a comprobar el pre-requisito del que depende su claim.

Estructura de los datos (verificada): 8 sujetos (HR+ET, misma población),
4044 memes comunes, 7782 pares (meme, username), y CADA MEME ES VISTO POR
COMO MÁXIMO 2 SUJETOS (media 1.92, mediana 2; 306 memes con 1 solo viewer).
El agregado por meme es por tanto la media sobre 1-2 sujetos.

Cinco cheques:

  1. ESTRUCTURA DE COBERTURA: distribution de viewers por meme y pares de
     sujetos (frozenset de usuarios por meme). El límite k<=2 es en sí mismo
     un hecho estructural de la desconexión.

  2. DESCOMPOSICIÓN DE VARIANZA (ICC): para cada métrica (componentes PSRI
     y métricas crudas) se estima la fracción de varianza atribuible a la
     identidad del SUJETO (ICC_subject), a la identidad del MEME (ICC_meme,
     crudo) y al meme NETA de rasgos de sujeto (ICC_meme_adj, sobre valores
     centrados por sujeto). Si ICC_meme_adj ~ 0, no hay señal de meme que
     conectar con el desacuerdo: la desconexión es cuantitativa, no solo
     conceptual.

  3. FIABILIDAD ENTRE VIEWERS DEL MISMO MEME (neta de sujeto): con k=2, la
     fiabilidad del agregado es la correlación entre los DOS viewers de cada
     meme (Spearman sobre valores centrados por sujeto), por par de sujetos y
     resumida por métrica. De ahí, con Spearman-Brown, la fiabilidad de la
     media de 2 viewers (r_mean) y el factor de atenuación sqrt(r_mean) que
     sufriría CUALQUIER correlación a nivel de meme.

  4. COTA DE POTENCIA EFECTIVA: N_eff = N_memes / (1 + (k-1)*ICC_meme_adj) y
     el tamaño de efecto mínimo detectable al 80% de potencia. Convierte el
     nulo en un claim acotado ("el diseño solo podría ver efectos > X"),
     no en un cero.

  5. SONDA DEL ESLABÓN (necesita etiquetas): dispersión entre viewers por
     meme (|v1 - v2|) de cada métrica contra las métricas de desacuerdo
     (entropy_21/22/23, soft_21_yes) y la valoración de sexismo. Es el
     eslabón de la cadena que conecta con la fisiología (meme ambiguo ->
     reacción MÁS VARIABLE); se espera nulo precisamente porque los cheques
     2-3 muestran que no hay señal de meme que dispersar.

Todas las pruebas son exploratorias: NO hay corrección por comparaciones
múltiples (el objetivo no es rechazar una hipótesis sino caracterizar el
diseño) y ninguna debe leerse como evidencia sobre la existencia o no de la
cadena causal, sino sobre la capacidad del diseño de detectarla.

Garantía de revert limpio: este módulo SOLO lee
  results/input/EXIST/HR_.xlsx, ET_.xlsx y EXIST2026_training.json
y escribe en results/output/EXIST/meme_reliability/. No importa ni modifica
build_features.py, sanity_check.py ni final_check.py; borrar el directorio de
salida restaura la situación anterior.

Ejecución:
  .venv/bin/python -m sources.exist.meme_reliability

Autor: Enrique
"""
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sources.common.common import logger, writeLog
from sources.psri.calculator import (
    compute_z_score, compute_coherence,
)
from sources.exist.loader import load_dataframes
from sources.exist.aggregator import add_psri_subject

DATA_ROOT = Path("results/input/EXIST")
LABELS_PATH = Path("results/input/EXIST/EXIST2026_training.json")

# Métricas a diagnosticar: componentes PSRI (fiabilidad) y métricas crudas.
# `S_estab_trial` y `PSRI_trial` se construyen a nivel de trial.
PSRI_METRICS = ["PSRI_trial", "S_estab_trial", "PSRI_hr_subj", "PSRI_et_subj",
                "S_coher_trial", "S_cond_subj"]
RAW_METRICS = ["garmin_hr_mean", "garmin_hr_std",
               "3d_eye_states_pupil diameter left [mm]_mean",
               "3d_eye_states_pupil diameter left [mm]_std",
               "reaction_time", "blinks_count"]
ALL_METRICS = PSRI_METRICS + RAW_METRICS

DISAGREEMENT_TARGETS = ["entropy_21", "entropy_22", "entropy_23", "soft_21_yes"]

MIN_COMMON_MEMES_PER_PAIR = 20


# ---------------------------------------------------------------------------
# Carga y construcción de la tabla a nivel de trial (meme, username)
# ---------------------------------------------------------------------------

def _load_trial_table() -> pd.DataFrame:
    """Reconstruye la tabla a nivel de trial con todos los componentes.

    Réplica exacta de la población de `build_psri_dataframe`
    (filter_common=True + merge inner por (meme_id, username)) y añade los
    componentes PSRI a nivel de trial: PSRI_hr_subj/PSRI_et_subj/S_cond_subj
    (via `add_psri_subject`) y S_coher_trial (z-scores con baseline_prev,
    mismas estadísticas de sujeto que en build_features/final_check).

    Returns:
        pandas.DataFrame: (meme_id, username, componentes PSRI, métricas
            crudas) — una fila por trial.
    """
    df_hr, df_eeg, df_et = load_dataframes(DATA_ROOT)
    common_memes = set(df_hr["meme_id"]) & set(df_et["meme_id"])
    df_hr = df_hr[df_hr["meme_id"].isin(common_memes)].copy()
    df_et = df_et[df_et["meme_id"].isin(common_memes)].copy()

    df_hr, df_et = add_psri_subject(df_hr, df_et)
    pair = df_hr.merge(df_et, on=["meme_id", "username"], how="inner")
    pair["meme_id"] = pair["meme_id"].astype(str)

    pupil_mean_col = "3d_eye_states_pupil diameter left [mm]_mean"
    hr_std_subj = pair.groupby("username")["garmin_hr_mean"].transform("std").replace(0, np.nan)
    pupil_std_subj = pair.groupby("username")[pupil_mean_col].transform("std").replace(0, np.nan)
    z_hr = pair.apply(
        lambda r: compute_z_score(
            r["garmin_hr_mean"], r["garmin_hr_mean_baseline_prev"],
            hr_std_subj.loc[r.name]), axis=1)
    z_pupil = pair.apply(
        lambda r: compute_z_score(
            r[pupil_mean_col],
            r["3d_eye_states_pupil diameter left [mm]_mean_baseline_prev"],
            pupil_std_subj.loc[r.name]), axis=1)
    pair["S_coher_trial"] = [
        compute_coherence(a, b) if not (np.isnan(a) or np.isnan(b)) else np.nan
        for a, b in zip(z_hr, z_pupil)
    ]

    pair["S_estab_trial"] = (pair["PSRI_hr_subj"] + pair["PSRI_et_subj"]) / 2.0
    pair["PSRI_trial"] = (
        pair["S_estab_trial"] / 3.0 + pair["S_coher_trial"] / 3.0
        + pair["S_cond_subj"] / 3.0
    )
    return pair


def _load_labels() -> pd.DataFrame:
    """Carga las métricas de desacuerdo por meme (sin tqdm).

    Returns:
        pandas.DataFrame: (meme_id, entropy_21, entropy_22, entropy_23,
            soft_21_yes).
    """
    import json
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels_data = json.load(f)

    rows = []
    for meme_id, info in labels_data.items():
        labels_21 = info.get("labels_task2_1", [])
        p_yes = np.nan
        entropy_21 = np.nan
        if labels_21:
            n = len(labels_21)
            p_yes = labels_21.count("YES") / n
            p_no = labels_21.count("NO") / n
            probs = [p for p in (p_yes, p_no) if p > 0]
            entropy_21 = -float(np.sum(np.array(probs) * np.log(np.array(probs))))

        entropy_22 = np.nan
        labels_22 = info.get("labels_task2_2", [])
        if labels_22:
            unique, counts = np.unique(labels_22, return_counts=True)
            probs = counts / len(labels_22)
            entropy_22 = -float(np.sum(probs * np.log(probs)))

        entropy_23 = np.nan
        labels_23 = info.get("labels_task2_3", [])
        if labels_23:
            flat = [item for sublist in labels_23 for item in sublist]
            if flat:
                unique, counts = np.unique(flat, return_counts=True)
                probs = counts / len(flat)
                entropy_23 = -float(np.sum(probs * np.log(probs)))

        rows.append({"meme_id": str(meme_id), "entropy_21": entropy_21,
                     "entropy_22": entropy_22, "entropy_23": entropy_23,
                     "soft_21_yes": p_yes})
    out = pd.DataFrame(rows)
    out["meme_id"] = out["meme_id"].astype(str)
    return out


# ---------------------------------------------------------------------------
# Cheque 1 + 2: estructura de cobertura y descomposición de varianza (ICC)
# ---------------------------------------------------------------------------

def _one_way_icc(y: np.ndarray, group: np.ndarray) -> tuple[float, float, float]:
    """ICC one-way (método de momentos / ANOVA) para grupos desbalanceados.

    Args:
        y (ndarray): Valores.
        group (ndarray): Identificadores de grupo (misma longitud que y).

    Returns:
        tuple: (var_between, var_within, icc). NaN si no computable.
    """
    df = pd.DataFrame({"y": np.asarray(y, dtype=float),
                       "g": np.asarray(group)}).dropna()
    k = df["g"].nunique()
    n = len(df)
    if k < 2 or n <= k:
        return np.nan, np.nan, np.nan
    counts = df.groupby("g")["y"].size()
    means = df.groupby("g")["y"].mean()
    grand = df["y"].mean()
    msb = (counts * (means - grand) ** 2).sum() / (k - 1)
    ssw = df.groupby("g")["y"].apply(lambda s: float(((s - s.mean()) ** 2).sum())).sum()
    msw = ssw / (n - k)
    n0 = (n - (counts ** 2).sum() / n) / (k - 1)
    var_between = (msb - msw) / n0 if n0 > 0 else 0.0
    total = var_between + msw
    icc = var_between / total if total > 0 else np.nan
    return var_between, msw, icc


def _center_by_subject(pair: pd.DataFrame, metric: str) -> pd.Series:
    """Resta la media por sujeto (rasgo) para aislar la señal de meme."""
    return pair[metric] - pair.groupby("username")[metric].transform("mean")


def run_variance_decomposition(pair: pd.DataFrame) -> pd.DataFrame:
    """ICC por métrica: sujeto, meme crudo y meme neta de sujeto.

    Args:
        pair (pandas.DataFrame): Tabla de trials.

    Returns:
        pandas.DataFrame: Filas por métrica con ICC_subject, ICC_meme,
            ICC_meme_adj y componentes de varianza.
    """
    rows = []
    for metric in ALL_METRICS:
        if metric not in pair.columns:
            continue
        sub = pair[["username", "meme_id", metric]].dropna()
        if len(sub) < 50:
            continue
        n_subj = sub["username"].nunique()
        n_memes = sub["meme_id"].nunique()
        vb_subj, vw_subj, icc_subj = _one_way_icc(sub[metric], sub["username"])
        vb_meme, vw_meme, icc_meme = _one_way_icc(sub[metric], sub["meme_id"])
        centered = _center_by_subject(sub, metric)
        vb_adj, vw_adj, icc_adj = _one_way_icc(centered, sub["meme_id"])
        # fiabilidad de la media de k viewers (Spearman-Brown) con ICC neto
        k_mean = len(sub) / n_memes
        r_mean = k_mean * icc_adj / (1 + (k_mean - 1) * icc_adj) if not np.isnan(icc_adj) else np.nan
        # ICC negativos = sin varianza de grupo (estimador de momentos los puede
        # dar ligeramente < 0); se aplanan a 0 para la lectura.
        def _clip(v):
            return 0.0 if (v is None or (isinstance(v, float) and np.isnan(v)) or v < 0) else float(v)
        rows.append({
            "metric": metric, "kind": "PSRI" if metric in PSRI_METRICS else "raw",
            "n_trials": len(sub), "n_subjects": n_subj, "n_memes": n_memes,
            "k_mean": k_mean,
            "ICC_subject": _clip(icc_subj), "ICC_meme": _clip(icc_meme),
            "ICC_meme_adj": _clip(icc_adj),
            "reliability_mean_SB": _clip(r_mean),
            "attenuation": np.sqrt(max(r_mean, 0.0)) if not np.isnan(r_mean) else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cheque 3: fiabilidad entre viewers del mismo meme (neta de sujeto)
# ---------------------------------------------------------------------------

def run_pairwise_reliability(pair: pd.DataFrame) -> pd.DataFrame:
    """Fiabilidad entre los dos viewers de cada meme, por par de sujetos.

    Con k=2, la "fiabilidad split-half del agregado por meme" coincide con la
    correlación entre los DOS viewers de cada meme, neta de rasgo de sujeto
    (valores centrados por sujeto). Se calcula por par de sujetos sobre los
    memes que ambos vieron y se resume por métrica (mediana, IQR, pool Fisher).

    Args:
        pair (pandas.DataFrame): Tabla de trials.

    Returns:
        pandas.DataFrame: Filas por métrica con resumen de la fiabilidad
            entre viewers y de la fiabilidad de la media (Spearman-Brown).
    """
    meme_users = pair.groupby("meme_id")["username"].apply(lambda s: tuple(sorted(set(s))))
    pairs = {}
    for meme_id, users in meme_users.items():
        if len(users) == 2:
            pairs.setdefault(frozenset(users), []).append(meme_id)

    rows = []
    for metric in ALL_METRICS:
        if metric not in pair.columns:
            continue
        centered = _center_by_subject(pair, metric)
        df = pair.assign(_y=centered)
        pair_rs = []
        pair_ns = []
        for key, memes in pairs.items():
            if len(memes) < MIN_COMMON_MEMES_PER_PAIR:
                continue
            a, b = sorted(key)
            wide = (df[df["meme_id"].isin(memes)]
                    .pivot_table(index="meme_id", columns="username", values="_y"))
            if not {a, b} <= set(wide.columns):
                continue
            ab = wide[[a, b]].dropna()
            if len(ab) < MIN_COMMON_MEMES_PER_PAIR:
                continue
            r, _ = spearmanr(ab[a], ab[b])
            if np.isnan(r):
                continue
            pair_rs.append(r)
            pair_ns.append(len(ab))
        if not pair_rs:
            continue
        pair_rs = np.asarray(pair_rs)
        pair_ns = np.asarray(pair_ns)
        # pool fijo Fisher (pesos = n-3)
        z = np.arctanh(np.clip(pair_rs, -0.999, 0.999))
        z_pool = np.sum((pair_ns - 3) * z) / np.sum(pair_ns - 3)
        r_pool = float(np.tanh(z_pool))
        r_med = float(np.median(pair_rs))
        # fiabilidad de la media de 2 viewers (Spearman-Brown)
        r_mean_med = 2 * r_med / (1 + r_med)
        r_mean_pool = 2 * r_pool / (1 + r_pool)
        rows.append({
            "metric": metric, "kind": "PSRI" if metric in PSRI_METRICS else "raw",
            "n_pairs": len(pair_rs), "memes_per_pair_median": float(np.median(pair_ns)),
            "r_viewers_median": r_med,
            "r_viewers_iqr": float(np.nanpercentile(pair_rs, 75) - np.nanpercentile(pair_rs, 25)),
            "r_viewers_pooled_fisher": r_pool,
            "r_mean_SB_median": r_mean_med, "r_mean_SB_pooled": r_mean_pool,
            "attenuation_median": np.sqrt(max(r_mean_med, 0.0)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cheque 4: cota de potencia efectiva del diseño
# ---------------------------------------------------------------------------

def run_power_bound(pair: pd.DataFrame) -> pd.DataFrame:
    """Tamaño de efecto mínimo detectable a nivel de meme.

    N_eff = N_memes / deff con deff = 1 + (k-1)*ICC_meme_adj; mínimo |r|
    detectable (Spearman) al 80% de potencia y alpha=0.05 a dos colas.

    Args:
        pair (pandas.DataFrame): Tabla de trials.

    Returns:
        pandas.DataFrame: Filas por métrica con N_eff y r_min.
    """
    n_memes = pair["meme_id"].nunique()
    z_alpha = 1.959964
    z_beta = 0.8416212
    rows = []
    for metric in ALL_METRICS:
        if metric not in pair.columns:
            continue
        sub = pair[["username", "meme_id", metric]].dropna()
        if len(sub) < 50:
            continue
        centered = _center_by_subject(sub, metric)
        _, _, icc_adj = _one_way_icc(centered, sub["meme_id"])
        k_mean = len(sub) / sub["meme_id"].nunique()
        icc_adj = 0.0 if np.isnan(icc_adj) or icc_adj < 0 else float(icc_adj)
        deff = 1 + (k_mean - 1) * icc_adj
        n_eff = n_memes / deff
        r_min = (z_alpha + z_beta) / np.sqrt(n_eff) if n_eff > 0 else np.nan
        # El efecto OBSERVABLE queda atenuado por la fiabilidad del agregado
        # (sqrt(reliability_mean_SB)): el efecto TRUE mínimo detectable es mayor.
        r_mean_sb = (k_mean * icc_adj / (1 + (k_mean - 1) * icc_adj)) if icc_adj > 0 else 0.0
        att = np.sqrt(r_mean_sb)
        r_true_min = (r_min / att) if (att > 0 and not np.isnan(r_min)) else np.nan
        rows.append({
            "metric": metric, "kind": "PSRI" if metric in PSRI_METRICS else "raw",
            "n_memes": n_memes, "k_mean": k_mean, "ICC_meme_adj": icc_adj,
            "deff": deff, "N_eff": n_eff, "r_min_80pct": r_min,
            "reliability_mean_SB": r_mean_sb,
            "r_true_min_80pct": r_true_min,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cheque 5: sonda del eslabón (dispersión entre viewers vs desacuerdo)
# ---------------------------------------------------------------------------

def run_dispersion_probe(pair: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Correlación de la dispersión entre viewers por meme vs desacuerdo.

    El eslabón de la cadena que toca la fisiología es "meme ambiguo ->
    reacción MÁS VARIABLE entre viewers"; se opera como dispersión por meme
    (desviación absoluta media de los viewers) de cada métrica, correlacionada
    (Spearman) con las métricas de desacuerdo. Solo memes con >= 2 viewers.

    Args:
        pair (pandas.DataFrame): Tabla de trials.
        labels (pandas.DataFrame): (meme_id, entropy_*, soft_21_yes).

    Returns:
        pandas.DataFrame: Filas (metric, target, n, rho, p_value).
    """
    disp_rows = []
    n_viewers = pair.groupby("meme_id")["username"].transform("nunique")
    pair2 = pair[n_viewers >= 2]
    if pair2.empty:
        return pd.DataFrame(columns=["metric", "target", "n", "rho", "p_value"])
    for metric in ALL_METRICS:
        if metric not in pair2.columns:
            continue
        disp = (pair2.groupby("meme_id")[metric]
                .apply(lambda s: float(np.nanmean(np.abs(s - s.mean()))) if s.notna().sum() >= 2 else np.nan)
                .rename("dispersion"))
        m = disp.reset_index().merge(labels, on="meme_id", how="inner").dropna(subset=["dispersion"])
        for target in DISAGREEMENT_TARGETS:
            if target not in m.columns:
                continue
            sub = m[["dispersion", target]].dropna()
            if len(sub) < 30:
                continue
            r, p = spearmanr(sub["dispersion"], sub[target])
            disp_rows.append({"metric": metric, "target": target, "n": len(sub),
                              "rho": r, "p_value": p})
    return pd.DataFrame(disp_rows)


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

def run_meme_reliability(out_dir: Path | None = None) -> dict:
    """Ejecuta el diagnóstico de fiabilidad del agregado por meme.

    Args:
        out_dir (Path or None, optional): Directorio de salida. Si es None se
            usa `results/output/EXIST/meme_reliability`.

    Returns:
        dict: Con las tablas de estructura, varianza, fiabilidad, potencia y
            sonda de dispersión.
    """
    if out_dir is None:
        out_dir = Path("results/output/EXIST/meme_reliability")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pair = _load_trial_table()
    writeLog("info", logger,
             f"[exist_meme_reliability] {len(pair)} trials; "
             f"{pair['meme_id'].nunique()} memes; {pair['username'].nunique()} sujetos")

    # Cheque 1: estructura de cobertura
    vc = pair.groupby("meme_id")["username"].agg(["nunique", lambda s: tuple(sorted(set(s)))])
    vc.columns = ["n_viewers", "viewers"]
    vc = vc.reset_index()
    n_pairs_struct = vc[vc["n_viewers"] >= 2]["viewers"].nunique()
    coverage = pd.DataFrame([{
        "memes": len(vc), "memes_1_viewer": int((vc["n_viewers"] == 1).sum()),
        "memes_2_viewers": int((vc["n_viewers"] == 2).sum()),
        "k_mean": vc["n_viewers"].mean(), "k_median": vc["n_viewers"].median(),
        "k_max": vc["n_viewers"].max(), "n_subject_pairs": int(n_pairs_struct),
    }])

    variance = run_variance_decomposition(pair)
    reliability = run_pairwise_reliability(pair)
    power = run_power_bound(pair)
    labels = _load_labels()
    dispersion = run_dispersion_probe(pair, labels)

    coverage.to_csv(out_dir / "coverage_structure.csv", index=False)
    variance.to_csv(out_dir / "variance_decomposition.csv", index=False)
    reliability.to_csv(out_dir / "pairwise_reliability.csv", index=False)
    power.to_csv(out_dir / "power_bound.csv", index=False)
    dispersion.to_csv(out_dir / "dispersion_probe.csv", index=False)

    # ---- Resumen de consola ----
    c = coverage.iloc[0]
    print("\n" + "=" * 72)
    print("== [EXIST meme_reliability] ¿el agregado por meme lleva señal? ==")
    print("=" * 72)
    print(f"\nESTRUCTURA: {int(c.memes)} memes, {int(c.memes_1_viewer)} con 1 viewer, "
          f"{int(c.memes_2_viewers)} con 2. k media={c.k_mean:.2f}, max={c.k_max}. "
          f"{int(c.n_subject_pairs)} pares de sujetos distintos.")

    print("\nVARIANZA (ICC): fracción de varianza atribuible a cada nivel.")
    print(variance[["metric", "ICC_subject", "ICC_meme", "ICC_meme_adj",
                    "reliability_mean_SB", "attenuation"]].round(4).to_string(index=False))

    print("\nFIABILIDAD ENTRE VIEWERS (neta de rasgo de sujeto; mediana de pares):")
    print(reliability[["metric", "n_pairs", "r_viewers_median", "r_viewers_pooled_fisher",
                       "r_mean_SB_median", "attenuation_median"]].round(4).to_string(index=False))

    print("\nCOTA DE POTENCIA (mínimo |r| detectable a nivel de meme, 80% pot.):")
    print(power[["metric", "N_eff", "r_min_80pct",
                 "reliability_mean_SB", "r_true_min_80pct"]].round(4).to_string(index=False))

    if not dispersion.empty:
        print("\nSONDA DEL ESLABÓN (dispersión entre viewers vs desacuerdo):")
        print(dispersion[["metric", "target", "n", "rho", "p_value"]].round(4).to_string(index=False))
        n_raw = int((dispersion["p_value"] < 0.05).sum())
        print(f"  p<0.05 sin corregir: {n_raw}/{len(dispersion)} (todo el bloque es diagnóstico)")
    else:
        print("\nSONDA DEL ESLABÓN: sin datos suficientes.")

    print("\n" + "=" * 72)
    print("LECTURA: si ICC_meme_adj ~ 0 y la fiabilidad entre viewers ~ 0, el")
    print("agregado por meme es ruido promediado sobre 1-2 sujetos: la correlación")
    print("con el desacuerdo es imposible POR DISEÑO, exista o no la cadena causal.")
    print("La desconexión queda medida con datos del propio PSRI, sin etiquetas.")
    print("=" * 72 + "\n")
    return {"coverage": coverage, "variance": variance,
            "reliability": reliability, "power": power, "dispersion": dispersion}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Diagnóstico de fiabilidad del agregado por meme en EXIST "
                    "(desalineación poblacional medida con datos del PSRI)")
    parser.add_argument("--out", type=Path, default=None,
                        help="directorio de salida (default: "
                             "results/output/EXIST/meme_reliability)")
    args = parser.parse_args()
    run_meme_reliability(args.out)
    print(f"\n[exist_meme_reliability] salidas en "
          f"{args.out or 'results/output/EXIST/meme_reliability'}/")
