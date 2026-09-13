"""
kemocon/task_validity.py

Módulo de diagnóstico de VALIDEZ DE TAREA (task-validity) para K-EmoCon.

Basado en las recomendaciones de `docs/dialogo.txt`: el tramo PRE-DEBATE
(inicio de la grabación E4 -> inicio del debate, sin interacción) actúa como
condición control negativo frente al DEBATE (conversación). Un componente
conductual (S_cond) es "válido como módulo contextual" solo si:

  1. distingue pre-debate de debate DENTRO de sujeto (modelo mixto beta1 != 0
     con efecto aleatorio por sujeto, y Wilcoxon pareado sobre medias por
     sujeto);
  2. es robusto a la duración de las ventanas y a excluir las primeras
     ventanas de cada periodo (adaptación al dispositivo);
  3. no depende de uno o pocos sujetos (leave-one-subject-out);
  4. tiene cobertura suficiente;
  5. se compara contra controles fisiológicos simples (movimiento = acc_std,
     amplitud = bvp_mean/eda_mean/hr_mean) para ver si S_cond tiene validez
     ESPECÍFICA o es solo una transformación de la señal genérica.

Este módulo es DIAGNÓSTICO y EXPLORATORIO (no es el resultado confirmatorio
del paper). No modifica ni la feature table ni el compuesto; solo la lee y
exporta CSVs de diagnóstico.

Ejecución: dentro de run_pipeline.process_kemocon().
Autor: Enrique
"""
from sources.common.common import logger, writeLog

import numpy as np
import pandas as pd

from statsmodels.formula.api import mixedlm
from scipy.stats import wilcoxon

WINDOW_S = 5.0


# ---------------------------------------------------------------------------
# Asignación de periodo (pre-debate vs. debate)
# ---------------------------------------------------------------------------
def assign_period(feature_table: pd.DataFrame, subj_times: pd.DataFrame,
                  window_s: float = WINDOW_S) -> pd.DataFrame:
    """Añade la columna `period` ('pre'|'debate') y `win_in_period` a la tabla.

    La feature table ancla la ventana 0 al inicio de grabación E4
    (`initTime`). El debate empieza en la ventana floor((startTime -
    initTime)/window_s) y termina en floor((endTime - initTime)/window_s).
    Las ventanas anteriores al inicio del debate se etiquetan 'pre'; el resto
    'debate' (incluyendo, si existieran, ventanas posteriores a endTime, que
    son marginales en este dataset).

    Args:
        feature_table (pandas.DataFrame): Tabla con `subject_id` y `win`.
        subj_times (pandas.DataFrame): Metadatos con índices `initTime`,
            `startTime`, `endTime` (en ms) indexados por `pid`.
        window_s (float, optional): Tamaño de ventana. Por defecto es 5.0.

    Returns:
        pandas.DataFrame: Copia con las columnas `period` y `win_in_period`
            añadidas.
    """
    df = feature_table.copy()
    periods = {}
    win_in_period = {}
    for sid, g in df.groupby("subject_id"):
        if sid not in subj_times.index:
            periods[sid] = "debate"  # sin metadatos, asumir debate
            win_in_period[sid] = np.arange(len(g))
            continue
        init = subj_times.loc[sid, "initTime"] / 1000.0
        start = subj_times.loc[sid, "startTime"] / 1000.0
        end = subj_times.loc[sid, "endTime"] / 1000.0
        w0 = int((start - init) // window_s)
        w1 = int((end - init) // window_s)
        wins = g["win"].to_numpy()
        is_debate = (wins >= w0) & (wins <= w1)
        periods[sid] = np.where(is_debate, "debate", "pre")
        # índice de ventana DENTRO de su periodo (para la robustez de
        # excluir las primeras ventanas)
        win_ip = np.zeros_like(wins)
        for pname in ("pre", "debate"):
            mask = np.array(periods[sid]) == pname
            win_ip[mask] = np.arange(mask.sum())
        win_in_period[sid] = win_ip
    df["period"] = np.concatenate([periods[s] for s in df["subject_id"].unique()])
    df["win_in_period"] = np.concatenate(
        [win_in_period[s] for s in df["subject_id"].unique()])
    return df


# ---------------------------------------------------------------------------
# Estadísticos por periodo (medias por sujeto)
# ---------------------------------------------------------------------------
def subject_period_means(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Media de `col` por (sujeto, periodo), pivote a columnas 'pre'/'debate'.

    Args:
        df (pandas.DataFrame): Tabla con `subject_id`, `period` y `col`.
        col (str): Nombre de la columna a agregar.

    Returns:
        pandas.DataFrame: Índice `subject_id`, columnas 'pre' y 'debate'
            (NaN si el sujeto no tiene datos válidos en ese periodo).
    """
    g = df.dropna(subset=[col]).groupby(["subject_id", "period"])[col].mean()
    return g.unstack("period").reindex(columns=["pre", "debate"])


def paired_wilcoxon(df: pd.DataFrame, col: str) -> dict:
    """Test de Wilcoxon pareado (por sujeto) entre pre-debate y debate.

    Para cada sujeto se toma la media de `col` en cada periodo; se hace el
    test de rangos con signo de Wilcoxon sobre esos pares.

    Args:
        df (pandas.DataFrame): Tabla con `subject_id`, `period` y `col`.
        col (str): Nombre de la columna.

    Returns:
        dict: Estadísticos (estat, p, n_pares, n_a_favor_debate,
            n_en_contra_debate, mediana_diff).
    """
    piv = subject_period_means(df, col).dropna()
    if len(piv) < 3:
        return {"estat": np.nan, "p": np.nan, "n_pares": 0,
                "n_a_favor_debate": 0, "n_en_contra_debate": 0,
                "mediana_diff": np.nan, "media_diff": np.nan}
    diff = piv["debate"] - piv["pre"]
    stat, p = wilcoxon(diff, zero_method="wilcox")
    return {
        "estat": float(stat), "p": float(p), "n_pares": len(piv),
        "n_a_favor_debate": int((diff > 0).sum()),
        "n_en_contra_debate": int((diff < 0).sum()),
        "mediana_diff": float(np.median(diff)),
        "media_diff": float(np.mean(diff)),
    }


# ---------------------------------------------------------------------------
# Modelo mixto
# ---------------------------------------------------------------------------
def fit_mixed_model(df: pd.DataFrame, col: str) -> dict:
    """Ajusta `col ~ C(period) + (1|subject_id)`.

    `C(period)` usa 'pre' como categoría de referencia, así que `beta1` es la
    diferencia media (debate - pre) DENTRO de sujeto.

    Args:
        df (pandas.DataFrame): Tabla con `subject_id`, `period` y `col`.
        col (str): Nombre de la columna.

    Returns:
        dict: Estadísticos del modelo mixto (beta1, se, t, p, n, n_subjects,
            convergencia).
    """
    sub = df.dropna(subset=[col]).copy()
    # Orden de referencia explícito: 'pre' es la categoría de referencia y
    # 'debate' el término de tratamiento (independiente del dtype de pandas).
    sub["period"] = pd.Categorical(sub["period"], categories=["pre", "debate"])
    n_subjects = sub["subject_id"].nunique()
    if len(sub) < 30 or n_subjects < 3:
        return {"beta1": np.nan, "se": np.nan, "t": np.nan, "p": np.nan,
                "n": len(sub), "n_subjects": n_subjects, "converged": False}
    try:
        md = mixedlm(f"{col} ~ C(period)", sub, groups=sub["subject_id"])
        mf = md.fit(reml=True, maxiter=2000)
        b = mf.params
        se = mf.bse
        p = mf.pvalues
        coef_name = "C(period)[T.debate]"
        beta1 = float(b[coef_name])
        se1 = float(se[coef_name])
        p1 = float(p[coef_name])
        return {"beta1": beta1, "se": se1, "t": beta1 / se1, "p": p1,
                "n": len(sub), "n_subjects": n_subjects,
                "converged": bool(mf.converged)}
    except Exception as exc:  # noqa: BLE001 -- captura fallos del optimizador
        writeLog("warning", logger,
                 f"[task_validity] modelo mixto falló para {col}: {exc}")
        return {"beta1": np.nan, "se": np.nan, "t": np.nan, "p": np.nan,
                "n": len(sub), "n_subjects": n_subjects, "converged": False}


# ---------------------------------------------------------------------------
# Cobertura por periodo
# ---------------------------------------------------------------------------
def coverage_by_period(df: pd.DataFrame, col: str) -> dict:
    """Fracción de ventanas con valor real (no NaN) en `col` por periodo.

    Args:
        df (pandas.DataFrame): Tabla con `period` y `col`.
        col (str): Nombre de la columna.

    Returns:
        dict: `pre` y `debate` con la fracción de cobertura y el conteo.
    """
    out = {}
    for pname in ("pre", "debate"):
        g = df[df["period"] == pname]
        n = len(g)
        n_valid = int(g[col].notna().sum()) if col in g.columns else 0
        out[pname] = {"frac": n_valid / n if n else np.nan, "n": n}
    return out


# ---------------------------------------------------------------------------
# Leave-one-subject-out sobre el modelo mixto
# ---------------------------------------------------------------------------
def loso_mixed(df: pd.DataFrame, col: str) -> dict:
    """Deja un sujeto fuera cada vez y reajusta el modelo mixto.

    Devuelve el rango de beta1 y si el signo se invierte en algún sujeto.

    Args:
        df (pandas.DataFrame): Tabla con `subject_id`, `period` y `col`.
        col (str): Nombre de la columna.

    Returns:
        dict: `beta_min`, `beta_max`, `delta_max`, `signo_invierte`,
            `n_subjects`.
    """
    betas = []
    subj_ids = sorted(df["subject_id"].unique())
    for sid in subj_ids:
        rest = df[df["subject_id"] != sid]
        r = fit_mixed_model(rest, col)
        if r["converged"]:
            betas.append(r["beta1"])
    if not betas:
        return {"beta_min": np.nan, "beta_max": np.nan, "delta_max": np.nan,
                "signo_invierte": None, "n_subjects": len(subj_ids)}
    full = fit_mixed_model(df, col)
    delta_max = max(abs(np.array(betas) - full["beta1"])) if full["converged"] else np.nan
    if full["converged"] and full["beta1"] != 0 and betas:
        # El signo se "invierte" en LOSO solo si algún reajuste cruza a cero
        # respecto del beta del modelo completo (p. ej. beta_full > 0 y algún
        # beta_loso < 0, o viceversa).
        b_full = full["beta1"]
        signo_invierte = bool((b_full > 0 and any(np.array(betas) < 0)) or
                              (b_full < 0 and any(np.array(betas) > 0)))
    else:
        signo_invierte = None
    return {
        "beta_min": float(np.min(betas)),
        "beta_max": float(np.max(betas)),
        "delta_max": float(delta_max),
        "signo_invierte": signo_invierte,
        "n_subjects": len(subj_ids),
    }


# ---------------------------------------------------------------------------
# Robustez: excluir las primeras k ventanas de cada periodo
# ---------------------------------------------------------------------------
def drop_first_windows(df: pd.DataFrame, col: str, k: int) -> dict:
    """Rehace el modelo mixto excluyendo las `k` primeras ventanas de cada periodo.

    Args:
        df (pandas.DataFrame): Tabla con `win_in_period`, `period` y `col`.
        col (str): Nombre de la columna.
        k (int): Número de ventanas iniciales a descartar por periodo.

    Returns:
        dict: Estadísticos del modelo mixto sobre el subconjunto.
    """
    sub = df[df["win_in_period"] >= k]
    return fit_mixed_model(sub, col)


# ---------------------------------------------------------------------------
# Análisis principal
# ---------------------------------------------------------------------------
def run_task_validity(feature_table: pd.DataFrame, subj_times: pd.DataFrame,
                      drop_k_values: tuple[int, ...] = (1, 2)) -> pd.DataFrame:
    """Ejecuta el análisis de task-validity completo.

    Compara cada componente PSRI (S_cond 7 variantes, S_estab, S_coher) y
    controles fisiológicos simples (acc_std, amplitud bvp/eda/hr) entre
    pre-debate y debate: modelo mixto, Wilcoxon pareado, cobertura, LOSO y
    robustez a las primeras ventanas.

    Args:
        feature_table (pandas.DataFrame): Feature table final.
        subj_times (pandas.DataFrame): Metadatos con initTime/startTime/endTime
            indexados por pid.
        drop_k_values (tuple, optional): Ventanas iniciales a descartar en la
            robustez. Por defecto es (1, 2).

    Returns:
        pandas.DataFrame: Una fila por (categoría, columna) con todos los
            estadísticos.
    """
    df = assign_period(feature_table, subj_times)

    componentes = {
        "S_cond": ["S_cond_att", "S_cond_att_acc", "S_cond_att_med",
                   "S_cond_att_std", "S_cond_att_med_std",
                   "S_cond_self", "S_cond_combined",
                   # Variantes MONOTÓNICAS (no-U, docs/dialogo.txt §2)
                   "S_cond_att_mono", "S_cond_att_med_mono",
                   "S_cond_att_mono_acc_hi", "S_cond_att_mono_acc_lo"],
        "S_estab": ["S_estab", "S_estab_imputed"],
        "S_coher": ["S_coher_prev1", "S_coher_prev5"],
        "control_movimiento": ["acc_std"],
        "control_amplitud": ["bvp_mean", "eda_mean", "hr_mean"],
    }

    rows = []
    for categoria, cols in componentes.items():
        for col in cols:
            if col not in df.columns:
                continue
            mixed = fit_mixed_model(df, col)
            wilc = paired_wilcoxon(df, col)
            cov = coverage_by_period(df, col)
            loso = loso_mixed(df, col)
            row = {
                "categoria": categoria,
                "columna": col,
                "n": mixed["n"],
                "n_subjects": mixed["n_subjects"],
                "beta1": mixed["beta1"],
                "se": mixed["se"],
                "t": mixed["t"],
                "p_mixto": mixed["p"],
                "converged": mixed["converged"],
                "wilcox_p": wilc["p"],
                "wilcox_n_pares": wilc["n_pares"],
                "wilcox_a_favor_debate": wilc["n_a_favor_debate"],
                "wilcox_en_contra_debate": wilc["n_en_contra_debate"],
                "wilcox_mediana_diff": wilc["mediana_diff"],
                "cobertura_pre": cov["pre"]["frac"],
                "cobertura_debate": cov["debate"]["frac"],
                "loso_beta_min": loso["beta_min"],
                "loso_beta_max": loso["beta_max"],
                "loso_delta_max": loso["delta_max"],
                "loso_signo_invierte": loso["signo_invierte"],
            }
            for k in drop_k_values:
                rk = drop_first_windows(df, col, k)
                row[f"beta1_drop{k}"] = rk["beta1"]
                row[f"p_mixto_drop{k}"] = rk["p"]
            rows.append(row)

    return pd.DataFrame(rows)


def export_period_assignments(feature_table: pd.DataFrame,
                              subj_times: pd.DataFrame) -> pd.DataFrame:
    """Exporta la asignación de periodo por ventana (window_task_validity).

    Materializa el nivel Q_window del marco de tres niveles de validez
    (metodologia_esqueleto.md §6.1): una fila por (sujeto, ventana) con el
    periodo pre-debate/debate. Permite auditar qué ventanas sirven de control
    negativo (reposo) y cuáles pertenecen a la tarea.

    Args:
        feature_table (pandas.DataFrame): Feature table final.
        subj_times (pandas.DataFrame): Metadatos con initTime/startTime/endTime
            indexados por pid.

    Returns:
        pandas.DataFrame: Columnas `subject_id`, `win`, `period`,
            `win_in_period`.
    """
    df = assign_period(feature_table, subj_times)
    return df[["subject_id", "win", "period", "win_in_period"]].reset_index(drop=True)
