"""
kemocon/build_features.py

PSRI real por ventana, replicando EXACTAMENTE la formulación de la Sección 5
(antigua Sección 3-4) de tu documento -- no una reconstrucción libre:

  S_estab -> compute_psri_gaussian_log aplicado por canal, POBLACIÓN GLOBAL
             (todos los sujetos y ventanas juntos, no por sujeto -- Tabla 10.2.1)
  S_coher -> compute_z_score con baseline_prev (media del/los trial(s)
             INMEDIATAMENTE ANTERIOR(es), no la media de toda la sesión) +
             std histórica del sujeto, luego compute_coherence(z_a, z_b)
  S_cond  -> compute_psri_gaussian_log aplicado dos veces (misma función que
             S_estab, no una función nueva) y promediado -- en EXIST sobre
             reaction_time y blinks_count; en K-EmoCon (sin eye-tracking)
             se propone Attention (NeuroSky) + acc_std como sustituto
             equivalente -- PENDIENTE DE CONFIRMAR por Si.

Imputación: valores faltantes en S_estab/S_coher/S_cond se imputan con la
mediana de esa dimensión antes de combinar (Sección 5.5 del documento).

Autor: Enrique
"""
from sources.psri.calculator import (
    compute_psri_gaussian_log,
    compute_coherence,
    compute_z_score,
    compute_weighted_psri,
)

from sources.common.common import logger, writeLog

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# S_estab -- población global por canal (Tabla 10.2.1: mediana/MAD sobre
# TODOS los pares sujeto-ventana juntos, no por sujeto)
# ---------------------------------------------------------------------------

def add_s_estab(window_table: pd.DataFrame,
                 channels: tuple[str, ...] = ("bvp", "eda", "hr", "temp", "ibi")) -> pd.DataFrame:
    """Calcula S_estab por canal con población global y promedia por ventana.

    Args:
        window_table (pandas.DataFrame): Tabla de ventanas por sujeto.
        channels (tuple, optional): Canales E4 a usar. Por defecto
            ("bvp", "eda", "hr", "temp", "ibi").

    Returns:
        pandas.DataFrame: Copia con las columnas `{ch}_S_estab` y `S_estab`.
    """
    df = window_table.copy()
    estab_cols = []
    for ch in channels:
        std_col = f"{ch}_std"
        if std_col not in df.columns:
            continue
        vals = df[std_col].to_numpy(dtype=float)
        mask = ~np.isnan(vals)
        R = np.full(len(vals), np.nan)
        if mask.sum() > 0:
            R[mask] = compute_psri_gaussian_log(vals[mask])
        out_col = f"{ch}_S_estab"
        df[out_col] = R
        estab_cols.append(out_col)

    df["S_estab"] = df[estab_cols].mean(axis=1) if estab_cols else np.nan
    return df

# ---------------------------------------------------------------------------
# S_coher -- HR (E4) vs EDA, dos sistemas fisiológicos distintos (no la
# misma magnitud medida por dos instrumentos), con baseline_prev
# (rolling, trial(s) inmediatamente anterior(es)) y std histórica del sujeto
# ---------------------------------------------------------------------------

def _rolling_baseline_prev(df: pd.DataFrame, col: str, n_prev: int = 1) -> pd.Series:
    """Media de los n_prev valores inmediatamente anteriores, por sujeto.

    Análogo a `garmin_hr_mean_baseline_prev(N)` de EXIST: para cada ventana
    se usa la media de las n_prev ventanas anteriores del mismo sujeto,
    ordenadas por `win`.

    Args:
        df (pandas.DataFrame): Tabla con `subject_id` y `win`.
        col (str): Columna sobre la que calcular la baseline.
        n_prev (int, optional): Número de trials anteriores. Por defecto es 1.

    Returns:
        pandas.Series: Baseline alineada con el índice original de `df`.
    """
    d = df.sort_values(["subject_id", "win"])
    baseline = d.groupby("subject_id")[col].transform(
        lambda s: s.shift(1).rolling(n_prev, min_periods=1).mean()
    )
    return baseline.reindex(df.index)


def add_s_coher(window_table: pd.DataFrame, n_prev: int = 1,
                 out_col: str = "S_coher") -> pd.DataFrame:
    """Calcula la coherencia cruzada HR (E4) vs EDA con baseline_prev.

    HR (E4) vs EDA son dos sistemas fisiológicos genuinamente distintos
    (cardiovascular vs electrodérmico/simpático puro), análogo conceptual al
    par HR-pupila de EXIST (dos sistemas, no dos instrumentos midiendo lo
    mismo). Sustituye por completo a la versión anterior (HR E4 vs Polar_HR):
    esa pareja medía la MISMA magnitud fisiológica (frecuencia cardíaca) con
    dos instrumentos distintos — una pérdida de acuerdo ahí sería ruido de
    medición o idiosincrasia de sensor/colocación, no una señal de
    descoordinación entre sistemas fisiológicos ante el estímulo, que es lo
    que S_coher pretende capturar (Sección 5.3 / 9.3.4 del documento).

    n_prev=1 -> baseline_prev (solo el trial inmediatamente anterior).
    n_prev=5 -> baseline_prev5 (media de los 5 trials anteriores).
    Ambas variantes usan el MISMO subject_std (variabilidad histórica
    completa del sujeto) — lo que cambia es únicamente la ventana de baseline.

    Args:
        window_table (pandas.DataFrame): Tabla de ventanas por sujeto.
        n_prev (int, optional): Ventana de baseline. Por defecto es 1.
        out_col (str, optional): Columna de salida. Por defecto es "S_coher".

    Returns:
        pandas.DataFrame: Copia con la columna de coherencia añadida.
    """
    df = window_table.copy()
    df[out_col] = np.nan

    col_a, col_b = "hr_mean", "eda_mean"
    if col_a not in df.columns or col_b not in df.columns:
        return df

    df = df.sort_values(["subject_id", "win"]).reset_index(drop=True)

    baseline_a = _rolling_baseline_prev(df, col_a, n_prev)
    baseline_b = _rolling_baseline_prev(df, col_b, n_prev)
    subj_std_a = df.groupby("subject_id")[col_a].transform("std")
    subj_std_b = df.groupby("subject_id")[col_b].transform("std")

    z_a = [compute_z_score(v, b, s) for v, b, s in zip(df[col_a], baseline_a, subj_std_a)]
    z_b = [compute_z_score(v, b, s) for v, b, s in zip(df[col_b], baseline_b, subj_std_b)]

    coher = [
        compute_coherence(a, b) if not (np.isnan(a) or np.isnan(b)) else np.nan
        for a, b in zip(z_a, z_b)
    ]
    df[out_col] = coher
    df.attrs["s_coher_pair"] = (col_a, col_b)
    return df


# ---------------------------------------------------------------------------
# S_cond -- MISMA compute_psri_gaussian_log aplicada dos veces, promediada.
# En EXIST: reaction_time + blinks_count. En K-EmoCon (sin eye-tracking):
# Attention (NeuroSky) + acc_std -- PROPUESTA, pendiente de confirmación.
# ---------------------------------------------------------------------------

def _apply_u_transform(series: pd.Series) -> np.ndarray:
    """Aplica la transformación en U a una columna con población global.

    Args:
        series (pandas.Series): Columna a transformar.

    Returns:
        ndarray: Fiabilidades; NaN donde la entrada era NaN.
    """
    vals = series.to_numpy(dtype=float)
    mask = ~np.isnan(vals)
    R = np.full(len(vals), np.nan)
    if mask.sum() > 0:
        R[mask] = compute_psri_gaussian_log(vals[mask])
    return R


def _apply_monotonic_transform(series: pd.Series, direction: int = 1) -> np.ndarray:
    """Aplica una transformación MONOTÓNICA (no en U) con población global.

    Para variables cuya interpretación es unívoca en una dirección — p. ej.
    más atención/participación = mejor compromiso — la función en U de
    `compute_psri_gaussian_log` no es adecuada: penaliza también el extremo
    "alto", que aquí es deseable. Esta transformación mapea la cola alta
    (si `direction=1`) o la cola baja (si `direction=-1`) a fiabilidades en
    (0,1] de forma estrictamente monotónica, usando estadísticos robustos de
    población global (mediana/MAD), consistente con el estilo de la versión
    en U.

    Args:
        series (pandas.Series): Columna a transformar.
        direction (int, optional): 1 si "más es mejor" (cola alta), -1 si
            "menos es mejor" (cola baja). Por defecto es 1.

    Returns:
        ndarray: Fiabilidades en (0,1]; NaN donde la entrada era NaN.
    """
    vals = series.to_numpy(dtype=float)
    mask = ~np.isnan(vals)
    R = np.full(len(vals), np.nan)
    if mask.sum() == 0:
        return R
    v = vals[mask]
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    mad = mad if mad > 0 else np.std(v)
    mad = mad if mad > 0 else 1.0
    z = direction * (v - med) / (1.4826 * mad)
    # R = 1 - exp(-max(z,0)/k): en z=0 -> 0, crece a 1 asintóticamente en la
    # cola buena; el valor del percentil ~98 de una normal (z≈2) da R≈0.63.
    R[mask] = 1.0 - np.exp(-np.clip(z, 0.0, None) / 2.0)
    return R


def add_self_consistency(window_table: pd.DataFrame, roll_window: int = 5) -> pd.DataFrame:
    """Añade la variabilidad local de la autoanotación por sujeto.

    Variabilidad LOCAL (rolling, centrada, `roll_window` trials) de
    `self_valence`/`self_arousal` — análogo conceptual a `garmin_hr_std` de
    EXIST pero aplicado a la autoanotación: tramos anormalmente PLANOS
    sugieren 'straight-lining' (el sujeto deja el valor fijo), tramos
    anormalmente ERRÁTICOS sugieren respuesta poco reflexiva. Mismo principio
    de "insufficient effort responding" que reaction_time/blinks_count en
    EXIST (Sección 9.3.5), pero aplicado a la tarea de autoanotación, la
    única conducta respecto-a-la-tarea que K-EmoCon SÍ registra para el
    sujeto fisiológico.

    Args:
        window_table (pandas.DataFrame): Tabla de ventanas por sujeto.
        roll_window (int, optional): Tamaño de la ventana rolling. Por
            defecto es 5.

    Returns:
        pandas.DataFrame: Copia con `{col}_local_std` para cada columna de
            autoanotación presente.
    """
    df = window_table.sort_values(["subject_id", "win"]).reset_index(drop=True)
    for col in ["self_valence", "self_arousal"]:
        if col not in df.columns:
            continue
        local_std = df.groupby("subject_id")[col].transform(
            lambda s: s.rolling(roll_window, center=True, min_periods=3).std()
        )
        df[f"{col}_local_std"] = local_std
    return df


def add_s_cond_variants(window_table: pd.DataFrame) -> pd.DataFrame:
    """Calcula las variantes candidatas de S_cond en paralelo.

    Siete variantes candidatas de S_cond, calculadas EN PARALELO (no se
    sustituyen entre sí) para comparar cuál captura señal trial-a-trial real
    antes de decidir cuál usar en el compuesto final:

    - S_cond_att        : solo Attention (NeuroSky) — variante mínima, un solo
                          R_* sin promediar
    - S_cond_att_acc    : Attention (NeuroSky) + acc_std — la implementación
                          original (atención + inestabilidad de movimiento)
    - S_cond_att_med    : Attention + Meditation (NeuroSky, ambas señales del
                          mismo dispositivo — compromiso activo vs. desconexión)
    - S_cond_self       : consistencia de autoanotación (self_valence/arousal
                          local_std) — la más próxima en espíritu al original
                          de EXIST (conducta respecto a la TAREA, no fisiología)
    - S_cond_combined   : media de todos los R_* disponibles (att, acc, med,
                          self_valence, self_arousal) — variante "todo junto"
    - S_cond_att_std    : solo attention_std (variabilidad intra-ventana de la
                          atención) — análogo a S_estab sobre un canal
                          conductual, variante exploratoria
    - S_cond_att_med_std: media de attention_std + meditation_std — como
                          S_cond_att_med pero sobre las stds en vez de las
                          medias, variante exploratoria

    Args:
        window_table (pandas.DataFrame): Tabla de ventanas por sujeto.

    Returns:
        pandas.DataFrame: Copia con las columnas R_* y S_cond_*.
    """
    df = window_table.copy()
    df = add_self_consistency(df)

    r_cols = {}
    if "attention_mean" in df.columns:
        r_cols["R_attention"] = _apply_u_transform(df["attention_mean"])
    if "acc_std" in df.columns:
        r_cols["R_acc_movement"] = _apply_u_transform(df["acc_std"])
    if "meditation_mean" in df.columns:
        r_cols["R_meditation"] = _apply_u_transform(df["meditation_mean"])
    if "attention_std" in df.columns:
        r_cols["R_attention_std"] = _apply_u_transform(df["attention_std"])
    if "meditation_std" in df.columns:
        r_cols["R_meditation_std"] = _apply_u_transform(df["meditation_std"])
    if "self_valence_local_std" in df.columns:
        r_cols["R_self_valence"] = _apply_u_transform(df["self_valence_local_std"])
    if "self_arousal_local_std" in df.columns:
        r_cols["R_self_arousal"] = _apply_u_transform(df["self_arousal_local_std"])

    # Variantes MONOTÓNICAS (no-U) -- docs/dialogo.txt §2: para variables con
    # interpretación unívoca (más atención/participación = mejor compromiso)
    # la función en U penaliza indebidamente el extremo alto. Se calculan en
    # paralelo para comparar si capturan señal que la U no captura.
    mono_cols = {}
    if "attention_mean" in df.columns:
        mono_cols["R_attention_mono"] = _apply_monotonic_transform(df["attention_mean"], direction=1)
    if "meditation_mean" in df.columns:
        mono_cols["R_meditation_mono"] = _apply_monotonic_transform(df["meditation_mean"], direction=1)
    if "acc_std" in df.columns:
        # movimiento: sin interpretación unívoca (documento); se prueban las
        # dos direcciones por separado para decidir empíricamente
        mono_cols["R_acc_mono_hi"] = _apply_monotonic_transform(df["acc_std"], direction=1)
        mono_cols["R_acc_mono_lo"] = _apply_monotonic_transform(df["acc_std"], direction=-1)

    for name, arr in mono_cols.items():
        df[name] = arr
        r_cols[name] = arr

    for name, arr in r_cols.items():
        df[name] = arr

    def _mean_of(*names):
        """Calcula la media de las columnas R_* presentes.

        Args:
            *names: Nombres de columnas R_*.

        Returns:
            pandas.Series: Media de las columnas presentes, o NaN si no hay
                ninguna.
        """
        present = [n for n in names if n in r_cols]
        if not present:
            return np.nan
        return df[present].mean(axis=1)

    df["S_cond_att"] = _mean_of("R_attention")
    df["S_cond_att_acc"] = _mean_of("R_attention", "R_acc_movement")
    df["S_cond_att_med"] = _mean_of("R_attention", "R_meditation")
    df["S_cond_att_std"] = _mean_of("R_attention_std")
    df["S_cond_att_med_std"] = _mean_of("R_attention_std", "R_meditation_std")
    df["S_cond_self"] = _mean_of("R_self_valence", "R_self_arousal")
    df["S_cond_combined"] = _mean_of("R_attention", "R_acc_movement", "R_meditation",
                                       "R_self_valence", "R_self_arousal")

    # Variantes monotónicas (no-U) -- comparables con las de arriba
    df["S_cond_att_mono"] = _mean_of("R_attention_mono")
    df["S_cond_att_med_mono"] = _mean_of("R_attention_mono", "R_meditation_mono")
    df["S_cond_att_mono_acc_hi"] = _mean_of("R_attention_mono", "R_acc_mono_hi")
    df["S_cond_att_mono_acc_lo"] = _mean_of("R_attention_mono", "R_acc_mono_lo")
    return df


# ---------------------------------------------------------------------------
# PSRI compuesto -- imputación por mediana (Sección 5.5) + compute_weighted_psri
# ---------------------------------------------------------------------------

def add_psri_composite(window_table: pd.DataFrame, s_coher_col: str = "S_coher",
                        s_cond_col: str = "S_cond_att_acc",
                        s_estab_col: str = "S_estab",
                        out_col: str = "psri_composite",
                        w1=1 / 3, w2=1 / 3, w3=1 / 3) -> pd.DataFrame:
    """Combina los tres componentes en el PSRI compuesto con imputación por mediana.

    Imputa cada componente con su mediana (Sección 5.5) y aplica
    `compute_weighted_psri` con los pesos dados. Si `s_cond_col` es None, se
    calcula el compuesto de DOS patas (S_estab + S_coher) con pesos
    renormalizados a 1/2–1/2, dejando la tercera dimensión fuera — la
    alternativa `PSRI_validated = w1·S_estab + w2·S_coher` recomendada en
    docs/dialogo.txt para no diluir los dos componentes validados con una
    tercera pata débil.

    Args:
        window_table (pandas.DataFrame): Tabla de ventanas por sujeto.
        s_coher_col (str, optional): Columna de S_coher. Por defecto es
            "S_coher".
        s_cond_col (str or None, optional): Columna de S_cond. Por defecto es
            "S_cond_att_acc". Si es None, se genera el compuesto de 2 patas.
        s_estab_col (str, optional): Columna de S_estab. Por defecto es
            "S_estab".
        out_col (str, optional): Columna compuesta de salida. Por defecto es
            "psri_composite".
        w1 (float, optional): Peso de S_estab. Por defecto es 1/3.
        w2 (float, optional): Peso de S_coher. Por defecto es 1/3.
        w3 (float, optional): Peso de S_cond. Por defecto es 1/3.

    Returns:
        pandas.DataFrame: Copia con las columnas `{col}_imputed` y `out_col`.
    """
    df = window_table.copy()
    for col in [s_estab_col, s_coher_col] + ([s_cond_col] if s_cond_col else []):
        median_val = df[col].median()
        df[f"{col}_imputed"] = df[col].fillna(median_val)

    if s_cond_col is None:
        # Compuesto de dos patas: renormalaiza a 1/2-1/2 los pesos de las
        # dos dimensiones nucleares (independientemente de w1/w2/w3).
        df[out_col] = (
            0.5 * df[f"{s_estab_col}_imputed"].to_numpy(dtype=float)
            + 0.5 * df[f"{s_coher_col}_imputed"].to_numpy(dtype=float)
        )
    else:
        df[out_col] = compute_weighted_psri(
            df[f"{s_estab_col}_imputed"].to_numpy(dtype=float),
            df[f"{s_coher_col}_imputed"].to_numpy(dtype=float),
            df[f"{s_cond_col}_imputed"].to_numpy(dtype=float),
            w1=w1, w2=w2, w3=w3,
        )
    return df


# ---------------------------------------------------------------------------
# Desacuerdo entre anotadores
# ---------------------------------------------------------------------------

def annotator_disagreement_features(window_table: pd.DataFrame) -> pd.DataFrame:
    """Añade métricas de desacuerdo entre anotadores externos.

    IMPORTANTE: patrón EXPLÍCITO R1..R5, no "R*" — las columnas
    R_self_valence/R_self_arousal (autoanotación) también empiezan por "R"
    y terminan en "_valence"/"_arousal"; incluirlas contaminaba la varianza
    "externa" con la autoanotación del propio sujeto (bug pre-existente).
    Las medias y varianzas se calculan sobre LOS MISMOS raters (skipna, NaN
    excluidos), como requiere el Experimento 3 (Control M4).

    Args:
        window_table (pandas.DataFrame): Tabla con anotaciones R1..R5.

    Returns:
        pandas.DataFrame: Copia con `external_*_mean/_var/_range`,
            `n_valid_raters_*`, `self_partner_diff` y
            `self_external_mean_diff`.
    """
    df = window_table.copy()
    ext_val_cols = [f"R{i}_valence" for i in range(1, 6) if f"R{i}_valence" in df.columns]
    if ext_val_cols:
        # media y varianza sobre LOS MISMOS raters (skipna, NaN excluidos) --
        # requerido por el Experimento 3 (Control M4): varianza y media deben
        # calcularse sobre el mismo conjunto de anotadores por ventana.
        df["external_valence_mean"] = df[ext_val_cols].mean(axis=1, skipna=True)
        df["external_valence_var"] = df[ext_val_cols].var(axis=1, skipna=True)
        df["external_valence_range"] = df[ext_val_cols].max(axis=1) - df[ext_val_cols].min(axis=1)
        df["n_valid_raters_valence"] = df[ext_val_cols].notna().sum(axis=1)

    ext_aro_cols = [f"R{i}_arousal" for i in range(1, 6) if f"R{i}_arousal" in df.columns]
    if ext_aro_cols:
        df["external_arousal_mean"] = df[ext_aro_cols].mean(axis=1, skipna=True)
        df["external_arousal_var"] = df[ext_aro_cols].var(axis=1, skipna=True)
        df["n_valid_raters_arousal"] = df[ext_aro_cols].notna().sum(axis=1)

    if "self_valence" in df.columns and "partner_valence" in df.columns:
        df["self_partner_diff"] = (df["self_valence"] - df["partner_valence"]).abs()

    if ext_val_cols and "self_valence" in df.columns:
        df["self_external_mean_diff"] = (df["self_valence"] - df[ext_val_cols].mean(axis=1)).abs()

    return df


# ---------------------------------------------------------------------------
# Orquestación + validación
# ---------------------------------------------------------------------------

def build_full_feature_table(subject_tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Construye la tabla de características completa para K-EmoCon.

    Concatena las tablas por sujeto, aplica S_estab, las variantes de
    S_cond, S_coher en sus dos variantes de baseline (prev1/prev5), el PSRI
    compuesto para ambas y las métricas de desacuerdo entre anotadores.
    Emite avisos si detecta filas duplicadas en (subject_id, win) o ventanas
    con todos los canales E4 en NaN.

    Args:
        subject_tables (list): Tablas por sujeto de `build_subject_table`.

    Returns:
        pandas.DataFrame: Tabla de características final.
    """
    combined = pd.concat(subject_tables, ignore_index=True)

    dup_mask = combined.duplicated(subset=["subject_id", "win"], keep=False)
    if dup_mask.any():
        n_dup = dup_mask.sum()
        dup_subjects = sorted(combined.loc[dup_mask, "subject_id"].unique().tolist())
        print(f"[build_features] AVISO: {n_dup} filas duplicadas en (subject_id, win) "
              f"tras el merge -- sujetos afectados: {dup_subjects}. Esto invalida "
              f"cualquier correlación calculada sobre esta tabla; revisar build_subject_table.")

    e4_channels = [c for c in ["bvp_mean", "eda_mean", "hr_mean", "temp_mean", "ibi_mean", "acc_std"]
                   if c in combined.columns]
    if e4_channels:
        e4_all_nan = combined[e4_channels].isna().all(axis=1)
        n_e4_all_nan = int(e4_all_nan.sum())
        extra_cols = [c for c in ["attention_mean", "meditation_mean", "polar_hr_mean"]
                      if c in combined.columns]
        n_extra_only = {
            c: int((e4_all_nan & combined[c].notna()).sum()) for c in extra_cols
        }
        print(f"[build_features] AVISO: {n_e4_all_nan} ventanas con TODOS los canales E4 en NaN "
              f"(de {len(combined)} totales). De esas, con valor válido en: {n_extra_only} "
              f"-- si estos números son altos, NeuroSky/Polar está expandiendo el rango de "
              f"ventanas más allá de lo que cubre E4, lo que cambia la población global de "
              f"referencia de S_estab aunque sus propias filas queden en NaN correctamente "
              f"filtradas.")

    combined = add_s_estab(combined)
    combined = add_s_cond_variants(combined)

    # Dos variantes de baseline para S_coher, comparables entre sí.
    # s_cond_col fijo a S_cond_att_acc (variante ya validada) mientras se
    # comparan las alternativas por separado -- ver validate_components_vs_disagreement
    combined = add_s_coher(combined, n_prev=1, out_col="S_coher_prev1")
    combined = add_psri_composite(combined, s_coher_col="S_coher_prev1",
                                   s_cond_col="S_cond_att_acc",
                                   out_col="psri_composite_prev1")
    # Alternativa de 2 patas (S_estab+S_coher) -- PSRI_validated de
    # docs/dialogo.txt: la tercera pata se deja fuera del compuesto
    # confirmatorio para no diluir los dos componentes nucleares.
    combined = add_psri_composite(combined, s_coher_col="S_coher_prev1",
                                   s_cond_col=None,
                                   out_col="psri_composite_2leg_prev1")

    combined = add_s_coher(combined, n_prev=5, out_col="S_coher_prev5")
    combined = add_psri_composite(combined, s_coher_col="S_coher_prev5",
                                   s_cond_col="S_cond_att_acc",
                                   out_col="psri_composite_prev5")
    combined = add_psri_composite(combined, s_coher_col="S_coher_prev5",
                                   s_cond_col=None,
                                   out_col="psri_composite_2leg_prev5")

    combined = annotator_disagreement_features(combined)
    return combined


def validate_components_vs_disagreement(feature_table: pd.DataFrame) -> pd.DataFrame:
    """Desglosa el compuesto en sus dimensiones y las compara contra el desacuerdo.

    Análisis DIAGNÓSTICO adicional (no sustituye a
    `validate_psri_vs_disagreement_combined`, que sigue siendo el resultado
    confirmatorio principal a citar en el paper).

    Desmenuza el compuesto en sus tres dimensiones — S_estab, S_cond, y
    S_coher en sus dos variantes de baseline (prev1/prev5) — para ver si la
    asimetría de signos observada en el compuesto (valence: negativo vs.
    arousal/self_partner_diff: positivo) proviene de una dimensión concreta
    o está presente en las tres. Usa las columnas YA IMPUTADAS (mediana)
    para ser comparable con el compuesto, que también imputa antes de
    combinar.

    Bloque de corrección PROPIO (10 componentes x 5 métricas = 50 tests),
    separado del bloque de 10 tests del compuesto — son preguntas distintas
    (¿hay efecto? vs. ¿de dónde viene el efecto / qué variante de S_cond
    usar?), mezclarlas diluiría la potencia de ambos análisis sin necesidad.

    Args:
        feature_table (pandas.DataFrame): Tabla de características final.

    Returns:
        pandas.DataFrame: Filas con rho/p y p-valores corregidos (Bonferroni
            y FDR).
    """
    from scipy.stats import spearmanr
    from statsmodels.stats.multitest import multipletests

    component_cols = ["S_estab_imputed",
                       "S_cond_att", "S_cond_att_acc", "S_cond_att_med",
                       "S_cond_att_std", "S_cond_att_med_std",
                       "S_cond_self", "S_cond_combined",
                       # Variantes MONOTÓNICAS (no-U, docs/dialogo.txt §2)
                       "S_cond_att_mono", "S_cond_att_med_mono",
                       "S_cond_att_mono_acc_hi", "S_cond_att_mono_acc_lo",
                       "S_coher_prev1_imputed", "S_coher_prev5_imputed"]
    targets = [c for c in ["external_valence_var", "external_valence_range",
                            "external_arousal_var", "self_partner_diff",
                            "self_external_mean_diff"] if c in feature_table.columns]

    rows = []
    for comp in component_cols:
        if comp not in feature_table.columns:
            continue
        valid = feature_table.dropna(subset=[comp])
        for t in targets:
            sub = valid.dropna(subset=[t])
            if len(sub) < 10:
                continue
            r, p = spearmanr(sub[comp], sub[t])
            rows.append({"component": comp, "target": t, "n": len(sub), "rho": r, "p": p})

    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_bonferroni"] = multipletests(result["p"], method="bonferroni")[1]
        result["p_fdr"] = multipletests(result["p"], method="fdr_bh")[1]
    return result


def leverage_diagnostics(feature_table: pd.DataFrame,
                          component_col: str = "S_estab_imputed",
                          target_cols: tuple[str, ...] = (
                              "external_valence_var", "external_valence_range",
                              "external_arousal_var",
                              "self_partner_diff", "self_external_mean_diff",
                          )) -> pd.DataFrame:
    """Diagnostica el leverage/clustering por sujeto de un componente.

    - rho_between_subject: correlación usando SOLO las medias por sujeto
      (n=nº sujetos) — si es tan alta como rho_full, el efecto podría ser
      puramente "rasgo estable del sujeto", no una relación trial-a-trial.
    - rho_within_subject: correlación tras centrar cada variable por su
      propia media de sujeto (person-mean-centering) — aísla la relación
      dentro de sujeto, que es la que S_estab pretende capturar
      conceptualmente (fiabilidad del TRIAL, no rasgo del sujeto — Sección
      9.3 del documento).
    - loso_*: rango de rho al excluir un sujeto cada vez; loso_max_abs_delta
      identifica si un único sujeto concentra la mayor parte del efecto.

    Args:
        feature_table (pandas.DataFrame): Tabla de características.
        component_col (str, optional): Columna del componente. Por defecto es
            "S_estab_imputed".
        target_cols (tuple, optional): Métricas de desacuerdo objetivo. Por
            defecto son las 5 métricas externas.

    Returns:
        pandas.DataFrame: Una fila por (component, target) con las
            correlaciones between/within y el análisis leave-one-subject-out.
    """
    from scipy.stats import spearmanr

    df = feature_table.dropna(subset=[component_col])
    results = []

    for target in target_cols:
        if target not in df.columns:
            continue
        sub = df.dropna(subset=[target])
        if len(sub) < 10:
            continue

        full_rho, full_p = spearmanr(sub[component_col], sub[target])

        subj_means = sub.groupby("subject_id")[[component_col, target]].mean()
        if len(subj_means) >= 3:
            between_rho, between_p = spearmanr(subj_means[component_col], subj_means[target])
        else:
            between_rho, between_p = np.nan, np.nan

        centered = sub.copy()
        centered[component_col] = centered[component_col] - centered.groupby("subject_id")[component_col].transform("mean")
        centered[target] = centered[target] - centered.groupby("subject_id")[target].transform("mean")
        within_rho, within_p = spearmanr(centered[component_col], centered[target])

        loso_rhos = {}
        for sid in sub["subject_id"].unique():
            rest = sub[sub["subject_id"] != sid]
            if len(rest) < 10:
                continue
            r, _ = spearmanr(rest[component_col], rest[target])
            loso_rhos[sid] = r
        loso_series = pd.Series(loso_rhos)
        deltas = (loso_series - full_rho).abs()
        most_influential = deltas.idxmax() if len(deltas) else None
        max_delta = deltas.max() if len(deltas) else np.nan

        results.append({
            "component": component_col, "target": target, "n": len(sub),
            "n_subjects": len(subj_means),
            "rho_full": full_rho,
            "rho_between_subject": between_rho,
            "rho_within_subject": within_rho, "p_within": within_p,
            "loso_rho_min": loso_series.min() if len(loso_series) else np.nan,
            "loso_rho_max": loso_series.max() if len(loso_series) else np.nan,
            "loso_max_abs_delta": max_delta,
            "loso_most_influential_subject": most_influential,
        })

    return pd.DataFrame(results)


def sweep_s_coher_baseline_window(
    feature_table: pd.DataFrame,
    n_prev_values: tuple[int, ...] = (1, 2, 3, 5, 8, 10),
    targets: tuple[str, ...] = ("external_valence_var", "external_valence_range"),
) -> pd.DataFrame:
    """Barre la ventana de baseline de S_coher con descomposición between/within.

    Para ver si rho_within decae de forma suave y monótona con el tamaño de
    la ventana (evidencia de que el efecto es genuinamente de corto plazo) o
    de forma errática (evidencia de que prev1 vs prev5 era más ruido de
    composición que señal real). No sustituye a `psri_composite_prev1/prev5`,
    que se mantienen fijos como las dos variantes "oficiales" del compuesto.

    Args:
        feature_table (pandas.DataFrame): Tabla de características.
        n_prev_values (tuple, optional): Tamaños de ventana a probar. Por
            defecto (1, 2, 3, 5, 8, 10).
        targets (tuple, optional): Métricas de desacuerdo objetivo. Por
            defecto ("external_valence_var", "external_valence_range").

    Returns:
        pandas.DataFrame: Filas con `n_prev` y el diagnóstico de leverage
            para cada punto del barrido.
    """
    rows = []
    for n_prev in n_prev_values:
        df = add_s_coher(feature_table, n_prev=n_prev, out_col=f"S_coher_sweep_{n_prev}")
        col = f"S_coher_sweep_{n_prev}"
        median_val = df[col].median()
        df[f"{col}_imputed"] = df[col].fillna(median_val)
        lev = leverage_diagnostics(df, component_col=f"{col}_imputed", target_cols=targets)
        lev.insert(0, "n_prev", n_prev)
        rows.append(lev)
    return pd.concat(rows, ignore_index=True)


def cross_component_independence(feature_table: pd.DataFrame,
                                  s_coher_col: str = "S_coher_prev1_imputed",
                                  s_cond_col: str = "S_cond_att_acc") -> pd.DataFrame:
    """Comprueba la independencia entre las tres dimensiones del PSRI.

    Réplica en K-EmoCon del chequeo de independencia que en EXIST se reporta
    en la Tabla 5 (correlación de Pearson S_estab-S_coher, S_estab-S_cond,
    S_coher-S_cond). Si las tres dimensiones fueran redundantes entre sí,
    esperaríamos correlaciones altas; valores bajos en ambos estudios
    refuerza que el principio de "tres eslabones independientes" no es
    específico de una instanciación concreta (HR-pupila/reaction_time en
    EXIST vs. HR-EDA/Attention+acc en K-EmoCon), sino del modelo en general.

    Usa Pearson (no Spearman) para que el número sea directamente comparable
    con la Tabla 5 de EXIST, que también usa Pearson.

    Args:
        feature_table (pandas.DataFrame): Tabla de características.
        s_coher_col (str, optional): Columna de S_coher. Por defecto es
            "S_coher_prev1_imputed".
        s_cond_col (str, optional): Columna de S_cond. Por defecto es
            "S_cond_att_acc".

    Returns:
        pandas.DataFrame: Una fila por par de dimensiones con `r_pearson`.
    """
    cols = {"S_estab": "S_estab_imputed", "S_coher": s_coher_col, "S_cond": s_cond_col}
    df = feature_table[list(cols.values())].rename(columns={v: k for k, v in cols.items()})

    pairs = [("S_estab", "S_coher"), ("S_estab", "S_cond"), ("S_coher", "S_cond")]
    rows = []
    for a, b in pairs:
        sub = df[[a, b]].dropna()
        r = sub[a].corr(sub[b], method="pearson")
        rows.append({"par": f"{a}-{b}", "r_pearson": r, "n": len(sub)})

    return pd.DataFrame(rows)


def validate_psri_vs_disagreement(feature_table: pd.DataFrame,
                                   psri_col: str = "psri_composite") -> pd.DataFrame:
    """Correlaciona una variante del PSRI contra las métricas de desacuerdo.

    Corrección por comparaciones múltiples DENTRO de una sola variante (5
    tests). Útil para inspección exploratoria; el resultado a reportar en el
    paper es `validate_psri_vs_disagreement_combined()` por consistencia con
    el Estudio 1.

    Args:
        feature_table (pandas.DataFrame): Tabla de características.
        psri_col (str, optional): Columna de la variante PSRI. Por defecto es
            "psri_composite".

    Returns:
        pandas.DataFrame: Filas con rho/p y p-valores FDR ajustados.
    """
    from scipy.stats import spearmanr
    from statsmodels.stats.multitest import multipletests

    targets = [c for c in ["external_valence_var", "external_valence_range",
                            "external_arousal_var", "self_partner_diff",
                            "self_external_mean_diff"] if c in feature_table.columns]

    rows = []
    valid = feature_table.dropna(subset=[psri_col])
    for t in targets:
        sub = valid.dropna(subset=[t])
        if len(sub) < 10:
            continue
        r, p = spearmanr(sub[psri_col], sub[t])
        rows.append({"psri_variant": psri_col, "target": t, "n": len(sub), "rho": r, "p": p})

    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_adj"] = multipletests(result["p"], method="fdr_bh")[1]
    return result


def validate_psri_vs_disagreement_combined(
    feature_table: pd.DataFrame,
    psri_cols: tuple[str, ...] = ("psri_composite_prev1", "psri_composite_prev5"),
) -> pd.DataFrame:
    """Valida el PSRI compuesto contra el desacuerdo con corrección unificada.

    Bloque ÚNICO de corrección por comparaciones múltiples (Bonferroni +
    FDR) sobre TODOS los tests del Estudio 2 juntos (todas las variantes de
    PSRI x todas las métricas de desacuerdo) — consistente con el criterio
    del Estudio 1 (EXIST), donde los 20 tests se corrigieron como un único
    bloque. No se mezcla con los tests de EXIST: cada estudio mantiene su
    propio bloque de corrección, como preguntas de investigación distintas.

    Args:
        feature_table (pandas.DataFrame): Tabla de características.
        psri_cols (tuple, optional): Variantes de PSRI. Por defecto
            ("psri_composite_prev1", "psri_composite_prev5").

    Returns:
        pandas.DataFrame: Filas con rho/p y p-valores Bonferroni y FDR.
    """
    from scipy.stats import spearmanr
    from statsmodels.stats.multitest import multipletests

    targets = [c for c in ["external_valence_var", "external_valence_range",
                            "external_arousal_var", "self_partner_diff",
                            "self_external_mean_diff"] if c in feature_table.columns]

    rows = []
    for psri_col in psri_cols:
        if psri_col not in feature_table.columns:
            continue
        valid = feature_table.dropna(subset=[psri_col])
        for t in targets:
            sub = valid.dropna(subset=[t])
            if len(sub) < 10:
                continue
            r, p = spearmanr(sub[psri_col], sub[t])
            rows.append({"psri_variant": psri_col, "target": t, "n": len(sub), "rho": r, "p": p})

    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_bonferroni"] = multipletests(result["p"], method="bonferroni")[1]
        result["p_fdr"] = multipletests(result["p"], method="fdr_bh")[1]
    return result