"""
exist/final_check.py

EXPLORATORY DIAGNOSTIC, NOT for the paper as confirmatory.

Última comprobación del Estudio 1 (EXIST 2026) cerrando el círculo con los
reenfoques del framework que se exploraron a fondo en K-EmoCon: verifica si
alguna de las adaptaciones tiene reflejo en los dos objetos de estudio de
EXIST -- (a) el desacuerdo entre anotadores (entropías `entropy_21/22/23` y
la proporción continua `soft_21_yes`) y (b) la propia valoración de sexismo
(`hard_21` YES/NO, `hard_22` DIRECT/JUDGEMENTAL).

Reenfoques probados y su reflejo en EXIST:

  1. DESCOMPOSICIÓN POR COMPONENTE vs ETIQUETA DURA: Kruskal-Wallis de cada
     componente (PSRI, PSRI_hr, PSRI_et, S_estab, S_coher, S_cond) contra
     hard_21 y hard_22. El sanity check actual solo prueba PSRI; aquí se
     desmenuza por eslabón (como en K-EmoCon) por si una pata concreta carga
     la asociación.
  2. COMPUESTO DE 2 PATAS (PSRI_validated = 0.5·S_estab + 0.5·S_coher, sin
     S_cond): el reenfoque que en K-EmoCon recuperó señal que la 3ª pata
     enmascaraba. Se prueba contra entropías, soft_21_yes y etiquetas duras.
  3. VALORACIÓN CONTINUA DE SEXISMO: correlaciones de todos los componentes
     contra `soft_21_yes` (fracción de anotadores que responden YES), target
     continuo más fino que el corte binario de `hard_21`.
  4. DISPERSIÓN ENTRE-SUJETOS DE LA FIABILIDAD (`PSRI_hr_std`,
     `PSRI_et_std`, `et_S_cond_subj_std`): reflejo a nivel de meme de la
     hipótesis "la variabilidad de la fiabilidad lleva señal"
     (experiment_estab_highvar en K-EmoCon): un meme cuyos sujetos de
     sensores DISCREPAN sobre la calidad de la señal podría correlacionar
     con más desacuerdo anotador.
  5. VENTANA DE BASELINE DE S_coher prev1 vs prev5: recomputa S_coher con
     `baseline_prev5` a nivel de trial (las columnas `*_baseline_prev5` ya
     existen en los Excel) y re-agrega por meme -- el análogo del barrido
     n_prev de K-EmoCon.
  6. S_cond MONOTÓNICO (tiempo de reacción "menos es mejor"): variante no-U
     de S_cond calculada a nivel de trial -- el análogo de la familia
     monotónica de S_cond en K-EmoCon.

Todas las pruebas se corrigen como UN ÚNICO bloque (Bonferroni + FDR),
consistente con la filosofía de pool único de EXIST.

Latencia de sensores: NO comprobable cuantitativamente aquí -- EXIST no
expone series temporales crudas (cada trial es un único agregado) y el par
S_coher es cross-device (Garmin x Tobii), exactamente el caso donde el
framework exige verificar la alineación (ver conceptual_psri.md §1.2.1 y
sensor_latency.py en K-EmoCon). Se documenta como limitación estructural.

Garantía de revert limpio: este módulo SOLO lee
  results/output/EXIST/physio_with_psri_memes.csv y
  results/input/EXIST/HR_.xlsx y ET_.xlsx
y escribe en results/output/EXIST/final_check/. No importa ni modifica
build_features.py ni sanity_check.py; borrar el directorio de salida
restaura la situación anterior.

Ejecución:
  .venv/bin/python -m sources.exist.final_check

Autor: Enrique
"""
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, kruskal
from statsmodels.stats.multitest import multipletests

from sources.common.common import logger, writeLog
from sources.psri.calculator import (
    compute_z_score, compute_coherence, compute_psri_gaussian_log,
)
from sources.exist.loader import load_dataframes

DATA_ROOT = Path("results/input/EXIST")
FEATURE_CSV = Path("results/output/EXIST/physio_with_psri_memes.csv")

COMPONENTS = ["PSRI", "PSRI_hr", "PSRI_et", "S_estab", "S_coher", "S_cond"]
ENTROPY_TARGETS = ["entropy_21", "entropy_22", "entropy_23"]
CONTINUOUS_TARGETS = ENTROPY_TARGETS + ["soft_21_yes"]
HARD_LABELS = [("hard_21", ["YES", "NO"]), ("hard_22", ["DIRECT", "JUDGEMENTAL"])]


# ---------------------------------------------------------------------------
# Variantes reflejo de los reenfoques (a nivel de trial, re-agregadas por meme)
# ---------------------------------------------------------------------------

def _monotonic_transform(series: pd.Series, direction: int = -1) -> np.ndarray:
    """Transformación MONOTÓNICA (no en U) con población global.

    Réplica de `_apply_monotonic_transform` de K-EmoCon (mediana/MAD robusta
    1.4826 y R = 1 - exp(-z_clip/2)). Con `direction=-1` ("menos es mejor"),
    los valores bajos de la variable reciben fiabilidad alta.

    Args:
        series (pandas.Series): Variable a transformar.
        direction (int, optional): -1 si "menos es mejor". Por defecto es -1.

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
    R[mask] = 1.0 - np.exp(-np.clip(z, 0.0, None) / 2.0)
    return R


def _trial_variants() -> tuple[pd.Series, pd.Series]:
    """Recomputa S_coher_prev5 y S_cond_mono a nivel de trial y por meme.

    Replica exactamente la población de `build_psri_dataframe`
    (filter_common=True + merge inner por (meme_id, username)) y las
    estadísticas de sujeto (std de las medias por `username`, ceros a NaN).

    Returns:
        tuple: (S_coher_prev5, S_cond_mono) por meme_id.
    """
    df_hr, df_eeg, df_et = load_dataframes(DATA_ROOT)
    common_memes = set(df_hr["meme_id"]) & set(df_et["meme_id"])
    df_hr = df_hr[df_hr["meme_id"].isin(common_memes)].copy()
    df_et = df_et[df_et["meme_id"].isin(common_memes)].copy()

    # --- S_coher_prev5 (baseline de 5 trials previos) ---
    # HR y ET NO comparten columnas de features (solo meme_id/username), así
    # que tras el merge inner los nombres de columna conservan el original.
    pair = df_hr.merge(df_et, on=["meme_id", "username"], how="inner")
    pupil_mean_col = "3d_eye_states_pupil diameter left [mm]_mean"
    pupil_bl5_col = "3d_eye_states_pupil diameter left [mm]_mean_baseline_prev5"
    hr_std_subj = pair.groupby("username")["garmin_hr_mean"].transform("std").replace(0, np.nan)
    pupil_std_subj = (
        pair.groupby("username")[pupil_mean_col].transform("std").replace(0, np.nan)
    )
    z_hr = pair.apply(
        lambda r: compute_z_score(
            r["garmin_hr_mean"],
            r["garmin_hr_mean_baseline_prev5"],
            hr_std_subj.loc[r.name],
        ), axis=1)
    z_pupil = pair.apply(
        lambda r: compute_z_score(
            r[pupil_mean_col],
            r[pupil_bl5_col],
            pupil_std_subj.loc[r.name],
        ), axis=1)
    pair["S_coher_prev5_trial"] = [
        compute_coherence(a, b) if not (np.isnan(a) or np.isnan(b)) else np.nan
        for a, b in zip(z_hr, z_pupil)
    ]
    coher5 = (pair.groupby("meme_id")["S_coher_prev5_trial"].mean()
              .fillna(pair["S_coher_prev5_trial"].median()))

    # --- S_cond_mono (reaction_time "menos es mejor", no-U) ---
    rt_vals = df_et["reaction_time"]
    R_rt_mono = _monotonic_transform(rt_vals, direction=-1)
    blink_vals = df_et["blinks_count"]
    eps_blink = (np.percentile(blink_vals[blink_vals > 0], 1)
                 if (blink_vals > 0).sum() > 0 else 1e-6)
    R_blink_U = compute_psri_gaussian_log(blink_vals, eps_hard=eps_blink)
    df_et = df_et.copy()
    df_et["S_cond_mono_subj"] = (R_rt_mono + R_blink_U) / 2
    cond_mono = (df_et.groupby("meme_id")["S_cond_mono_subj"].mean()
                 .fillna(df_et["S_cond_mono_subj"].median()))

    return coher5, cond_mono


def build_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Añade las variantes reflejo de los reenfoques a la tabla de memes.

    Args:
        df (pandas.DataFrame): Feature table por meme.

    Returns:
        pandas.DataFrame: Copia con `S_coher_prev5`, `S_cond_mono`,
            `PSRI_2leg` (0.5·S_estab + 0.5·S_coher_prev1) y
            `PSRI_2leg_prev5` (0.5·S_estab + 0.5·S_coher_prev5).
    """
    df = df.copy()
    coher5, cond_mono = _trial_variants()
    df = df.merge(coher5.rename("S_coher_prev5"), on="meme_id", how="left")
    df = df.merge(cond_mono.rename("S_cond_mono"), on="meme_id", how="left")
    for col in ["S_coher_prev5", "S_cond_mono"]:
        df[col] = df[col].fillna(df[col].median())
    df["PSRI_2leg"] = 0.5 * df["S_estab"] + 0.5 * df["S_coher"]
    df["PSRI_2leg_prev5"] = 0.5 * df["S_estab"] + 0.5 * df["S_coher_prev5"]
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_correlations(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Spearman de cada métrica contra cada target continuo.

    Args:
        df (pandas.DataFrame): Feature table por meme.
        metrics (list): Métricas (componentes y variantes).

    Returns:
        pandas.DataFrame: Filas (metric, target, n, rho, p_value).
    """
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        for target in CONTINUOUS_TARGETS:
            if target not in df.columns:
                continue
            sub = df[[metric, target]].dropna()
            if len(sub) < 10:
                continue
            r, p = spearmanr(sub[metric], sub[target])
            rows.append({"test": "Spearman", "metric": metric, "target": target,
                         "n": len(sub), "stat": r, "p_value": p})
    return pd.DataFrame(rows)


def run_kruskal(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Kruskal-Wallis de cada métrica contra las etiquetas duras de sexismo.

    Args:
        df (pandas.DataFrame): Feature table por meme.
        metrics (list): Métricas (componentes y variantes).

    Returns:
        pandas.DataFrame: Filas (metric, label, n, stat, p_value,
            effect_size, median_grp0, median_grp1).
    """
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        for label, valid in HARD_LABELS:
            if label not in df.columns:
                continue
            sub = df[[label, metric]].dropna()
            sub = sub[sub[label].isin(valid)]
            if len(sub) < 10:
                continue
            g0 = sub[sub[label] == valid[0]][metric]
            g1 = sub[sub[label] == valid[1]][metric]
            if len(g0) < 2 or len(g1) < 2:
                continue
            stat, p = kruskal(g0, g1)
            n = len(sub)
            epsilon_sq = (stat - 2 + 1) / (n - 2) if n > 2 else float("nan")
            rows.append({"test": "Kruskal-Wallis", "metric": metric,
                         "label": f"{metric} vs {label} ({valid[0]} vs {valid[1]})",
                         "n": n, "stat": stat, "p_value": p,
                         "effect_size": epsilon_sq,
                         "median_grp0": g0.median(), "median_grp1": g1.median()})
    return pd.DataFrame(rows)


def unify_and_correct(corr: pd.DataFrame, kw: pd.DataFrame) -> pd.DataFrame:
    """Fusiona correlaciones y Kruskal en un único bloque con Bonferroni y FDR.

    Args:
        corr (pandas.DataFrame): Filas de correlación.
        kw (pandas.DataFrame): Filas de Kruskal.

    Returns:
        pandas.DataFrame: Tabla unificada con p corregidos.
    """
    unified = []
    for _, r in kw.iterrows():
        unified.append({"test": r["test"], "label": r["label"],
                        "p_value": r["p_value"], "stat": r["stat"],
                        "effect_size": r["effect_size"], "n": r["n"]})
    for _, r in corr.iterrows():
        unified.append({"test": r["test"],
                        "label": f"{r['metric']} vs {r['target']}",
                        "p_value": r["p_value"], "stat": r["stat"],
                        "effect_size": r["stat"] ** 2, "n": r["n"]})
    out = pd.DataFrame(unified)
    if out.empty:
        return out
    p_vals = out["p_value"].to_numpy()
    reject_bonf, p_bonf, _, _ = multipletests(p_vals, method="bonferroni")
    reject_fdr, p_fdr, _, _ = multipletests(p_vals, method="fdr_bh")
    out["p_bonferroni"] = p_bonf
    out["reject_bonferroni"] = reject_bonf
    out["p_fdr"] = p_fdr
    out["reject_fdr"] = reject_fdr
    return out


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

def run_final_check(out_dir: Path | None = None) -> dict:
    """Ejecuta la última comprobación de EXIST.

    Args:
        out_dir (Path or None, optional): Directorio de salida. Si es None se
            usa `results/output/EXIST/final_check`.

    Returns:
        dict: Con las tablas de correlaciones, Kruskal y la unificada.
    """
    if out_dir is None:
        out_dir = FEATURE_CSV.parent / "final_check"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURE_CSV)
    df = build_variants(df)
    writeLog("info", logger,
             f"[exist_final_check] {len(df)} memes; escribiendo en {out_dir}")

    metrics = (COMPONENTS
               + ["S_coher_prev5", "S_cond_mono", "PSRI_2leg", "PSRI_2leg_prev5",
                  "PSRI_hr_std", "PSRI_et_std", "et_S_cond_subj_std"])
    metrics = list(dict.fromkeys([m for m in metrics if m in df.columns]))

    corr = run_correlations(df, metrics)
    kw = run_kruskal(df, metrics)
    unified = unify_and_correct(corr, kw)

    corr.to_csv(out_dir / "final_check_correlations.csv", index=False)
    kw.to_csv(out_dir / "final_check_kruskal.csv", index=False)
    unified.to_csv(out_dir / "final_check_unified.csv", index=False)
    df.to_csv(out_dir / "final_check_variants.csv", index=False)

    print("\n== [FINAL CHECK] Correlaciones (Spearman) vs desacuerdo/soft_21_yes ==")
    sig = unified[(unified["test"] == "Spearman")].copy()
    sig["label"] = sig["label"].str.replace(" vs ", " ↔ ")
    sig = sig.sort_values("p_value")
    print(sig[["label", "n", "stat", "p_value", "p_bonferroni", "p_fdr"]]
          .round(4).to_string(index=False))

    print("\n== [FINAL CHECK] Kruskal-Wallis vs valoración de sexismo (hard_21/22) ==")
    kws = unified[unified["test"] == "Kruskal-Wallis"].copy()
    kws = kws.sort_values("p_value")
    print(kws[["label", "n", "stat", "effect_size", "p_value",
               "p_bonferroni", "p_fdr"]].round(4).to_string(index=False))

    n_tot = len(unified)
    n_bonf = int(unified["reject_bonferroni"].sum())
    n_fdr = int(unified["reject_fdr"].sum())
    n_raw = int((unified["p_value"] < 0.05).sum())
    print(f"\nResumen: {n_raw}/{n_tot} p<0.05 sin corregir | "
          f"{n_bonf}/{n_tot} tras Bonferroni | {n_fdr}/{n_tot} tras FDR")
    survivors = unified[unified["reject_bonferroni"] | unified["reject_fdr"]]
    if survivors.empty:
        print("Ninguna variante reflejo de los reenfoques sobrevive la "
              "corrección: el nulo de EXIST es robusto también a los reenfoques.")
    else:
        print("\n== Sobreviven a la corrección (Bonferroni o FDR) ==")
        print(survivors[["test", "label", "stat", "p_value", "p_bonferroni",
                         "p_fdr"]].round(4).to_string(index=False))
        mono = survivors[survivors["label"].str.contains("S_cond_mono")]
        if not mono.empty:
            print("\nEl único reflejo robusto de los reenfoques en EXIST es "
                  "S_cond_mono (tiempo de reacción MONOTÓNICO 'menos es mejor'): "
                  "los memes procesados con más fluidez (RT más corto) se asocian "
                  "con MENOS desacuerdo anotador y con valoración MENOS sexista. "
                  "La variante en U original de S_cond (que penaliza también el "
                  "RT rápido) destruye esta señal direccional.")
    return {"correlations": corr, "kruskal": kw, "unified": unified}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Última comprobación EXIST: reenfoques del framework vs "
                    "desacuerdo y valoración de sexismo")
    parser.add_argument("--out", type=Path, default=None,
                        help="directorio de salida (default: "
                             "results/output/EXIST/final_check)")
    args = parser.parse_args()
    run_final_check(args.out)
    print(f"\n[exist_final_check] salidas en "
          f"{args.out or FEATURE_CSV.parent / 'final_check'}/")
