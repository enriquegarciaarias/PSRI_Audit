"""
kemocon/experiment_estab_highvar.py

EXPERIMENTAL diagnostic, NOT for the paper (confirmatory block is fixed).
Response to a conceptual reflection: the U of S_estab penalizes BOTH high and
low variability as "unreliable". But for the SUBJECTIVITY question (does
physiological (un)reliability predict annotator disagreement?), the HIGH
variability side might carry information that the U-mask hides -- "there things
happen". Concretely:

  - Is there signal between variability LEVEL (log-sigma z) and disagreement
    within the HIGH variability windows only?
  - Is there a threshold above which those highs are informative, and does the
    signal collapse for the EXTREME highs (plateau/inversion)?
  - Does S_coher act as a regulator: high-variability windows "go well" (are
    informative) when S_coher is not out of control (coordinated), and are
    uninformative/artifact when S_coher is chaotic?

Guarantee of clean revert: este módulo SOLO lee
results/output/KEMOCON/kemocon_feature_table.csv y escribe en
results/output/KEMOCON/experiment_estab_highvar/. No importa ni modifica
build_features.py, run_pipeline.py ni calculator.py; borrar el archivo
restaura la situación anterior.

Operationalization:
  - z_var = mean over E4 channels of the robust log-sigma z-score
    (median/MAD, population-global, same stats as compute_psri_gaussian_log).
    This is the "variability level" before the U is applied.
  - HIGH region   = windows with z_var > 0 (on the descending/chaotic side of
    the U; flat windows have z_var << 0). Complements S_estab: same input,
    but re-labels the high-variability branch as "region of interest" instead
    of "unreliable".
  - Analyses, all within-subject (person-mean-centering) on the HIGH region:
      1. rho(z_var, target) -- raw within-subject Spearman.
      2. rho(z_var, target) controlling acc_std within-subject (partial rank).
      3. Threshold sweep: rho within windows with z_var >= t, for t in a grid
         of quantiles -- does the signal appear, stabilize, or invert?
      4. S_coher regulator: split HIGH windows by S_coher_prev1 median
         (coordinated vs uncoordinated) and compare the within rho.

Ejecución:
  .venv/bin/python -m sources.kemocon.experiment_estab_highvar

Autor: Enrique
"""
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, rankdata, t as t_dist

from sources.common.common import logger, writeLog

DEFAULT_INPUT = Path("results/output/KEMOCON/kemocon_feature_table.csv")

TARGETS = ["external_valence_var", "external_valence_range",
           "external_arousal_var", "self_partner_diff",
           "self_external_mean_diff"]

CHANNELS = ["bvp_std", "eda_std", "hr_std", "temp_std", "ibi_std"]

# Quantiles del barrido de umbral (lado alto, z_var > 0)
SWEEP_QUANTILES = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]


def _robust_z_var(feature: pd.DataFrame,
                  channels: tuple[str, ...] = tuple(CHANNELS)) -> pd.Series:
    """Media por ventana de los z-scores robustos de log-sigma por canal.

    Replica las estadísticas de `compute_psri_gaussian_log` (log-transform,
    mediana y MAD robusta con 1.4826) para expresar el NIVEL de variabilidad
    de la ventana antes de aplicar la campana en U.

    Args:
        feature (pandas.DataFrame): Feature table.
        channels (tuple, optional): Columnas *_std de los canales E4.

    Returns:
        pandas.Series: z_var (media de los z por canal); NaN donde todos los
            canales eran NaN.
    """
    z_cols = []
    for col in channels:
        if col not in feature.columns:
            continue
        vals = feature[col].to_numpy(dtype=float)
        mask = ~np.isnan(vals)
        z = np.full(len(vals), np.nan)
        if mask.sum() > 0:
            log_sigma = np.log(vals[mask] + 1e-9)
            median_log = np.median(log_sigma)
            mad_log = 1.4826 * np.median(np.abs(log_sigma - median_log))
            if mad_log == 0:
                mad_log = 1.0
            z[mask] = (log_sigma - median_log) / mad_log
        out = f"{col}__z"
        feature[out] = z
        z_cols.append(out)
    if not z_cols:
        return pd.Series(np.nan, index=feature.index)
    return feature[z_cols].mean(axis=1)


def _partial_rho_within(sub: pd.DataFrame, x: str, y: str,
                        controls: list[str]) -> tuple[float, float]:
    """Correlación parcial de Spearman within-subject (centrado por sujeto).

    Person-mean-centering de x, y y controles, luego rango y regresión de los
    rangos contra los controles (residuos). Devolver (rho, p) o NaN si no
    converge.

    Args:
        sub (pandas.DataFrame): Subconjunto con subject_id.
        x (str): Columna predictor.
        y (str): Columna target.
        controls (list): Columnas de control (centradas igualmente).

    Returns:
        tuple: (rho_parcial_within, p).
    """
    df = sub.dropna(subset=[x, y] + controls).copy()
    if len(df) < 10 or df["subject_id"].nunique() < 3:
        return np.nan, np.nan
    for c in [x, y] + controls:
        df[f"{c}_c"] = df[c] - df.groupby("subject_id")[c].transform("mean")
    rx = rankdata(df[f"{x}_c"].to_numpy())
    ry = rankdata(df[f"{y}_c"].to_numpy())
    n = len(df)
    try:
        if controls:
            X = np.column_stack([np.ones(n)] +
                                [rankdata(df[f"{c}_c"].to_numpy())
                                 for c in controls])
            res_x = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]
            res_y = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
        else:
            res_x, res_y = rx, ry
        rho = np.corrcoef(res_x, res_y)[0, 1]
        dfree = n - len(controls) - 2
        if dfree < 1 or not np.isfinite(rho):
            return np.nan, np.nan
        t_stat = rho * np.sqrt(dfree / (1 - rho ** 2))
        return rho, 2 * t_dist.sf(abs(t_stat), dfree)
    except Exception as exc:  # noqa: BLE001
        writeLog("warning", logger,
                 f"[estab_highvar] parcial within falló ({x}~{y}): {exc}")
        return np.nan, np.nan


def run_estab_highvar(feature_table: pd.DataFrame,
                      out_dir: Path) -> pd.DataFrame:
    """Ejecuta el diagnóstico de la región de alta variabilidad de S_estab.

    Args:
        feature_table (pandas.DataFrame): Feature table final.
        out_dir (Path): Directorio de salida.

    Returns:
        pandas.DataFrame: Tabla resumen del barrido de umbral (para
            inspección).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = feature_table.copy()
    df["z_var"] = _robust_z_var(df)

    high = df.dropna(subset=["z_var", "S_coher_prev1"]).copy()
    high_region = high[high["z_var"] > 0].copy()
    n_high = len(high_region)
    n_subj_high = high_region["subject_id"].nunique()
    writeLog("info", logger,
             f"[estab_highvar] región alta z_var>0: {n_high} ventanas, "
             f"{n_subj_high} sujetos (de {len(high)} con z_var y S_coher)")

    high_region.to_csv(out_dir / "kemocon_highvar_windows.csv", index=False)

    # --- 1) within-subject rho(z_var, target) en la región alta ---
    rows = []
    for target in TARGETS:
        if target not in high_region.columns:
            continue
        r0, p0 = _partial_rho_within(high_region, "z_var", target, [])
        r1, p1 = _partial_rho_within(high_region, "z_var", target, ["acc_std"])
        rows.append({"region": "high (z_var>0)", "target": target,
                     "n": n_high,
                     "rho_within": r0, "p_within": p0,
                     "rho_within_ctrl_acc": r1, "p_within_ctrl_acc": p1})
    base_df = pd.DataFrame(rows)
    base_df.to_csv(out_dir / "kemocon_highvar_within.csv", index=False)

    print("\n== [1] Región alta (z_var>0): rho within z_var↔desacuerdo ==")
    print(base_df[["target", "n", "rho_within", "p_within",
                   "rho_within_ctrl_acc", "p_within_ctrl_acc"]]
          .round(4).to_string(index=False))

    # --- 2) barrido de umbral: rho within para z_var >= quantile ---
    sweep_rows = []
    for q in SWEEP_QUANTILES:
        thr = np.quantile(df["z_var"].dropna(), q)
        sub = high[high["z_var"] >= thr]
        for target in TARGETS:
            if target not in sub.columns:
                continue
            r0, p0 = _partial_rho_within(sub, "z_var", target, [])
            r1, p1 = _partial_rho_within(sub, "z_var", target, ["acc_std"])
            sweep_rows.append({"q": q, "thr_zvar": thr,
                               "n": len(sub),
                               "n_subj": sub["subject_id"].nunique(),
                               "target": target,
                               "rho_within": r0, "p_within": p0,
                               "rho_within_ctrl_acc": r1,
                               "p_within_ctrl_acc": p1})
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(out_dir / "kemocon_highvar_threshold_sweep.csv", index=False)

    print("\n== [2] Barrido de umbral (rho within z_var↔desacuerdo) ==")
    print(sweep_df.pivot_table(index="target", columns="q",
                               values="rho_within").round(3).to_string())
    print("\n  -- controlando acc_std (within) --")
    print(sweep_df.pivot_table(index="target", columns="q",
                               values="rho_within_ctrl_acc").round(3).to_string())

    # --- 3) S_coher como regulador: split por mediana dentro de la región alta ---
    med_coher = high_region["S_coher_prev1"].median()
    coord = high_region[high_region["S_coher_prev1"] >= med_coher]
    uncoord = high_region[high_region["S_coher_prev1"] < med_coher]
    reg_rows = []
    for label, grp in [("coord (S_coher>=med)", coord),
                       ("uncoord (S_coher<med)", uncoord)]:
        for target in TARGETS:
            if target not in grp.columns:
                continue
            r0, p0 = _partial_rho_within(grp, "z_var", target, [])
            r1, p1 = _partial_rho_within(grp, "z_var", target, ["acc_std"])
            reg_rows.append({"region": label, "target": target,
                             "n": len(grp),
                             "rho_within": r0, "p_within": p0,
                             "rho_within_ctrl_acc": r1,
                             "p_within_ctrl_acc": p1})
    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(out_dir / "kemocon_highvar_scoher_moderator.csv", index=False)

    print(f"\n== [3] S_coher regulador dentro de la región alta (med={med_coher:.3f}) ==")
    print(reg_df[["region", "target", "n", "rho_within", "p_within",
                  "rho_within_ctrl_acc", "p_within_ctrl_acc"]]
          .round(4).to_string(index=False))

    return sweep_df


def run_from_csv(input_csv: Path = DEFAULT_INPUT,
                 out_dir: Path | None = None) -> pd.DataFrame:
    """Carga la feature table del CSV y ejecuta el diagnóstico.

    Args:
        input_csv (Path, optional): Feature table. Por defecto es
            DEFAULT_INPUT.
        out_dir (Path or None, optional): Directorio de salida. Si es None se
            usa `results/output/KEMOCON/experiment_estab_highvar`.

    Returns:
        pandas.DataFrame: Barrido de umbral.
    """
    feature = pd.read_csv(input_csv)
    if out_dir is None:
        out_dir = DEFAULT_INPUT.parent / "experiment_estab_highvar"
    return run_estab_highvar(feature, out_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Diagnóstico exploratorio: región de alta variabilidad de S_estab")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="feature table de entrada (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=None,
                        help="directorio de salida (default: "
                             "results/output/KEMOCON/experiment_estab_highvar)")
    args = parser.parse_args()
    run_from_csv(args.input, args.out)
    print("\n[experiment_estab_highvar] salidas en "
          f"{args.out or DEFAULT_INPUT.parent / 'experiment_estab_highvar'}/")