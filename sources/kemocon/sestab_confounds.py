"""
kemocon/sestab_confounds.py

Análisis de ROBUSTEZ del Estudio 2 (K-EmoCon): confounds del efecto
entre-sujetos de S_estab sobre el desacuerdo.

Pregunta: un sujeto con señal crónicamente inestable puede tener también más
movimiento (acc_std), otro nivel de EDA, HR o TEMP. Si S_estab es un proxy de
algún confound, su valor entre-sujetos es parcialmente espurio. Este módulo
cuantifica qué parte del efecto entre-sujetos sobrevive al control por esos
confounds, y por qué canal (bvp/eda/hr/temp/ibi) llega.

Resultado empírico (agosto 2026, n=23 sujetos): el efecto entre-sujetos de
S_estab sobre valence/arousal está explicado por el rasgo de movimiento
(control por acc_std: ρ ≈ −0.28 → 0.00 en valence_var). El valor independiente
se limita a self_partner_diff/self_external_mean_diff (entre-sujetos) y a la
señal within-subject de self_partner_diff (el movimiento no la confunde). Ver
EstudioKemocon.md §4.

Análisis (nivel sujeto):
  1. rho simple: S_estab medio vs desacuerdo medio (réplica rho_between).
  2. rho parcial (rangos): control individual y conjunto (acc_std, eda_mean,
     hr_mean, temp_mean).
  3. Descomposición por canal: qué S_estab_canal arrastra el efecto.
  4. OLS sujeto (estandarizado): target ~ S_estab + confounds.

Se invoca desde run_pipeline.process_kemocon() con la feature table en memoria;
también se puede ejecutar en standalone para regenerar sin re-ejecutar el
pipeline.

Ejecución standalone:
  .venv/bin/python -m sources.kemocon.sestab_confounds

Autor: Enrique
"""
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, rankdata, t as t_dist
from statsmodels.formula.api import ols

from sources.common.common import logger, writeLog

DEFAULT_INPUT = Path("results/output/KEMOCON/kemocon_feature_table.csv")

TARGETS = ["external_valence_var", "external_valence_range",
           "external_arousal_var", "self_partner_diff",
           "self_external_mean_diff"]

CONFOUNDS = ["acc_std", "eda_mean", "hr_mean", "temp_mean"]
CHANNELS = ["bvp_S_estab", "eda_S_estab", "hr_S_estab", "temp_S_estab", "ibi_S_estab"]


def _partial_rho(x: pd.Series, y: pd.Series, controls: list[str],
                 df_ctrl: pd.DataFrame) -> tuple[float, float]:
    """Correlación parcial de Spearman (rangos + residuos) con controles.

    Args:
        x (pandas.Series): Variable de interés.
        y (pandas.Series): Variable objetivo.
        controls (list): Nombres de columnas control en df_ctrl.
        df_ctrl (pandas.DataFrame): Tabla sujeto con los controles.

    Returns:
        tuple: (rho_parcial, p). NaN si no converge.
    """
    n = len(x)
    r_x = rankdata(x)
    r_y = rankdata(y)
    try:
        if controls:
            X = np.column_stack([np.ones(n)] + [rankdata(df_ctrl[c].to_numpy())
                                                for c in controls])
            beta_x = np.linalg.lstsq(X, r_x, rcond=None)[0]
            beta_y = np.linalg.lstsq(X, r_y, rcond=None)[0]
            res_x = r_x - X @ beta_x
            res_y = r_y - X @ beta_y
        else:
            res_x, res_y = r_x, r_y
        rho = np.corrcoef(res_x, res_y)[0, 1]
        df = n - len(controls) - 2
        if df < 1 or not np.isfinite(rho):
            return np.nan, np.nan
        t = rho * np.sqrt(df / (1 - rho ** 2))
        p = 2 * t_dist.sf(abs(t), df)
        return rho, p
    except Exception as exc:  # noqa: BLE001
        writeLog("warning", logger, f"[sestab_confounds] partial rho falló: {exc}")
        return np.nan, np.nan


def run_sestab_confounds(feature_table: pd.DataFrame,
                         out_dir: Path) -> pd.DataFrame:
    """Ejecuta el control de confounds del efecto entre-sujetos de S_estab.

    Args:
        feature_table (pandas.DataFrame): Feature table final (una fila por
            sujeto-ventana).
        out_dir (Path): Directorio de salida.

    Returns:
        pandas.DataFrame: Tabla sujeto agregada (para inspección).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg_cols = ["S_estab"] + CHANNELS + CONFOUNDS + TARGETS
    agg_cols = [c for c in agg_cols if c in feature_table.columns]
    subj = feature_table.groupby("subject_id")[agg_cols].mean().reset_index()
    n = len(subj)
    writeLog("info", logger, f"[sestab_confounds] sujetos con cobertura: {n}")
    subj.to_csv(out_dir / "kemocon_sestab_subject_level.csv", index=False)

    rows = []
    for target in TARGETS:
        r, p = spearmanr(subj["S_estab"], subj[target])
        rows.append({"target": target, "control": "none",
                     "rho": r, "p": p, "n": n})

    for target in TARGETS:
        for control in ["none"] + CONFOUNDS + ["ALL"]:
            if control == "none":
                r, p = _partial_rho(subj["S_estab"], subj[target], [], subj)
                ctrl_label = "none"
            elif control == "ALL":
                r, p = _partial_rho(subj["S_estab"], subj[target], CONFOUNDS, subj)
                ctrl_label = "acc+eda+hr+temp"
            else:
                r, p = _partial_rho(subj["S_estab"], subj[target], [control], subj)
                ctrl_label = control
            rows.append({"target": target, "control": ctrl_label, "rho": r,
                         "p": p, "n": n})

    controls_df = pd.DataFrame(rows)
    controls_df.to_csv(out_dir / "kemocon_sestab_between_controls.csv", index=False)

    ch_rows = []
    for ch in CHANNELS:
        if ch not in subj.columns:
            continue
        r, p = spearmanr(subj[ch], subj["external_valence_var"])
        r2, p2 = spearmanr(subj[ch], subj["external_arousal_var"])
        ch_rows.append({"channel": ch, "rho_valence_var": r, "p_valence": p,
                        "rho_arousal_var": r2, "p_arousal": p2, "n": n})
    pd.DataFrame(ch_rows).to_csv(
        out_dir / "kemocon_sestab_channel_decomposition.csv", index=False)

    std = subj.copy()
    for c in ["S_estab"] + CONFOUNDS + TARGETS:
        if c in std.columns:
            std[f"z_{c}"] = (std[c] - std[c].mean()) / std[c].std(ddof=0)
    ols_rows = []
    for target in TARGETS:
        formula = (f"z_{target} ~ z_S_estab + z_acc_std + z_eda_mean "
                   "+ z_hr_mean + z_temp_mean")
        m = ols(formula, std).fit()
        for coef in ["z_S_estab", "z_acc_std", "z_eda_mean", "z_hr_mean", "z_temp_mean"]:
            ols_rows.append({"target": target, "predictor": coef,
                             "beta": m.params[coef], "se": m.bse[coef],
                             "p": m.pvalues[coef]})
    pd.DataFrame(ols_rows).to_csv(out_dir / "kemocon_sestab_ols.csv", index=False)

    print("\n== [ROBUSTEZ] Confounds del efecto entre-sujetos de S_estab ==")
    print("rho simple y parcial (S_estab medio vs desacuerdo medio, n=%d):" % n)
    print(controls_df[controls_df["control"].isin(
        ["none", "acc_std", "eda_mean", "acc+eda+hr+temp"])]
          .pivot_table(index="target", columns="control",
                       values="rho").round(3).to_string())
    print("Canal que arrastra el efecto (valence_var):")
    print(pd.DataFrame(ch_rows).set_index("channel")
          [["rho_valence_var", "rho_arousal_var"]].round(3).to_string())
    return subj


def run_from_csv(input_csv: Path = DEFAULT_INPUT,
                 out_dir: Path | None = None) -> pd.DataFrame:
    """Carga la feature table del CSV y ejecuta el análisis.

    Args:
        input_csv (Path, optional): Feature table. Por defecto es
            DEFAULT_INPUT.
        out_dir (Path or None, optional): Directorio de salida. Si es None se
            usa el directorio padre del CSV (results/output/KEMOCON/).

    Returns:
        pandas.DataFrame: Tabla sujeto agregada.
    """
    feature = pd.read_csv(input_csv)
    if out_dir is None:
        out_dir = Path(input_csv).parent
    return run_sestab_confounds(feature, out_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Robustez: confounds del efecto entre-sujetos de S_estab")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="feature table de entrada (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=None,
                        help="directorio de salida (default: junto al CSV)")
    args = parser.parse_args()
    run_from_csv(args.input, args.out)
    print(f"\n[sestab_confounds] salidas en "
          f"{args.out or Path(args.input).parent}/")