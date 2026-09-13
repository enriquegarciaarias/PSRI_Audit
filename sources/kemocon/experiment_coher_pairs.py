"""
kemocon/experiment_coher_pairs.py

EXPLORATORY DIAGNOSTIC, NOT for the paper as confirmatory (confirmatory block
is fixed to the HR-EDA pair).

Response to a reviewer question (docs/ECIR2027/review.md, Q8): "Beyond HR-EDA,
do other cross-system pairs (e.g., BVP-EDA, EEG-derived engagement-EDA)
replicate the within-subject S_coher effect, even if weaker?"

The claim to audit: the within-subject association S_coher <-> annotator
disagreement (external_valence_var rho_within ~ +0.060, p~0.001 in the
reference HR-EDA pair) should be a property of cross-SYSTEM coordination, not
an accident of one specific sensor pairing. If it replicates (even weaker)
across other physiologically-distinct pairs and collapses for same-system
pairs, the effect is generic; if it only lives in HR-EDA, it is pair-specific.

Design:
  - Recompute S_coher_prev1 for every pair among the E4 channels
    (bvp, eda, hr, temp, ibi) with the IDENTICAL formula used for the
    reference pair (rolling baseline prev1 per subject + historical
    subject_std + compute_z_score + compute_coherence) -- see
    build_features.add_s_coher.
  - Cross-system replication candidates (cardiovascular vs sympathetic vs
    thermoregulatory): bvp-eda, hr-temp, bvp-temp, eda-temp, ibi-eda,
    ibi-temp.
  - Same-system control pairs (redundancy: two channels of the same
    physiological system, like the discarded HR(Ed)-HR(Polar)):
    bvp-hr, hr-ibi, bvp-ibi. A true cross-system effect should NOT appear
    here.
  - Reference pair for sanity check: hr-eda (must reproduce
    rho_within ~ +0.060, p ~ 0.001 for external_valence_var).
  - Per pair x target: pooled / between-subject / within-subject Spearman
    (person-mean-centering) + leave-one-subject-out, via
    build_features.leverage_diagnostics. All p-values of the family are
    corrected as ONE block (Bonferroni + FDR), consistent with the project's
    multiple-comparison policy.

Guarantee of clean revert: este módulo SOLO lee
results/output/KEMOCON/kemocon_feature_table.csv y escribe en
results/output/KEMOCON/experiment_coher_pairs/. No importa ni modifica
run_pipeline.py ni calculator.py; borrar la carpeta de salida restaura la
situación anterior.

Ejecución:
  .venv/bin/python -m sources.kemocon.experiment_coher_pairs

Autor: Enrique
"""
from pathlib import Path

import numpy as np
import pandas as pd

from statsmodels.stats.multitest import multipletests

from sources.psri.calculator import compute_coherence, compute_z_score
from sources.kemocon.build_features import _rolling_baseline_prev, leverage_diagnostics
from sources.common.common import logger, writeLog

DEFAULT_INPUT = Path("results/output/KEMOCON/kemocon_feature_table.csv")
OUT_DIR = Path("results/output/KEMOCON/experiment_coher_pairs/")
N_PREV = 1

TARGETS = ["external_valence_var", "external_valence_range",
           "external_arousal_var", "self_partner_diff",
           "self_external_mean_diff"]

# (col_a, col_b, system_a, system_b, role)
PAIRS = [
    ("hr_mean", "eda_mean", "cardiovascular", "sympathetic", "reference"),
    ("bvp_mean", "eda_mean", "cardiovascular", "sympathetic", "replication"),
    ("hr_mean", "temp_mean", "cardiovascular", "thermoregulatory", "replication"),
    ("bvp_mean", "temp_mean", "cardiovascular", "thermoregulatory", "replication"),
    ("eda_mean", "temp_mean", "sympathetic", "thermoregulatory", "replication"),
    ("ibi_mean", "eda_mean", "cardiovascular", "sympathetic", "replication"),
    ("ibi_mean", "temp_mean", "cardiovascular", "thermoregulatory", "replication"),
    ("bvp_mean", "hr_mean", "cardiovascular", "cardiovascular", "same-system control"),
    ("hr_mean", "ibi_mean", "cardiovascular", "cardiovascular", "same-system control"),
    ("bvp_mean", "ibi_mean", "cardiovascular", "cardiovascular", "same-system control"),
]


def _coher_pair(feature: pd.DataFrame, col_a: str, col_b: str,
                n_prev: int = N_PREV) -> pd.Series:
    """S_coher para un par arbitrario, replicando `add_s_coher` (HR-EDA).

    Misma fórmula que el par de referencia: baseline rolling prev1 por sujeto,
    std histórica del sujeto, Z-scores y decaimiento exponencial.

    Args:
        feature (pandas.DataFrame): Feature table.
        col_a (str): Canal A.
        col_b (str): Canal B.
        n_prev (int, optional): Ventana de baseline. Por defecto es 1.

    Returns:
        pandas.Series: S_coher del par, NaN donde algún Z-score es NaN.
    """
    df = feature.sort_values(["subject_id", "win"]).reset_index(drop=True)
    baseline_a = _rolling_baseline_prev(df, col_a, n_prev)
    baseline_b = _rolling_baseline_prev(df, col_b, n_prev)
    subj_std_a = df.groupby("subject_id")[col_a].transform("std")
    subj_std_b = df.groupby("subject_id")[col_b].transform("std")

    z_a = [compute_z_score(v, b, s)
           for v, b, s in zip(df[col_a], baseline_a, subj_std_a)]
    z_b = [compute_z_score(v, b, s)
           for v, b, s in zip(df[col_b], baseline_b, subj_std_b)]

    coher = [
        compute_coherence(a, b) if not (np.isnan(a) or np.isnan(b)) else np.nan
        for a, b in zip(z_a, z_b)
    ]
    return pd.Series(coher, index=df.index)


def main() -> None:
    writeLog("info", logger, "[experiment_coher_pairs] Inicio")
    feature = pd.read_csv(DEFAULT_INPUT)
    logger.info("Feature table: %d filas, %d sujetos",
                len(feature), feature["subject_id"].nunique())

    rows = []
    for col_a, col_b, sys_a, sys_b, role in PAIRS:
        if col_a not in feature.columns or col_b not in feature.columns:
            logger.info("Par %s-%s omitido (canal ausente)", col_a, col_b)
            continue
        feat = feature.copy()
        s_coher = _coher_pair(feat, col_a, col_b, n_prev=N_PREV)
        tag = f"{col_a.split('_')[0]}_{col_b.split('_')[0]}"
        col = f"S_coher_{tag}"
        feat[col] = s_coher
        med = feat[col].median()
        feat[f"{col}_imputed"] = feat[col].fillna(med)

        lev = leverage_diagnostics(feat, component_col=f"{col}_imputed",
                                   target_cols=TARGETS)
        lev.insert(0, "pair_tag", tag)
        lev.insert(0, "role", role)
        lev.insert(0, "system_b", sys_b)
        lev.insert(0, "system_a", sys_a)
        lev.insert(0, "col_b", col_b)
        lev.insert(0, "col_a", col_a)
        rows.append(lev)
        logger.info("Par %s (%s): rho_within valence_var=%.4f (p=%.4g)",
                    tag, role,
                    lev.loc[lev["target"] == "external_valence_var",
                            "rho_within_subject"].iloc[0],
                    lev.loc[lev["target"] == "external_valence_var",
                            "p_within"].iloc[0])

    out = pd.concat(rows, ignore_index=True)
    out["p_within_bonf"] = multipletests(out["p_within"], method="bonferroni")[1]
    out["p_within_fdr"] = multipletests(out["p_within"], method="fdr_bh")[1]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "coher_pairs_within.csv", index=False)
    logger.info("Salida completa: %s", OUT_DIR / "coher_pairs_within.csv")

    # Tabla resumen enfocada en los targets de valencia (los del efecto
    # confirmatorio) para juzgar si la réplica existe.
    focus = out[out["target"].isin(["external_valence_var",
                                    "external_valence_range"])]
    summary = focus.pivot_table(
        index=["pair_tag", "role", "col_a", "col_b", "system_a", "system_b"],
        columns="target",
        values="rho_within_subject",
    ).reset_index()
    summary_p = focus.pivot_table(
        index="pair_tag", columns="target", values="p_within"
    ).reset_index()
    summary_bonf = focus.pivot_table(
        index="pair_tag", columns="target", values="p_within_bonf"
    ).reset_index()

    print("\n=== S_coher WITHIN-SUBJECT por par (rho) ===\n")
    print(summary.to_string(index=False))
    print("\n=== p_within (cruda) ===\n")
    print(summary_p.to_string(index=False))
    print("\n=== p_within Bonferroni (familia pares x targets) ===\n")
    print(summary_bonf.to_string(index=False))
    writeLog("info", logger, "[experiment_coher_pairs] Fin")


if __name__ == "__main__":
    main()
