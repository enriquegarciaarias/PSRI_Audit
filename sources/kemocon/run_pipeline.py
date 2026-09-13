"""
kemocon/run_pipeline.py
https://zenodo.org/records/3931963

Orquesta el pipeline completo K-EmoCon (Study 2 de PSRI).
Ejecutar con: python -m kemocon.run_pipeline

Autor: Enrique
"""
from sources.common.common import processControl, logger, writeLog
from sources.common.utils import inicioModulo
from sources.kemocon.loader import (
    load_metadata, eligible_subjects, load_e4_subject,
    load_neurosky_polar_subject, load_annotations_subject,
)
from sources.kemocon.aggregator import build_subject_table
from sources.kemocon.build_features import build_full_feature_table, validate_psri_vs_disagreement
from sources.kemocon.sanity_check import load_quality_tables, flag_low_quality_subjects, sanity_check_feature_table


def process_kemocon():
    """Ejecuta el pipeline completo de K-EmoCon.

    Carga metadatos, filtra sujetos elegibles y por calidad, construye las
    tablas por sujeto y la tabla de características, y lanza validaciones y
    diagnósticos (cobertura, leverage, barrido de baseline, independencia de
    componentes). Exporta todos los CSV y figuras al directorio de salida.

    Returns:
        None
    """
    in_dir, out_dir = inicioModulo("process_kemocon")
    meta = load_metadata(in_dir)
    subjects = eligible_subjects(meta)
    writeLog("info", logger, f"Sujetos elegibles: {len(subjects)}")
    subj_times = meta["subjects"].set_index("pid")

    quality = load_quality_tables(in_dir)
    excluded = flag_low_quality_subjects(quality)
    writeLog("info", logger, f"Sujetos excluidos por calidad: {sorted(excluded)}")

    subject_tables = []

    for sid in subjects:
        e4 = load_e4_subject(sid, in_dir)
        if not e4:
            continue
        polar = load_neurosky_polar_subject(sid, in_dir)
        ann = load_annotations_subject(sid, in_dir)

        t0 = min(df["timestamp"].iloc[0] for df in e4.values() if "timestamp" in df.columns)

        # Alineación temporal de anotaciones: la columna 'seconds' es relativa
        # al INICIO DEL DEBATE (startTime), mientras que la fisiología se ancla
        # a t0 (initTime). Este offset expresa ambas series en el mismo reloj.
        start_sec = subj_times.loc[sid, "startTime"] / 1000.0
        ann_offset_s = start_sec - t0

        subject_tables.append(build_subject_table(sid, e4, polar, ann, t0, ann_offset_s=ann_offset_s))

    feature_table = build_full_feature_table(subject_tables)
    feature_table = sanity_check_feature_table(feature_table, excluded)

    def _coverage(col):
        """Calcula la cobertura real (no imputada) de una columna.

        Args:
            col (str): Nombre de la columna.

        Returns:
            str: Texto con la cobertura o aviso de columna no encontrada.
        """
        if col not in feature_table.columns:
            return "columna no encontrada"
        n = len(feature_table)
        non_null = feature_table[col].notna().sum()
        return f"{non_null}/{n} ({100 * non_null / n:.1f}%) ventanas con valor calculado (no imputado)"

    print("\n== Cobertura real (pre-imputación) por componente ==")
    for col in ["S_estab", "S_cond_att", "S_cond_att_acc", "S_cond_att_med",
                "S_cond_att_std", "S_cond_att_med_std",
                "S_cond_self", "S_cond_combined",
                "S_coher_prev1", "S_coher_prev5"]:
        print(f"  {col}: {_coverage(col)}")

    print("\n== [DIAGNÓSTICO] Distribución de self_valence_local_std / "
          "self_arousal_local_std (para investigar si S_cond_self es casi binario) ==")
    for col in ["self_valence_local_std", "self_arousal_local_std"]:
        if col in feature_table.columns:
            s = feature_table[col].dropna()
            frac_zero = (s == 0).mean()
            print(f"  {col}: n={len(s)}, % exactamente 0 = {100 * frac_zero:.1f}%, "
                  f"percentiles [5,25,50,75,95] = "
                  f"{s.quantile([.05, .25, .5, .75, .95]).round(4).tolist()}")
        else:
            print(f"  {col}: columna no encontrada")

    from sources.kemocon.build_features import (
        validate_psri_vs_disagreement_combined,
        validate_components_vs_disagreement,
    )

    validation_combined = validate_psri_vs_disagreement_combined(feature_table)
    print("\n== Validación PSRI vs desacuerdo (BLOQUE ÚNICO, 10 tests -- "
          "consistente con el criterio de corrección del Estudio 1) ==")
    print(validation_combined.to_string(index=False))

    validation_2leg = validate_psri_vs_disagreement_combined(
        feature_table, psri_cols=("psri_composite_2leg_prev1", "psri_composite_2leg_prev5"))
    print("\n== [COMPARACIÓN 2 patas vs 3 patas] Compuesto S_estab+S_coher "
          "(sin S_cond) -- alternativa PSRI_validated de docs/dialogo.txt ==")
    print(validation_2leg.to_string(index=False))

    validation_components = validate_components_vs_disagreement(feature_table)
    print("\n== [DIAGNÓSTICO, bloque de corrección propio, 70 tests] "
          "S_estab / S_cond / S_coher (prev1, prev5) vs desacuerdo ==")
    print(validation_components.to_string(index=False))

    print("\n== [DIAGNÓSTICO] Distribución de local_std de autoanotación (para interpretar S_cond_self) ==")
    for col in ["self_valence_local_std", "self_arousal_local_std"]:
        if col not in feature_table.columns:
            print(f"  {col}: columna no encontrada")
            continue
        s = feature_table[col]
        n_total = len(s)
        n_nan = s.isna().sum()
        n_zero = (s == 0).sum()
        print(f"  {col}: {n_nan}/{n_total} NaN ({100 * n_nan / n_total:.1f}%), "
              f"{n_zero}/{n_total} exactamente 0 ({100 * n_zero / n_total:.1f}%)")
        print(f"    describe: {s.describe().to_dict()}")

    from sources.kemocon.build_features import leverage_diagnostics

    print("\n== [DIAGNÓSTICO] Ventanas por sujeto (tras exclusión de calidad) ==")
    print(feature_table.groupby("subject_id").size().to_string())

    leverage_estab = leverage_diagnostics(feature_table, component_col="S_estab_imputed")
    print("\n== [DIAGNÓSTICO] Leverage por sujeto sobre S_estab "
          "(between vs within-subject, leave-one-subject-out) ==")
    print(leverage_estab.to_string(index=False))

    leverage_coher_prev1 = leverage_diagnostics(feature_table, component_col="S_coher_prev1_imputed")
    print("\n== [DIAGNÓSTICO] Leverage por sujeto sobre S_coher_prev1 (HR-EDA) "
          "(between vs within-subject, leave-one-subject-out) ==")
    print(leverage_coher_prev1.to_string(index=False))

    leverage_coher_prev5 = leverage_diagnostics(feature_table, component_col="S_coher_prev5_imputed")
    print("\n== [DIAGNÓSTICO] Leverage por sujeto sobre S_coher_prev5 (HR-EDA) "
          "(between vs within-subject, leave-one-subject-out) ==")
    print(leverage_coher_prev5.to_string(index=False))

    leverage_scond = {}
    for variant in ["S_cond_att", "S_cond_att_acc", "S_cond_att_med",
                    "S_cond_att_std", "S_cond_att_med_std",
                    "S_cond_self", "S_cond_combined"]:
        lev = leverage_diagnostics(feature_table, component_col=variant)
        leverage_scond[variant] = lev
        print(f"\n== [DIAGNÓSTICO] Leverage por sujeto sobre {variant} "
              "(between vs within-subject, leave-one-subject-out) ==")
        print(lev.to_string(index=False))

    from sources.kemocon.build_features import sweep_s_coher_baseline_window

    sweep = sweep_s_coher_baseline_window(feature_table)
    print("\n== [DIAGNÓSTICO] Barrido de n_prev (ventana de baseline) para S_coher "
          "-- rho_within por tamaño de ventana, valence_var/range ==")
    print(sweep.to_string(index=False))

    from sources.kemocon.build_features import cross_component_independence

    independence = cross_component_independence(feature_table)
    print("\n== Independencia entre componentes del PSRI (Pearson, réplica de la "
          "Tabla 5 de EXIST) -- justifica los pesos iguales (1/3,1/3,1/3) del compuesto ==")
    print(independence.to_string(index=False))

    leverage_estab_vs_cond = leverage_diagnostics(
        feature_table, component_col="S_estab_imputed", target_cols=("S_cond_att_acc",)
    )
    print("\n== [DIAGNÓSTICO] Descomposición between/within del par S_estab-S_cond_att_acc "
          "(r=0.193 pooled -- ¿rasgo entre-sujetos o solapamiento real a nivel de trial?) ==")
    print(leverage_estab_vs_cond.to_string(index=False))

    for variant in ["psri_composite_prev1", "psri_composite_2leg_prev1"]:
        lev_comp = leverage_diagnostics(feature_table, component_col=variant)
        print(f"\n== [COMPARACIÓN 2 vs 3 patas] Leverage de {variant} "
              "(between vs within, leave-one-subject-out) ==")
        print(lev_comp.to_string(index=False))
        lev_comp.to_csv(out_dir / f"kemocon_psri_leverage_{variant}.csv", index=False)

    validation_prev1 = validate_psri_vs_disagreement(feature_table, psri_col="psri_composite_prev1")
    validation_prev5 = validate_psri_vs_disagreement(feature_table, psri_col="psri_composite_prev5")
    print("\n-- (exploratorio, no reportar en el paper) prev1 por separado --")
    print(validation_prev1.to_string(index=False))
    print("\n-- (exploratorio, no reportar en el paper) prev5 por separado --")
    print(validation_prev5.to_string(index=False))

    from sources.kemocon.task_validity import run_task_validity, export_period_assignments

    task_validity = run_task_validity(feature_table, subj_times)
    print("\n== [DIAGNÓSTICO] Task-validity: pre-debate (control negativo) vs debate ==")
    print("Compara S_cond / S_estab / S_coher / controles fisiológicos entre\n"
          "pre-debate (sin interacción) y debate, con modelo mixto + Wilcoxon\n"
          "pareado por sujeto + LOSO + robustez a las primeras ventanas.")
    print(task_validity.to_string(index=False))
    task_validity.to_csv(out_dir / "kemocon_task_validity.csv", index=False)

    # Nivel Q_window del marco de tres niveles de validez: asignación por
    # ventana (reposo pre-debate vs tarea debate) para auditoría.
    period_assign = export_period_assignments(feature_table, subj_times)
    period_assign.to_csv(out_dir / "kemocon_period_assignments.csv", index=False)

    # Análisis de robustez: confounds del efecto entre-sujetos de S_estab
    # (control por rasgo de movimiento/EDA/HR/TEMP). Ver EstudioKemocon.md §4.
    from sources.kemocon.sestab_confounds import run_sestab_confounds
    run_sestab_confounds(feature_table, out_dir)


    feature_table.to_csv(out_dir / "kemocon_feature_table.csv", index=False)
    validation_combined.to_csv(out_dir / "kemocon_psri_validation_combined.csv", index=False)
    validation_2leg.to_csv(out_dir / "kemocon_psri_validation_combined_2leg.csv", index=False)
    validation_components.to_csv(out_dir / "kemocon_psri_validation_components_diagnostic.csv", index=False)
    leverage_estab.to_csv(out_dir / "kemocon_psri_leverage_diagnostic_estab.csv", index=False)
    leverage_coher_prev1.to_csv(out_dir / "kemocon_psri_leverage_diagnostic_coher_prev1.csv", index=False)
    leverage_coher_prev5.to_csv(out_dir / "kemocon_psri_leverage_diagnostic_coher_prev5.csv", index=False)
    for variant, lev in leverage_scond.items():
        lev.to_csv(out_dir / f"kemocon_psri_leverage_diagnostic_{variant.lower()}.csv", index=False)
    sweep.to_csv(out_dir / "kemocon_psri_scoher_baseline_sweep.csv", index=False)
    independence.to_csv(out_dir / "kemocon_psri_component_independence.csv", index=False)
    leverage_estab_vs_cond.to_csv(out_dir / "kemocon_psri_leverage_estab_vs_cond.csv", index=False)

    from sources.kemocon.plots import plot_scoher_baseline_sweep, plot_between_within_decomposition

    plot_scoher_baseline_sweep(
        sweep,
        out_dir / "fig1_scoher_baseline_sweep.png",
        out_dir / "fig1_scoher_baseline_sweep.pdf",
    )
    plot_between_within_decomposition(
        leverage_estab, leverage_coher_prev1,
        out_dir / "fig2_between_within_decomposition.png",
        out_dir / "fig2_between_within_decomposition.pdf",
    )
    print("\n[plots] Figuras generadas en outputs/: "
          "fig1_scoher_baseline_sweep.{png,pdf}, fig2_between_within_decomposition.{png,pdf}")
    validation_prev1.to_csv(out_dir / "kemocon_psri_validation_prev1_exploratory.csv", index=False)
    validation_prev5.to_csv(out_dir / "kemocon_psri_validation_prev5_exploratory.csv", index=False)

if __name__ == "__main__":
    process_kemocon()