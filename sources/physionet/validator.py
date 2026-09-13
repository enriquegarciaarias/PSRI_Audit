# src/physionet/validator.py
# https://physionet.org/content/challenge-2011/1.0.0/
"""
Validación del instrumento PSRI contra PhysioNet/CinC Challenge 2011.

Orquesta las cinco comprobaciones del Estudio 3: comparativa de AUC frente a
los SQIs de referencia, significación estadística (DeLong + bootstrap),
sensibilidad de calibración, análisis del solapamiento estructural y el
estudio de ablación metodológica sobre la transformación de `std_signal`.

Autor: Enrique
"""
from sources.common.common import logger, processControl, writeLog
from sources.common.utils import inicioModulo

import os
import numpy as np

from sources.psri.calculator import (
    compute_psri_gaussian_log,
    compute_psri_mono_decay,
    compute_psri_classic_zscore,
    compute_psri_empirical_percentile,
    compute_psri_minmax,
)
from sources.psri.validation import compare_all_vs_valid
from sources.physionet.loader import load_ground_truth, load_record_ids
from sources.physionet.features import extract_variability
from sources.physionet.reporting import print_validation_report, plot_roc, plot_boxplot_by_class

from sources.common.metrics import compute_classification_metrics
from sources.physionet.reference_sqi import compute_ksqi, compute_interlead_correlation_sqi


def validate_reference_sqi(sqi_func, sqi_name):
    """Valida un SQI de referencia contra el ground truth de PhysioNet.

    Aplica el manejo de casos degenerados (sentinel para derivación plana,
    descarte de no evaluables) y calcula AUC/G-mean sobre el umbral mediana.

    Args:
        sqi_func (callable): Función que recibe la ruta de un registro y
            devuelve el score (float, None o np.nan).
        sqi_name (str): Nombre del SQI, usado en logs.

    Returns:
        dict o None: Métricas de `compute_classification_metrics`, o None si
            no se obtuvo ningún valor finito.
    """
    input_dir, output_dir = inicioModulo("validate_psri")
    data_dir = input_dir / "set-a"
    gt = load_ground_truth(data_dir)
    record_ids = load_record_ids(data_dir)

    raw = []  # (rec_id, val, label) — val: float | None (degenerado) | np.nan (no evaluable)
    for rec_id in record_ids:
        if rec_id not in gt:
            continue
        path = os.path.join(data_dir, rec_id)
        try:
            val = sqi_func(path)
        except Exception as e:
            writeLog("error", logger, f"Error calculando {sqi_name} para {rec_id}: {e}")
            val = np.nan
        raw.append((rec_id, val, gt[rec_id]))

    finite_vals = [v for _, v, _ in raw if isinstance(v, float) and np.isfinite(v)]
    if not finite_vals:
        writeLog("error", logger, f"{sqi_name}: no se obtuvo ningún valor finito. Abortando.")
        return None
    min_finite = min(finite_vals)
    sentinel = min_finite - (abs(min_finite) * 0.1 + 1.0)  # finito, por debajo de todo score real

    scores, y_true = [], []
    n_degenerate = 0
    n_dropped_acceptable = 0
    n_dropped_unacceptable = 0

    for rec_id, val, label in raw:
        if val is None:
            scores.append(sentinel)
            y_true.append(label)
            n_degenerate += 1
        elif isinstance(val, float) and np.isnan(val):
            if label == 1:
                n_dropped_acceptable += 1
            else:
                n_dropped_unacceptable += 1
        else:
            scores.append(val)
            y_true.append(label)

    scores = np.array(scores)
    y_true = np.array(y_true)

    writeLog("info", logger,
             f"{sqi_name}: n_evaluados={len(scores)} | degenerados_reasignados={n_degenerate} "
             f"(sentinel={sentinel:.4f}) | no_evaluables: {n_dropped_acceptable} aceptables, "
             f"{n_dropped_unacceptable} inaceptables")

    n_degenerate_acceptable = sum(1 for _, val, label in raw if val is None and label == 1)
    n_degenerate_unacceptable = sum(1 for _, val, label in raw if val is None and label == 0)
    writeLog("info", logger,
             f"{sqi_name}: degenerados por clase -> {n_degenerate_acceptable} aceptables, "
             f"{n_degenerate_unacceptable} inaceptables (de {n_degenerate} totales)")

    metrics = compute_classification_metrics(y_true, scores, umbral=np.median(scores))
    writeLog("info", logger, f"{sqi_name}: AUC={metrics['auc']:.4f} G-mean={metrics['gmean']:.4f}")
    return metrics


def compare_psri_vs_reference_sqis():
    """Compara el AUC del PSRI (std_signal) contra los SQIs de referencia.

    Ejecuta `validate_psri` y `validate_reference_sqi` para kSQI y la
    Correlación inter-derivación sobre el mismo set-a, e imprime la tabla
    comparativa.

    Returns:
        dict: {'psri': ..., 'ksqi': ..., 'interlead_corr': ...} con las
            métricas de cada método.
    """
    results_psri = validate_psri(feature_method='std_signal', plot_roc_curve=False,
                                  plot_box=False, verbose=False)
    results_ksqi = validate_reference_sqi(compute_ksqi, 'kSQI (curtosis)')
    results_corr = validate_reference_sqi(compute_interlead_correlation_sqi,
                                           'Correlación entre derivaciones')

    print("\n" + "=" * 78)
    print("=== COMPARATIVA: PSRI vs SQIs DE REFERENCIA (PhysioNet Challenge 2011) ===")
    print("=" * 78)
    print(f"{'Método':32s} | {'AUC':>8s} | {'G-mean':>8s}")
    print("-" * 78)
    print(f"{'PSRI (std_signal)':32s} | {results_psri['auc_all']:>8.4f} | {results_psri['gmean_all']:>8.4f}")
    print(f"{'kSQI (curtosis)':32s} | {results_ksqi['auc']:>8.4f} | {results_ksqi['gmean']:>8.4f}")
    print(f"{'Correlación entre derivaciones':32s} | {results_corr['auc']:>8.4f} | {results_corr['gmean']:>8.4f}")
    print("=" * 78)

    return {'psri': results_psri, 'ksqi': results_ksqi, 'interlead_corr': results_corr}

def _extract_all_variability(data_dir, record_ids, gt, feature_method):
    """Extrae la variabilidad y el flag de fallo de todos los registros.

    Args:
        data_dir (Path): Directorio con los registros.
        record_ids (list): IDs de registros.
        gt (dict): Mapa {record_id: etiqueta}.
        feature_method (str): Método de extracción de variabilidad.

    Returns:
        tuple: (variability_values, y_true, fail_flags) como arrays numpy.
    """
    variability_values, y_true, fail_flags = [], [], []

    for rec_id in record_ids:
        if rec_id not in gt:
            continue
        path = os.path.join(data_dir, rec_id)
        try:
            val, fail = extract_variability(path, method=feature_method, return_fail_flag=True)
        except Exception as e:
            writeLog("error", logger, f"Error inesperado procesando {rec_id}: {e}")
            val, fail = 0.0, True
        variability_values.append(val)
        y_true.append(gt[rec_id])
        fail_flags.append(fail)

    return np.array(variability_values), np.array(y_true), np.array(fail_flags)


def validate_psri(umbral=0.5, feature_method='std_signal', plot_roc_curve=True,
                  plot_box=True, verbose=True):
    """Valida el PSRI (función en U) sobre PhysioNet Challenge 2011.

    Args:
        umbral (float, optional): Umbral para clasificar binariamente (score
            >= umbral -> aceptable). Por defecto es 0.5.
        feature_method (str, optional): Método de variabilidad ('std_signal',
            'std_window', 'rr_std'). Por defecto es 'std_signal'.
        plot_roc_curve (bool, optional): Si guardar la figura ROC. Por
            defecto es True.
        plot_box (bool, optional): Si guardar el boxplot por clase. Por
            defecto es True.
        verbose (bool, optional): Si imprimir el reporte completo por consola.
            Por defecto es True.

    Returns:
        dict o None: Diccionario con y_true, y_scores, fail_count,
            fail_indices, auc_all, gmean_all, auc_valid, gmean_valid,
            valid_count y valid_mask; None si no se pudo extraer variabilidad.
    """
    input_dir, output_dir = inicioModulo("validate_psri")
    data_dir = input_dir / "set-a"

    gt = load_ground_truth(data_dir)
    record_ids = load_record_ids(data_dir)
    writeLog("info", logger, f"Total de registros listados: {len(record_ids)} | Con etiqueta: {len(gt)}")

    variability_values, y_true, fail_flags = _extract_all_variability(
        data_dir, record_ids, gt, feature_method
    )

    if len(variability_values) == 0:
        writeLog("error", logger, "No se pudo extraer variabilidad de ningún registro. Abortando.")
        return None

    valid_mask = ~fail_flags
    fail_count = int(np.sum(fail_flags))

    # Cálculo del score PSRI sobre todos los datos (incluye fallos, que ya vienen a 0.0)
    scores = compute_psri_gaussian_log(variability_values, eps_hard=None)

    metrics_all, metrics_valid, diagnosis = compare_all_vs_valid(
        y_true, scores, valid_mask, umbral=umbral
    )

    if verbose:
        print_validation_report(
            feature_method, umbral, fail_count, len(variability_values),
            metrics_all, metrics_valid, diagnosis
        )

    if plot_roc_curve:
        plot_roc(feature_method, metrics_all, metrics_valid)

    if plot_box:
        plot_boxplot_by_class(feature_method, y_true, scores)

    return {
        'y_true': y_true,
        'y_scores': scores,
        'fail_count': fail_count,
        'fail_indices': np.where(fail_flags)[0],
        'auc_all': metrics_all['auc'],
        'gmean_all': metrics_all['gmean'],
        'auc_valid': metrics_valid['auc'],
        'gmean_valid': metrics_valid['gmean'],
        'valid_count': int(np.sum(valid_mask)),
        'valid_mask': valid_mask,
    }


def compare_feature_methods(methods=('std_signal', 'rr_std'), umbral=0.5,
                            plot_roc_curve=False, plot_box=False, verbose=False):
    """Compara el PSRI con distintos métodos de extracción de variabilidad.

    Args:
        methods (tuple, optional): Métodos a comparar. Por defecto
            ('std_signal', 'rr_std').
        umbral (float, optional): Umbral de decisión. Por defecto es 0.5.
        plot_roc_curve (bool, optional): Si guardar figuras ROC.
        plot_box (bool, optional): Si guardar boxplots.
        verbose (bool, optional): Si imprimir reportes completos.

    Returns:
        dict: {feature_method: resultado de `validate_psri`}.
    """
    from sources.physionet.reporting import print_comparison_table

    results = {}
    for method in methods:
        writeLog("info", logger, f"Validando PSRI con método: {method}")
        results[method] = validate_psri(
            umbral=umbral, feature_method=method,
            plot_roc_curve=plot_roc_curve, plot_box=plot_box, verbose=verbose
        )

    print_comparison_table(results)
    return results

def process_physio():
    """Orquesta las cinco validaciones del Estudio 3 (PhysioNet).

    Ejecuta la comparativa de AUC, la significación estadística, la
    sensibilidad de calibración, el análisis de solapamiento y la ablación
    metodológica, en ese orden.

    Returns:
        None
    """
    compare_psri_vs_reference_sqis()
    compare_psri_vs_reference_sqis_with_significance()
    check_psri_calibration_sensitivity()
    analyze_psri_overlap_limitation()
    run_ablation_study()
    return

    resultados = compare_feature_methods(
        methods=('std_signal', 'rr_std'),
        umbral=0.5,
        plot_roc_curve=True,   # guarda una figura ROC por método (roc_physionet_<method>.png)
        plot_box=True,         # guarda un boxplot por método (boxplot_psri_<method>.png)
        verbose=True           # imprime el reporte completo (todos/válidos) de cada método
    )

    # 'resultados' ya trae la tabla comparativa impresa por print_comparison_table.
    # Si necesitáis los datos crudos para el paper (p. ej. exportar a CSV/LaTeX):
    for method, r in resultados.items():
        writeLog("info", logger,
                 f"{method}: AUC(todos)={r['auc_all']:.4f} AUC(válidos)={r['auc_valid']:.4f} "
                 f"fallos={r['fail_count']}/{len(r['y_true'])}")

    compare_psri_vs_reference_sqis()


def compare_psri_vs_reference_sqis_with_significance(feature_method='std_signal',
                                                     n_boot=2000, plot_figure=True):
    """Compara PSRI vs SQIs con significación estadística (DeLong + bootstrap).

    Recalcula los tres scores por registro para garantizar una comparación
    pareada (misma composición de registros) y añade PR-AUC y figuras ROC.

    Args:
        feature_method (str, optional): Método de variabilidad. Por defecto
            es 'std_signal'.
        n_boot (int, optional): Número de réplicas de bootstrap. Por defecto
            es 2000.
        plot_figure (bool, optional): Si generar las figuras ROC. Por defecto
            es True.

    Returns:
        dict: Resultados con 'pr_auc' y una entrada por SQI comparado
            (DeLong + bootstrap).
    """
    from sources.physionet.auc_comparison import (
        delong_roc_test, bootstrap_auc_diff, plot_roc_comparison, plot_roc_comparison_with_thresholds
    )

    input_dir, output_dir = inicioModulo("validate_psri")
    data_dir = input_dir / "set-a"
    gt = load_ground_truth(data_dir)
    record_ids = load_record_ids(data_dir)

    per_record = {}
    for rec_id in record_ids:
        if rec_id not in gt:
            continue
        per_record[rec_id] = {"label": gt[rec_id]}

    # --- PSRI: reutiliza _extract_all_variability, igual que validate_psri() ---
    variability_values, _, _ = _extract_all_variability(
        data_dir, list(per_record.keys()), gt, feature_method
    )
    psri_scores_all = compute_psri_gaussian_log(variability_values, eps_hard=None)
    for rec_id, score in zip(per_record.keys(), psri_scores_all):
        per_record[rec_id]["psri"] = float(score)

    # --- kSQI y Correlación: misma lógica de sentinel que validate_reference_sqi() ---
    for sqi_func, key in [(compute_ksqi, "ksqi"), (compute_interlead_correlation_sqi, "corr")]:
        raw = []
        for rec_id in per_record:
            path = os.path.join(data_dir, rec_id)
            try:
                val = sqi_func(path)
            except Exception as e:
                writeLog("error", logger, f"Error calculando {key} para {rec_id}: {e}")
                val = np.nan
            raw.append((rec_id, val))

        finite_vals = [v for _, v in raw if isinstance(v, float) and np.isfinite(v)]
        min_finite = min(finite_vals)
        sentinel = min_finite - (abs(min_finite) * 0.1 + 1.0)

        for rec_id, val in raw:
            if val is None:
                per_record[rec_id][key] = sentinel
            elif isinstance(val, float) and np.isnan(val):
                per_record[rec_id][key] = None  # no evaluable -> descartado de la intersección
            else:
                per_record[rec_id][key] = float(val)

    # --- Intersección: solo registros con los 3 scores disponibles ---
    common_ids = [rid for rid, d in per_record.items()
                  if d.get("psri") is not None and d.get("ksqi") is not None
                  and d.get("corr") is not None]

    n_dropped = len(per_record) - len(common_ids)
    writeLog("info", logger,
             f"Comparación pareada: {len(common_ids)}/{len(per_record)} registros con los 3 "
             f"scores disponibles ({n_dropped} descartados por no-evaluable en algún SQI)")

    labels = np.array([per_record[rid]["label"] for rid in common_ids])
    psri_scores = np.array([per_record[rid]["psri"] for rid in common_ids], dtype=float)
    ksqi_scores = np.array([per_record[rid]["ksqi"] for rid in common_ids], dtype=float)
    corr_scores = np.array([per_record[rid]["corr"] for rid in common_ids], dtype=float)

    results = {}
    # --- Punto 6: PR-AUC como comprobación adicional bajo desbalance de clases ---
    # (773 aceptables / 225 inaceptables, ~77%/23% -- el ROC-AUC puede ser
    # optimista bajo desbalance; PR-AUC (average precision) es más sensible
    # a los falsos positivos sobre la clase minoritaria)
    from sklearn.metrics import average_precision_score

    pr_auc_psri = average_precision_score(labels, psri_scores)
    pr_auc_ksqi = average_precision_score(labels, ksqi_scores)
    pr_auc_corr = average_precision_score(labels, corr_scores)

    writeLog("info", logger,
             f"PR-AUC (average precision, clase positiva='acceptable', "
             f"n={len(labels)}, {int(labels.sum())} positivos / "
             f"{int((1 - labels).sum())} negativos): "
             f"PSRI={pr_auc_psri:.4f} | kSQI={pr_auc_ksqi:.4f} | "
             f"Correlación inter-derivación={pr_auc_corr:.4f}")

    results["pr_auc"] = {
        "psri": pr_auc_psri, "ksqi": pr_auc_ksqi, "corr": pr_auc_corr,
    }

    for name, scores_b in [("kSQI", ksqi_scores), ("Correlación inter-derivación", corr_scores)]:
        d = delong_roc_test(labels, psri_scores, scores_b)
        b = bootstrap_auc_diff(labels, psri_scores, scores_b, n_boot=n_boot)
        writeLog("info", logger,
                 f"PSRI vs {name}: DeLong diff={d['diff']:.4f} z={d['z']:.3f} p={d['p_value']:.4g} | "
                 f"Bootstrap diff={b['diff']:.4f} IC95%=[{b['ci_low']:.4f},{b['ci_high']:.4f}] "
                 f"p={b['p_value']:.4g}")
        results[name] = {"delong": d, "bootstrap": b}

    if plot_figure:
        plot_roc_comparison(
            labels,
            {"PSRI (std_signal)": psri_scores, "kSQI": ksqi_scores,
             "Correlación inter-derivación": corr_scores},
            out_png=output_dir / "fig_physionet_roc_comparison.png",
            out_pdf=output_dir / "fig_physionet_roc_comparison.pdf",
        )
        plot_roc_comparison_with_thresholds(
            labels,
            {"PSRI (std_signal)": psri_scores, "kSQI": ksqi_scores, "Correlación inter-derivación": corr_scores},
            out_png="outputs/fig_physionet_roc_thresholds.png",
            out_pdf="outputs/fig_physionet_roc_thresholds.pdf",
            operating_thresholds={"PSRI (std_signal)": 0.5},  # kSQI/Correlación usan su propia mediana automáticamente
        )


    return results


def check_psri_calibration_sensitivity(feature_method='std_signal', n_splits=200,
                                       calib_frac=0.5, in_sample_auc=0.887,
                                       plot_figure=True):
    """Comprueba cuánto del AUC in-sample depende de calibrar sobre la muestra.

    PSRI calibra su mediana/MAD/eps_hard sobre la misma población que puntúa
    (ventaja estructural frente a los SQIs de fórmula cerrada). Este chequeo
    evalúa el AUC fuera de muestra con particiones calibración/evaluación.

    Args:
        feature_method (str, optional): Método de variabilidad. Por defecto
            es 'std_signal'.
        n_splits (int, optional): Número de particiones aleatorias. Por
            defecto es 200.
        calib_frac (float, optional): Fracción de calibración. Por defecto
            es 0.5.
        in_sample_auc (float, optional): AUC in-sample de referencia. Por
            defecto es 0.887.
        plot_figure (bool, optional): Si generar la figura. Por defecto es
            True.

    Returns:
        dict: Resultado de `split_half_sensitivity`.
    """
    from sources.physionet.calibration_sensitivity import (
        split_half_sensitivity, plot_calibration_sensitivity,
    )

    input_dir, output_dir = inicioModulo("validate_psri")
    data_dir = input_dir / "set-a"
    gt = load_ground_truth(data_dir)
    record_ids = load_record_ids(data_dir)

    variability_values, y_true, fail_flags = _extract_all_variability(
        data_dir, record_ids, gt, feature_method
    )

    result = split_half_sensitivity(
        variability_values, y_true, n_splits=n_splits, calib_frac=calib_frac
    )

    writeLog("info", logger,
             f"Sensibilidad de calibración PSRI ({n_splits} splits aleatorios "
             f"{int(calib_frac * 100)}/{int((1 - calib_frac) * 100)}): "
             f"AUC fuera de muestra media={result['mean']:.4f} "
             f"mediana={result['median']:.4f} std={result['std']:.4f} "
             f"IC95%=[{result['ci_low']:.4f},{result['ci_high']:.4f}] "
             f"(splits válidos: {result['n_valid_splits']}/{n_splits}) "
             f"| AUC in-sample original={in_sample_auc:.4f} (referencia)")

    if plot_figure:
        plot_calibration_sensitivity(
            result, in_sample_auc=in_sample_auc,
            out_png=output_dir / "fig_physionet_calibration_sensitivity.png",
            out_pdf=output_dir / "fig_physionet_calibration_sensitivity.pdf",
        )

    return result


# AÑADIR a src/sources/physionet/validator.py
# (junto a las funciones de comparación anteriores, reutilizando los mismos
#  imports ya presentes: os, np, logger, writeLog, inicioModulo,
#  load_ground_truth, load_record_ids, _extract_all_variability,
#  compute_psri_gaussian_log)

def analyze_psri_overlap_limitation(feature_method='std_signal', threshold=0.5,
                                    plot_figure=True):
    """Comprueba si los errores del PSRI se concentran en la zona de solapamiento.

    La limitación estructural de reducir toda la señal a un único escalar
    (std_signal): este análisis localiza la zona donde las distribuciones de
    las dos clases se solapan y evalúa si ahí caen los errores.

    Args:
        feature_method (str, optional): Método de variabilidad. Por defecto
            es 'std_signal'.
        threshold (float, optional): Umbral de clasificación del PSRI. Por
            defecto es 0.5.
        plot_figure (bool, optional): Si generar la figura. Por defecto es
            True.

    Returns:
        dict: Resultado de `analyze_classification_overlap`.
    """
    from sources.physionet.overlap_analysis import (
        analyze_classification_overlap, plot_overlap_diagnostic,
    )

    input_dir, output_dir = inicioModulo("validate_psri")
    data_dir = input_dir / "set-a"
    gt = load_ground_truth(data_dir)
    record_ids = load_record_ids(data_dir)

    variability_values, y_true, fail_flags = _extract_all_variability(
        data_dir, record_ids, gt, feature_method
    )
    psri_scores = compute_psri_gaussian_log(variability_values, eps_hard=None)

    result = analyze_classification_overlap(
        variability_values, y_true, psri_scores, threshold=threshold
    )

    lo_std, hi_std = result["overlap_zone_std_signal"]
    writeLog("info", logger,
             f"Zona de solapamiento std_signal (P5-P95 de ambas clases): "
             f"[{lo_std:.4f}, {hi_std:.4f}] | "
             f"mal clasificados en esa zona: {result['rate_misclass_in_overlap']:.1%} "
             f"(n={result['n_misclassified']}) | "
             f"bien clasificados en esa zona: {result['rate_correct_in_overlap']:.1%} "
             f"(n={result['n_correct']}) | "
             f"tasa global en zona: {result['rate_overall_in_overlap']:.1%}")

    writeLog("info", logger,
             f"Desglose por tipo de error -- "
             f"FP (n={result['n_fp']}) en zona: {result['rate_fp_in_overlap']:.1%} "
             f"(baseline de inaceptables: {result['rate_unacceptable_baseline']:.1%}) | "
             f"FN (n={result['n_fn']}) en zona: {result['rate_fn_in_overlap']:.1%} "
             f"(baseline de aceptables: {result['rate_acceptable_baseline']:.1%})")

    if plot_figure:
        plot_overlap_diagnostic(
            result,
            out_png=output_dir / "fig_physionet_overlap_diagnostic.png",
            out_pdf=output_dir / "fig_physionet_overlap_diagnostic.pdf",
        )

    return result

# En process_physio(), añadir tras las funciones anteriores:
#
#     compare_psri_vs_reference_sqis()
#     compare_psri_vs_reference_sqis_with_significance()
#     check_psri_calibration_sensitivity()
#     analyze_psri_overlap_limitation()
#     run_ablation_study()
#     return

ABLATION_VARIANTS = [
    # (nombre, etiqueta normalización, forma funcional, función)
    ("Propuesta (PSRI)", "Log + Mediana/MAD", "Gaussiana Invertida (U)",
     compute_psri_gaussian_log),
    ("Alt. A: Mono-Decay", "Log + Mediana/MAD", "Monótona Decreciente",
     compute_psri_mono_decay),
    ("Alt. B: Z-Score Clásico", "Log + Media/Std", "Gaussiana Invertida (U)",
     compute_psri_classic_zscore),
    ("Alt. C: Percentil Empírico", "Ninguna (Ranking)", "Escalonado No Paramétrico",
     compute_psri_empirical_percentile),
    ("Alt. D: Min-Max Crudo", "Mínimo/Máximo", "Lineal Acotada [0,1]",
     compute_psri_minmax),
]


def run_ablation_study(feature_method='std_signal', n_boot=2000, plot_figure=True):
    """Ejecuta el estudio de ablación metodológica sobre la transformación de σ.

    Todas las variantes reciben exactamente el mismo input (std_signal por
    registro) y solo difieren en la transformación estadística aplicada.
    Compara el Modelo Propuesto contra sus alternativas con AUC, DeLong y
    bootstrap pareado, e imprime/exporta la tabla LaTeX y la ROC superpuesta.

    Args:
        feature_method (str, optional): Método de variabilidad. Por defecto
            es 'std_signal'.
        n_boot (int, optional): Réplicas de bootstrap. Por defecto es 2000.
        plot_figure (bool, optional): Si generar la ROC superpuesta. Por
            defecto es True.

    Returns:
        list: Filas de resultados por variante (AUC y comparaciones).
    """
    from sources.physionet.auc_comparison import (
        delong_roc_test, bootstrap_auc_diff, plot_roc_comparison,
    )
    from sklearn.metrics import roc_auc_score

    input_dir, output_dir = inicioModulo("validate_psri")
    data_dir = input_dir / "set-a"
    gt = load_ground_truth(data_dir)
    record_ids = load_record_ids(data_dir)

    variability_values, y_true, fail_flags = _extract_all_variability(
        data_dir, record_ids, gt, feature_method
    )
    labels = np.asarray(y_true)
    valid_mask = ~np.asarray(fail_flags)

    # Todas las variantes puntúan el MISMO array de std_signal
    scores_dict = {}
    for name, _, _, func in ABLATION_VARIANTS:
        scores_dict[name] = func(variability_values)

    proposed_name = "Propuesta (PSRI)"

    rows = []
    for name, scores in scores_dict.items():
        auc_all = float(roc_auc_score(labels, scores))
        auc_valid = float(roc_auc_score(labels[valid_mask], scores[valid_mask]))
        row = {"variante": name, "auc_all": auc_all, "auc_valid": auc_valid}
        if name != proposed_name:
            d = delong_roc_test(labels, scores_dict[proposed_name], scores)
            b = bootstrap_auc_diff(labels, scores_dict[proposed_name], scores, n_boot=n_boot)
            row["delong_diff"] = d["diff"]
            row["delong_p"] = d["p_value"]
            row["boot_diff"] = b["diff"]
            row["boot_ci"] = (b["ci_low"], b["ci_high"])
            row["boot_p"] = b["p_value"]
            writeLog("info", logger,
                     f"Ablación: {name} vs {proposed_name} -> "
                     f"DeLong diff={d['diff']:.4f} p={d['p_value']:.4g} | "
                     f"Bootstrap diff={b['diff']:.4f} "
                     f"IC95%=[{b['ci_low']:.4f},{b['ci_high']:.4f}] p={b['p_value']:.4g}")
        rows.append(row)

    _print_ablation_table(rows, labels)
    _print_ablation_latex(rows, output_dir, n_records=len(labels))

    if plot_figure:
        plot_roc_comparison(
            labels, scores_dict,
            out_png=output_dir / "fig_physionet_ablation_roc.png",
            out_pdf=output_dir / "fig_physionet_ablation_roc.pdf",
        )

    return rows


def _print_ablation_table(rows, labels):
    """Imprime la tabla comparativa de la ablación por consola.

    Args:
        rows (list): Filas de resultados de `run_ablation_study`.
        labels (array-like): Etiquetas reales, para el conteo por clase.

    Returns:
        None
    """
    n = len(labels)
    n_pos = int(labels.sum())
    print("\n" + "=" * 92)
    print(f"=== ABLATION STUDY: TRANSFORMACIÓN DE std_signal (n={n}, "
          f"{n_pos} aceptables / {n - n_pos} inaceptables) ===")
    print("=" * 92)
    header = (f"{'Variante':24s} | {'AUC(todos)':>10s} | {'AUC(válid)':>10s} | "
              f"{'ΔAUC':>8s} | {'p (DeLong)':>12s} | {'IC95% bootstrap':>18s}")
    print(header)
    print("-" * len(header))
    for row in rows:
        if "delong_diff" in row:
            print(f"{row['variante']:24s} | {row['auc_all']:>10.4f} | {row['auc_valid']:>10.4f} | "
                  f"{row['delong_diff']:>+8.4f} | {row['delong_p']:>12.4g} | "
                  f"[{row['boot_ci'][0]:.4f}, {row['boot_ci'][1]:.4f}]")
        else:
            print(f"{row['variante']:24s} | {row['auc_all']:>10.4f} | {row['auc_valid']:>10.4f} | "
                  f"{'—':>8s} | {'—':>12s} | {'—':>18s}")
    print("=" * 92)
    print("ΔAUC = AUC(Propuesta) - AUC(Alternativa) | p(DeLong): test de DeLong para AUCs correlacionadas.")
    print("IC95% bootstrap: intervalo de confianza de la diferencia de AUC (2000 réplicas pareadas).")


def _print_ablation_latex(rows, output_dir, n_records):
    """Genera y guarda la tabla LaTeX de la matriz de ablación.

    Args:
        rows (list): Filas de resultados de `run_ablation_study`.
        output_dir (Path): Directorio donde guardar el .tex.
        n_records (int): Número de registros, usado en la caption.

    Returns:
        None
    """
    meta = {name: (norm, forma) for name, norm, forma, _ in ABLATION_VARIANTS}
    proposed_name = "Propuesta (PSRI)"

    lines = []
    lines.append("% Matriz de ablación sobre la transformación de $\\sigma$ (std\\_signal).")
    lines.append("% Todas las variantes operan exclusivamente sobre la desviación estándar cruda")
    lines.append("% como única entrada, sin conocimiento morfológico del ECG.")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Diseño de ablación sobre la transformación de $\\sigma$ (std\\_signal). "
                 f"Todas las variantes operan exclusivamente sobre la desviación estándar cruda como "
                 f"única entrada, sin conocimiento morfológico del ECG. AUC sobre PhysioNet/CinC 2011 "
                 f"(n={n_records} registros); $p$ del test de DeLong frente a la propuesta.}}")
    lines.append("\\label{tab:ablation_std}")
    lines.append("\\begin{tabular}{@{}llllll@{}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Variante} & \\textbf{Normalización} & \\textbf{Forma Funcional} & "
                 "\\textbf{AUC} & $\\mathbf{\\Delta}$\\textbf{AUC} & \\textbf{p (DeLong)} \\\\")
    lines.append("\\midrule")

    for i, row in enumerate(rows):
        name = row["variante"]
        norm, forma = meta[name]
        auc = row["auc_all"]
        if name == proposed_name:
            lines.append(f"\\textbf{{{name}}} & {norm} & {forma} & {auc:.4f} & — & — \\\\")
        else:
            diff = row["delong_diff"]
            p = row["delong_p"]
            p_str = f"{p:.4g}" if p >= 1e-4 else "< $10^{-4}$"
            lines.append(f"{name} & {norm} & {forma} & {auc:.4f} & {diff:+.4f} & {p_str} \\\\")
        if name == proposed_name and i < len(rows) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)
    out_tex = output_dir / "ablation_study_table.tex"
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(latex, encoding="utf-8")
    writeLog("info", logger, f"Tabla LaTeX de ablación guardada en: {out_tex}")

    print("\n=== TABLA LATEX PARA EL PAPER ===")
    print(latex)