# src/psri/validation.py
"""
Metodología de validación del instrumento PSRI contra datasets con ground
truth de calidad conocida. Reutilizable en cualquier dataset de validación
(PhysioNet Challenge 2011, futuros datasets con etiquetas de calidad).

Autor: Enrique
"""
import numpy as np
from sources.common.metrics import compute_classification_metrics


def compare_all_vs_valid(y_true, scores, valid_mask, umbral=0.5, auc_diff_threshold=0.05):
    """Compara métricas sobre todos los datos vs. solo la extracción válida.

    Diagnostica si los fallos de extracción de características explican por
    sí solos la separación observada: si el AUC difiere en más de
    `auc_diff_threshold` entre todos los registros y el subconjunto válido,
    los fallos contribuyen significativamente.

    Args:
        y_true (array-like): Etiquetas reales binarias.
        scores (array-like): Scores continuos del modelo.
        valid_mask (array-like de bool): Máscara de registros con extracción
            válida (sin fallos).
        umbral (float, optional): Umbral de decisión. Por defecto es 0.5.
        auc_diff_threshold (float, optional): Umbral de ΔAUC considerado
            significativo. Por defecto es 0.05.

    Returns:
        tuple: (metrics_all, metrics_valid, diagnosis) donde cada `metrics_*`
            es el dict de `compute_classification_metrics` y `diagnosis`
            contiene `diff_auc`, `diff_gmean` y `fails_drive_auc`.
    """
    metrics_all = compute_classification_metrics(y_true, scores, umbral=umbral)

    y_true_valid = np.asarray(y_true)[valid_mask]
    scores_valid = np.asarray(scores)[valid_mask]
    metrics_valid = compute_classification_metrics(y_true_valid, scores_valid, umbral=umbral)

    diff_auc = metrics_all['auc'] - metrics_valid['auc']
    diff_gmean = metrics_all['gmean'] - metrics_valid['gmean']
    fails_drive_auc = bool(diff_auc > auc_diff_threshold) if not np.isnan(diff_auc) else False

    diagnosis = {
        'diff_auc': diff_auc,
        'diff_gmean': diff_gmean,
        'fails_drive_auc': fails_drive_auc,
    }
    return metrics_all, metrics_valid, diagnosis