# src/common/metrics.py
"""
Métricas de evaluación de clasificación binaria, genéricas y reutilizables
en cualquier parte del proyecto (PSRI, clasificación de sexismo, etc.).
Sin dependencia de dominio ni de I/O.

Autor: Enrique
"""
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_curve, auc


def compute_classification_metrics(y_true, scores, umbral=0.5):
    """Calcula las métricas de clasificación binaria para un conjunto dado.

    Args:
        y_true (array-like): Etiquetas reales binarias (0/1).
        scores (array-like): Puntuaciones continuas (mayor = clase 1).
        umbral (float, optional): Umbral de decisión. Por defecto es 0.5.

    Returns:
        dict: Diccionario con la matriz de confusión (`cm`), exactitud
            (`acc`), sensibilidad (`sens`), especificidad (`spec`),
            G-mean (`gmean`), curvas ROC (`fpr`, `tpr`), AUC (`auc`) y
            tamaño de muestra (`n`). Si no hay datos devuelve métricas
            vacías con `n=0`.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    if len(y_true) == 0:
        return _empty_metrics()

    y_pred = (scores >= umbral).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    gmean = np.sqrt(sens * spec) if sens > 0 and spec > 0 else 0.0

    if len(np.unique(y_true)) < 2:
        fpr, tpr, roc_auc = np.array([]), np.array([]), np.nan
    else:
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)

    return {
        'cm': cm, 'acc': acc, 'sens': sens, 'spec': spec, 'gmean': gmean,
        'fpr': fpr, 'tpr': tpr, 'auc': roc_auc, 'n': len(y_true),
    }


def _empty_metrics():
    """Devuelve un diccionario de métricas vacío (n=0).

    Returns:
        dict: Métricas con valores NaN o arrays vacíos y `n=0`.
    """
    return {
        'cm': np.zeros((2, 2), dtype=int),
        'acc': np.nan, 'sens': np.nan, 'spec': np.nan, 'gmean': np.nan,
        'fpr': np.array([]), 'tpr': np.array([]), 'auc': np.nan, 'n': 0,
    }