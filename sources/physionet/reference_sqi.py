# src/physionet/reference_sqi.py
"""
Índices de calidad de señal (SQI) de referencia de la literatura.

Implementa kSQI (curtosis) y la correlación inter-derivación, los dos índices
diseñados específicamente para ECG contra los que se compara el PSRI en la
validación de PhysioNet.

Autor: Enrique
"""
import numpy as np
from scipy.stats import kurtosis
import wfdb


def compute_ksqi(record_path, flat_lead_threshold=1e-6):
    """Calcula el kSQI (curtosis) de un registro ECG.

    Args:
        record_path (str): Ruta del registro WFDB.
        flat_lead_threshold (float, optional): Umbral de std para considerar
            una derivación plana. Por defecto es 1e-6.

    Returns:
        float o None: Score válido, o None si la señal es degenerada
            (derivación plana / curtosis indefinida), indicando peor calidad.
    """
    record = wfdb.rdrecord(record_path)
    signals = record.p_signal

    std_per_lead = np.std(signals, axis=0)
    if np.any(std_per_lead < flat_lead_threshold):
        return None

    kurt_per_lead = kurtosis(signals, axis=0, fisher=True, bias=False)
    if np.any(~np.isfinite(kurt_per_lead)):
        return None

    return float(np.mean(kurt_per_lead))


def compute_interlead_correlation_sqi(record_path, flat_lead_threshold=1e-6):
    """Calcula la correlación media entre pares de derivaciones del ECG.

    Args:
        record_path (str): Ruta del registro WFDB.
        flat_lead_threshold (float, optional): Umbral de std para considerar
            una derivación plana. Por defecto es 1e-6.

    Returns:
        float, None o np.nan: Score válido, None si hay una derivación plana
            (peor calidad), o np.nan si el registro es legítimamente no
            evaluable (menos de 2 derivaciones).
    """
    record = wfdb.rdrecord(record_path)
    signals = record.p_signal
    n_leads = signals.shape[1]
    if n_leads < 2:
        return np.nan  # no es degradación de calidad, es que no hay pares que correlacionar

    std_per_lead = np.std(signals, axis=0)
    if np.any(std_per_lead < flat_lead_threshold):
        return None

    corr_matrix = np.corrcoef(signals.T)
    iu = np.triu_indices(n_leads, k=1)
    return float(np.mean(corr_matrix[iu]))