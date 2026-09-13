# src/physionet/features.py
"""
Extracción de métricas de variabilidad de señales ECG (PhysioNet Challenge 2011).

Dado un registro WFDB, extrae un escalar de variabilidad por tres métodos
(`std_signal`, `std_window` y `rr_std`), que sirve de entrada al PSRI.

Autor: Enrique
"""
from sources.common.common import logger, processControl, writeLog
import wfdb
import numpy as np

def extract_variability(record_path, method='std_signal', return_fail_flag=False):
    """Extrae una métrica de variabilidad de la señal ECG de un registro.

    Args:
        record_path (str): Ruta del registro WFDB (sin extensión).
        method (str, optional): Método de variabilidad: 'std_signal',
            'std_window' o 'rr_std'. Por defecto es 'std_signal'.
        return_fail_flag (bool, optional): Si es True devuelve
            (valor, fail_flag) donde fail_flag indica que el método no pudo
            extraer una variabilidad válida. Por defecto es False.

    Returns:
        float o tuple: Valor de variabilidad; si `return_fail_flag` es True,
            devuelve la tupla (valor, bool).

    Raises:
        ValueError: Si el método no está soportado.
    """
    record = wfdb.rdrecord(record_path)
    signals = record.p_signal
    fs = record.fs

    if method == 'std_signal':
        std_leads = np.std(signals, axis=0)
        val = np.mean(std_leads)
        return (val, False) if return_fail_flag else val

    elif method == 'std_window':
        window_samples = int(fs)
        n_windows = signals.shape[0] // window_samples
        std_windows = []
        for i in range(n_windows):
            window = signals[i*window_samples:(i+1)*window_samples, :]
            std_window = np.std(window, axis=0).mean()
            std_windows.append(std_window)
        val = np.median(std_windows) if std_windows else 0.0
        return (val, False) if return_fail_flag else val

    elif method == 'rr_std':
        try:
            from wfdb import processing
            lead_index = 1 if signals.shape[1] > 1 else 0
            signal = signals[:, lead_index]
            # API actual de wfdb: qrs_detect fue renombrado a xqrs_detect
            qrs_inds = processing.xqrs_detect(sig=signal, fs=fs, verbose=False)
            if len(qrs_inds) < 2:
                if return_fail_flag:
                    return (0.0, True)
                else:
                    return 0.0
            rr_intervals = np.diff(qrs_inds) / fs
            val = np.std(rr_intervals)
            return (val, False) if return_fail_flag else val
        except Exception as e:
            if return_fail_flag:
                return (0.0, True)
            else:
                writeLog("error", logger, f"Advertencia: error en detección de picos para {record_path}: {e}")
                return 0.0
    else:
        raise ValueError(f"Método {method} no soportado.")