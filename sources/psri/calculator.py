# src/psri/calculator.py
"""
Módulo de cálculo puro para el PSRI multimodal.

Contiene las funciones matemáticas del núcleo del PSRI (S_estab, S_coher,
S_cond y el compuesto ponderado) sin dependencia de DataFrames: toda la
lógica opera sobre arrays NumPy, lo que lo hace reutilizable en cualquier
pipeline (PhysioNet, EXIST, K-EmoCon). Incluye además las alternativas de la
ablación metodológica y utilidades de fiabilidad espectral (fase literatura
2026, sin uso activo aún).

Autor: Enrique
"""
from sources.common.common import logger, processControl, writeLog
import numpy as np

def compute_weighted_psri(s_estab, s_coher, s_cond, w1=1/3, w2=1/3, w3=1/3):
    """Combina las tres dimensiones de fiabilidad en el PSRI compuesto.

    Args:
        s_estab (array-like): Fiabilidad de estabilidad intra-trial (S_estab).
        s_coher (array-like): Fiabilidad de coherencia cross-canal (S_coher).
        s_cond (array-like): Fiabilidad de plausibilidad condicional (S_cond).
        w1 (float, optional): Peso de S_estab. Por defecto es 1/3.
        w2 (float, optional): Peso de S_coher. Por defecto es 1/3.
        w3 (float, optional): Peso de S_cond. Por defecto es 1/3.

    Returns:
        ndarray: PSRI compuesto = w1*S_estab + w2*S_coher + w3*S_cond.

    Raises:
        ValueError: Si los pesos no suman 1.0 (tolerancia 1e-6).
    """
    total_w = w1 + w2 + w3
    if not np.isclose(total_w, 1.0, atol=1e-6):
        raise ValueError(f"Los pesos deben sumar 1.0 (actual: {total_w:.4f})")
    return w1 * s_estab + w2 * s_coher + w3 * s_cond


def apply_plausibility_gate(values, s_cond, s_cond_threshold=0.5):
    """Aplica una compuerta dura de plausibilidad (hard mask) a valores.

    Los valores cuya ventana tiene S_cond por debajo del umbral se anulan a
    0.0 en lugar de contribuir parcialmente a la fusión. Modela el gating de
    dos etapas: primero rechazar lo fisiológicamente imposible, después
    ponderar lo restante.

    Args:
        values (array-like): Fiabilidades continuas en [0,1] (o np.nan).
        s_cond (array-like o float): S_cond por ventana en [0,1]; si es
            escalar se aplica el mismo umbral a todas las ventanas.
        s_cond_threshold (float, optional): Umbral de plausibilidad. Por
            defecto es 0.5.

    Returns:
        ndarray: Copia de `values` con 0.0 donde S_cond < umbral. Los NaN se
            conservan.

    Raises:
        ValueError: Si `s_cond` no es escalar ni tiene el mismo tamaño que
            `values`.
    """
    values = np.asarray(values, dtype=float)
    s_cond = np.asarray(s_cond, dtype=float)
    out = values.copy()
    if s_cond.ndim == 0 and values.ndim > 0:
        s_cond = np.full(values.shape, float(s_cond))
    elif s_cond.size != values.size:
        raise ValueError(
            f"s_cond debe ser escalar o tener el mismo tamaño que values "
            f"(values: {values.size}, s_cond: {s_cond.size})"
        )
    out[s_cond < s_cond_threshold] = 0.0
    return out

def compute_psri_gaussian_log(sigma_vals, eps_hard=None, epsilon=1e-9):
    """Calcula S_estab con una función en U (campana gaussiana en log-espacio).

    El Modelo Propuesto: log-transform, z-score robusto (mediana/MAD) y
    campana gaussiana invertida, de modo que tanto las señales planas
    (varianza casi nula) como las ruidosas (varianza extrema) reciben una
    fiabilidad baja.

    Args:
        sigma_vals (array-like): Desviaciones estándar intra-trial.
        eps_hard (float, optional): Umbral por debajo del cual la señal se
            considera plana (R=0). Si es None, se usa el percentil 1 de la
            distribución.
        epsilon (float, optional): Pequeño valor para evitar log(0). Por
            defecto es 1e-9.

    Returns:
        ndarray: Fiabilidades en [0,1], una por valor de `sigma_vals`.
    """
    sigma_vals = np.array(sigma_vals)
    R = np.ones_like(sigma_vals, dtype=float)

    # Si no se especifica eps_hard, usar percentil 1 (excluyendo ceros si los hay)
    if eps_hard is None:
        non_zero = sigma_vals[sigma_vals > 0]
        if len(non_zero) > 0:
            eps_hard = np.percentile(non_zero, 1)
        else:
            eps_hard = 1e-6

    # Caso degenerado: señal plana
    R[sigma_vals < eps_hard] = 0.0

    # Log-transform (solo para valores > eps_hard para evitar log(0))
    valid = sigma_vals >= eps_hard
    if valid.sum() > 0:
        log_sigma = np.log(sigma_vals[valid] + epsilon)

        # Estadísticos robustos (mediana y MAD)
        median_log = np.median(log_sigma)
        mad_log = 1.4826 * np.median(np.abs(log_sigma - median_log))
        if mad_log == 0:
            mad_log = 1.0

        # Z-score robusto
        z = (log_sigma - median_log) / mad_log

        # Fiabilidad como campana gaussiana
        R[valid] = np.exp(-0.5 * z ** 2)

    return R


def compute_psri_mono_decay(sigma_vals, eps_hard=None, epsilon=1e-9):
    """Alternativa A de la ablación metodológica ('El Mono-Decay').

    Mantiene la pre-transformación del modelo propuesto (log + mediana/MAD)
    pero sustituye la campana de Gauss invertida por una función monótona
    decreciente R = 1/(1+|z|). Si su AUC es menor, demuestra que asignar
    máxima fiabilidad a una señal plana es un error conceptual y la forma en
    U es obligatoria.

    Args:
        sigma_vals (array-like): Desviaciones estándar intra-trial.
        eps_hard (float, optional): Umbral de señal plana (None -> percentil
            1 de los no-cero).
        epsilon (float, optional): Pequeño valor para evitar log(0). Por
            defecto es 1e-9.

    Returns:
        ndarray: Fiabilidades en [0,1].
    """
    sigma_vals = np.array(sigma_vals)
    R = np.ones_like(sigma_vals, dtype=float)

    if eps_hard is None:
        non_zero = sigma_vals[sigma_vals > 0]
        if len(non_zero) > 0:
            eps_hard = np.percentile(non_zero, 1)
        else:
            eps_hard = 1e-6

    # Caso degenerado: señal plana
    R[sigma_vals < eps_hard] = 0.0

    valid = sigma_vals >= eps_hard
    if valid.sum() > 0:
        log_sigma = np.log(sigma_vals[valid] + epsilon)

        # Estadísticos robustos idénticos al modelo propuesto
        median_log = np.median(log_sigma)
        mad_log = 1.4826 * np.median(np.abs(log_sigma - median_log))
        if mad_log == 0:
            mad_log = 1.0

        z = (log_sigma - median_log) / mad_log

        # En vez de la campana de Gauss: función monótona decreciente
        R[valid] = 1.0 / (1.0 + np.abs(z))

    return R


def compute_psri_classic_zscore(sigma_vals, eps_hard=None, epsilon=1e-9):
    """Alternativa B de la ablación metodológica ('El Z-Score Clásico').

    Mantiene el logaritmo y la campana de Gauss, pero calcula el Z-score con
    la media y la desviación estándar clásicas en lugar de la mediana y el
    MAD. Si su AUC es menor, demuestra que la media clásica se desplaza con
    sensores rotos (varianzas extremas) y justifica la estadística robusta.

    Args:
        sigma_vals (array-like): Desviaciones estándar intra-trial.
        eps_hard (float, optional): Umbral de señal plana (None -> percentil
            1 de los no-cero).
        epsilon (float, optional): Pequeño valor para evitar log(0). Por
            defecto es 1e-9.

    Returns:
        ndarray: Fiabilidades en [0,1].
    """
    sigma_vals = np.array(sigma_vals)
    R = np.ones_like(sigma_vals, dtype=float)

    if eps_hard is None:
        non_zero = sigma_vals[sigma_vals > 0]
        if len(non_zero) > 0:
            eps_hard = np.percentile(non_zero, 1)
        else:
            eps_hard = 1e-6

    # Caso degenerado: señal plana
    R[sigma_vals < eps_hard] = 0.0

    valid = sigma_vals >= eps_hard
    if valid.sum() > 0:
        log_sigma = np.log(sigma_vals[valid] + epsilon)

        # Estadísticos CLÁSICOS en lugar de los robustos
        mean_log = np.mean(log_sigma)
        std_log = np.std(log_sigma)
        if std_log == 0:
            std_log = 1.0

        z = (log_sigma - mean_log) / std_log

        # Misma campana de Gauss que el modelo propuesto
        R[valid] = np.exp(-0.5 * z ** 2)

    return R


def compute_psri_empirical_percentile(sigma_vals):
    """Alternativa C de la ablación metodológica ('Percentil Empírico').

    Competidor no paramétrico: no asume ninguna distribución, asigna a cada
    ECG el percentil que ocupa su std_signal dentro del conjunto de
    referencia (ranking empírico). Si el modelo propuesto (U de Gauss) le
    gana, demuestra que una forma paramétrica suave generaliza mejor que un
    simple ranking discreto.

    Args:
        sigma_vals (array-like): Desviaciones estándar intra-trial.

    Returns:
        ndarray: Percentiles en (0,1], mismo orden que `sigma_vals`.
    """
    from scipy.stats import rankdata

    sigma_vals = np.array(sigma_vals)
    n = len(sigma_vals)
    if n == 0:
        return np.array([], dtype=float)
    return rankdata(sigma_vals, method="average") / n


def compute_psri_minmax(sigma_vals):
    """Alternativa D de la ablación metodológica ('Min-Max Crudo').

    Normaliza la std_signal linealmente entre el mínimo y el máximo
    observados. Si falla estrepitosamente, demuestra que cualquier
    normalización basada en extremos es frágil ante un sensor con varianza
    extrema que arrastra el máximo a un valor absurdo.

    Args:
        sigma_vals (array-like): Desviaciones estándar intra-trial.

    Returns:
        ndarray: Fiabilidades en [0,1].
    """
    sigma_vals = np.array(sigma_vals, dtype=float)
    n = len(sigma_vals)
    if n == 0:
        return np.array([], dtype=float)

    s_min = sigma_vals.min()
    s_max = sigma_vals.max()
    span = s_max - s_min
    if span <= 0:
        # Caso degenerado: sin rango observable, valor neutral
        return np.full(n, 0.5)
    return (sigma_vals - s_min) / span


def compute_spectral_snr(window, fs, band=(0.5, 40.0)):
    """Calcula el SNR espectral de una ventana cruda.

    Potencia dentro de la banda fisiológica de interés del canal frente a la
    de fuera (ruido y artefactos de alta frecuencia). Complementa S_estab
    distinguiendo una señal plana pero ruidosa de una estable de verdad.

    Args:
        window (array-like): Muestras crudas de la ventana.
        fs (float): Frecuencia de muestreo en Hz.
        band (tuple, optional): (fmin, fmax) en Hz de la banda fisiológica.
            Por defecto es (0.5, 40.0).

    Returns:
        float: SNR en dB, np.nan si la ventana es inválida (menos de 2
            muestras no-NaN o fmax >= Nyquist), o -inf si la señal es plana.
    """
    arr = np.asarray(window, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return np.nan

    fmin, fmax = band
    if fmax >= fs / 2.0:
        return np.nan

    spectrum = np.fft.rfft(arr - arr.mean())
    freqs = np.fft.rfftfreq(len(arr), d=1.0 / fs)

    dc = freqs > 0.0
    power = np.abs(spectrum) ** 2
    total_power = power[dc].sum()
    if total_power <= 0:
        return -np.inf

    signal_power = power[dc & (freqs >= fmin) & (freqs <= fmax)].sum()
    noise_power = total_power - signal_power
    if noise_power <= 0:
        noise_power = np.finfo(float).eps
    return 10.0 * np.log10(signal_power / noise_power)


def snr_to_reliability(snr_db, snr_ref=10.0, sharpness=0.5):
    """Mapea el SNR espectral (dB) a una fiabilidad en [0,1] con una sigmoide.

    La sigmoide está centrada en `snr_ref` (por defecto 10 dB): en ese punto
    vale 0.5, por debajo tiende a 0 y por encima a 1 de forma monótona.

    Args:
        snr_db (float o array-like): SNR en dB. Los NaN se propagan como NaN;
            -inf (señal plana) mapea a 0.0.
        snr_ref (float, optional): Centro de la sigmoide en dB. Por defecto
            es 10.0.
        sharpness (float, optional): Pendiente de la sigmoide. Por defecto es
            0.5.

    Returns:
        ndarray o float: Fiabilidad en [0,1] (misma forma que `snr_db`).
    """
    snr_db = np.asarray(snr_db, dtype=float)
    flat = snr_db == -np.inf
    nan_mask = np.isnan(snr_db)
    R = np.exp(-sharpness * (snr_db - snr_ref))
    R = 1.0 / (1.0 + R)
    R = np.where(flat, 0.0, R)
    R = np.where(nan_mask, np.nan, R)
    return R


def compute_z_score(value, baseline, subject_std):
    """Calcula el Z-score de un valor respecto a su baseline y su std de sujeto.

    Args:
        value (float): Valor observado (ej. HR media en un trial).
        baseline (float): Valor de referencia (ej. media previa del sujeto).
        subject_std (float): Desviación típica del sujeto en esa métrica a lo
            largo de sus trials.

    Returns:
        float: Puntuación Z, o np.nan si `subject_std` es 0 o NaN.
    """
    if subject_std == 0 or np.isnan(subject_std):
        return np.nan
    return (value - baseline) / subject_std


def compute_coherence(z_hr, z_pupil, scale=2.0):
    """Calcula la coherencia (S_coher) entre dos Z-scores.

    Usa un decaimiento exponencial en forma de V de la discrepancia entre
    los Z-scores: discrepancia nula produce máxima coherencia.

    Args:
        z_hr (float): Z-score de la frecuencia cardíaca.
        z_pupil (float): Z-score del diámetro pupilar.
        scale (float, optional): Factor de escala del exponente. Por defecto
            es 2.0.

    Returns:
        float: Valor de coherencia en [0,1], o np.nan si alguno de los
            Z-scores es NaN.
    """
    if np.isnan(z_hr) or np.isnan(z_pupil):
        return np.nan
    return np.exp(-np.abs(z_hr - z_pupil) / scale)

"""
AÑADIR a src/psri/calculator.py (junto a las otras funciones puras del módulo).

Propuesta de compute_s_cond siguiendo el patrón de nomenclatura del framework:
  - S_estab  -> estabilidad intra-trial      (compute_psri_gaussian_log)
  - S_coher  -> coherencia cross-canal        (compute_coherence)
  - S_cond   -> plausibilidad condicional     (ESTA FUNCIÓN)

Definición propuesta: fracción de muestras crudas dentro de la ventana/trial
que caen en un rango fisiológicamente plausible para ese canal. Sirve tanto
como detector de "dispositivo no puesto" (ej. TEMP < 25°C) como de valores
imposibles/artefactuales (ej. HR > 220 bpm).

*** VERIFICAR contra la definición original de la Sección 3.1 antes de usar
en producción -- esto es una reconstrucción razonada a partir del nombre y
del rol que ocupa en compute_weighted_psri, no una recuperación del código
original. ***
"""



def compute_s_cond(window_values, valid_range, min_samples=1):
    """Calcula S_cond por ventana/trial como fracción de muestras plausibles.

    La fracción de muestras crudas de la ventana que caen dentro del rango
    fisiológicamente plausible para el canal. Sirve como detector de
    dispositivo no puesto y de valores imposibles/artefactuales.

    Args:
        window_values (iterable de arrays): Una entrada por ventana/trial,
            cada una con las muestras crudas de ese canal.
        valid_range (tuple): (min, max) del rango fisiológicamente válido.
        min_samples (int, optional): Ventanas con menos muestras que este
            umbral devuelven NaN. Por defecto es 1.

    Returns:
        ndarray: Fiabilidades en [0,1], una por ventana. NaN si la ventana
            no tiene suficientes muestras.
    """
    S_cond = np.full(len(window_values), np.nan, dtype=float)
    for i, w in enumerate(window_values):
        arr = np.asarray(w, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) < min_samples:
            continue
        in_range = (arr >= valid_range[0]) & (arr <= valid_range[1])
        S_cond[i] = in_range.mean()
    return S_cond