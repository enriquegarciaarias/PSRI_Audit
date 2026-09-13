# AÑADIR a src/sources/physionet/ como calibration_sensitivity.py
"""
physionet/calibration_sensitivity.py

Punto 3 de la revisión metodológica del Experimento 2 (validación PhysioNet):
kSQI y la Correlación inter-derivación son fórmulas CERRADAS -- no tienen
ningún parámetro ajustado a los 998 registros de set-a. compute_psri_gaussian_log,
en cambio, calcula su mediana/MAD/eps_hard de referencia SOBRE LA MISMA
población que luego puntúa -- una calibración interna a la muestra que los
otros dos índices no tienen. No es fuga de etiqueta (el cálculo es no
supervisado, nunca usa acceptable/unacceptable), pero sí es una ventaja
estructural que podría explicar parte del AUC=0.887, no solo "el mecanismo
es mejor".

Este módulo cuantifica cuánto: separa estrictamente un conjunto de
CALIBRACIÓN (fija mediana/MAD/eps_hard) de un conjunto de EVALUACIÓN
(se puntúa con esa referencia externa, nunca con la suya propia), repetido
en muchas particiones aleatorias para obtener una distribución del AUC
fuera de muestra, comparable contra el AUC in-sample original (0.887).

Autor: Enrique
"""

from pathlib import Path
import numpy as np


def compute_psri_scores_with_reference(sigma_vals, ref_sigma_vals, epsilon: float = 1e-9):
    """Puntúa un conjunto de evaluación con una referencia de calibración externa.

    Misma fórmula que `compute_psri_gaussian_log` (mediana/MAD robustas en
    log-espacio + campana gaussiana), pero eps_hard/mediana/MAD se calculan
    sobre `ref_sigma_vals` (calibración) y se aplican a `sigma_vals`
    (evaluación): nunca se calibra un registro con su propio conjunto.

    Args:
        sigma_vals (array-like): Valores del conjunto de EVALUACIÓN.
        ref_sigma_vals (array-like): Valores del conjunto de CALIBRACIÓN.
        epsilon (float, optional): Pequeño valor para evitar log(0). Por
            defecto es 1e-9.

    Returns:
        ndarray: Fiabilidades en [0,1] para `sigma_vals`.

    Raises:
        ValueError: Si el conjunto de calibración no tiene valores válidos
            por encima de eps_hard.
    """
    sigma_vals = np.asarray(sigma_vals, dtype=float)
    ref_sigma_vals = np.asarray(ref_sigma_vals, dtype=float)

    ref_non_zero = ref_sigma_vals[ref_sigma_vals > 0]
    eps_hard = np.percentile(ref_non_zero, 1) if len(ref_non_zero) > 0 else 1e-6

    ref_valid = ref_sigma_vals[ref_sigma_vals >= eps_hard]
    if len(ref_valid) == 0:
        raise ValueError("Conjunto de calibración sin valores válidos por encima de eps_hard")
    ref_log = np.log(ref_valid + epsilon)
    median_log = np.median(ref_log)
    mad_log = 1.4826 * np.median(np.abs(ref_log - median_log))
    if mad_log == 0:
        mad_log = 1.0

    R = np.zeros_like(sigma_vals, dtype=float)
    valid = sigma_vals >= eps_hard
    log_sigma = np.log(sigma_vals[valid] + epsilon)
    z = (log_sigma - median_log) / mad_log
    R[valid] = np.exp(-0.5 * z ** 2)
    return R


def split_half_sensitivity(variability_values, y_true, n_splits: int = 200,
                            calib_frac: float = 0.5, seed: int = 42):
    """Mide el AUC fuera de muestra repitiendo particiones calibración/evaluación.

    Args:
        variability_values (array-like): std_signal por registro.
        y_true (array-like): Etiquetas binarias.
        n_splits (int, optional): Número de particiones. Por defecto es 200.
        calib_frac (float, optional): Fracción de calibración. Por defecto es
            0.5.
        seed (int, optional): Semilla aleatoria. Por defecto es 42.

    Returns:
        dict: Distribución completa ('aucs') y resumen (mean, median, std,
            ci_low, ci_high, n_valid_splits, n_requested_splits).
    """
    from sklearn.metrics import roc_auc_score

    variability_values = np.asarray(variability_values, dtype=float)
    y_true = np.asarray(y_true)
    n = len(variability_values)
    rng = np.random.default_rng(seed)

    aucs = []
    for _ in range(n_splits):
        idx = rng.permutation(n)
        n_calib = int(n * calib_frac)
        calib_idx, eval_idx = idx[:n_calib], idx[n_calib:]

        y_eval = y_true[eval_idx]
        if y_eval.sum() == 0 or y_eval.sum() == len(y_eval):
            continue  # split degenerado, sin las 2 clases en evaluación

        scores_eval = compute_psri_scores_with_reference(
            variability_values[eval_idx], variability_values[calib_idx]
        )
        aucs.append(roc_auc_score(y_eval, scores_eval))

    aucs = np.array(aucs)
    return {
        "aucs": aucs,
        "mean": float(aucs.mean()),
        "median": float(np.median(aucs)),
        "std": float(aucs.std()),
        "ci_low": float(np.percentile(aucs, 2.5)),
        "ci_high": float(np.percentile(aucs, 97.5)),
        "n_valid_splits": int(len(aucs)),
        "n_requested_splits": n_splits,
    }


def plot_calibration_sensitivity(result: dict, in_sample_auc: float = 0.887,
                                  out_png=None, out_pdf=None):
    """Dibuja el histograma del AUC fuera de muestra con referencia in-sample.

    Args:
        result (dict): Salida de `split_half_sensitivity`.
        in_sample_auc (float, optional): AUC in-sample de referencia. Por
            defecto es 0.887.
        out_png (str o Path, optional): Ruta del PNG.
        out_pdf (str o Path, optional): Ruta del PDF.

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=200)
    ax.hist(result["aucs"], bins=30, color="#1a5276", alpha=0.75, edgecolor="#0e2f44")
    ax.axvline(in_sample_auc, color="#a04000", linewidth=2, linestyle="--",
               label=f"AUC in-sample (original) = {in_sample_auc:.3f}")
    ax.axvline(result["mean"], color="#1a5276", linewidth=2,
               label=f"Out-of-sample AUC (mean) = {result['mean']:.3f}")
    ax.set_xlabel("PSRI AUC evaluated out-of-sample")
    ax.set_ylabel(f"Number of partitions (of {result['n_valid_splits']})")
    ax.set_title("PSRI sensitivity to calibrating on the same sample it evaluates",
                 fontsize=11.5, pad=12)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    if out_png:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _self_check():
    """Sanity check con datos sintéticos que replican la estructura real.

    Construye una señal con separación genuina en forma de U ('aceptable' =
    variabilidad típica; 'inaceptable' = mezcla de señal plana y ruidosa) y
    verifica que el AUC in-sample y fuera de muestra son altos y estables.

    Returns:
        None
    """
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    n = 998
    y = rng.integers(0, 2, size=n)  # 1 = aceptable, 0 = inaceptable

    sigma = np.empty(n, dtype=float)
    acc_mask = y == 1
    sigma[acc_mask] = rng.lognormal(mean=0.0, sigma=0.3, size=acc_mask.sum())
    unacc_idx = np.where(~acc_mask)[0]
    half = len(unacc_idx) // 2
    rng.shuffle(unacc_idx)
    sigma[unacc_idx[:half]] = rng.uniform(0.0005, 0.01, size=half)  # plana
    sigma[unacc_idx[half:]] = rng.lognormal(mean=2.2, sigma=0.3, size=len(unacc_idx) - half)  # ruidosa

    in_sample_scores = compute_psri_scores_with_reference(sigma, sigma)
    in_sample_auc = roc_auc_score(y, in_sample_scores)

    result = split_half_sensitivity(sigma, y, n_splits=200)
    print(f"AUC in-sample (control): {in_sample_auc:.4f}")
    print(f"AUC fuera de muestra: media={result['mean']:.4f} "
          f"IC95%=[{result['ci_low']:.4f},{result['ci_high']:.4f}] "
          f"(splits válidos {result['n_valid_splits']}/{result['n_requested_splits']})")
    assert in_sample_auc > 0.85, "el control in-sample debería separar bien con esta señal U sintética"
    assert result["mean"] > 0.75, "con señal U informativa sintética, el AUC fuera de muestra debe ser alto"
    print("OK: el chequeo produce un AUC fuera de muestra alto y estable con señal sintética informativa.")


if __name__ == "__main__":
    _self_check()