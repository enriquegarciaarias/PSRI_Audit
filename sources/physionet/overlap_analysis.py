# AÑADIR a src/sources/physionet/ como overlap_analysis.py
"""
physionet/overlap_analysis.py

Comprueba empíricamente la limitación conceptual señalada: std_signal no
puede distinguir, por diseño, una señal limpia con QRS marcado (std alta
por actividad cardíaca genuina) de una señal ruidosa con artefactos de
amplitud similar (std alta por perturbación externa) -- ambas producen el
mismo número de entrada.

Estrategia: para cada clase (aceptable/inaceptable), se calcula el rango
[percentil 5, percentil 95] de log(std_signal). La INTERSECCIÓN de ambos
rangos es la "zona de solapamiento" -- la región donde, por construcción,
un valor de std_signal es compatible con ambas clases. Si los registros mal
clasificados caen desproporcionadamente en esa zona (frente a los bien
clasificados), confirma que el mecanismo de fallo es justo el señalado:
ambigüedad estructural del único dato de entrada, no un fallo aleatorio.

Autor: Enrique
"""

from pathlib import Path
import numpy as np
import pandas as pd


def compute_overlap_zone(log_std_accept: np.ndarray, log_std_unaccept: np.ndarray,
                          pct_low: float = 5, pct_high: float = 95) -> tuple[float, float]:
    """Calcula la intersección de los rangos de cada clase en espacio log.

    Args:
        log_std_accept (ndarray): log(std_signal) de la clase aceptable.
        log_std_unaccept (ndarray): log(std_signal) de la clase inaceptable.
        pct_low (float, optional): Percentil inferior. Por defecto es 5.
        pct_high (float, optional): Percentil superior. Por defecto es 95.

    Returns:
        tuple: (low, high) de la zona de solapamiento; si no hay
            solapamiento, low > high.
    """
    lo_a, hi_a = np.percentile(log_std_accept, [pct_low, pct_high])
    lo_u, hi_u = np.percentile(log_std_unaccept, [pct_low, pct_high])
    low = max(lo_a, lo_u)
    high = min(hi_a, hi_u)
    return low, high


def analyze_classification_overlap(std_signal, labels, psri_scores, threshold: float = 0.5,
                                    epsilon: float = 1e-9) -> dict:
    """Analiza si los errores del PSRI se concentran en la zona de solapamiento.

    Args:
        std_signal (array-like): Variabilidad cruda ANTES de la transformación
            PSRI (variability_values).
        labels (array-like): 1=aceptable, 0=inaceptable.
        psri_scores (array-like): Score PSRI ya calculado
            (`compute_psri_gaussian_log(std_signal)`).
        threshold (float, optional): Umbral de clasificación. Por defecto es
            0.5.
        epsilon (float, optional): Pequeño valor para log. Por defecto es
            1e-9.

    Returns:
        dict: Tabla completa por registro, límites de la zona de
            solapamiento y contraste de tasas dentro/fuera entre mal y bien
            clasificados, con desglose FP/FN y baselines por clase.
    """
    std_signal = np.asarray(std_signal, dtype=float)
    labels = np.asarray(labels)
    psri_scores = np.asarray(psri_scores, dtype=float)

    log_std = np.log(std_signal + epsilon)
    predicted = (psri_scores >= threshold).astype(int)
    correct = predicted == labels

    log_accept = log_std[labels == 1]
    log_unaccept = log_std[labels == 0]
    low, high = compute_overlap_zone(log_accept, log_unaccept)
    in_overlap = (log_std >= low) & (log_std <= high)

    table = pd.DataFrame({
        "std_signal": std_signal, "log_std_signal": log_std,
        "label": labels, "psri_score": psri_scores, "predicted": predicted,
        "correct": correct, "in_overlap_zone": in_overlap,
    })

    misclassified = table[~table["correct"]]
    well_classified = table[table["correct"]]

    rate_misclass_in_overlap = misclassified["in_overlap_zone"].mean() if len(misclassified) else np.nan
    rate_correct_in_overlap = well_classified["in_overlap_zone"].mean() if len(well_classified) else np.nan
    rate_overall_in_overlap = table["in_overlap_zone"].mean()

    # --- Desglose por tipo de error (FP vs FN) ---
    # Hipótesis afinada: FP (inaceptable predicho aceptable) debería concentrarse
    # DENTRO de la zona (parece "típico" sin serlo); FN (aceptable predicho
    # inaceptable) debería concentrarse FUERA (atípico sin dejar de ser válido).
    # Mezclar ambos en una sola tasa puede cancelar o invertir el patrón.
    fp_mask = (table["label"] == 0) & (table["predicted"] == 1)
    fn_mask = (table["label"] == 1) & (table["predicted"] == 0)

    rate_fp_in_overlap = table.loc[fp_mask, "in_overlap_zone"].mean() if fp_mask.any() else np.nan
    rate_fn_in_overlap = table.loc[fn_mask, "in_overlap_zone"].mean() if fn_mask.any() else np.nan

    # Tasas base: ¿qué fracción de CADA clase completa cae en la zona?
    # (para saber si FP/FN están por encima o por debajo de lo esperable
    # por azar dentro de su propia clase)
    rate_unacceptable_baseline = table.loc[table["label"] == 0, "in_overlap_zone"].mean()
    rate_acceptable_baseline = table.loc[table["label"] == 1, "in_overlap_zone"].mean()

    return {
        "table": table,
        "overlap_zone_log": (low, high),
        "overlap_zone_std_signal": (np.exp(low) - epsilon, np.exp(high) - epsilon),
        "n_misclassified": int((~table["correct"]).sum()),
        "n_correct": int(table["correct"].sum()),
        "rate_misclass_in_overlap": rate_misclass_in_overlap,
        "rate_correct_in_overlap": rate_correct_in_overlap,
        "rate_overall_in_overlap": rate_overall_in_overlap,
        "n_fp": int(fp_mask.sum()), "n_fn": int(fn_mask.sum()),
        "rate_fp_in_overlap": rate_fp_in_overlap,
        "rate_fn_in_overlap": rate_fn_in_overlap,
        "rate_unacceptable_baseline": rate_unacceptable_baseline,
        "rate_acceptable_baseline": rate_acceptable_baseline,
    }


def plot_overlap_diagnostic(result: dict, out_png=None, out_pdf=None):
    """Dibuja histogramas por clase con la zona de solapamiento marcada.

    Args:
        result (dict): Salida de `analyze_classification_overlap`.
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

    table = result["table"]
    low, high = result["overlap_zone_log"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200)

    ax.hist(table.loc[table["label"] == 1, "log_std_signal"], bins=40, alpha=0.55,
            color="#1a5276", label="Acceptable (true label)", density=True)
    ax.hist(table.loc[table["label"] == 0, "log_std_signal"], bins=40, alpha=0.55,
            color="#a04000", label="Unacceptable (true label)", density=True)

    if high > low:
        ax.axvspan(low, high, color="#888888", alpha=0.15, zorder=0,
                   label="Overlap zone (P5\u2013P95 of both classes)")

    mis = table[~table["correct"]]
    ax.scatter(mis["log_std_signal"], np.full(len(mis), -0.02) * 0 + ax.get_ylim()[1] * -0.03,
               marker="|", color="black", s=40, label=f"Misclassified (n={len(mis)})",
               clip_on=False)

    ax.set_xlabel("log(std_signal)")
    ax.set_ylabel("Density")
    ax.set_title("Do misclassified records fall in the overlap zone?",
                 fontsize=11.5, pad=12)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()

    if out_png:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def breakdown_by_error_type(result: dict) -> dict:
    """Desglosa los errores en FN y FP comparándolos contra su propia clase.

    Más específico que la zona de solapamiento compartida: el PSRI no separa
    'parecido a aceptable vs. inaceptable', separa 'típico vs. atípico
    respecto a la mediana poblacional' (penalización simétrica en U).
    Permite distinguir los FN en la cola atípica de 'aceptable' de los FP
    cerca de la mediana global.

    Args:
        result (dict): Salida de `analyze_classification_overlap`.

    Returns:
        dict: Conteos de FN/FP y sus distancias a la mediana global.
    """
    table = result["table"]

    accept_log = table.loc[table["label"] == 1, "log_std_signal"]
    unaccept_log = table.loc[table["label"] == 0, "log_std_signal"]
    accept_median = accept_log.median()
    global_median = table["log_std_signal"].median()

    fn = table[(table["label"] == 1) & (~table["correct"])]  # aceptable -> mal clasificado
    fp = table[(table["label"] == 0) & (~table["correct"])]  # inaceptable -> mal clasificado

    accept_p10, accept_p90 = np.percentile(accept_log, [10, 90])
    fn_in_own_tail = ((fn["log_std_signal"] < accept_p10) | (fn["log_std_signal"] > accept_p90)).mean() \
        if len(fn) else float("nan")

    fp_dist_to_global_median = (fp["log_std_signal"] - global_median).abs().median() if len(fp) else float("nan")
    unaccept_dist_to_global_median = (unaccept_log - global_median).abs().median()

    return {
        "n_fn": len(fn), "n_fp": len(fp),
        "fn_pct_en_cola_propia_clase_p10_p90": fn_in_own_tail,
        "fp_distancia_mediana_global": fp_dist_to_global_median,
        "inaceptable_distancia_mediana_global_tipica": unaccept_dist_to_global_median,
        "fp_mas_cerca_de_mediana_que_tipico": (
            fp_dist_to_global_median < unaccept_dist_to_global_median
            if len(fp) else None
        ),
    }


def _self_check():
    """Sanity check sintético del análisis de solapamiento.

    Construye un caso con solapamiento deliberado entre clases y verifica que
    los mal clasificados se concentran más en la zona de solapamiento que los
    bien clasificados.

    Returns:
        None
    """
    rng = np.random.default_rng(0)
    n = 998
    labels = rng.integers(0, 2, size=n)
    # aceptable: std típica ~ lognormal(0, 0.3); inaceptable: mezcla plana/ruidosa
    # con una fracción de "ruido moderado" que se solapa a propósito con aceptable
    std_signal = np.empty(n)
    acc = labels == 1
    std_signal[acc] = rng.lognormal(0.0, 0.3, size=acc.sum())
    unacc_idx = np.where(~acc)[0]
    rng.shuffle(unacc_idx)
    third = len(unacc_idx) // 3
    std_signal[unacc_idx[:third]] = rng.uniform(0.0005, 0.01, size=third)  # plana, fácil
    std_signal[unacc_idx[third:2*third]] = rng.lognormal(0.05, 0.35, size=third)  # SOLAPADA a propósito
    std_signal[unacc_idx[2*third:]] = rng.lognormal(3.0, 0.3, size=len(unacc_idx)-2*third)  # ruidosa extrema, fácil

    from sources.psri.calculator import compute_psri_gaussian_log  # ajustar import si hace falta
    psri_scores = compute_psri_gaussian_log(std_signal, eps_hard=None)

    result = analyze_classification_overlap(std_signal, labels, psri_scores)
    print(f"Zona de solapamiento (log): {result['overlap_zone_log']}")
    print(f"% mal clasificados en zona de solapamiento: {result['rate_misclass_in_overlap']:.1%}")
    print(f"% bien clasificados en zona de solapamiento: {result['rate_correct_in_overlap']:.1%}")
    print(f"FP (n={result['n_fp']}) en zona: {result['rate_fp_in_overlap']:.1%} "
          f"(baseline inaceptables: {result['rate_unacceptable_baseline']:.1%})")
    print(f"FN (n={result['n_fn']}) en zona: {result['rate_fn_in_overlap']:.1%} "
          f"(baseline aceptables: {result['rate_acceptable_baseline']:.1%})")
    assert result["rate_misclass_in_overlap"] > result["rate_correct_in_overlap"], \
        "los mal clasificados deberían concentrarse más en la zona de solapamiento"
    print("OK: los mal clasificados se concentran más en la zona de solapamiento, como se esperaba.")


if __name__ == "__main__":
    _self_check()