"""
physionet/auc_comparison.py

Comparación estadística de AUCs CORRELACIONADAS (mismos 998 registros
evaluados por PSRI, kSQI y Correlación inter-derivación -- no son muestras
independientes, así que un test para AUCs independientes sería incorrecto).

Dos métodos, ambos calculados sobre los mismos datos para que se puedan
contrastar entre sí:

1. Test de DeLong (DeLong et al. 1988; implementación rápida de Sun & Xu 2014)
   -- cerrado, exacto, el estándar de facto en biomedicina para esta
   comparación exacta (comparar AUCs de la misma cohorte).
2. Bootstrap pareado -- remuestrea los REGISTROS (no las predicciones por
   separado, para preservar el emparejamiento: cada bootstrap re-evalúa los
   tres scores sobre el mismo subconjunto remuestreado de pacientes),
   recalcula la diferencia de AUC en cada réplica, y construye un IC del 95%
   y un p-valor empírico. Más simple de auditar a ojo que DeLong, sirve de
   contraste independiente del resultado cerrado.

Uso: ver el bloque `if __name__ == "__main__"` al final con datos sintéticos
de comprobación, y `EJEMPLO_DE_USO_REAL` con la plantilla para tus arrays.

Autor: Enrique
"""
from sources.common.common import processControl, logger, writeLog
import numpy as np


# ---------------------------------------------------------------------------
# DeLong
# ---------------------------------------------------------------------------

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Calcula los midranks (rango con empates promediado) de un array.

    Args:
        x (ndarray): Array de scores.

    Returns:
        ndarray: Midranks en el mismo orden que la entrada.
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed: np.ndarray, m: int):
    """Implementación rápida del test de DeLong (Sun & Xu, 2014).

    Args:
        predictions_sorted_transposed (ndarray): Shape (k, n_total), columnas
            ordenadas con los m positivos (clase 'acceptable') primero, luego
            los n negativos.
        m (int): Número de positivos.

    Returns:
        tuple: (aucs, delongcov) para las k curvas.
    """
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    sx = np.atleast_2d(sx)
    sy = np.atleast_2d(sy)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray):
    """Compara dos AUCs correlacionadas mediante el test de DeLong.

    Args:
        labels (ndarray): Array binario, 1 = clase positiva ('acceptable'),
            0 = negativa.
        scores_a (ndarray): Scores continuos del primer método (más alto =
            más probable 'acceptable'), en el mismo orden que `labels`.
        scores_b (ndarray): Scores continuos del segundo método.

    Returns:
        dict: Con `auc_a`, `auc_b`, `diff`, `se_diff`, `z` y `p_value`
            (dos colas).
    """
    labels = np.asarray(labels).astype(int)
    order = np.argsort(-labels)  # positivos primero
    labels_sorted = labels[order]
    m = int(labels_sorted.sum())  # nº de positivos

    preds = np.vstack([
        np.asarray(scores_a, dtype=float)[order],
        np.asarray(scores_b, dtype=float)[order],
    ])

    aucs, delongcov = _fast_delong(preds, m)

    auc_a, auc_b = aucs[0], aucs[1]
    var_diff = delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]
    var_diff = max(var_diff, 1e-12)  # evita división por 0 numérica
    z = (auc_a - auc_b) / np.sqrt(var_diff)

    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(z)))

    return {
        "auc_a": auc_a, "auc_b": auc_b, "diff": auc_a - auc_b,
        "se_diff": np.sqrt(var_diff), "z": z, "p_value": p_value,
    }


# ---------------------------------------------------------------------------
# Bootstrap pareado (contraste independiente de DeLong)
# ---------------------------------------------------------------------------

def bootstrap_auc_diff(labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray,
                        n_boot: int = 2000, seed: int = 42, ci: float = 0.95):
    """Bootstrap pareado de la diferencia de AUC entre dos scores.

    En cada réplica se remuestrean los REGISTROS (no los scores por
    separado), preservando el emparejamiento: cada registro remuestreado
    aporta su valor de `scores_a` Y `scores_b` simultáneamente.

    Args:
        labels (ndarray): Etiquetas binarias.
        scores_a (ndarray): Scores del primer método.
        scores_b (ndarray): Scores del segundo método.
        n_boot (int, optional): Número de réplicas. Por defecto es 2000.
        seed (int, optional): Semilla del generador aleatorio. Por defecto es
            42.
        ci (float, optional): Nivel de confianza del intervalo. Por defecto
            es 0.95.

    Returns:
        dict: Con `auc_a`, `auc_b`, `diff`, `ci_low`, `ci_high`, `p_value`
            (bilateral), `n_boot_valid` y el array `diffs` completo.
    """
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    n = len(labels)

    auc_a_obs = roc_auc_score(labels, scores_a)
    auc_b_obs = roc_auc_score(labels, scores_b)
    diff_obs = auc_a_obs - auc_b_obs

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    i = 0
    attempts = 0
    while i < n_boot and attempts < n_boot * 10:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        y = labels[idx]
        if y.sum() == 0 or y.sum() == n:  # réplica degenerada, sin las 2 clases
            continue
        a = roc_auc_score(y, scores_a[idx])
        b = roc_auc_score(y, scores_b[idx])
        diffs[i] = a - b
        i += 1
    diffs = diffs[:i]

    alpha = 1 - ci
    ci_low, ci_high = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # p-valor bilateral: proporción de réplicas al otro lado de 0 respecto al signo observado
    p_value = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    p_value = min(p_value, 1.0)

    return {
        "auc_a": auc_a_obs, "auc_b": auc_b_obs, "diff": diff_obs,
        "ci_low": ci_low, "ci_high": ci_high, "p_value": p_value,
        "n_boot_valid": i, "diffs": diffs,
    }


# ---------------------------------------------------------------------------
# Comprobación con datos sintéticos (ejecutar: python -m physionet.auc_comparison)
# ---------------------------------------------------------------------------

def _self_check():
    """Sanity check con datos sintéticos del test de DeLong y del bootstrap.

    Verifica que: (1) con el mismo score dos veces la diferencia es 0 y el
    p-valor ~1, y (2) con un score claramente mejor ambos métodos detectan
    la mejora con significación.

    Returns:
        None
    """
    rng = np.random.default_rng(0)
    n = 500
    labels = rng.integers(0, 2, size=n)
    true_signal = labels + rng.normal(0, 0.3, size=n)  # score informativo

    print("== Caso 1: mismo score dos veces (diff debe ser ~0, p~1.0) ==")
    r = delong_roc_test(labels, true_signal, true_signal)
    print(f"  DeLong: auc_a={r['auc_a']:.4f} auc_b={r['auc_b']:.4f} "
          f"diff={r['diff']:.6f} p={r['p_value']:.4f}")
    assert abs(r["diff"]) < 1e-9, "diff debería ser exactamente 0"
    assert r["p_value"] > 0.99, "p-valor debería ser ~1.0"

    print("\n== Caso 2: score_a mucho mejor que score_b (ruido puro) ==")
    noise_signal = rng.normal(0, 1, size=n)
    r_delong = delong_roc_test(labels, true_signal, noise_signal)
    r_boot = bootstrap_auc_diff(labels, true_signal, noise_signal, n_boot=1000)
    print(f"  DeLong:    auc_a={r_delong['auc_a']:.4f} auc_b={r_delong['auc_b']:.4f} "
          f"diff={r_delong['diff']:.4f} p={r_delong['p_value']:.2e}")
    print(f"  Bootstrap: auc_a={r_boot['auc_a']:.4f} auc_b={r_boot['auc_b']:.4f} "
          f"diff={r_boot['diff']:.4f} IC95%=[{r_boot['ci_low']:.4f}, {r_boot['ci_high']:.4f}] "
          f"p={r_boot['p_value']:.4f}")
    assert r_delong["diff"] > 0.2, "score_a debería ganar claramente"
    assert r_delong["p_value"] < 0.001, "debería ser muy significativo"
    assert r_boot["ci_low"] > 0, "el IC bootstrap no debería cruzar el 0"
    print("\nOK: ambos métodos coinciden en signo y significación en los dos casos de control.")


def plot_threshold_curves(labels: np.ndarray, scores: np.ndarray, out_png, out_pdf=None,
                           mark_threshold: float = 0.5, n_points: int = 300,
                           title: str = "PSRI: Sensitivity and FPR as a function of threshold"):
    """Dibuja TPR y FPR en función del umbral de decisión.

    Muestra explícitamente lo que la curva ROC oculta por diseño: el eje X
    aquí es el propio umbral (de 0 a 1), no el FPR. Solo tiene sentido para
    scores en escala [0,1] como PSRI.

    Args:
        labels (ndarray): Etiquetas binarias.
        scores (ndarray): Scores en [0,1].
        out_png (str o Path): Ruta del PNG de salida.
        out_pdf (str o Path, optional): Ruta del PDF de salida.
        mark_threshold (float, optional): Umbral a destacar. Por defecto es
            0.5.
        n_points (int, optional): Número de puntos del barrido. Por defecto
            es 300.
        title (str, optional): Título de la figura.

    Returns:
        None
    """
    from pathlib import Path
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)

    thresholds = np.linspace(0, 1, n_points)
    tpr_vals, fpr_vals = [], []
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        tpr_vals.append(tp / n_pos if n_pos > 0 else np.nan)
        fpr_vals.append(fp / n_neg if n_neg > 0 else np.nan)

    tpr_vals, fpr_vals = np.array(tpr_vals), np.array(fpr_vals)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=200)
    ax.plot(thresholds, tpr_vals, color="#1a5276", linewidth=1.8,
            label="Sensitivity / TPR (threshold)")
    ax.plot(thresholds, fpr_vals, color="#a04000", linewidth=1.8,
            label="FPR / 1\u2212Specificity (threshold)")

    # Punto marcado en el umbral de interés (por defecto 0.5, el de la G-mean)
    idx_mark = np.argmin(np.abs(thresholds - mark_threshold))
    ax.axvline(mark_threshold, color="#555555", linestyle="--", linewidth=1)
    ax.scatter([mark_threshold], [tpr_vals[idx_mark]], color="#1a5276", zorder=5, s=60)
    ax.scatter([mark_threshold], [fpr_vals[idx_mark]], color="#a04000", zorder=5, s=60)
    ax.annotate(f"threshold={mark_threshold}\nTPR={tpr_vals[idx_mark]:.3f}\nFPR={fpr_vals[idx_mark]:.3f}",
                xy=(mark_threshold, (tpr_vals[idx_mark] + fpr_vals[idx_mark]) / 2),
                xytext=(mark_threshold + 0.08, 0.5), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#555555"))

    ax.set_xlabel("Decision threshold (PSRI score)")
    ax.set_ylabel("Rate")
    ax.set_title(title, fontsize=11.5, pad=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=9, loc="center left")
    fig.tight_layout()

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


EJEMPLO_DE_USO_REAL = """
# Una vez tengas los tres arrays de scores YA CALCULADOS por tu pipeline real
# (los 998 registros, con los casos degenerados ya resueltos con el sentinel
# para kSQI/Correlación tal como describe el documento), y el array de
# etiquetas (1='acceptable', 0='unacceptable'):

from physionet.auc_comparison import delong_roc_test, bootstrap_auc_diff, plot_roc_comparison

# labels, psri_scores, ksqi_scores, corr_scores = ... (cargar de tu pipeline real)

for name_b, scores_b in [("kSQI", ksqi_scores), ("Correlación inter-derivación", corr_scores)]:
    print(f"\\n== PSRI vs {name_b} ==")
    d = delong_roc_test(labels, psri_scores, scores_b)
    b = bootstrap_auc_diff(labels, psri_scores, scores_b, n_boot=2000)
    print(f"DeLong:    diff_AUC={d['diff']:.4f}  z={d['z']:.3f}  p={d['p_value']:.4g}")
    print(f"Bootstrap: diff_AUC={b['diff']:.4f}  IC95%=[{b['ci_low']:.4f}, {b['ci_high']:.4f}]  "
          f"p={b['p_value']:.4g}")

plot_roc_comparison(
    labels,
    {"PSRI (std_signal)": psri_scores, "kSQI": ksqi_scores, "Correlación inter-derivación": corr_scores},
    out_png="outputs/fig_physionet_roc_comparison.png",
    out_pdf="outputs/fig_physionet_roc_comparison.pdf",
)
"""


def plot_roc_comparison_with_thresholds(
    labels: np.ndarray, scores_dict: dict, out_png, out_pdf=None,
    quantiles: tuple = (0.25, 0.5, 0.75),
    operating_thresholds: dict | None = None,
):
    """Dibuja las ROC comparadas anotando cuartiles y umbrales operativos.

    Cada curva usa su propia escala y se anotan marcadores en los cuartiles
    de su distribución y una estrella con el umbral operativo real.

    Args:
        labels (ndarray): Etiquetas binarias.
        scores_dict (dict): {nombre: scores} por método.
        out_png (str o Path): Ruta del PNG de salida.
        out_pdf (str o Path, optional): Ruta del PDF de salida.
        quantiles (tuple, optional): Cuantiles a marcar. Por defecto
            (0.25, 0.5, 0.75).
        operating_thresholds (dict, optional): {nombre: umbral operativo}.
            Para métodos no listados se usa su propia mediana.

    Returns:
        None
    """
    from pathlib import Path
    from sklearn.metrics import roc_curve, roc_auc_score
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    operating_thresholds = operating_thresholds or {}

    colors = ["#1a5276", "#a04000", "#117864", "#7d3c98"]
    fig, ax = plt.subplots(figsize=(6.5, 6.2), dpi=200)

    def _fpr_tpr_at_threshold(scores, thr):
        """Calcula FPR/TPR para un umbral de clasificación dado.

        Args:
            scores (ndarray): Scores continuos del método.
            thr (float): Umbral de decisión.

        Returns:
            tuple: (fpr, tpr) en ese umbral.
        """
        pred = (scores >= thr).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        return fpr, tpr

    for i, (name, scores) in enumerate(scores_dict.items()):
        scores = np.asarray(scores, dtype=float)
        color = colors[i % len(colors)]
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        ax.plot(fpr, tpr, color=color, linewidth=1.8, zorder=2,
                label=f"{name} (AUC={auc:.3f})")

        # Cuartiles de la propia distribución de scores de este método
        for q in quantiles:
            thr = np.quantile(scores, q)
            q_fpr, q_tpr = _fpr_tpr_at_threshold(scores, thr)
            ax.scatter(q_fpr, q_tpr, color=color, s=35, marker="o",
                       edgecolors="black", linewidths=0.5, zorder=3, alpha=0.85)

        # Umbral operativo real (destacado)
        op_thr = operating_thresholds.get(name, float(np.median(scores)))
        op_fpr, op_tpr = _fpr_tpr_at_threshold(scores, op_thr)
        ax.scatter(op_fpr, op_tpr, color=color, s=170, marker="*",
                   edgecolors="black", linewidths=0.9, zorder=5)
        ax.annotate(f"threshold={op_thr:.3g}", (op_fpr, op_tpr),
                    fontsize=7.5, color=color, xytext=(6, -8),
                    textcoords="offset points", fontweight="bold")

    ax.plot([0, 1], [0, 1], color="#999999", linewidth=0.9, linestyle="--", zorder=1)
    ax.scatter([], [], color="#555555", s=35, marker="o", edgecolors="black",
               linewidths=0.5,                label=f"Quartiles ({', '.join(f'P{int(q*100)}' for q in quantiles)})")
    ax.scatter([], [], color="#555555", s=170, marker="*", edgecolors="black",
               linewidths=0.9, label="Operating threshold (\u2605 = the one used for G-mean)")
    ax.set_xlabel("False Positive Rate (1 \u2212 Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("PhysioNet/CinC 2011 Validation: PSRI vs. Reference SQIs\n"
                 "(with threshold annotations per curve)", fontsize=11, pad=12)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    writeLog("info", logger, f"Grafico generado: {out_png}")


def plot_roc_comparison(labels: np.ndarray, scores_dict: dict, out_png, out_pdf=None):
    """Dibuja curvas ROC superpuestas para varios métodos.

    Las curvas se generan con `sklearn.metrics.roc_curve` sobre los arrays
    reales; el AUC de la leyenda es el mismo que reportan
    `delong_roc_test`/`bootstrap_auc_diff`.

    Args:
        labels (ndarray): Etiquetas binarias.
        scores_dict (dict): {nombre: scores} por método.
        out_png (str o Path): Ruta del PNG de salida.
        out_pdf (str o Path, optional): Ruta del PDF de salida.

    Returns:
        None
    """
    from pathlib import Path
    from sklearn.metrics import roc_curve, roc_auc_score
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    colors = ["#1a5276", "#a04000", "#117864", "#7d3c98"]
    fig, ax = plt.subplots(figsize=(5.5, 5.2), dpi=200)

    for i, (name, scores) in enumerate(scores_dict.items()):
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=1.8,
                 label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], color="#999999", linewidth=0.9, linestyle="--", zorder=1)
    ax.set_xlabel("False Positive Rate (1 \u2212 Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("PhysioNet/CinC 2011 Validation: PSRI vs. Reference SQIs (n=998)",
                 fontsize=11, pad=12)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    writeLog("info", logger, f"Grafico generado: {out_png}")

if __name__ == "__main__":
    _self_check()
    print(EJEMPLO_DE_USO_REAL)