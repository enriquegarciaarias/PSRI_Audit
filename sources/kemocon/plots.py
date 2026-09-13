"""
kemocon/plots.py

Genera las figuras del Estudio 2 directamente a partir de los DataFrames que
ya produce run_pipeline.py (sweep_s_coher_baseline_window, leverage_diagnostics)
-- no a partir de números transcritos a mano. Se ejecuta automáticamente al
final de run_pipeline.py; también se puede llamar de forma independiente
pasando los CSV ya guardados en outputs/ (ver reload_and_plot() al final).

Autor: Enrique
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
})


def plot_scoher_baseline_sweep(
    sweep_df: pd.DataFrame,
    out_png: Path,
    out_pdf: Path | None = None,
    targets: tuple[str, ...] = ("external_valence_var", "external_valence_range"),
    alpha: float = 0.05,
):
    """Dibuja rho_within de S_coher en función del tamaño de baseline.

    Args:
        sweep_df (pandas.DataFrame): Salida de
            `build_features.sweep_s_coher_baseline_window()` (columnas:
            n_prev, target, rho_within_subject, p_within, ...).
        out_png (Path): Ruta del PNG de salida.
        out_pdf (Path, optional): Ruta del PDF de salida.
        targets (tuple, optional): Métricas a graficar. Por defecto
            ("external_valence_var", "external_valence_range").
        alpha (float, optional): Nivel de significancia. Por defecto es 0.05.

    Returns:
        None
    """
    colors = ["#1a5276", "#a04000", "#117864", "#7d3c98"]
    markers = ["o", "s", "^", "D"]

    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=200)

    for i, target in enumerate(targets):
        sub = sweep_df[sweep_df["target"] == target].sort_values("n_prev")
        if sub.empty:
            continue
        x = sub["n_prev"].to_numpy()
        y = sub["rho_within_subject"].to_numpy(dtype=float)
        p = sub["p_within"].to_numpy(dtype=float)
        color, marker = colors[i % len(colors)], markers[i % len(markers)]

        ax.plot(x, y, color=color, linewidth=1.6, zorder=2)
        sig = p < alpha
        label = target.replace("external_", "")
        ax.scatter(x[sig], y[sig], color=color, marker=marker, s=70,
                   zorder=3, label=f"{label} (p<{alpha})")
        ax.scatter(x[~sig], y[~sig], facecolors="white", edgecolors=color,
                   marker=marker, s=70, linewidths=1.6, zorder=3, label=f"{label} (n.s.)")

    ax.axhline(0, color="#999999", linewidth=0.8, zorder=1)
    ax.set_xlabel("Tamaño de la ventana de baseline ($n_{prev}$, nº de trials previos)")
    ax.set_ylabel(r"$\rho_{within\text{-}subject}$ (S_coher vs. desacuerdo externo)")
    ax.set_title("Robustez del efecto de S_coher frente a la elección de baseline",
                 fontsize=12, pad=12)
    if "n_prev" in sweep_df.columns:
        ax.set_xticks(sorted(sweep_df["n_prev"].unique()))
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_between_within_decomposition(
    leverage_a_df: pd.DataFrame,
    leverage_b_df: pd.DataFrame,
    out_png: Path,
    out_pdf: Path | None = None,
    label_a: str = "S_estab",
    label_b: str = "S_coher (HR-EDA, prev1)",
    alpha: float = 0.05,
):
    """Dibuja la descomposición between/within-subject para dos componentes.

    Args:
        leverage_a_df (pandas.DataFrame): Salida de
            `build_features.leverage_diagnostics()` para el primer
            componente (columnas: target, rho_between_subject,
            rho_within_subject, p_within, ...), una fila por target.
        leverage_b_df (pandas.DataFrame): Igual para el segundo componente.
        out_png (Path): Ruta del PNG de salida.
        out_pdf (Path, optional): Ruta del PDF de salida.
        label_a (str, optional): Etiqueta del primer componente. Por defecto
            es "S_estab".
        label_b (str, optional): Etiqueta del segundo componente. Por
            defecto es "S_coher (HR-EDA, prev1)".
        alpha (float, optional): Nivel de significancia. Por defecto es 0.05.

    Returns:
        None
    """
    targets = list(dict.fromkeys(
        list(leverage_a_df["target"]) + list(leverage_b_df["target"])
    ))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=200, sharey=True)
    x = np.arange(len(targets))
    width = 0.32

    for ax, title, df in [(axes[0], label_a, leverage_a_df), (axes[1], label_b, leverage_b_df)]:
        df_idx = df.set_index("target").reindex(targets)
        between = df_idx["rho_between_subject"].to_numpy(dtype=float)
        within = df_idx["rho_within_subject"].to_numpy(dtype=float)
        p_within = df_idx["p_within"].to_numpy(dtype=float)

        ax.bar(x - width / 2, between, width, label=r"$\rho_{between}$",
               color="#bdc3c7", edgecolor="#7f8c8d")
        ax.bar(x + width / 2, within, width, label=r"$\rho_{within}$",
               color="#1a5276", edgecolor="#0e2f44")

        for xi, val, p in zip(x, within, p_within):
            if np.isnan(val) or np.isnan(p):
                continue
            marker = "*" if p < alpha else "n.s."
            offset = 0.02 * (1 if val >= 0 else -1.4)
            ax.text(xi + width / 2, val + offset, marker, ha="center",
                     fontsize=(13 if marker == "*" else 8),
                     color="#1a5276" if marker == "*" else "#888888")

        ax.axhline(0, color="#999999", linewidth=0.8)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace("_", "_\n") for t in targets], fontsize=8.5)

    axes[0].set_ylabel(r"$\rho$ (Spearman)")
    axes[0].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle("Descomposición between/within-subject: qué efectos sobreviven al centrar por sujeto",
                 fontsize=12.5, y=1.03)
    fig.tight_layout()

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def reload_and_plot(out_dir: Path = Path("outputs")):
    """Regenera las figuras a partir de los CSV ya guardados por `run_pipeline.py`.

    Utilidad para no tener que re-ejecutar todo el pipeline.

    Args:
        out_dir (Path, optional): Directorio con los CSV. Por defecto es
            Path("outputs").

    Returns:
        None
    """
    sweep = pd.read_csv(out_dir / "kemocon_psri_scoher_baseline_sweep.csv")
    leverage_estab = pd.read_csv(out_dir / "kemocon_psri_leverage_diagnostic_estab.csv")
    leverage_coher = pd.read_csv(out_dir / "kemocon_psri_leverage_diagnostic_coher_prev1.csv")

    plot_scoher_baseline_sweep(
        sweep,
        out_dir / "fig1_scoher_baseline_sweep.png",
        out_dir / "fig1_scoher_baseline_sweep.pdf",
    )
    plot_between_within_decomposition(
        leverage_estab, leverage_coher,
        out_dir / "fig2_between_within_decomposition.png",
        out_dir / "fig2_between_within_decomposition.pdf",
    )


if __name__ == "__main__":
    reload_and_plot()