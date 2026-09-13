"""
kemocon/experiment3.py

HERRAMIENTA EXPLORATORIA -- NO REPORTAR EN EL PAPER (decisión de diseño,
jul 2026). Se conserva como herramienta reusable para otros datasets/
estudios futuros.

Experimento 3 (Study 2, K-EmoCon): evaluar si el PSRI usado como filtro o
peso mejora OPERATIVAMENTE un modelo de regresión frente a usar la señal
cruda. Puede predecir dos grupos de target:

  - emoción continua    : external_valence_mean, external_arousal_mean
                          (media de raters externos con skipna, >=3 raters)
  - desacuerdo externo  : external_valence_var/range, external_arousal_var,
                          self_partner_diff, self_external_mean_diff
                          (los mismos targets del Estudio 2)

Diseño (según docs/dialogo.txt + decisiones de diseño):

  CV     : Leave-One-Subject-Out (los sujetos de test NUNCA entran en
           entrenamiento). Métricas por fold: CCC, RMSE, MAE.
  Modelo : RandomForestRegressor o HistGradientBoostingRegressor con
           hiperparámetros fijos y documentados (no optimizados -- la
           comparación es sobre el efecto del PSRI, no sobre ajuste fino).
           HGB maneja NaN nativamente (permite usar ibi/polar_hr/attention/
           meditation sin imputar).

  Modelos (entrenamiento -> evaluación, siempre el MISMO test completo):
    M1 Base               : todas las ventanas          -> test completo
    M2 PSRI-Filtro        : solo psri > umbral          -> test completo
    M3 PSRI-Ponderación   : todas, weight=psri en fit() -> test completo
    M4 Control Varianza   : todas, weight=1/(1+var)     -> test completo
                            (confound exacto de dificultad del target; solo
                            tiene sentido para el grupo "emoción")
    M5 Control Aleatorio  : drop aleatorio del mismo % que M2 (seed fija)
                            (si M2/M3 no superan a M5, el efecto es solo
                            "quitar datos", no "quitar mala señal")

  Significación: Wilcoxon pareado + t-test pareado sobre los deltas
  M1 vs cada modelo a través de los folds.

RESULTADO EMPÍRICO (K-EmoCon, LOSO 23 sujetos): en TODAS las configuraciones
probadas (RF y HGB, features core y todas, targets de emoción y de
desacuerdo), el CCC por fold es ~0 y ningún modelo PSRI supera
significativamente al base -- el control aleatorio M5 iguala o supera al
PSRI. La señal fisiológica agregada no contiene suficiente señal
cross-subject para que el PSRI demuestre mejora operativa.

Autor: Enrique
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from sources.common.common import logger, writeLog
from sources.kemocon.build_features import annotator_disagreement_features

# ---------------------------------------------------------------------------
# Configuración del experimento
# ---------------------------------------------------------------------------

PSRI_COL = "psri_composite_prev1"
PSRI_THRESHOLD = 0.6
RANDOM_SEED = 1234

FEATURES_CORE = ["hr_mean", "eda_mean", "bvp_mean", "temp_mean", "acc_std"]
FEATURES_ALL = [
    "bvp_mean", "bvp_std", "eda_mean", "eda_std", "hr_mean", "hr_std",
    "temp_mean", "temp_std", "ibi_mean", "ibi_std", "acc_std",
    "polar_hr_mean", "polar_hr_std",
    "attention_mean", "attention_std", "meditation_mean", "meditation_std",
]

# grupo de targets -> (target, columna de raters válidos o None, columna de
# varianza para el Control M4 o None)
TARGET_GROUPS = {
    "emotion": {
        "external_valence_mean": ("n_valid_raters_valence", "external_valence_var"),
        "external_arousal_mean": ("n_valid_raters_arousal", "external_arousal_var"),
    },
    "disagreement": {
        "external_valence_var": ("n_valid_raters_valence", None),
        "external_valence_range": ("n_valid_raters_valence", None),
        "external_arousal_var": ("n_valid_raters_arousal", None),
        "self_partner_diff": (None, None),
        "self_external_mean_diff": (None, None),
    },
}

RF_PARAMS = {
    "n_estimators": 200,
    "min_samples_leaf": 5,
    "max_depth": None,
    "random_state": 42,
    "n_jobs": -1,
}

HGB_PARAMS = {
    "max_iter": 300,
    "min_samples_leaf": 20,
    "learning_rate": 0.05,
    "random_state": 42,
}

MODEL_NAMES = ["M1_Base", "M2_PSRI_Filtro", "M3_PSRI_Peso",
               "M4_Control_Var", "M5_Control_Azar"]


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Concordance Correlation Coefficient (Lin, 1989).

    Args:
        y_true (ndarray): Valores reales.
        y_pred (ndarray): Valores predichos.

    Returns:
        float: CCC en [-1, 1], o NaN con menos de 2 muestras.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return np.nan
    mu_t, mu_p = y_true.mean(), y_pred.mean()
    s_t, s_p = y_true.var(ddof=1), y_pred.var(ddof=1)
    cov = np.cov(y_true, y_pred, ddof=1)[0, 1]
    return float(2 * cov / (s_t + s_p + (mu_t - mu_p) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula el error cuadrático medio.

    Args:
        y_true (ndarray): Valores reales.
        y_pred (ndarray): Valores predichos.

    Returns:
        float: RMSE, o NaN si no hay muestras.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula el error absoluto medio.

    Args:
        y_true (ndarray): Valores reales.
        y_pred (ndarray): Valores predichos.

    Returns:
        float: MAE, o NaN si no hay muestras.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)))


METRIC_FUNCS = {"ccc": ccc, "rmse": rmse, "mae": mae}


# ---------------------------------------------------------------------------
# Dataset del experimento
# ---------------------------------------------------------------------------

def build_experiment_dataset(feature_table: pd.DataFrame, target: str,
                             min_raters: int = 3) -> pd.DataFrame:
    """Construye el dataset de modelado para un target dado.

      - SIEMPRE recalcula external_*_mean/var/range/n_valid desde R1..R5
        (el CSV histórico contiene external_valence_var contaminada con la
        autoanotación -- R_self_valence matcheaba el patrón R*_valence)
      - filtra por >= `min_raters` raters externos si el target los requiere
      - descarta filas con NaN en features, target o PSRI
      - añade control_weight_var = 1/(1+var) para el Control M4

    Args:
        feature_table (pandas.DataFrame): Tabla de características.
        target (str): Nombre del target a predecir.
        min_raters (int, optional): Mínimo de raters externos. Por defecto
            es 3.

    Returns:
        pandas.DataFrame: Dataset listo para modelar.

    Raises:
        ValueError: Si la columna de raters válidos no está disponible.
    """
    df = annotator_disagreement_features(feature_table.copy())

    rater_col, var_col = TARGET_GROUPS["emotion"].get(
        target, TARGET_GROUPS["disagreement"].get(target, (None, None))
    )
    if rater_col is not None:
        if rater_col not in df.columns:
            raise ValueError(f"Columna {rater_col} no disponible")
        df = df[df[rater_col] >= min_raters].copy()

    needed = [target, PSRI_COL]
    df = df.dropna(subset=needed)

    if var_col is not None:
        df = df.dropna(subset=[var_col])
        df["control_weight_var"] = 1.0 / (1.0 + df[var_col])
    else:
        df["control_weight_var"] = np.nan
    return df


# ---------------------------------------------------------------------------
# LOSO-CV
# ---------------------------------------------------------------------------

def _new_model(model_name: str):
    """Instancia el modelo según el tipo elegido.

    Args:
        model_name (str): "RF" o cualquier otro valor (HGB).

    Returns:
        sklearn estimator: Modelo con los hiperparámetros fijos del módulo.
    """
    if model_name == "RF":
        return RandomForestRegressor(**RF_PARAMS)
    return HistGradientBoostingRegressor(**HGB_PARAMS)


def _predict_model(model_name: str, train: pd.DataFrame, test: pd.DataFrame,
                   target: str, model_kind: str, rng: np.random.Generator) -> np.ndarray:
    """Entrena según el tratamiento del modelo y predice sobre todo el test.

    Entrena en `train` (con el tratamiento específico del modelo) y predice
    sobre TODAS las filas de `test`. La evaluación se hace siempre sobre el
    test completo — filtrar/ponderar ocurre SOLO en entrenamiento.

    Args:
        model_name (str): Nombre del modelo (M1_Base, M2_PSRI_Filtro,
            M3_PSRI_Peso, M4_Control_Var, M5_Control_Azar).
        train (pandas.DataFrame): Conjunto de entrenamiento.
        test (pandas.DataFrame): Conjunto de test (completo).
        target (str): Columna objetivo.
        model_kind (str): "RF" o "HGB".
        rng (numpy.random.Generator): Generador aleatorio para M5.

    Returns:
        ndarray: Predicciones sobre todo `test`.

    Raises:
        ValueError: Si `model_name` no es conocido.
    """
    model = _new_model(model_kind)

    if model_name == "M1_Base":
        model.fit(train[FEATURES], train[target])
    elif model_name == "M2_PSRI_Filtro":
        keep = train[PSRI_COL] > PSRI_THRESHOLD
        model.fit(train.loc[keep, FEATURES], train.loc[keep, target])
    elif model_name == "M3_PSRI_Peso":
        model.fit(train[FEATURES], train[target],
                  sample_weight=train[PSRI_COL].to_numpy(dtype=float))
    elif model_name == "M4_Control_Var":
        w = train["control_weight_var"].to_numpy(dtype=float)
        model.fit(train[FEATURES], train[target], sample_weight=w)
    elif model_name == "M5_Control_Azar":
        drop_frac = float((train[PSRI_COL] <= PSRI_THRESHOLD).mean())
        n_drop = int(round(drop_frac * len(train)))
        drop_idx = rng.choice(len(train), size=n_drop, replace=False)
        keep = np.ones(len(train), dtype=bool)
        keep[drop_idx] = False
        model.fit(train.loc[keep, FEATURES], train.loc[keep, target])
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")

    return model.predict(test[FEATURES])


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula CCC, RMSE y MAE de una vez.

    Args:
        y_true (ndarray): Valores reales.
        y_pred (ndarray): Valores predichos.

    Returns:
        dict: Con `ccc`, `rmse` y `mae`.
    """
    return {name: fn(y_true, y_pred) for name, fn in METRIC_FUNCS.items()}


def run_experiment3(feature_table: pd.DataFrame,
                    out_dir: Path | None = None,
                    target_group: str = "disagreement",
                    model_kind: str = "HGB",
                    features: list[str] | None = None,
                    include_m4: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta el Experimento 3 (LOSO-CV por sujeto).

    Devuelve (per_fold, summary):

    per_fold: una fila por (model, target, subject) con ccc/rmse/mae/n_test.
    summary : agregado por (model, target) con media±std de los folds,
              métricas POOLED (todas las predicciones de test concatenadas)
              y test de significación pareado M1 vs cada modelo.

    Args:
        feature_table (pandas.DataFrame): Tabla de características.
        out_dir (Path, optional): Directorio para los CSV de salida.
        target_group (str, optional): Grupo de targets ("emotion" o
            "disagreement"). Por defecto es "disagreement".
        model_kind (str, optional): "RF" o "HGB". Por defecto es "HGB".
        features (list, optional): Features a usar. Si es None, `FEATURES_ALL`.
        include_m4 (bool, optional): Si incluir el Control M4. Por defecto es
            False.

    Returns:
        tuple: (per_fold, summary).

    Raises:
        ValueError: Si `target_group` no existe en `TARGET_GROUPS`.
    """
    global FEATURES
    FEATURES = features if features is not None else FEATURES_ALL
    if target_group not in TARGET_GROUPS:
        raise ValueError(f"Grupo de targets desconocido: {target_group}")
    targets = list(TARGET_GROUPS[target_group].keys())

    models = [m for m in MODEL_NAMES if m != "M4_Control_Var" or include_m4]

    subjects = sorted(feature_table["subject_id"].unique())
    writeLog("info", logger, f"[Experimento 3] Grupo={target_group}, "
                             f"modelo={model_kind}, features={len(FEATURES)}, "
                             f"sujetos LOSO={len(subjects)}")

    per_fold_rows = []
    pooled = {t: {m: {"y": [], "p": []} for m in models} for t in targets}

    for target in targets:
        df = build_experiment_dataset(feature_table, target)
        writeLog("info", logger,
                 f"[Experimento 3] Target {target}: {len(df)} ventanas, "
                 f"{df['subject_id'].nunique()} sujetos")

        for test_sid in subjects:
            train = df[df["subject_id"] != test_sid]
            test = df[df["subject_id"] == test_sid]
            y_true = test[target].to_numpy(dtype=float)
            rng = np.random.default_rng(RANDOM_SEED)

            for model_name in models:
                y_pred = _predict_model(model_name, train, test, target,
                                        model_kind, rng)
                met = _metrics(y_true, y_pred)
                per_fold_rows.append({
                    "model": model_name, "target": target, "subject": test_sid,
                    "n_test": len(test), **met,
                })
                pooled[target][model_name]["y"].extend(y_true.tolist())
                pooled[target][model_name]["p"].extend(y_pred.tolist())

    per_fold = pd.DataFrame(per_fold_rows)

    summary_rows = []
    for target in targets:
        for model_name in models:
            fold = per_fold[(per_fold["target"] == target) & (per_fold["model"] == model_name)]
            y = np.array(pooled[target][model_name]["y"])
            p = np.array(pooled[target][model_name]["p"])
            row = {
                "model": model_name, "target": target,
                "n_folds": len(fold), "n_windows_pooled": len(y),
            }
            for metric in METRIC_FUNCS:
                row[f"{metric}_mean"] = fold[metric].mean()
                row[f"{metric}_std"] = fold[metric].std()
                row[f"{metric}_pooled"] = METRIC_FUNCS[metric](y, p)

            if model_name != "M1_Base":
                base = per_fold[(per_fold["target"] == target) & (per_fold["model"] == "M1_Base")]
                for metric in METRIC_FUNCS:
                    if len(fold) >= 2 and len(base) >= 2:
                        d = base[metric].to_numpy(dtype=float) - fold[metric].to_numpy(dtype=float)
                        row[f"{metric}_delta_mean"] = d.mean()
                        row[f"{metric}_wilcoxon_p"] = wilcoxon(d).pvalue
                        row[f"{metric}_ttest_p"] = ttest_rel(base[metric], fold[metric]).pvalue
                    else:
                        row[f"{metric}_delta_mean"] = np.nan
                        row[f"{metric}_wilcoxon_p"] = np.nan
                        row[f"{metric}_ttest_p"] = np.nan
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        per_fold.to_csv(out_dir / "kemocon_experiment3_per_fold.csv", index=False)
        summary.to_csv(out_dir / "kemocon_experiment3_summary.csv", index=False)

    return per_fold, summary


# ---------------------------------------------------------------------------
# Salida legible
# ---------------------------------------------------------------------------

def print_experiment3_results(summary: pd.DataFrame) -> None:
    """Imprime los resultados del Experimento 3 por consola.

    Args:
        summary (pandas.DataFrame): Salida de `run_experiment3`.

    Returns:
        None
    """
    print("\n== Experimento 3 (HERRAMIENTA EXPLORATORIA, no reportar) ==")
    for target in summary["target"].unique():
        print(f"\n-- Target: {target} --")
        view = summary[summary["target"] == target]
        print(f"{'modelo':<18}{'CCC_mean':>9}{'CCC_pool':>10}{'RMSE_mean':>11}"
              f"{'MAE_mean':>10}{'n_folds':>9}")
        for _, r in view.iterrows():
            print(f"{r['model']:<18}{r['ccc_mean']:>9.3f}{r['ccc_pooled']:>10.3f}"
                  f"{r['rmse_mean']:>11.3f}{r['mae_mean']:>10.3f}{int(r['n_folds']):>9}")
        print("\n  Significación pareada vs M1_Base (delta_mean [wilcoxon_p | ttest_p]):")
        for _, r in view.iterrows():
            if r["model"] == "M1_Base":
                continue
            print(f"  {r['model']:<18} "
                  f"CCC: {r['ccc_delta_mean']:+.3f} [p_w={r['ccc_wilcoxon_p']:.3f} "
                  f"p_t={r['ccc_ttest_p']:.3f}]   "
                  f"RMSE: {r['rmse_delta_mean']:+.3f} [p_w={r['rmse_wilcoxon_p']:.3f} "
                  f"p_t={r['rmse_ttest_p']:.3f}]   "
                  f"MAE: {r['mae_delta_mean']:+.3f} [p_w={r['mae_wilcoxon_p']:.3f} "
                  f"p_t={r['mae_ttest_p']:.3f}]")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_experiment3_ccc(summary: pd.DataFrame, out_png: Path,
                         out_pdf: Path | None = None, alpha: float = 0.05):
    """Dibuja el CCC medio por fold (±std) por modelo y target.

    Marca con * los modelos con mejora significativa vs M1 (Wilcoxon).

    Args:
        summary (pandas.DataFrame): Salida de `run_experiment3`.
        out_png (Path): Ruta del PNG de salida.
        out_pdf (Path, optional): Ruta del PDF de salida.
        alpha (float, optional): Nivel de significancia. Por defecto es 0.05.

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#333333", "axes.labelcolor": "#222222",
        "text.color": "#222222", "xtick.color": "#333333", "ytick.color": "#333333",
    })
    colors = ["#7f8c8d", "#a04000", "#1a5276", "#7d3c98", "#117864"]
    models = list(dict.fromkeys(summary["model"]))
    targets = summary["target"].unique()

    fig, axes = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 4.4),
                             dpi=200, sharey=True)
    if len(targets) == 1:
        axes = [axes]
    x = np.arange(len(models))

    for ax, target in zip(axes, targets):
        view = summary[summary["target"] == target].set_index("model").reindex(models)
        means = view["ccc_mean"].to_numpy(dtype=float)
        stds = view["ccc_std"].to_numpy(dtype=float)
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.bar(i, mean, 0.6, color=colors[i % len(colors)], edgecolor="#222222",
                   linewidth=0.6, yerr=std, capsize=3, alpha=0.85)
        for i, model in enumerate(models):
            if model == "M1_Base" or np.isnan(view.loc[model, "ccc_wilcoxon_p"]):
                continue
            if view.loc[model, "ccc_wilcoxon_p"] < alpha:
                ax.text(i, means[i] + stds[i] + 0.005, "*", ha="center",
                        fontsize=13, color="#a04000")
        ax.axhline(0, color="#999999", linewidth=0.8)
        ax.set_title(target, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([m.split("_", 1)[1] for m in models], rotation=15, fontsize=8.5)
        ax.set_xlabel("Modelo")
    axes[0].set_ylabel("CCC medio por fold (LOSO)")
    fig.suptitle("Experimento 3 (exploratorio): CCC de predicción continua "
                 "(* = mejora significativa vs Base, Wilcoxon)", fontsize=12.5, y=1.02)
    fig.tight_layout()

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    demo = pd.read_csv("results/output/KEMOCON/kemocon_feature_table.csv")
    out = Path("results/output/KEMOCON/experiment3")
    group = sys.argv[1] if len(sys.argv) > 1 else "disagreement"
    per_fold, summary = run_experiment3(demo, out_dir=out, target_group=group,
                                        model_kind="HGB", features=FEATURES_ALL)
    print_experiment3_results(summary)
    plot_experiment3_ccc(summary, out / f"fig_experiment3_ccc_{group}.png",
                         out / f"fig_experiment3_ccc_{group}.pdf")
