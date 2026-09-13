"""
exist/downstream_transfer.py

EXPLORATORY UTILITY-TRANSFER TEST (no confirmatorio; se reporta solo si el
resultado lo justifica).

Pone a prueba la predicción de la auditoría de agregación de EXIST: si el
agregado fisiológico por meme no tiene señal reproducible (ICC_meme≈0,
correlación entre viewers≈0), entonces NO debe aportar utilidad downstream
sobre un baseline de contenido; el canal conductual (tiempo de reacción,
parpadeos), que sí tiene señal de meme reproducible, sí debería aportar algo.

Diseño (unidad = meme, n≈3984):

  Targets:
    - regresión     : entropy_22, entropy_23, soft_21_yes
    - clasificación : hard_21 (YES/NO)

  Bloques de features:
    - content            : texto OCR (TF-IDF) + 64 dims de imagen (img_pca)
    - behavioral         : reaction_time y blinks (canal fiable)
    - physiological      : componentes PSRI + HR/pupila crudos (canal no fiable)
    - content_behavioral : content + behavioral
    - content_physio     : content + physiological
    - content_combined   : content + behavioral + physiological

  Modelos (fijos, no optimizados): Ridge (regresión) y LogisticRegression
  (clasificación). El preprocesado (imputación por mediana, estandarización,
  TF-IDF) va dentro del Pipeline y se ajusta SOLO en train.

  CV:
    - random   : KFold(5, shuffle, seed)
    - grouped  : GroupKFold(5) por conjunto de viewers del meme (robustez
                 frente a fuga de rasgo de sujeto en los bloques fisiológicos)

  Métricas (out-of-fold): Spearman/RMSE (regresión), macro-F1/ROC-AUC
  (clasificación). Se reporta la métrica pooled sobre las predicciones OOF y
  la media±std por fold. Control de azar: target permutado sobre content_combined.

Hipótesis / lectura esperada: content domina; content_physio ≈ content (la
fisiología no aporta); content_behavioral ≈ o > content (el canal fiable sí).
Si la fisiología mejora sobre el contenido, la auditoría de agregación quedaría
en cuestión (resultado a revisar, no a esconder).

RESULTADO EMPÍRICO (EXIST, 3984 memes; pooled out-of-fold; CV aleatoria y
agrupada por viewers coinciden):

  | Bloque               | entropy_22 | entropy_23 | soft_21_yes | hard_21 (AUC) |
  |----------------------|-----------|-----------|------------|---------------|
  | content              | 0.210     | 0.287     | 0.453      | 0.736         |
  | content + conductual | 0.222     | 0.300     | 0.458      | 0.742         |
  | content + fisiológico| 0.211     | 0.286     | 0.452      | 0.736         |
  | content + ambos      | 0.223     | 0.298     | 0.456      | 0.740         |
  | control permutado    | ≈0        | ≈0        | ≈0         | ≈0.5          |

El contenido es un baseline fuerte (ρ≈0.45 en `soft_21_yes`; AUC≈0.74 en
`hard_21`). Añadir el canal conductual (RT, parpadeos — el único con fiabilidad
de meme reproducible) mejora de forma pequeña pero consistente en los cuatro
objetivos (ρ +0.006..+0.016; AUC +0.007); añadir el fisiológico (PSRI, HR,
pupila) no cambia el resultado y el combinado iguala al conductual. La
auditoría de agregación, por tanto, ANTICIPA qué canal aporta utilidad
downstream: solo el que tiene fiabilidad de meme reproducible.

Ejecución:
  .venv/bin/python -m sources.exist.downstream_transfer

Salidas en results/output/EXIST/downstream_transfer/.

Autor: Enrique
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, f1_score, roc_auc_score
from sklearn.model_selection import KFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sources.common.common import logger, writeLog
from sources.exist.loader import load_dataframes

DATA_ROOT = Path("results/input/EXIST")
CORPUS_JSON = DATA_ROOT / "corpus_EXIST2026_training.json"
FEATURE_CSV = Path("results/output/EXIST/physio_with_psri_memes.csv")
OUT_DIR = FEATURE_CSV.parent / "downstream_transfer"

RANDOM_SEED = 1234
N_SPLITS = 5

REGRESSION_TARGETS = ["entropy_22", "entropy_23", "soft_21_yes"]
CLASSIFICATION_TARGETS = ["hard_21"]

TEXT_COL = "text"

BEHAVIORAL = [
    "et_reaction_time_mean", "et_reaction_time_std",
    "et_blinks_count_mean", "et_blinks_count_std",
]
PHYSIOLOGICAL = [
    "PSRI", "PSRI_hr_mean", "PSRI_hr_std", "PSRI_et_mean", "PSRI_et_std",
    "S_estab", "S_coher", "S_cond",
    "hr_garmin_hr_mean_mean", "hr_garmin_hr_mean_std",
    "hr_garmin_hr_std_mean", "hr_garmin_hr_std_std",
    "et_3d_eye_states_pupil diameter left [mm]_mean_mean",
    "et_3d_eye_states_pupil diameter left [mm]_mean_std",
    "et_3d_eye_states_pupil diameter left [mm]_std_mean",
]
IMG_COLS = [f"img_pca_{i}" for i in range(64)] + ["img_mean", "img_std"]

CONTENT = [TEXT_COL] + IMG_COLS

BLOCKS = {
    "content": CONTENT,
    "behavioral": BEHAVIORAL,
    "physiological": PHYSIOLOGICAL,
    "content_behavioral": CONTENT + BEHAVIORAL,
    "content_physio": CONTENT + PHYSIOLOGICAL,
    "content_combined": CONTENT + BEHAVIORAL + PHYSIOLOGICAL,
}


# ---------------------------------------------------------------------------
# Tablas auxiliares
# ---------------------------------------------------------------------------

def _content_table() -> pd.DataFrame:
    """Tabla por meme con texto OCR y features de imagen del corpus.

    Returns:
        pandas.DataFrame: `meme_id`, `text`, `img_*` (una fila por meme).
    """
    with open(CORPUS_JSON, encoding="utf-8") as f:
        data = json.load(f)
    rows = {}
    for r in data:
        mid = int(r["id_EXIST"])
        if mid in rows:
            continue
        row = {"meme_id": mid, TEXT_COL: str(r.get("text", ""))}
        for k, v in r.items():
            if k.startswith("img_"):
                row[k] = v
        rows[mid] = row
    return pd.DataFrame(rows.values())


def _viewer_groups() -> pd.Series:
    """Conjunto de viewers por meme, como string ordenada `u1|u2`.

    Returns:
        pandas.Series: índice `meme_id`, valor grupo.
    """
    df_hr, _, df_et = load_dataframes(DATA_ROOT)
    common = set(df_hr["meme_id"]) & set(df_et["meme_id"])
    hr = df_hr[df_hr["meme_id"].isin(common)][["meme_id", "username"]]
    et = df_et[df_et["meme_id"].isin(common)][["meme_id", "username"]]
    viewers = pd.concat([hr, et]).drop_duplicates()
    return viewers.groupby("meme_id")["username"].apply(
        lambda s: "|".join(sorted(s.astype(str)))
    )


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

def _make_preprocessor(features: list[str]) -> ColumnTransformer:
    """Preprocesado: TF-IDF para texto y (imputación+escalado) para numéricas.

    Args:
        features (list): Columnas del bloque.

    Returns:
        ColumnTransformer: Transformador ajustable dentro del Pipeline.
    """
    transformers = []
    if TEXT_COL in features:
        transformers.append((
            "txt",
            TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2,
                            sublinear_tf=True),
            TEXT_COL,
        ))
    numeric = [f for f in features if f != TEXT_COL]
    transformers.append((
        "num",
        Pipeline([("imp", SimpleImputer(strategy="median")),
                  ("sc", StandardScaler())]),
        numeric,
    ))
    return ColumnTransformer(transformers, remainder="drop")


def _make_model(features: list[str], task: str) -> Pipeline:
    """Pipeline preprocesado + modelo (Ridge o LogisticRegression)."""
    if task == "regression":
        model = Ridge(alpha=1.0)
    else:
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("pre", _make_preprocessor(features)), ("model", model)])


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

def _evaluate(df: pd.DataFrame, target: str, features: list[str], task: str,
              cv, groups: np.ndarray | None = None,
              permute: bool = False) -> dict:
    """Evalúa un bloque de features con CV out-of-fold.

    Args:
        df (pandas.DataFrame): Feature table por meme (incluye texto).
        target (str): Columna objetivo.
        features (list): Columnas de entrada.
        task (str): 'regression' o 'classification'.
        cv: Splitter de sklearn.
        groups (ndarray or None, optional): Grupos para GroupKFold.
        permute (bool, optional): Si True, permuta el target (control de azar).

    Returns:
        dict: métricas pooled, media±std por fold y nº de folds válidos.
    """
    X = df[features]
    if task == "regression":
        y = df[target].astype(float)
    else:
        y = (df[target].astype(str) == "YES").astype(int)
    model = _make_model(features, task)

    rng = np.random.default_rng(RANDOM_SEED)
    if permute:
        y = pd.Series(rng.permutation(y.to_numpy()), index=y.index)

    oof = np.full(len(df), np.nan)
    folds = []
    split_groups = groups if isinstance(cv, GroupKFold) else None
    for tr, te in cv.split(X, y, split_groups):
        model.fit(X.iloc[tr], y.iloc[tr])
        if task == "regression":
            pred = model.predict(X.iloc[te])
            oof[te] = pred
            folds.append({
                "spearman": spearmanr(y.iloc[te], pred).correlation,
                "rmse": np.sqrt(mean_squared_error(y.iloc[te], pred)),
            })
        else:
            pred = model.predict(X.iloc[te])
            proba = model.predict_proba(X.iloc[te])[:, 1]
            oof[te] = proba
            try:
                auc = roc_auc_score(y.iloc[te], proba)
            except ValueError:
                auc = np.nan
            folds.append({
                "f1": f1_score(y.iloc[te], pred, average="macro"),
                "auc": auc,
            })

    fold_df = pd.DataFrame(folds)
    mask = ~np.isnan(oof)
    pooled = {}
    if task == "regression":
        pooled["spearman"] = spearmanr(y[mask], oof[mask]).correlation
        pooled["rmse"] = np.sqrt(mean_squared_error(y[mask], oof[mask]))
        n_valid = int(fold_df["spearman"].notna().sum())
    else:
        pooled["f1"] = f1_score(y[mask], (oof[mask] >= 0.5).astype(int),
                                average="macro")
        pooled["auc"] = roc_auc_score(y[mask], oof[mask])
        n_valid = int(fold_df["auc"].notna().sum())

    out = {"n": int(mask.sum()), "n_folds": n_valid}
    for col in fold_df.columns:
        out[f"{col}_mean"] = float(fold_df[col].mean(skipna=True))
        out[f"{col}_std"] = float(fold_df[col].std(skipna=True))
        out[f"{col}_pooled"] = float(pooled[col])
    return out


def run_downstream_transfer(out_dir: Path | None = None) -> dict:
    """Ejecuta el test de transferencia de utilidad en EXIST.

    Args:
        out_dir (Path or None, optional): Directorio de salida. Si es None se
            usa `results/output/EXIST/downstream_transfer`.

    Returns:
        dict: Con la tabla larga de resultados.
    """
    if out_dir is None:
        out_dir = OUT_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURE_CSV)
    df = df.merge(_content_table(), on="meme_id", how="left")
    groups = _viewer_groups()
    df = df.merge(groups.rename("viewer_group"), left_on="meme_id",
                  right_index=True, how="left")
    df["viewer_group"] = df["viewer_group"].fillna("unknown")
    df[TEXT_COL] = df[TEXT_COL].fillna("")

    writeLog("info", logger,
             f"[downstream_transfer] {len(df)} memes; salidas en {out_dir}")

    rows = []
    for task, targets in (("regression", REGRESSION_TARGETS),
                          ("classification", CLASSIFICATION_TARGETS)):
        for target in targets:
            sub = df[df[target].notna()].reset_index(drop=True)
            sub_groups = sub["viewer_group"].to_numpy()
            cv_schemes = {
                "random": KFold(n_splits=N_SPLITS, shuffle=True,
                                random_state=RANDOM_SEED),
                "grouped": GroupKFold(n_splits=N_SPLITS),
            }
            for cv_name, cv in cv_schemes.items():
                for block, feats in BLOCKS.items():
                    feats = [f for f in feats if f in sub.columns]
                    res = _evaluate(sub, target, feats, task, cv, sub_groups)
                    rows.append({"target": target, "task": task,
                                 "cv": cv_name, "block": block, **res})
                feats = [f for f in BLOCKS["content_combined"]
                         if f in sub.columns]
                res = _evaluate(sub, target, feats, task, cv, sub_groups,
                                permute=True)
                rows.append({"target": target, "task": task, "cv": cv_name,
                             "block": "permuted_control", **res})

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "downstream_transfer_results.csv", index=False)

    print("\n== [DOWNSTREAM TRANSFER] Métrica pooled out-of-fold ==")
    order = ["content", "behavioral", "physiological", "content_behavioral",
             "content_physio", "content_combined", "permuted_control"]
    for task in ["regression", "classification"]:
        sub = results[results["task"] == task]
        if sub.empty:
            continue
        metric = "spearman_pooled" if task == "regression" else "auc_pooled"
        print(f"\n-- {task} ({metric}) --")
        piv = sub.pivot_table(index=["target", "cv"], columns="block",
                              values=metric)
        cols = [c for c in order if c in piv.columns]
        print(piv[cols].round(4).to_string())

    return {"results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test de transferencia de utilidad downstream en EXIST "
                    "(contenido vs contenido+fisiología)")
    parser.add_argument("--out", type=Path, default=None,
                        help="directorio de salida (default: "
                             "results/output/EXIST/downstream_transfer)")
    args = parser.parse_args()
    run_downstream_transfer(args.out)
    print(f"\n[downstream_transfer] salidas en "
          f"{args.out or OUT_DIR}/")
