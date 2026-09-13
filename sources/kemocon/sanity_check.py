"""
kemocon/sanity_check.py

Filtra sujetos/ventanas de baja calidad usando las tablas ya calculadas en
data_quality_tables/.

Esquema real confirmado (sin columna pid -- la fila N tras la cabecera
corresponde al sujeto pid=N; filas 'n/a' = sujeto sin ese archivo E4):
  - e4_completeness.csv: fracción numérica directa por columna de modalidad
    (ACC,BVP,EDA,HR,IBI,TEMP), ej. 0.998
  - e4_zeros.csv: texto "count (fracción)" por celda, ej. "4007 (0.994)",
    columnas EDA,HR,IBI,TEMP (sin ACC/BVP)
  - e4_durations.csv / e4_outliers.csv: formato no confirmado -- se aplica
    el mismo parser genérico (_extract_fraction), que maneja tanto números
    directos como "count (fracción)" y 'n/a'; revisar si el formato real
    difiere.

Autor: Enrique
"""

from pathlib import Path
import re
import pandas as pd
import numpy as np

ROOT = Path("k-emocon")

_FRACTION_IN_PARENS = re.compile(r"\(([-\d.]+)\)")


def _extract_fraction(cell) -> float:
    """Parsea una celda de calidad a fracción numérica.

    Soporta numérico directo (0.998), texto "count (fracción)"
    (e4_zeros: "4007 (0.994)" -> 0.994) y 'n/a'/NaN.

    Args:
        cell: Valor de la celda.

    Returns:
        float: Fracción parseada, o np.nan si no se puede interpretar.
    """
    if pd.isna(cell):
        return np.nan
    if isinstance(cell, (int, float)):
        return float(cell)
    s = str(cell).strip()
    if s.lower() in ("n/a", "na", ""):
        return np.nan
    try:
        return float(s)
    except ValueError:
        pass
    m = _FRACTION_IN_PARENS.search(s)
    if m:
        return float(m.group(1))
    return np.nan


def _load_and_parse(path: Path) -> pd.DataFrame:
    """Lee una tabla de calidad y asigna pid por número de fila.

    Args:
        path (Path): Ruta al CSV de calidad.

    Returns:
        pandas.DataFrame: Tabla con `pid` (1-indexado) y celdas parseadas
            con `_extract_fraction`.
    """
    raw = pd.read_csv(path)
    parsed = raw.apply(lambda col: col.map(_extract_fraction))
    parsed.insert(0, "pid", range(1, len(parsed) + 1))
    return parsed


def load_quality_tables(root: Path = ROOT) -> dict[str, pd.DataFrame]:
    """Carga las cuatro tablas de calidad de datos E4.

    Args:
        root (Path, optional): Raíz del dataset. Por defecto es "k-emocon".

    Returns:
        dict: Con `e4_completeness`, `e4_durations`, `e4_outliers` y
            `e4_zeros`.
    """
    qdir = root / "data_quality_tables"
    return {
        "e4_completeness": _load_and_parse(qdir / "e4_completeness.csv"),
        "e4_durations": _load_and_parse(qdir / "e4_durations.csv"),
        "e4_outliers": _load_and_parse(qdir / "e4_outliers.csv"),
        "e4_zeros": _load_and_parse(qdir / "e4_zeros.csv"),
    }


def flag_low_quality_subjects(
    quality: dict[str, pd.DataFrame],
    min_completeness: float = 0.90,
    max_zero_frac: float = 0.20,
    max_outlier_frac: float = 0.10,
    min_duration_ratio: float = 0.50,
    verbose: bool = True,
) -> set[int]:
    """Identifica sujetos de baja calidad según criterios de señal E4.

    Excluye sujetos que:
    - tienen completeness < `min_completeness` en cualquier modalidad, o
    - tienen fracción de ceros > `max_zero_frac` en cualquier modalidad, o
    - tienen fracción de outliers > `max_outlier_frac` en cualquier
      modalidad, o
    - tienen una modalidad cuya duración cae muy por debajo de la duración
      máxima de sus propias modalidades (`min_duration_ratio`) — esto detecta
      caídas de señal específicas de una modalidad dentro de un sujeto por lo
      demás válido (ej. fallo del detector de picos IBI: sujeto con ACC=984s
      pero IBI=54s no es "sesión corta", es pérdida de señal IBI).

    Con verbose=True imprime el desglose por criterio para que las
    exclusiones queden documentadas (coherente con el estándar de
    transparencia metodológica del resto del framework).

    Args:
        quality (dict): Salida de `load_quality_tables`.
        min_completeness (float, optional): Mínimo de completeness. Por
            defecto es 0.90.
        max_zero_frac (float, optional): Máxima fracción de ceros. Por
            defecto es 0.20.
        max_outlier_frac (float, optional): Máxima fracción de outliers. Por
            defecto es 0.10.
        min_duration_ratio (float, optional): Ratio mínimo de duración entre
            modalidades. Por defecto es 0.50.
        verbose (bool, optional): Si imprime el desglose. Por defecto es True.

    Returns:
        set: PIDs de los sujetos excluidos.
    """
    reasons: dict[str, set[int]] = {}

    comp = quality["e4_completeness"]
    metric_cols = [c for c in comp.columns if c != "pid"]
    low_comp_mask = (comp[metric_cols] < min_completeness).any(axis=1)
    reasons["completeness"] = set(comp.loc[low_comp_mask, "pid"].tolist())

    zeros = quality["e4_zeros"]
    metric_cols_z = [c for c in zeros.columns if c != "pid"]
    high_zero_mask = (zeros[metric_cols_z] > max_zero_frac).any(axis=1)
    reasons["zeros"] = set(zeros.loc[high_zero_mask, "pid"].tolist())

    outliers = quality["e4_outliers"]
    metric_cols_o = [c for c in outliers.columns if c != "pid"]
    high_outlier_mask = (outliers[metric_cols_o] > max_outlier_frac).any(axis=1)
    reasons["outliers"] = set(outliers.loc[high_outlier_mask, "pid"].tolist())

    durations = quality["e4_durations"]
    dur_cols = [c for c in durations.columns if c != "pid"]
    row_max = durations[dur_cols].max(axis=1)
    row_min = durations[dur_cols].min(axis=1)
    duration_ratio = row_min / row_max
    low_duration_mask = duration_ratio < min_duration_ratio
    reasons["duration_dropout"] = set(durations.loc[low_duration_mask, "pid"].tolist())

    bad = set().union(*reasons.values())

    if verbose:
        for reason, pids in reasons.items():
            if pids:
                print(f"[sanity_check] excluidos por {reason}: {sorted(pids)}")
        print(f"[sanity_check] total excluidos por baja calidad: {sorted(bad)}")

    return bad


def sanity_check_feature_table(feature_table: pd.DataFrame,
                                excluded_subjects: set[int]) -> pd.DataFrame:
    """Filtra ventanas de sujetos excluidos y con fisiología completamente NaN.

    Args:
        feature_table (pandas.DataFrame): Tabla de características.
        excluded_subjects (set): Sujetos excluidos por baja calidad.

    Returns:
        pandas.DataFrame: Tabla filtrada y reiniciada en índice.
    """
    before = len(feature_table)
    df = feature_table[~feature_table["subject_id"].isin(excluded_subjects)].copy()

    physio_cols = [c for c in df.columns if c.endswith("_mean")]
    df = df.dropna(subset=physio_cols, how="all")

    print(f"[sanity_check] ventanas: {before} -> {len(df)} "
          f"(excluidos {len(excluded_subjects)} sujetos por baja calidad)")
    return df.reset_index(drop=True)