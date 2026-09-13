"""
kemocon/loader.py

Carga de datos crudos de K-EmoCon (esquema real confirmado):
  - metadata (subjects.csv, data_availability.csv)
  - señales E4 (ACC, BVP, EDA, HR, IBI, TEMP): CSVs ya tabulares, columnas
    timestamp, pid, [x,y,z | value], device_serial, device_number, entry_time
  - anotaciones (self, partner, external R1-R5, aggregated_external)

NOTA (pendiente): fuente Polar/BrainWave (columnas Attention, BrainWave,
Meditation, Polar_HR en data_availability.csv) no está en el árbol descargado
todavía -- ver load_neuro_polar_subject() como stub a completar cuando
localices esa carpeta/tarball.

Autor: Enrique
"""

from pathlib import Path
import pandas as pd

ROOT = Path("k-emocon")  # ajustar a la raíz real del dataset descargado

E4_MODALITIES = ["ACC", "BVP", "EDA", "HR", "IBI", "TEMP"]
EXTERNAL_RATERS = [f"R{i}" for i in range(1, 6)]

# Confirmado: Polar_HR.csv usa timestamp en MILISEGUNDOS. Los E4_*.csv no
# están confirmados (Empatica suele exportar en segundos) -- para no romper
# la sincronización silenciosamente, se normaliza TODO a segundos por
# magnitud: timestamps unix en ms rondan 1e12-1e13 hoy en día, en s rondan
# 1e9-1e10. Umbral conservador en 1e12.
_MS_THRESHOLD = 1e12


def _normalize_timestamp(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    """Normaliza los timestamps unix a segundos por magnitud.

    Los timestamps en milisegundos rondan 1e12-1e13; en segundos rondan
    1e9-1e10. Si la mediana supera `_MS_THRESHOLD` (1e12) se divide por 1000.

    Args:
        df (pandas.DataFrame): DataFrame con la columna de tiempo.
        col (str, optional): Columna de timestamp. Por defecto es
            "timestamp".

    Returns:
        pandas.DataFrame: Copia con el timestamp normalizado a segundos.
    """
    if col not in df.columns or df.empty:
        return df
    df = df.copy()
    median_val = df[col].median()
    if median_val > _MS_THRESHOLD:
        df[col] = df[col] / 1000.0
    return df


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def load_metadata(root: Path = ROOT) -> dict:
    """Carga los metadatos de sujetos y disponibilidad de datos.

    Args:
        root (Path, optional): Raíz del dataset. Por defecto es "k-emocon".

    Returns:
        dict: Con `subjects` y `availability`.
    """
    subjects = pd.read_csv(root / "metadata" / "subjects.csv")
    availability = pd.read_csv(root / "metadata" / "data_availability.csv")
    return {"subjects": subjects, "availability": availability}


def eligible_subjects(meta: dict, require_external_min: int = 3) -> list[int]:
    """Devuelve los sujetos elegibles para el análisis.

    Un sujeto es elegible si tiene E4 completo + self + partner + al menos
    `require_external_min` anotaciones externas.

    Args:
        meta (dict): Salida de `load_metadata`.
        require_external_min (int, optional): Mínimo de raters externos. Por
            defecto es 3.

    Returns:
        list: PIDs de los sujetos elegibles, ordenados.
    """
    avail = meta["availability"].copy()

    e4_cols = [f"E4_{m}" for m in E4_MODALITIES]
    ext_cols = [f"external_annotations_{r}" for r in EXTERNAL_RATERS]

    e4_ok = avail[e4_cols].all(axis=1)
    self_ok = avail["self_annotations"].astype(bool)
    partner_ok = avail["partner_annotations"].astype(bool)
    n_external = avail[ext_cols].astype(bool).sum(axis=1)

    mask = e4_ok & self_ok & partner_ok & (n_external >= require_external_min)
    return sorted(avail.loc[mask, "pid"].astype(int).tolist())


# ---------------------------------------------------------------------------
# Señales E4 (ya tabulares -- lectura directa)
# ---------------------------------------------------------------------------

def load_e4_subject(subject_id: int, root: Path = ROOT) -> dict[str, pd.DataFrame]:
    """Carga las señales E4 de un sujeto.

    Args:
        subject_id (int): Identificador del sujeto.
        root (Path, optional): Raíz del dataset. Por defecto es "k-emocon".

    Returns:
        dict: Modalidad -> DataFrame con columnas originales (timestamp,
            pid, value[/x,y,z], device_serial, device_number, entry_time).
            `timestamp` se normaliza a segundos.
    """
    subj_dir = root / "e4_data" / str(subject_id)
    out = {}
    for mod in E4_MODALITIES:
        f = subj_dir / f"E4_{mod}.csv"
        if f.exists():
            out[mod] = _normalize_timestamp(pd.read_csv(f))
    return out


# ---------------------------------------------------------------------------
# Fuente Polar/NeuroSky (confirmado: neurosky_polar_data/<pid>/...)
# Cobertura irregular por sujeto -- no todos tienen los 4 archivos
# (ej. sujeto 20 solo tiene BrainWave.csv; sujeto 1 no tiene Attention/Meditation).
# ---------------------------------------------------------------------------

NEUROSKY_POLAR_MODALITIES = ["Attention", "BrainWave", "Meditation", "Polar_HR"]


def load_neurosky_polar_subject(subject_id: int, root: Path = ROOT) -> dict[str, pd.DataFrame]:
    """Carga las señales NeuroSky/Polar de un sujeto.

    La cobertura es irregular por sujeto — no todos tienen los 4 archivos
    (ej. sujeto 20 solo tiene BrainWave.csv; sujeto 1 no tiene
    Attention/Meditation).

    Args:
        subject_id (int): Identificador del sujeto.
        root (Path, optional): Raíz del dataset. Por defecto es "k-emocon".

    Returns:
        dict: Modalidad -> DataFrame (los disponibles), con timestamp
            normalizado a segundos.
    """
    subj_dir = root / "neurosky_polar_data" / str(subject_id)
    out = {}
    for mod in NEUROSKY_POLAR_MODALITIES:
        f = subj_dir / f"{mod}.csv"
        if f.exists():
            df = pd.read_csv(f)
            # Polar_HR confirmado en ms; Attention/BrainWave/Meditation sin
            # confirmar -- _normalize_timestamp es un no-op si no hay
            # columna 'timestamp' o si ya está en segundos.
            out[mod] = _normalize_timestamp(df)
    return out


# ---------------------------------------------------------------------------
# Anotaciones
# ---------------------------------------------------------------------------

def load_annotations_subject(subject_id: int, root: Path = ROOT) -> dict:
    """Carga todas las anotaciones de un sujeto.

    Args:
        subject_id (int): Identificador del sujeto.
        root (Path, optional): Raíz del dataset. Por defecto es "k-emocon".

    Returns:
        dict: Con `self`, `partner`, `aggregated_external` y
            `external_raters` (dict R1-R5 -> DataFrame o None si no existe).
    """
    ann_root = root / "emotion_annotations"
    pid = f"P{subject_id}"

    def _read(p):
        """Lee un CSV si existe.

        Args:
            p (Path): Ruta al archivo.

        Returns:
            pandas.DataFrame o None: DataFrame si existe, None si no.
        """
        return pd.read_csv(p) if p.exists() else None

    out = {
        "self": _read(ann_root / "self_annotations" / f"{pid}.self.csv"),
        "partner": _read(ann_root / "partner_annotations" / f"{pid}.partner.csv"),
        "aggregated_external": _read(
            ann_root / "aggregated_external_annotations" / f"{pid}.external.csv"
        ),
        "external_raters": {},
    }
    for r in EXTERNAL_RATERS:
        p = ann_root / "external_annotations" / f"{pid}.{r}.csv"
        df = _read(p)
        if df is not None:
            out["external_raters"][r] = df
    return out


# ---------------------------------------------------------------------------
# Inspección (correr primero sobre 1-2 sujetos reales)
# ---------------------------------------------------------------------------

def inspect_sample(root: Path = ROOT, subject_id: int = 1):
    """Inspecciona el esquema real de los datos sobre 1-2 sujetos.

    Imprime head de availability, E4 BVP (con rango de timestamp), self
    annotation y Polar_HR para verificar el esquema antes de usar el
    agregador.

    Args:
        root (Path, optional): Raíz del dataset. Por defecto es "k-emocon".
        subject_id (int, optional): Sujeto a inspeccionar. Por defecto es 1.

    Returns:
        None
    """
    print("== data_availability ==")
    meta = load_metadata(root)
    print(meta["availability"].head())

    print("\n== E4 BVP head (timestamp ya normalizado a segundos si hacía falta) ==")
    e4 = load_e4_subject(subject_id, root)
    if "BVP" in e4:
        print(e4["BVP"].head())
        ts_min, ts_max = e4["BVP"]["timestamp"].min(), e4["BVP"]["timestamp"].max()
        print(f"timestamp range: {ts_min} - {ts_max}")
        print("  (unix actual en segundos ronda ~1.75e9 -- si este rango no se "
              "parece, revisa _MS_THRESHOLD o el esquema real de la columna)")

    print("\n== self annotation head ==")
    ann = load_annotations_subject(subject_id, root)
    if ann["self"] is not None:
        print(ann["self"].head())

    print("\n== Polar_HR head (verificar nombre de columna de valor/tiempo) ==")
    np_data = load_neurosky_polar_subject(subject_id, root)
    if "Polar_HR" in np_data:
        print(np_data["Polar_HR"].head())
        print("columns:", list(np_data["Polar_HR"].columns))


if __name__ == "__main__":
    inspect_sample()