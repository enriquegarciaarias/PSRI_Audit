"""
kemocon/aggregator.py

Sincroniza señales E4 + Polar_HR/Attention (frecuencias distintas por
modalidad) y anotaciones (~5s) en una tabla común de ventanas por sujeto.

Fix: window_signal() ahora prefija mean/std/n_samples POR CANAL dentro de la
propia función (en vez de renombrarlos después en build_subject_table), para
que al fusionar más de dos tablas no colisionen nombres genéricos como
'n_samples' entre canales (causaba MergeError: duplicate columns
{'n_samples_x','n_samples_y'} al fusionar la 3ª tabla en adelante).

Autor: Enrique
"""

import pandas as pd
import numpy as np

WINDOW_S = 5.0  # resolución nativa de las anotaciones K-EmoCon
TIME_COL = "timestamp"

VALUE_COL_CANDIDATES = ["value", "bpm", "hr", "HR"]


def _resolve_value_col(df: pd.DataFrame) -> str:
    """Resuelve el nombre de la columna de valores de una señal.

    Args:
        df (pandas.DataFrame): DataFrame con la señal.

    Returns:
        str: Nombre de la columna de valores.

    Raises:
        KeyError: Si ninguna columna candidata está presente.
    """
    for c in VALUE_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError(f"No se encontró columna de valor en {list(df.columns)}; "
                    f"añade el nombre real a VALUE_COL_CANDIDATES.")


def _to_relative_time(df: pd.DataFrame, t0: float, time_col: str = TIME_COL) -> pd.DataFrame:
    """Añade el tiempo relativo a `t0` como columna `t_rel`.

    Args:
        df (pandas.DataFrame): DataFrame con la columna de tiempo.
        t0 (float): Timestamp de referencia.
        time_col (str, optional): Columna de tiempo. Por defecto es
            "timestamp".

    Returns:
        pandas.DataFrame: Copia con la columna `t_rel` añadida.
    """
    df = df.copy()
    df["t_rel"] = df[time_col] - t0
    return df


def _window_index(t_rel: pd.Series, window_s: float = WINDOW_S) -> pd.Series:
    """Asigna el índice de ventana a partir del tiempo relativo.

    Args:
        t_rel (pandas.Series): Tiempos relativos.
        window_s (float, optional): Tamaño de ventana en segundos. Por
            defecto es 5.0.

    Returns:
        pandas.Series: Índices de ventana enteros.
    """
    return (t_rel // window_s).astype(int)


def window_signal(df: pd.DataFrame, value_cols: list[str], t0: float, prefix: str,
                   window_s: float = WINDOW_S, time_col: str = TIME_COL) -> pd.DataFrame:
    """Colapsa una señal a medias/desviaciones por ventana.

    Genera `{prefix}_mean`, `{prefix}_std`, `{prefix}_n_samples` por ventana
    de `window_s` segundos. Para señales multi-columna (ACC: x,y,z) no se
    devuelve una media por eje, solo std agregada (media de las stds de los
    3 ejes) y n_samples — igual que antes, pero ahora con nombres únicos
    garantizados para poder fusionar N tablas sin colisión.

    Args:
        df (pandas.DataFrame): DataFrame con la señal cruda.
        value_cols (list): Columnas de valores (1 para univariada, varias
            para ACC).
        t0 (float): Timestamp de referencia.
        prefix (str): Prefijo de las columnas de salida.
        window_s (float, optional): Tamaño de ventana en segundos. Por
            defecto es 5.0.
        time_col (str, optional): Columna de tiempo. Por defecto es
            "timestamp".

    Returns:
        pandas.DataFrame: Una fila por ventana con las métricas agregadas.
    """
    d = _to_relative_time(df, t0, time_col)
    d["win"] = _window_index(d["t_rel"], window_s)
    grouped = d.groupby("win")[value_cols]
    n_samples = grouped.size()

    if len(value_cols) == 1:
        mean_s = grouped.mean()[value_cols[0]]
        std_s = grouped.std()[value_cols[0]]
        out = pd.DataFrame({
            f"{prefix}_mean": mean_s,
            f"{prefix}_std": std_s,
            f"{prefix}_n_samples": n_samples,
        })
    else:
        std_df = grouped.std()
        out = pd.DataFrame({
            f"{prefix}_std": std_df.mean(axis=1),
            f"{prefix}_n_samples": n_samples,
        })

    return out.reset_index()


def raw_window_groups(df: pd.DataFrame, value_col: str, t0: float,
                       window_s: float = WINDOW_S, time_col: str = TIME_COL,
                       max_win: int | None = None) -> list[np.ndarray]:
    """Devuelve las muestras crudas agrupadas por ventana.

    Utilidad, no usada por el pipeline actual de S_cond, que ahora opera
    sobre columnas ya agregadas.

    Args:
        df (pandas.DataFrame): DataFrame con la señal.
        value_col (str): Columna de valores.
        t0 (float): Timestamp de referencia.
        window_s (float, optional): Tamaño de ventana. Por defecto es 5.0.
        time_col (str, optional): Columna de tiempo. Por defecto es
            "timestamp".
        max_win (int, optional): Número máximo de ventanas. Si es None se
            deriva del máximo de la señal.

    Returns:
        list: Un array numpy por ventana con sus muestras.
    """
    d = _to_relative_time(df, t0, time_col)
    d["win"] = _window_index(d["t_rel"], window_s)
    if max_win is None:
        max_win = int(d["win"].max()) if len(d) else 0
    grouped = d.groupby("win")[value_col].apply(lambda s: s.to_numpy())
    return [grouped.get(w, np.array([])) for w in range(max_win + 1)]


def window_annotation(df: pd.DataFrame, t0: float, window_s: float = WINDOW_S,
                       value_cols: tuple[str, ...] = ("valence", "arousal"),
                       offset_s: float = 0.0) -> pd.DataFrame:
    """Agrega anotaciones a una fila por ventana.

    Fix: si la anotación bruta viene muestreada más fina que `window_s` (ej.
    anotación continua cada 1-2s en vez de cada 5s), sin agregar por ventana
    se devolvían MÚLTIPLES filas por 'win' — al fusionar con how='left' en
    build_subject_table, eso duplicaba la fila fisiológica entera por cada
    muestra de anotación extra en esa ventana, contaminando cualquier
    correlación calculada después. Ahora se agrega con groupby('win').mean(),
    igual que `window_signal()` ya hace para las señales fisiológicas — una
    sola fila por ventana, siempre.

    Alineación temporal: en K-EmoCon la columna `seconds` de las anotaciones
    es relativa al INICIO DEL DEBATE (`startTime`), mientras que la fisiología
    (E4) se ancla a `t0 = initTime` (inicio de la grabación del sensor). El
    intervalo `initTime -> startTime` (mediana 8.6 min) es el tramo pre-debate
    sin interacción ni anotaciones. `offset_s` debe ser `startTime - t0` (en
    segundos) para expresar las anotaciones en el mismo reloj que la
    fisiología y alinearlas con sus ventanas reales.

    Args:
        df (pandas.DataFrame): Anotaciones crudas.
        t0 (float): Timestamp de referencia (inicio de la fisiología).
        window_s (float, optional): Tamaño de ventana. Por defecto es 5.0.
        value_cols (tuple, optional): Columnas de anotación. Por defecto
            ("valence", "arousal").
        offset_s (float, optional): Desplazamiento en segundos a sumar al
            tiempo de la anotación antes de ventanear (para alinear
            `seconds` relativos al debate con la fisiología anclada a t0).
            Por defecto es 0.0.

    Returns:
        pandas.DataFrame: Una fila por ventana con la media de cada
            anotación (o solo 'win' si no hay columnas presentes).
    """
    d = df.copy()
    time_col = "seconds" if "seconds" in d.columns else d.columns[0]
    t_raw = d[time_col] if time_col == "seconds" else d[time_col] - t0
    d["t_rel"] = t_raw + offset_s
    d["win"] = _window_index(d["t_rel"], window_s)
    cols_present = [c for c in value_cols if c in d.columns]
    if not cols_present:
        return d[["win"]].drop_duplicates()
    return d.groupby("win")[cols_present].mean().reset_index()


def build_subject_table(subject_id: int, e4: dict, polar: dict, ann: dict, t0: float,
                        ann_offset_s: float = 0.0) -> pd.DataFrame:
    """Une E4 + Polar/Attention/Meditation + anotaciones por ventana y sujeto.

    IMPORTANTE: las fuentes E4 (core_tables) definen la POBLACIÓN de
    ventanas vía merge 'outer' entre sí (todas cubren la misma sesión de
    debate, es correcto unir su rango). Las fuentes auxiliares de otro
    dispositivo (aux_tables: Polar_HR, Attention, Meditation) se fusionan con
    'left' SOBRE esa población ya fijada, nunca con 'outer' — si se fusionan
    con outer, un dispositivo con un rango de grabación distinto (empieza
    antes, termina después) EXPANDE silenciosamente el conjunto de ventanas
    y contamina la población de referencia global usada por S_estab (Tabla
    10.2.1: mediana/MAD sobre TODOS los pares sujeto-ventana). Bug real
    detectado: al añadir Meditation con outer, cambiaron números de S_estab
    que no dependen de Meditation en absoluto.

    `ann_offset_s` alinea el reloj de las anotaciones con el de la
    fisiología: en K-EmoCon la columna `seconds` es relativa al inicio del
    debate (`startTime`), mientras que la fisiología se ancla a
    `t0 = initTime`. Pasar `ann_offset_s = startTime - t0` (en segundos)
    expresa ambas en la misma escala (ver `window_annotation`).

    Args:
        subject_id (int): Identificador del sujeto.
        e4 (dict): Señales E4 (BVP, EDA, HR, TEMP, IBI, ACC).
        polar (dict): Señales del dispositivo Polar (Polar_HR, Attention,
            Meditation).
        ann (dict): Anotaciones ('self', 'partner', 'external_raters').
        t0 (float): Timestamp mínimo entre todas las modalidades E4.
        ann_offset_s (float, optional): Offset en segundos a sumar al
            tiempo de las anotaciones antes de ventanear. Por defecto es 0.0.

    Returns:
        pd.DataFrame: Tabla ventana x features por sujeto, ordenada por
            ventana.

    Raises:
        AssertionError: Si el merge de fuentes auxiliares cambia el número de
            ventanas o aparecen ventanas duplicadas.
    """
    core_tables = []
    aux_tables = []

    if "BVP" in e4:
        core_tables.append(window_signal(e4["BVP"], ["value"], t0, prefix="bvp"))
    if "EDA" in e4:
        core_tables.append(window_signal(e4["EDA"], ["value"], t0, prefix="eda"))
    if "HR" in e4:
        core_tables.append(window_signal(e4["HR"], ["value"], t0, prefix="hr"))
    if "TEMP" in e4:
        core_tables.append(window_signal(e4["TEMP"], ["value"], t0, prefix="temp"))
    if "IBI" in e4:
        core_tables.append(window_signal(e4["IBI"], ["value"], t0, prefix="ibi"))
    if "ACC" in e4:
        core_tables.append(window_signal(e4["ACC"], ["x", "y", "z"], t0, prefix="acc"))

    if "Polar_HR" in polar:
        val_col = _resolve_value_col(polar["Polar_HR"])
        aux_tables.append(window_signal(polar["Polar_HR"], [val_col], t0, prefix="polar_hr"))

    if "Attention" in polar:
        val_col = _resolve_value_col(polar["Attention"])
        aux_tables.append(window_signal(polar["Attention"], [val_col], t0, prefix="attention"))

    if "Meditation" in polar:
        val_col = _resolve_value_col(polar["Meditation"])
        aux_tables.append(window_signal(polar["Meditation"], [val_col], t0, prefix="meditation"))

    if not core_tables:
        return pd.DataFrame(columns=["subject_id", "win"])

    merged = core_tables[0][["win"]].drop_duplicates()
    for t in core_tables:
        merged = merged.merge(t, on="win", how="outer")  # E4 entre sí: outer OK, misma sesión

    n_windows_e4_only = len(merged)
    for t in aux_tables:
        merged = merged.merge(t, on="win", how="left")  # auxiliares: NUNCA expanden la población

    assert len(merged) == n_windows_e4_only, (
        f"BUG: el merge de fuentes auxiliares cambió el nº de ventanas "
        f"({n_windows_e4_only} -> {len(merged)}) para el sujeto {subject_id}; "
        f"revisar que aux_tables use 'left', no 'outer'."
    )
    assert not merged["win"].duplicated().any(), (
        f"BUG: ventanas 'win' duplicadas tras el merge para el sujeto {subject_id}."
    )

    if ann.get("self") is not None:
        merged = merged.merge(
            window_annotation(ann["self"], t0, offset_s=ann_offset_s).add_prefix("self_").rename(
                columns={"self_win": "win"}),
            on="win", how="left")
    if ann.get("partner") is not None:
        merged = merged.merge(
            window_annotation(ann["partner"], t0, offset_s=ann_offset_s).add_prefix("partner_").rename(
                columns={"partner_win": "win"}),
            on="win", how="left")

    for rater, df in ann.get("external_raters", {}).items():
        merged = merged.merge(
            window_annotation(df, t0, offset_s=ann_offset_s).add_prefix(f"{rater}_").rename(
                columns={f"{rater}_win": "win"}),
            on="win", how="left")

    merged.insert(0, "subject_id", subject_id)
    return merged.sort_values("win").reset_index(drop=True)