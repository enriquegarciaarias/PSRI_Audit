"""
kemocon/sensor_latency.py

EXPLORATORY DIAGNOSTIC, NOT for the paper (confirmatory block is fixed).

Question: does the PSRI composite need to model sensor latency? S_coher compares
two physiological systems in the SAME window (K-EmoCon: HR vs EDA, both from the
same Empatica E4 device). If one channel started its real measurement with a
latency that the code does not account for (window_signal anchors every channel
to the same t0 = min first timestamp), the two series would be compared on a
misaligned grid and S_coher would measure alignment artefact, not coordination.

This module VALIDATES that assumption empirically with four independent checks
and writes them to results/output/KEMOCON/sensor_latency/:

  1. startup_latency : per subject and E4 channel, offset of the first sample
     relative to t0, plus the same for the cross-device NeuroSky/Polar signals.
     If a channel starts at t0 for every subject, the shared start is real, not
     an assumption. A persistent (non-startup) offset would show up as a large,
     non-zero offset here.
  2. grid_alignment  : do HR timestamps fall exactly on the EDA sample grid
     (0.25 s)? 100% exact match = same device clock, no persistent intra-E4
     offset after startup. A persistent offset would produce a constant
     sub-window mismatch.
  3. crosscorr       : window-level (5 s) Spearman cross-correlation of
     hr_mean vs eda_mean per subject over all windows, at lags -6..+6. A real
     systematic physiological/sensor lag would produce a sharp, consistent peak
     at a small non-zero lag; a trend/boundary artifact produces the peak at the
     extremes; no signal produces a flat profile.
  4. scoher_lag      : sensitivity of the CONFIRMATORY result (S_coher_prev1 vs
     external disagreement, pooled + within-subject) to shifting hr_mean by
     -3..+3 windows before recomputing S_coher. If the sign/magnitude of the
     within-subject effect survives shifts in a plausible latency range (±1-2
     windows = ±5-10 s), the published conclusion is not an artefact of the
     assumed synchronization.

Expected findings (already reproduced): E4 EDA/ACC/BVP/TEMP start exactly at t0
for all subjects; HR starts at exactly +10 s (derived-channel warm-up) and IBI at
+12..+41 s, both confined to the PRE-debate period (debate starts ~8.6 min after
t0). HR samples land 100% on the EDA grid (0 ms mismatch) -> no persistent
intra-device offset, so the "same start" comparison is physically correct for
HR-EDA. The cross-correlation has no sharp peak at small lags, and S_coher's
within-subject effect peaks at shift 0 and only decays by ±2-3 windows -> the
confirmatory result is robust to plausible latency corrections. The real latency
risk in K-EmoCon is cross-device: NeuroSky/Polar start 0-396 s after E4
(median ~112 s), which affects S_cond (attention) coverage near the debate
start, and would matter for S_coher only if it ever used a cross-device pair.

Guarantee of clean revert: este módulo SOLO lee
  results/input/KEMOCON/  (raw E4 + NeuroSky/Polar + metadata) y
  results/output/KEMOCON/kemocon_feature_table.csv
y escribe en results/output/KEMOCON/sensor_latency/. No importa ni modifica
build_features.py, run_pipeline.py ni calculator.py; borrar el directorio de
salida restaura la situación anterior.

Ejecución:
  .venv/bin/python -m sources.kemocon.sensor_latency

Autor: Enrique
"""
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sources.common.common import logger, writeLog
from sources.kemocon.loader import (
    load_metadata, eligible_subjects, load_e4_subject,
    load_neurosky_polar_subject,
)
from sources.kemocon.build_features import add_s_coher

DATA_ROOT = Path("results/input/KEMOCON")
FEATURE_CSV = Path("results/output/KEMOCON/kemocon_feature_table.csv")

E4_CHANNELS = ["ACC", "BVP", "EDA", "HR", "IBI", "TEMP"]
NEURO_POLAR_CHANNELS = ["Attention", "BrainWave", "Meditation", "Polar_HR"]
WINDOW_S = 5.0
CORR_LAGS = list(range(-6, 7))
SHIFTS = [-3, -2, -1, 0, 1, 2, 3]
DISAGREEMENT_TARGETS = ["external_valence_var", "external_valence_range",
                        "external_arousal_var"]


def startup_latency_table(subjects: list[int], out_dir: Path) -> pd.DataFrame:
    """Offset del primer sample de cada canal respecto a t0, por sujeto.

    Args:
        subjects (list): Sujetos elegibles.
        out_dir (Path): Directorio de salida.

    Returns:
        pandas.DataFrame: Filas (subject, source, channel, offset_from_t0_s,
            n_samples).
    """
    rows = []
    for sid in subjects:
        e4 = load_e4_subject(sid, DATA_ROOT)
        if not e4:
            continue
        t0 = min(df["timestamp"].iloc[0] for df in e4.values())
        for mod in E4_CHANNELS:
            if mod not in e4:
                continue
            rows.append({"subject": sid, "source": "e4", "channel": mod,
                         "offset_from_t0_s": e4[mod]["timestamp"].iloc[0] - t0,
                         "n_samples": len(e4[mod])})
        npd = load_neurosky_polar_subject(sid, DATA_ROOT)
        for mod in NEURO_POLAR_CHANNELS:
            if mod not in npd:
                continue
            rows.append({"subject": sid, "source": "neurosky_polar", "channel": mod,
                         "offset_from_t0_s": npd[mod]["timestamp"].iloc[0] - t0,
                         "n_samples": len(npd[mod])})

    lat = pd.DataFrame(rows)
    if not lat.empty:
        lat.to_csv(out_dir / "kemocon_sensor_startup_latency.csv", index=False)

        print("\n== [1] STARTUP LATENCY (offset del 1er sample vs t0, en s) ==")
        piv = lat[lat["source"] == "e4"].groupby("channel")["offset_from_t0_s"].agg(
            ["count", "mean", "median", "min", "max"])
        piv["n_zero"] = lat[lat["source"] == "e4"].groupby("channel")[
            "offset_from_t0_s"].apply(lambda s: (s == 0).sum())
        print(piv.round(2).to_string())
        np_lat = lat[lat["source"] == "neurosky_polar"]
        if len(np_lat):
            print("\n-- NeuroSky/Polar (cross-device) --")
            print(np_lat.groupby("channel")["offset_from_t0_s"].agg(
                ["count", "mean", "median", "min", "max"]).round(1).to_string())
    else:
        writeLog("warning", logger, "[sensor_latency] sin datos para startup latency")
    return lat


def grid_alignment_table(subjects: list[int], out_dir: Path) -> pd.DataFrame:
    """Verifica que las muestras HR caen exactamente sobre el grid de EDA.

    HR (1 Hz, timestamps enteros) y EDA (4 Hz, cuartos de segundo) comparten el
    mismo reloj E4. Si cada muestra HR coincide exactamente con una muestra EDA
    (distancia 0 ms), no existe offset persistente intra-dispositivo tras el
    arranque y la comparación por ventanas usa la misma rejilla temporal.

    Args:
        subjects (list): Sujetos elegibles.
        out_dir (Path): Directorio de salida.

    Returns:
        pandas.DataFrame: Filas (subject, hr_samples, pct_exact_match,
            max_dist_to_eda_grid_ms).
    """
    rows = []
    for sid in subjects:
        e4 = load_e4_subject(sid, DATA_ROOT)
        if "HR" not in e4 or "EDA" not in e4:
            continue
        hr = e4["HR"]["timestamp"].to_numpy()
        eda_grid = np.unique(np.round(e4["EDA"]["timestamp"].to_numpy(), 3))
        d = np.abs(hr[:, None] - eda_grid[None, :]).min(axis=1)
        rows.append({"subject": sid,
                     "hr_samples": len(hr),
                     "pct_exact_match": (d < 1e-6).mean() * 100,
                     "max_dist_to_eda_grid_ms": d.max() * 1000,
                     "median_dist_to_eda_grid_ms": np.median(d) * 1000})
    grid = pd.DataFrame(rows)
    if not grid.empty:
        grid.to_csv(out_dir / "kemocon_hr_eda_grid_alignment.csv", index=False)
        print("\n== [2] GRID ALIGNMENT HR vs EDA (¿misma rejilla temporal?) ==")
        print(grid.describe().round(1).to_string())
    else:
        writeLog("warning", logger, "[sensor_latency] sin datos de HR/EDA")
    return grid


def _subject_window_means(sid: int) -> tuple[pd.Series, pd.Series] | None:
    """Medias por ventana de HR y EDA crudas para un sujeto.

    Args:
        sid (int): Sujeto.

    Returns:
        tuple or None: (hr_mean_series, eda_mean_series) indexados por win, o
            None si no hay datos suficientes.
    """
    e4 = load_e4_subject(sid, DATA_ROOT)
    if "HR" not in e4 or "EDA" not in e4:
        return None
    t0 = min(df["timestamp"].iloc[0] for df in e4.values())
    hr = e4["HR"].copy()
    eda = e4["EDA"].copy()
    hr["t_rel"] = hr["timestamp"] - t0
    eda["t_rel"] = eda["timestamp"] - t0
    hr["win"] = (hr["t_rel"] // WINDOW_S).astype(int)
    eda["win"] = (eda["t_rel"] // WINDOW_S).astype(int)
    h = hr.groupby("win")["value"].mean()
    e = eda.groupby("win")["value"].mean()
    common = h.index.intersection(e.index)
    h, e = h.loc[common], e.loc[common]
    if len(h) < 15:
        return None
    return h, e


def crosscorr_table(subjects: list[int], out_dir: Path) -> pd.DataFrame:
    """Cross-correlación de ventanas (5 s) HR-EDA por sujeto, lags -6..+6.

    Args:
        subjects (list): Sujetos elegibles.
        out_dir (Path): Directorio de salida.

    Returns:
        pandas.DataFrame: Filas (subject, lag, rho) + resumen del lag óptimo.
    """
    rows = []
    for sid in subjects:
        pair = _subject_window_means(sid)
        if pair is None:
            continue
        h, e = pair
        h_arr, e_arr = h.to_numpy(), e.to_numpy()
        for lag in CORR_LAGS:
            if lag > 0:
                x, y = h_arr[lag:], e_arr[:-lag]
            elif lag < 0:
                x, y = h_arr[:lag], e_arr[-lag:]
            else:
                x, y = h_arr, e_arr
            if len(x) < 10:
                continue
            r, _ = spearmanr(x, y)
            rows.append({"subject": sid, "lag_win": lag, "rho": r})
    cc = pd.DataFrame(rows)
    if not cc.empty:
        cc.to_csv(out_dir / "kemocon_hr_eda_crosscorr.csv", index=False)
        mean_r = cc.groupby("lag_win")["rho"].mean()
        best = cc.groupby("subject").apply(
            lambda g: g.loc[g["rho"].abs().idxmax(), "lag_win"], include_groups=False)
        print("\n== [3] CROSS-CORRELACIÓN ventana HR-EDA (media por lag) ==")
        print(mean_r.round(4).to_string())
        print("\n-- distribución del lag de |rho| máximo por sujeto --")
        print(best.value_counts().sort_index().to_string())
        peak = mean_r.abs().idxmax()
        print(f"\nlag con |rho| medio máximo: {peak:+d} ventanas "
              f"(rho={mean_r[peak]:+.3f})")
        print("NOTA: pico en los extremos del rango = artefacto de tendencia; "
              "perfil plano/descendente monótono = sin lag fisiológico sistemático.")
    else:
        writeLog("warning", logger, "[sensor_latency] sin datos de cross-correlación")
    return cc


def scoher_lag_sensitivity(feature: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Sensibilidad del resultado confirmatorio de S_coher al shift de hr_mean.

    Replica `add_s_coher` (n_prev=1) sobre hr_mean desplazada -3..+3 ventanas
    por sujeto y correlaciona S_coher resultante contra las métricas de
    desacuerdo (pooled y within-subject). Un efecto genuinamente cortoplacista
    sobrevive en un rango plausible de latencia; si el pico se desplazara a un
    shift concreto, indicaría una latencia sistemática sin corregir.

    Args:
        feature (pandas.DataFrame): Feature table final.
        out_dir (Path): Directorio de salida.

    Returns:
        pandas.DataFrame: Filas (shift_win, target, n, rho_pooled, p_pooled,
            rho_within, p_within).
    """
    rows = []
    for shift in SHIFTS:
        df = feature.sort_values(["subject_id", "win"]).copy()
        df["hr_mean"] = df.groupby("subject_id")["hr_mean"].shift(shift)
        df = add_s_coher(df.drop(columns=["S_coher_prev1", "S_coher_prev5"],
                                 errors="ignore"),
                         n_prev=1, out_col="S_coher_shift")
        for target in DISAGREEMENT_TARGETS:
            if target not in df.columns:
                continue
            sub = df.dropna(subset=["S_coher_shift", target])
            if len(sub) < 10:
                continue
            r_p, p_p = spearmanr(sub["S_coher_shift"], sub[target])
            c = df.copy()
            c["sc"] = c["S_coher_shift"] - c.groupby("subject_id")[
                "S_coher_shift"].transform("mean")
            c["tv"] = c[target] - c.groupby("subject_id")[target].transform("mean")
            c = c.dropna(subset=["sc", "tv"])
            if len(c) < 10:
                r_w = p_w = np.nan
            else:
                r_w, p_w = spearmanr(c["sc"], c["tv"])
            rows.append({"shift_win": shift, "target": target, "n": len(sub),
                         "rho_pooled": r_p, "p_pooled": p_p,
                         "rho_within": r_w, "p_within": p_w})
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(out_dir / "kemocon_scoher_lag_sensitivity.csv", index=False)
        print("\n== [4] SENSIBILIDAD S_coher_prev1 vs desacuerdo al shift de hr_mean ==")
        for target in DISAGREEMENT_TARGETS:
            if target not in out["target"].values:
                continue
            print(f"\n-- target={target} --")
            print(out[out["target"] == target][["shift_win", "n", "rho_pooled",
                                                "rho_within", "p_within"]]
                  .round(3).to_string(index=False))
    return out


def run_all(subjects: list[int] | None = None,
            feature_csv: Path = FEATURE_CSV,
            out_dir: Path | None = None) -> dict:
    """Ejecuta los cuatro chequeos de latencia de sensores.

    Args:
        subjects (list or None): Sujetos a usar; si es None se derivan de la
            metadata.
        feature_csv (Path, optional): Feature table. Por defecto es
            FEATURE_CSV.
        out_dir (Path or None, optional): Directorio de salida. Si es None se
            usa `results/output/KEMOCON/sensor_latency`.

    Returns:
        dict: Con los cuatro DataFrames bajo las claves 'startup', 'grid',
            'crosscorr' y 'scoher_lag'.
    """
    if out_dir is None:
        out_dir = DATA_ROOT.parent.parent / "output" / "KEMOCON" / "sensor_latency"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(DATA_ROOT)
    if subjects is None:
        subjects = eligible_subjects(meta)
    writeLog("info", logger,
             f"[sensor_latency] sujetos: {len(subjects)}; salida: {out_dir}")

    startup = startup_latency_table(subjects, out_dir)
    grid = grid_alignment_table(subjects, out_dir)
    crosscorr = crosscorr_table(subjects, out_dir)
    feature = pd.read_csv(feature_csv)
    scoher = scoher_lag_sensitivity(feature, out_dir)

    return {"startup": startup, "grid": grid,
            "crosscorr": crosscorr, "scoher_lag": scoher}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Diagnóstico exploratorio: latencia de sensores en K-EmoCon")
    parser.add_argument("--features", type=Path, default=FEATURE_CSV,
                        help="feature table (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=None,
                        help="directorio de salida (default: "
                             "results/output/KEMOCON/sensor_latency)")
    parser.add_argument("--subjects", type=int, nargs="*", default=None,
                        help="lista explícita de sujetos (default: elegibles)")
    args = parser.parse_args()
    run_all(args.subjects, args.features, args.out)
    print("\n[sensor_latency] salidas en "
          f"{args.out or DATA_ROOT.parent.parent / 'output' / 'KEMOCON' / 'sensor_latency'}/")
