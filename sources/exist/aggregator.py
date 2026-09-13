# src/psri/aggregator.py
"""
Agregación de datos y etiquetas para el pipeline de EXIST 2026.

Contiene las funciones que calculan el PSRI por sujeto sobre HR y ET,
agregan los DataFrames por meme_id y fusionan las etiquetas del JSON con
las entropías de las tareas 2.1-2.3.

Autor: Enrique
"""

import pandas as pd
import numpy as np
import json
from tqdm import tqdm

from sources.psri.calculator import compute_psri_gaussian_log


def add_psri_subject(df_hr, df_et):
    """Añade columnas de fiabilidad intra-trial (PSRI) a HR y ET.

    Calcula `PSRI_hr_subj`, `PSRI_et_subj` (función en U en log-espacio
    sobre las desviaciones estándar) y `S_cond_subj` (consistencia
    conductual a partir de `reaction_time` y `blinks_count`).

    Args:
        df_hr (pandas.DataFrame): Datos de HR con `garmin_hr_std`.
        df_et (pandas.DataFrame): Datos de eye-tracking con pupila,
            `reaction_time` y `blinks_count`.

    Returns:
        tuple: (df_hr, df_et) modificados con las columnas nuevas.
    """
    # PSRI para HR
    hr_std_vals = df_hr['garmin_hr_std']
    eps_hr = np.percentile(hr_std_vals[hr_std_vals > 0], 1) if (hr_std_vals > 0).sum() > 0 else 1e-6
    df_hr['PSRI_hr_subj'] = compute_psri_gaussian_log(hr_std_vals, eps_hard=eps_hr)

    # PSRI para ET (se usa pupila izquierda)
    et_std_vals = df_et['3d_eye_states_pupil diameter left [mm]_std']
    eps_et = np.percentile(et_std_vals[et_std_vals > 0], 1) if (et_std_vals > 0).sum() > 0 else 1e-6
    df_et['PSRI_et_subj'] = compute_psri_gaussian_log(et_std_vals, eps_hard=eps_et)

    # NUEVO: S_cond_subj — consistencia conductual (tiempo de reacción + parpadeos)
    # Misma lógica que estabilidad: tanto un tiempo de reacción/parpadeo anormalmente
    # bajo como anormalmente alto se alejan de la respuesta "típica" y penalizan fiabilidad.
    rt_vals = df_et['reaction_time']
    eps_rt = np.percentile(rt_vals[rt_vals > 0], 1) if (rt_vals > 0).sum() > 0 else 1e-6
    S_cond_rt = compute_psri_gaussian_log(rt_vals, eps_hard=eps_rt)

    blink_vals = df_et['blinks_count']
    eps_blink = np.percentile(blink_vals[blink_vals > 0], 1) if (blink_vals > 0).sum() > 0 else 1e-6
    S_cond_blink = compute_psri_gaussian_log(blink_vals, eps_hard=eps_blink)

    df_et['S_cond_subj'] = (S_cond_rt + S_cond_blink) / 2

    return df_hr, df_et


def aggregate_by_meme(df_hr, df_et):
    """Agrupa los DataFrames por meme_id y calcula medias y desviaciones.

    IMPORTANTE (decisión de diseño): la fusión es SOLO HR+ET. EEG queda
    excluido del merge poblacional porque su población de sujetos no está
    alineada con HR/ET (0 solapamiento a nivel de par (meme, username)).

    Args:
        df_hr (pandas.DataFrame): Datos de HR con PSRI por sujeto.
        df_et (pandas.DataFrame): Datos de ET con PSRI por sujeto.

    Returns:
        pandas.DataFrame: Agregación por meme_id con las métricas
            `hr_*_mean/std` y `et_*_mean/std`.
    """
    # HR
    hr_agg = df_hr.groupby('meme_id').agg({
        'garmin_hr_mean': ['mean', 'std'],
        'garmin_hr_std': ['mean', 'std'],
        'garmin_hr_max': ['mean', 'std'],
        'garmin_hr_min': ['mean', 'std'],
        'garmin_hr_mean_baseline_prev': ['mean', 'std'],
        'garmin_hr_mean_baseline_prev5': ['mean', 'std'],
        'PSRI_hr_subj': ['mean', 'std']
    }).reset_index()
    hr_agg.columns = ['meme_id'] + [f'hr_{col[0]}_{col[1]}' for col in hr_agg.columns[1:]]

    # ET (pupila, fijaciones, sacadas, parpadeos, tiempo de reacción)
    et_agg = df_et.groupby('meme_id').agg({
        '3d_eye_states_pupil diameter left [mm]_mean': ['mean', 'std'],
        '3d_eye_states_pupil diameter left [mm]_std': ['mean', 'std'],
        '3d_eye_states_pupil diameter right [mm]_mean': ['mean', 'std'],
        '3d_eye_states_pupil diameter right [mm]_std': ['mean', 'std'],
        'fixations_duration_mean_ns': ['mean', 'std'],
        'fixations_count': ['mean', 'std'],
        'saccades_count': ['mean', 'std'],
        'blinks_count': ['mean', 'std'],
        'reaction_time': ['mean', 'std'],
        'PSRI_et_subj': ['mean', 'std'],
        'S_cond_subj': ['mean', 'std']
    }).reset_index()
    et_agg.columns = ['meme_id'] + [f'et_{col[0]}_{col[1]}' for col in et_agg.columns[1:]]

    # Fusión de las dos agregaciones (misma población HR=ET)
    df_merged = hr_agg.merge(et_agg, on='meme_id', how='inner')

    return df_merged


def merge_labels(df_merged, labels_path):
    """Carga el JSON de etiquetas y añade hard labels y entropías.

    Args:
        df_merged (pandas.DataFrame): DataFrame agregado por meme.
        labels_path (str o Path): Ruta al JSON
            (EXIST2026_training.json).

    Returns:
        pandas.DataFrame: DataFrame con las columnas `hard_21`,
            `soft_21_yes`, `entropy_21`, `hard_22`, `entropy_22` y
            `entropy_23`.
    """
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)

    meme_labels = {}
    for meme_id, info in tqdm(labels_data.items(), total=len(labels_data), desc="Procesando etiquetas"):
        # Task 2.1
        labels_21 = info.get('labels_task2_1', [])
        if labels_21:
            n = len(labels_21)
            p_yes = labels_21.count('YES') / n
            p_no = labels_21.count('NO') / n
            entropy_21 = 0.0
            if p_yes > 0:
                entropy_21 -= p_yes * np.log(p_yes)
            if p_no > 0:
                entropy_21 -= p_no * np.log(p_no)
            hard_21 = 'YES' if p_yes >= 0.5 else 'NO'
        else:
            p_yes = np.nan
            entropy_21 = np.nan
            hard_21 = np.nan

        # Task 2.2
        labels_22 = info.get('labels_task2_2', [])
        if labels_22:
            unique, counts = np.unique(labels_22, return_counts=True)
            probs = counts / len(labels_22)
            entropy_22 = -np.sum(probs * np.log(probs))
            hard_22 = unique[np.argmax(counts)]
        else:
            entropy_22 = np.nan
            hard_22 = np.nan

        # Task 2.3
        labels_23 = info.get('labels_task2_3', [])
        if labels_23:
            flat = [item for sublist in labels_23 for item in sublist]
            if flat:
                unique, counts = np.unique(flat, return_counts=True)
                probs = counts / len(flat)
                entropy_23 = -np.sum(probs * np.log(probs))
            else:
                entropy_23 = np.nan
        else:
            entropy_23 = np.nan

        meme_labels[meme_id] = {
            'hard_21': hard_21,
            'soft_21_yes': p_yes,
            'entropy_21': entropy_21,
            'hard_22': hard_22,
            'entropy_22': entropy_22,
            'entropy_23': entropy_23
        }

    df_labels = pd.DataFrame.from_dict(meme_labels, orient='index').reset_index().rename(columns={'index': 'meme_id'})
    df_labels['meme_id'] = df_labels['meme_id'].astype(str)

    # Unir con el DataFrame de memes
    df_final = df_merged.merge(df_labels, on='meme_id', how='inner')
    return df_final