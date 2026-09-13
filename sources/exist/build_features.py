# src/pipeline/build_features.py
"""
Pipeline principal de construcción de características para EXIST 2026.

Construye el DataFrame final con el PSRI por sujeto, la coherencia cruzada
HR-ET (S_coher) y las etiquetas de las tareas, listo para el sanity check.

Autor: Enrique
"""
from sources.common.common import logger, processControl, writeLog
from sources.common.utils import inicioModulo

import pandas as pd
import numpy as np
from pathlib import Path

from sources.exist.loader import load_dataframes, diagnose_coverage
from sources.psri.calculator import compute_z_score, compute_coherence, compute_weighted_psri
from sources.exist.aggregator import add_psri_subject, aggregate_by_meme, merge_labels


def build_psri_dataframe(input_dir, labels_path, filter_common=True):
    """Construye el DataFrame final con PSRI, coherencia cruzada y etiquetas.

    Args:
        input_dir (Path): Directorio con los archivos Excel.
        labels_path (Path): Ruta al archivo JSON de etiquetas.
        filter_common (bool, optional): Si es True filtra solo los memes con
            HR y ET (la fusión multimodal es SOLO HR+ET; EEG queda excluido
            por desalineación poblacional). Por defecto es True.

    Returns:
        pandas.DataFrame: DataFrame con un registro por meme y las etiquetas.
    """
    # 1. Cargar datos
    df_hr, df_eeg, df_et = load_dataframes(input_dir)

    # 2. Diagnóstico de cobertura
    coverage_stats = diagnose_coverage(df_hr, df_eeg, df_et, labels_path)

    # 3. Filtrar memes comunes HR+ET si se solicita (EEG fuera del merge)
    if filter_common:
        common_memes = coverage_stats['hr_memes'] & coverage_stats['et_memes']
        if common_memes:
            print(f"Filtrando para quedarse solo con los {len(common_memes)} memes que tienen HR y ET...")
            df_hr = df_hr[df_hr['meme_id'].isin(common_memes)]
            df_et = df_et[df_et['meme_id'].isin(common_memes)]
        else:
            print("Advertencia: No hay memes comunes entre HR y ET. Se procederá con todos los datos.")

    # 4. Calcular PSRI por sujeto
    df_hr, df_et = add_psri_subject(df_hr, df_et)

    # 5. Calcular coherencia cruzada HR-ET a nivel de trial (usando baseline)
    # Unimos HR y ET por (meme_id, username) — misma población (HR=ET, 8 sujetos)
    df_hr_et = pd.merge(df_hr, df_et, on=['meme_id', 'username'], how='inner')

    # Para cada sujeto, calcular la std de HR y pupila a través de todos sus trials
    hr_std_subj = df_hr_et.groupby('username')['garmin_hr_mean'].transform('std')
    pupil_std_subj = df_hr_et.groupby('username')['3d_eye_states_pupil diameter left [mm]_mean'].transform('std')
    # Evitar división por cero
    hr_std_subj = hr_std_subj.replace(0, np.nan)
    pupil_std_subj = pupil_std_subj.replace(0, np.nan)

    # Calcular Z-scores por fila
    df_hr_et['z_hr'] = df_hr_et.apply(
        lambda row: compute_z_score(row['garmin_hr_mean'],
                                    row['garmin_hr_mean_baseline_prev'],
                                    hr_std_subj.loc[row.name]),
        axis=1
    )
    df_hr_et['z_pupil'] = df_hr_et.apply(
        lambda row: compute_z_score(row['3d_eye_states_pupil diameter left [mm]_mean'],
                                    row['3d_eye_states_pupil diameter left [mm]_mean_baseline_prev'],
                                    pupil_std_subj.loc[row.name]),
        axis=1
    )

    # Calcular coherencia por trial
    df_hr_et['S_coher_trial'] = df_hr_et.apply(
        lambda row: compute_coherence(row['z_hr'], row['z_pupil']),
        axis=1
    )

    # Agregar por meme: media de S_coher_trial
    coher_by_meme = df_hr_et.groupby('meme_id')['S_coher_trial'].mean().reset_index().rename(
        columns={'S_coher_trial': 'S_coher'}
    )
    coher_by_meme['S_coher'] = coher_by_meme['S_coher'].fillna(coher_by_meme['S_coher'].median())

    # 6. Agregar por meme (medias y desviaciones) — fusión SOLO HR+ET (misma población)
    df_merged = aggregate_by_meme(df_hr, df_et)

    # 7. Añadir S_coher al merged
    df_merged = df_merged.merge(coher_by_meme, on='meme_id', how='left')
    # Rellenar posibles NaN en S_coher
    if 'S_coher' in df_merged.columns:
        df_merged['S_coher'] = df_merged['S_coher'].fillna(df_merged['S_coher'].median())

    # 8. Componentes de fiabilidad y PSRI compuesto ponderado (S_estab, S_coher, S_cond)
    if 'hr_PSRI_hr_subj_mean' in df_merged.columns and 'et_PSRI_et_subj_mean' in df_merged.columns:
        df_merged['S_estab'] = (df_merged['hr_PSRI_hr_subj_mean'] + df_merged['et_PSRI_et_subj_mean']) / 2
        df_merged['S_estab'] = df_merged['S_estab'].fillna(df_merged['S_estab'].median())
    else:
        writeLog("error", logger, "Faltan columnas para calcular S_estab.")
        df_merged['S_estab'] = np.nan

    if 'et_S_cond_subj_mean' in df_merged.columns:
        df_merged['S_cond'] = df_merged['et_S_cond_subj_mean'].fillna(df_merged['et_S_cond_subj_mean'].median())
    else:
        writeLog("error", logger, "Falta columna et_S_cond_subj_mean para calcular S_cond.")
        df_merged['S_cond'] = np.nan

    if df_merged[['S_estab', 'S_coher', 'S_cond']].notna().all(axis=None):
        df_merged['PSRI'] = compute_weighted_psri(
            df_merged['S_estab'], df_merged['S_coher'], df_merged['S_cond'],
            w1=1 / 3, w2=1 / 3, w3=1 / 3
        )
    else:
        writeLog("error", logger,
                 "Hay NaN en S_estab/S_coher/S_cond tras imputación; revisar antes de calcular PSRI ponderado.")
        df_merged['PSRI'] = df_merged[['S_estab', 'S_coher', 'S_cond']].mean(axis=1)

    # 8b. NUEVO — Renombrar columnas por-modalidad para claridad y compatibilidad con sanity_check.py
    # (esto se perdió al sustituir el paso 9 original; PSRI_hr/PSRI_et son necesarios para el
    # diagnóstico por modalidad y para el pool ampliado de comparaciones múltiples)
    rename_map = {}
    if 'hr_PSRI_hr_subj_mean' in df_merged.columns:
        rename_map['hr_PSRI_hr_subj_mean'] = 'PSRI_hr_mean'
    if 'hr_PSRI_hr_subj_std' in df_merged.columns:
        rename_map['hr_PSRI_hr_subj_std'] = 'PSRI_hr_std'
    if 'et_PSRI_et_subj_mean' in df_merged.columns:
        rename_map['et_PSRI_et_subj_mean'] = 'PSRI_et_mean'
    if 'et_PSRI_et_subj_std' in df_merged.columns:
        rename_map['et_PSRI_et_subj_std'] = 'PSRI_et_std'
    df_merged.rename(columns=rename_map, inplace=True)

    # 10. Unir etiquetas
    df_final = merge_labels(df_merged, labels_path)

    print(f"Dataframe final (memes con todos los sensores y etiqueta): {df_final.shape}")
    return df_final


def process_dataframe():
    """Función de entrada del pipeline de EXIST.

    Lee input_dir y output_dir desde `utils.inicioModulo`, ejecuta
    `build_psri_dataframe` y exporta el CSV final.

    Returns:
        None
    """


    input_dir, output_dir = inicioModulo("process_dataframe")
    labels_path = input_dir / 'EXIST2026_training.json'

    df_final = build_psri_dataframe(input_dir, labels_path, filter_common=True)

    if df_final.empty:
        writeLog("error", logger, "No se pudo generar el dataframe. Revise los mensajes de error anteriores.")
        return

    output_file = output_dir / "physio_with_psri_memes.csv"
    df_final.to_csv(output_file, index=False)
    writeLog("info", logger, "Dataframes creados.")