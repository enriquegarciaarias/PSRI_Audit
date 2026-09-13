"""
Importación de datos y construcción de características de EXIST (versión antigua).

Este módulo contiene la versión previa del pipeline de EXIST: diagnóstico de
cobertura de las tres fuentes (HR, EEG, ET), cálculo del PSRI por sujeto con
la función en U y coherencia cruzada HR-ET, agregación por meme y fusión con
las etiquetas del JSON. Se conserva como referencia histórica; el pipeline
activo está en `sources.exist`.

Autor: Enrique
"""
from sources.common.common import logger, processControl, writeLog
from sources.common.utils import inicioModulo
import pandas as pd
import numpy as np
import json
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


def diagnose_coverage(df_hr, df_eeg, df_et, labels_path):
    """Imprime estadísticas de cobertura y solapamiento entre las fuentes.

    Args:
        df_hr (pandas.DataFrame): Datos de HR.
        df_eeg (pandas.DataFrame): Datos de EEG.
        df_et (pandas.DataFrame): Datos de eye-tracking.
        labels_path (str o Path): Ruta al JSON de etiquetas.

    Returns:
        dict: Conjuntos de memes y pares (meme_id, username) por fuente.
    """
    print("\n" + "=" * 60)
    print("📊 ESTADÍSTICAS DE COBERTURA Y SOLAPAMIENTO")
    print("=" * 60)

    # Cargar etiquetas
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
    meme_ids_labels = set(labels_data.keys())

    # Convertir a strings para consistencia
    for df in [df_hr, df_eeg, df_et]:
        df['meme_id'] = df['meme_id'].astype(str)

    # Conjuntos de pares (meme_id, username)
    hr_pairs = set(zip(df_hr['meme_id'], df_hr['username']))
    eeg_pairs = set(zip(df_eeg['meme_id'], df_eeg['username']))
    et_pairs = set(zip(df_et['meme_id'], df_et['username']))

    print(f"Total de memes con etiquetas (JSON): {len(meme_ids_labels)}")

    print("\n🔹 NÚMERO DE REGISTROS (meme_id + username):")
    print(f"  HR: {len(hr_pairs)}")
    print(f"  EEG: {len(eeg_pairs)}")
    print(f"  ET: {len(et_pairs)}")

    print("\n🔹 SOLAPAMIENTO ENTRE FUENTES (pares meme+user):")
    hr_eeg = hr_pairs & eeg_pairs
    hr_et = hr_pairs & et_pairs
    eeg_et = eeg_pairs & et_pairs
    all_three = hr_pairs & eeg_pairs & et_pairs
    print(
        f"  HR ∩ EEG: {len(hr_eeg)} ({len(hr_eeg) / len(hr_pairs) * 100:.1f}% de HR, {len(hr_eeg) / len(eeg_pairs) * 100:.1f}% de EEG)")
    print(
        f"  HR ∩ ET:  {len(hr_et)} ({len(hr_et) / len(hr_pairs) * 100:.1f}% de HR, {len(hr_et) / len(et_pairs) * 100:.1f}% de ET)")
    print(
        f"  EEG ∩ ET: {len(eeg_et)} ({len(eeg_et) / len(eeg_pairs) * 100:.1f}% de EEG, {len(eeg_et) / len(et_pairs) * 100:.1f}% de ET)")
    print(
        f"  HR ∩ EEG ∩ ET: {len(all_three)} ({len(all_three) / len(hr_pairs) * 100:.1f}% de HR, {len(all_three) / len(eeg_pairs) * 100:.1f}% de EEG, {len(all_three) / len(et_pairs) * 100:.1f}% de ET)")

    print("\n🔹 SOLAPAMIENTO POR MEME (sin considerar usuario):")
    hr_memes = set(df_hr['meme_id'])
    eeg_memes = set(df_eeg['meme_id'])
    et_memes = set(df_et['meme_id'])
    print(f"  Memes en HR: {len(hr_memes)}")
    print(f"  Memes en EEG: {len(eeg_memes)}")
    print(f"  Memes en ET: {len(et_memes)}")
    print(f"  Memes en HR ∩ EEG: {len(hr_memes & eeg_memes)}")
    print(f"  Memes en HR ∩ ET:  {len(hr_memes & et_memes)}")
    print(f"  Memes en EEG ∩ ET: {len(eeg_memes & et_memes)}")
    print(f"  Memes en las tres fuentes: {len(hr_memes & eeg_memes & et_memes)}")

    print("\n🔹 SOLAPAMIENTO CON ETIQUETAS (JSON):")
    hr_labels = hr_memes & meme_ids_labels
    eeg_labels = eeg_memes & meme_ids_labels
    et_labels = et_memes & meme_ids_labels
    all_labels = (hr_memes & eeg_memes & et_memes) & meme_ids_labels
    print(
        f"  Memes en HR con etiqueta: {len(hr_labels)} ({len(hr_labels) / len(meme_ids_labels) * 100:.1f}% del total de etiquetas)")
    print(f"  Memes en EEG con etiqueta: {len(eeg_labels)} ({len(eeg_labels) / len(meme_ids_labels) * 100:.1f}%)")
    print(f"  Memes en ET con etiqueta: {len(et_labels)} ({len(et_labels) / len(meme_ids_labels) * 100:.1f}%)")
    print(
        f"  Memes en las tres fuentes Y con etiqueta: {len(all_labels)} ({len(all_labels) / len(meme_ids_labels) * 100:.1f}%)")

    print("\n🔹 USUARIOS ÚNICOS:")
    hr_users = set(df_hr['username'])
    eeg_users = set(df_eeg['username'])
    et_users = set(df_et['username'])
    print(f"  HR: {len(hr_users)}")
    print(f"  EEG: {len(eeg_users)}")
    print(f"  ET: {len(et_users)}")
    print(f"  Comunes a las tres fuentes: {len(hr_users & eeg_users & et_users)}")

    # Mostrar algunos memes que faltan en EEG pero están en HR y ET
    missing_eeg = (hr_memes & et_memes) - eeg_memes
    if missing_eeg:
        print(f"\n🔹 EJEMPLOS DE MEMES FALTANTES:")
        print(f"  Memes en HR y ET pero NO en EEG (primeros 5): {list(missing_eeg)[:5]}")

    print("\n" + "=" * 60)
    print(
        f"📌 RESUMEN: Cobertura de memes con los tres sensores y etiqueta: {len(all_labels)}/{len(meme_ids_labels)} ({len(all_labels) / len(meme_ids_labels) * 100:.1f}%)")
    if len(all_labels) / len(meme_ids_labels) > 0.95:
        print("✅ La cobertura es aceptable. Puede proceder con el análisis completo.")
    else:
        print("⚠️ La cobertura es baja. Considere trabajar solo con los memes que tienen los tres sensores.")
    print("=" * 60 + "\n")

    return {
        'total_labels': len(meme_ids_labels),
        'common_memes': all_labels,
        'hr_pairs': hr_pairs,
        'eeg_pairs': eeg_pairs,
        'et_pairs': et_pairs,
        'hr_memes': hr_memes,
        'eeg_memes': eeg_memes,
        'et_memes': et_memes,
    }


def compute_psri_gaussian_log(sigma_vals, eps_hard=None, epsilon=1e-9):
    """Calcula el PSRI con una función en U (campana gaussiana en log-espacio).

    Args:
        sigma_vals (array-like): Desviaciones estándar intra-trial.
        eps_hard (float, optional): Umbral de señal plana (R=0). Si es None
            se usa el percentil 1 de la distribución.
        epsilon (float, optional): Pequeño valor para evitar log(0). Por
            defecto es 1e-9.

    Returns:
        ndarray: Fiabilidades en [0,1].
    """
    sigma_vals = np.array(sigma_vals)
    R = np.ones_like(sigma_vals, dtype=float)

    # Si no se especifica eps_hard, usar percentil 1 (excluyendo ceros si los hay)
    if eps_hard is None:
        non_zero = sigma_vals[sigma_vals > 0]
        if len(non_zero) > 0:
            eps_hard = np.percentile(non_zero, 1)
        else:
            eps_hard = 1e-6

    # Caso degenerado: señal plana
    R[sigma_vals < eps_hard] = 0.0

    # Log-transform (solo para valores > eps_hard para evitar log(0))
    valid = sigma_vals >= eps_hard
    if valid.sum() > 0:
        log_sigma = np.log(sigma_vals[valid] + epsilon)

        # Estadísticos robustos (mediana y MAD)
        median_log = np.median(log_sigma)
        mad_log = 1.4826 * np.median(np.abs(log_sigma - median_log))
        if mad_log == 0:
            mad_log = 1.0

        # Z-score robusto
        z = (log_sigma - median_log) / mad_log

        # Fiabilidad como campana gaussiana
        R[valid] = np.exp(-0.5 * z ** 2)

    return R


def data_to_dataframe(df_hr, df_eeg, df_et, labels_path='labels.json'):
    """Procesa los dataframes de HR, EEG y ET hasta el DataFrame por meme.

    Calcula el PSRI por sujeto para HR y ET, la coherencia cruzada HR-ET
    (S_coher), agrega por meme, fusiona las tres fuentes y añade las
    etiquetas del JSON (hard labels y entropías de las tareas 2.1-2.3).

    Args:
        df_hr (pandas.DataFrame): Datos de HR.
        df_eeg (pandas.DataFrame): Datos de EEG.
        df_et (pandas.DataFrame): Datos de eye-tracking.
        labels_path (str, optional): Ruta al JSON de etiquetas. Por defecto
            es 'labels.json'.

    Returns:
        pandas.DataFrame: DataFrame con un registro por meme y las etiquetas,
            o DataFrame vacío si no hay coincidencias.
    """
    print(f"HR: {df_hr.shape}, EEG: {df_eeg.shape}, ET: {df_et.shape}")

    # Convertir identificadores a string
    for df in [df_hr, df_eeg, df_et]:
        df['meme_id'] = df['meme_id'].astype(str)
        if 'username' in df.columns:
            df['username'] = df['username'].astype(str)

    # ============================================================
    # 1. Seleccionar columnas relevantes
    # ============================================================

    # HR
    hr_cols = [
        'meme_id', 'username',
        'garmin_hr_mean', 'garmin_hr_std', 'garmin_hr_max', 'garmin_hr_min',
        'garmin_hr_mean_baseline_prev', 'garmin_hr_mean_baseline_prev5'
    ]
    missing_hr = [c for c in hr_cols if c not in df_hr.columns]
    if missing_hr:
        print(f"Columnas HR faltantes: {missing_hr}")
    df_hr = df_hr[hr_cols].copy()

    # EEG: seleccionar canales (16 canales * 9 métricas)
    eeg_cols = [col for col in df_eeg.columns if col.startswith('EXG_Channel_')]
    eeg_keep = ['meme_id', 'username'] + eeg_cols
    df_eeg = df_eeg[eeg_keep].copy()

    # ET: seleccionar métricas clave
    et_cols = [
        'meme_id', 'username',
        'reaction_time',
        '3d_eye_states_pupil diameter left [mm]_mean',
        '3d_eye_states_pupil diameter left [mm]_std',
        '3d_eye_states_pupil diameter right [mm]_mean',
        '3d_eye_states_pupil diameter right [mm]_std',
        'fixations_duration_mean_ns', 'fixations_count',
        'saccades_duration_mean_ns', 'saccades_count',
        'blinks_count',
        'gaze_gaze x [px]_mean', 'gaze_gaze y [px]_mean',
        '3d_eye_states_pupil diameter left [mm]_mean_baseline_prev',
        '3d_eye_states_pupil diameter left [mm]_mean_baseline_prev5',
        '3d_eye_states_pupil diameter right [mm]_mean_baseline_prev',
        '3d_eye_states_pupil diameter right [mm]_mean_baseline_prev5'
    ]
    # Ya incluimos baseline en et_cols, no hace falta añadirlas después
    et_baseline_cols = [
        '3d_eye_states_pupil diameter left [mm]_mean_baseline_prev',
        '3d_eye_states_pupil diameter left [mm]_mean_baseline_prev5',
        '3d_eye_states_pupil diameter right [mm]_mean_baseline_prev',
        '3d_eye_states_pupil diameter right [mm]_mean_baseline_prev5'
    ]
    # Asegurar que están en et_cols (si no, añadirlas)
    for c in et_baseline_cols:
        if c not in et_cols and c in df_et.columns:
            et_cols.append(c)

    missing_et = [c for c in et_cols if c not in df_et.columns]
    if missing_et:
        print(f"Columnas ET faltantes: {missing_et}")
    df_et = df_et[et_cols].copy()

    # ============================================================
    # 2. Procesar EEG: promediar canales por banda
    # ============================================================

    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    for band in bands:
        band_cols = [col for col in eeg_cols if f'_{band}_power' in col]
        if band_cols:
            df_eeg[f'eeg_{band.lower()}_mean'] = df_eeg[band_cols].mean(axis=1, skipna=True)
        else:
            print(f"Advertencia: no se encontraron columnas para {band}")

    std_cols = [col for col in eeg_cols if col.endswith('_std')]
    if std_cols:
        df_eeg['eeg_std_mean'] = df_eeg[std_cols].mean(axis=1, skipna=True)
    else:
        df_eeg['eeg_std_mean'] = np.nan

    # Mantener solo las columnas procesadas
    eeg_final_cols = ['meme_id', 'username'] + [col for col in df_eeg.columns if col.startswith('eeg_')]
    df_eeg = df_eeg[eeg_final_cols].copy()

    # ============================================================
    # 3. Calcular PSRI por sujeto (intra‑trial) usando gaussiana
    # ============================================================

    # 3.1 PSRI para HR (estabilidad intra‑trial) con función en U
    hr_std_vals = df_hr['garmin_hr_std']
    # Usar percentil 1 como umbral de señal plana
    eps_hr = np.percentile(hr_std_vals[hr_std_vals > 0], 1) if (hr_std_vals > 0).sum() > 0 else 1e-6
    df_hr['PSRI_hr_subj'] = compute_psri_gaussian_log(hr_std_vals, eps_hard=eps_hr)

    # 3.2 PSRI para ET (estabilidad intra‑trial)
    et_std_vals = df_et['3d_eye_states_pupil diameter left [mm]_std']
    eps_et = np.percentile(et_std_vals[et_std_vals > 0], 1) if (et_std_vals > 0).sum() > 0 else 1e-6
    df_et['PSRI_et_subj'] = compute_psri_gaussian_log(et_std_vals, eps_hard=eps_et)

    print("\n✅ PSRI por sujeto calculado (función en U en log-espacio).")

    # ============================================================
    # 4. Calcular S_coher a nivel de trial (usando baseline)
    # ============================================================

    # Unimos HR y ET por (meme_id, username)
    df_hr_et = pd.merge(df_hr, df_et, on=['meme_id', 'username'], how='inner')

    # Para cada sujeto, calcular la desviación estándar de HR y pupila a través de todos sus trials
    # (se usa esto como referencia de variabilidad individual)
    hr_std_subj = df_hr_et.groupby('username')['garmin_hr_mean'].transform('std')
    pupil_std_subj = df_hr_et.groupby('username')['3d_eye_states_pupil diameter left [mm]_mean'].transform('std')

    # Z-score respecto a baseline_prev y normalizado por la std del sujeto
    # Si la std del sujeto es 0, evitamos división por cero
    hr_std_subj = hr_std_subj.replace(0, np.nan)
    pupil_std_subj = pupil_std_subj.replace(0, np.nan)

    df_hr_et['z_hr'] = (df_hr_et['garmin_hr_mean'] - df_hr_et['garmin_hr_mean_baseline_prev']) / hr_std_subj
    df_hr_et['z_pupil'] = (df_hr_et['3d_eye_states_pupil diameter left [mm]_mean'] - df_hr_et[
        '3d_eye_states_pupil diameter left [mm]_mean_baseline_prev']) / pupil_std_subj

    # Coherencia: 1 - diferencia_absoluta/2 (valores entre 0 y 1)
    df_hr_et['S_coher_trial'] = np.exp(-np.abs(df_hr_et['z_hr'] - df_hr_et['z_pupil']) / 2)

    # Agregar por meme: media de S_coher_trial
    coher_by_meme = df_hr_et.groupby('meme_id')['S_coher_trial'].mean().reset_index().rename(
        columns={'S_coher_trial': 'S_coher'})
    # Rellenar posibles NaN
    coher_by_meme['S_coher'] = coher_by_meme['S_coher'].fillna(coher_by_meme['S_coher'].median())

    print("\n✅ Coherencia cruzada HR-ET (S_coher) calculada a nivel de trial usando baseline.")

    # ============================================================
    # 5. Agregar por meme (promedio de PSRI y otros estadísticos)
    # ============================================================

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

    # EEG
    eeg_agg = df_eeg.groupby('meme_id').agg({
        'eeg_alpha_mean': ['mean', 'std'],
        'eeg_beta_mean': ['mean', 'std'],
        'eeg_theta_mean': ['mean', 'std'],
        'eeg_delta_mean': ['mean', 'std'],
        'eeg_gamma_mean': ['mean', 'std'],
        'eeg_std_mean': ['mean', 'std']
    }).reset_index()
    eeg_agg.columns = ['meme_id'] + [f'eeg_{col[0]}_{col[1]}' for col in eeg_agg.columns[1:]]

    # ET
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
        'PSRI_et_subj': ['mean', 'std']
    }).reset_index()
    et_agg.columns = ['meme_id'] + [f'et_{col[0]}_{col[1]}' for col in et_agg.columns[1:]]

    # ============================================================
    # 6. Fusionar agregaciones y añadir S_coher
    # ============================================================

    df_merged = hr_agg.merge(eeg_agg, on='meme_id', how='inner')
    df_merged = df_merged.merge(et_agg, on='meme_id', how='inner')
    # Añadir S_coher
    df_merged = df_merged.merge(coher_by_meme, on='meme_id', how='left')

    print(f"Dataframe fusionado (memes): {df_merged.shape}")

    if df_merged.empty:
        print("ERROR: No se encontraron coincidencias de memes entre las fuentes.")
        return pd.DataFrame()

    # ============================================================
    # 7. PSRI compuesto y renombrado
    # ============================================================

    # PSRI compuesto: media de PSRI_hr y PSRI_et
    df_merged['PSRI'] = (df_merged['hr_PSRI_hr_subj_mean'] + df_merged['et_PSRI_et_subj_mean']) / 2

    # Rellenar posibles NaN (para S_coher y PSRI)
    for col in ['hr_PSRI_hr_subj_mean', 'et_PSRI_et_subj_mean', 'PSRI', 'S_coher']:
        if col in df_merged.columns:
            median_val = df_merged[col].median()
            df_merged[col] = df_merged[col].fillna(median_val)

    # Renombrar para claridad
    df_merged.rename(columns={
        'hr_PSRI_hr_subj_mean': 'PSRI_hr_mean',
        'hr_PSRI_hr_subj_std': 'PSRI_hr_std',
        'et_PSRI_et_subj_mean': 'PSRI_et_mean',
        'et_PSRI_et_subj_std': 'PSRI_et_std'
    }, inplace=True)

    print("\n✅ PSRI compuesto calculado (media de fiabilidades HR y ET).")
    print("✅ S_coher (coherencia cruzada) añadida.")

    # ============================================================
    # 8. Cargar etiquetas del JSON
    # ============================================================

    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)

    meme_labels = {}
    for meme_id, info in tqdm(labels_data.items(), total=len(labels_data), desc="Processing memes"):
        labels = info.get('labels_task2_1', [])
        if labels:
            n = len(labels)
            p_yes = labels.count('YES') / n
            p_no = labels.count('NO') / n
            entropy = 0.0
            if p_yes > 0:
                entropy -= p_yes * np.log(p_yes)
            if p_no > 0:
                entropy -= p_no * np.log(p_no)
            hard = 'YES' if p_yes >= 0.5 else 'NO'
        else:
            p_yes = np.nan
            entropy = np.nan
            hard = np.nan

        labels_22 = info.get('labels_task2_2', [])
        if labels_22:
            unique, counts = np.unique(labels_22, return_counts=True)
            probs = counts / len(labels_22)
            entropy_22 = -np.sum(probs * np.log(probs))
            hard_22 = unique[np.argmax(counts)]
        else:
            entropy_22 = np.nan
            hard_22 = np.nan

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
            'hard_21': hard,
            'soft_21_yes': p_yes,
            'entropy_21': entropy,
            'hard_22': hard_22,
            'entropy_22': entropy_22,
            'entropy_23': entropy_23
        }

    df_labels = pd.DataFrame.from_dict(meme_labels, orient='index').reset_index().rename(columns={'index': 'meme_id'})
    df_labels['meme_id'] = df_labels['meme_id'].astype(str)

    # ============================================================
    # 9. Unir etiquetas al dataframe de memes
    # ============================================================

    df_final = df_merged.merge(df_labels, on='meme_id', how='inner')
    print(f"Dataframe final (memes con todos los sensores y etiqueta): {df_final.shape}")

    if df_final.empty:
        print("ERROR: No se encontraron etiquetas para los memes fusionados.")
        return pd.DataFrame()

    return df_final


def process_dataframe():
    """Función de entrada del pipeline antiguo de EXIST.

    Carga los tres Excel y el JSON de etiquetas, filtra los memes comunes,
    calcula el PSRI/coherencia y exporta el CSV final a `results/output`.

    Returns:
        None
    """
    input_dir, output_dir = inicioModulo("process_dataframe")

    df_hr = pd.read_excel(input_dir / 'HR_.xlsx')
    df_eeg = pd.read_excel(input_dir / 'EEG_.xlsx')
    df_et = pd.read_excel(input_dir / 'ET_.xlsx')
    labels_path = input_dir / 'EXIST2026_training.json'

    # 1. Diagnóstico de cobertura
    coverage_stats = diagnose_coverage(df_hr, df_eeg, df_et, labels_path)

    # 2. Filtrar memes comunes
    common_memes = coverage_stats['common_memes']
    if common_memes:
        print(f"Filtrando para quedarse solo con los {len(common_memes)} memes que tienen los tres sensores...")
        df_hr = df_hr[df_hr['meme_id'].isin(common_memes)]
        df_eeg = df_eeg[df_eeg['meme_id'].isin(common_memes)]
        df_et = df_et[df_et['meme_id'].isin(common_memes)]
    else:
        print(
            "Advertencia: No hay memes con los tres sensores. Se procederá con todos los datos, pero el PSRI será menos fiable.")

    # 3. Procesar datos y calcular PSRI y S_coher
    df_meme = data_to_dataframe(df_hr, df_eeg, df_et, labels_path=labels_path)

    if df_meme.empty:
        writeLog("error", logger, "No se pudo generar el dataframe. Revise los mensajes de error anteriores.")
        return

    output_file = output_dir / "physio_with_psri_memes.csv"
    df_meme.to_csv(output_file, index=False)
    writeLog("info", logger, "Dataframes creados.")