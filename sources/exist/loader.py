# src/data/loader.py
"""
Carga de datos de EXIST 2026.

Lee los archivos Excel (HR, EEG, ET) y el JSON de etiquetas, y proporciona
funciones de diagnóstico de cobertura y solapamiento entre las fuentes.

Autor: Enrique
"""
from sources.common.common import logger, processControl, writeLog
import pandas as pd
import json
from pathlib import Path


def load_dataframes(input_dir):
    """Carga los tres DataFrames desde los archivos Excel del directorio.

    Args:
        input_dir (str o Path): Directorio con `HR_.xlsx`, `EEG_.xlsx` y
            `ET_.xlsx`.

    Returns:
        tuple: (df_hr, df_eeg, df_et).
    """
    df_hr = pd.read_excel(input_dir / 'HR_.xlsx')
    df_eeg = pd.read_excel(input_dir / 'EEG_.xlsx')
    df_et = pd.read_excel(input_dir / 'ET_.xlsx')
    return df_hr, df_eeg, df_et


def diagnose_coverage(df_hr, df_eeg, df_et, labels_path):
    """Imprime estadísticas de cobertura y solapamiento entre fuentes y etiquetas.

    Args:
        df_hr (pandas.DataFrame): Datos de HR.
        df_eeg (pandas.DataFrame): Datos de EEG.
        df_et (pandas.DataFrame): Datos de eye-tracking.
        labels_path (str o Path): Ruta al JSON de etiquetas.

    Returns:
        dict: Conjuntos de memes y pares (meme_id, username) por fuente, con
            `total_labels` y `common_memes`.
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