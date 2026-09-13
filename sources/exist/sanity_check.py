# src/exist/sanity_check.py
"""
Sanity check del PSRI calculado sobre EXIST 2026.

Genera estadísticas descriptivas, gráficos, pruebas de hipótesis y el
experimento de correlación PSRI/S_coher vs entropía con corrección por
comparaciones múltiples (Bonferroni y FDR/Benjamini-Hochberg).

Autor: Enrique
"""

from sources.common.utils import inicioModulo
from sources.common.common import logger, processControl, writeLog

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kruskal
from statsmodels.stats.multitest import multipletests


# ============================================================
# CARGA Y PREPARACIÓN
# ============================================================

def load_and_prepare(df_path):
    """Carga el CSV y normaliza nombres de columnas heredados de versiones previas.

    Args:
        df_path (str o Path): Ruta al CSV generado por `build_features.py`.

    Returns:
        pandas.DataFrame: DataFrame cargado con las columnas renombradas.
    """
    df = pd.read_csv(df_path)
    df['meme_id'] = df['meme_id'].astype(str)

    if 'PSRI_hr_mean' in df.columns:
        df.rename(columns={
            'PSRI_hr_mean': 'PSRI_hr',
            'PSRI_et_mean': 'PSRI_et'
        }, inplace=True)
        writeLog("info", logger,"📌 Columnas adaptadas: PSRI_hr_mean -> PSRI_hr, PSRI_et_mean -> PSRI_et")

    required_cols = ['PSRI', 'PSRI_hr', 'PSRI_et']
    for col in required_cols:
        if col not in df.columns:
            writeLog("info", logger,f"⚠️ Advertencia: Columna '{col}' no encontrada. Se omitirá en los análisis.")

    return df


# ============================================================
# BLOQUE 1: ESTADÍSTICAS DESCRIPTIVAS Y RANGOS
# ============================================================

def print_descriptive_stats(df, cols_psri_exist):
    """Imprime estadísticas descriptivas y verificación de rangos del PSRI.

    Args:
        df (pandas.DataFrame): DataFrame con las columnas PSRI.
        cols_psri_exist (list): Columnas PSRI presentes.

    Returns:
        None
    """
    df_clean_psri = df[cols_psri_exist].dropna()
    writeLog("info", logger,"\n🔹 ESTADÍSTICAS DESCRIPTIVAS:")
    writeLog("info", logger,df_clean_psri.describe().round(4))

    writeLog("info", logger,"\n🔹 VERIFICACIÓN DE LA CORRECCIÓN DE ESCALA (PSRI):")
    for col in ['PSRI_hr', 'PSRI_et']:
        if col in df.columns:
            writeLog("info", logger,f"  - {col}: Media = {df[col].mean():.4f} | Mediana = {df[col].median():.4f}")

    writeLog("info", logger,"\n🔹 VERIFICACIÓN DE RANGOS:")
    for col in cols_psri_exist:
        min_val = df[col].min()
        max_val = df[col].max()
        has_outside = (df[col] < 0).any() or (df[col] > 1).any()
        writeLog("info", logger,f"  {col}: min={min_val:.4f}, max={max_val:.4f}, ¿fuera de [0,1]? {has_outside}")


def plot_distributions(df, cols_psri_exist, output_dir=None):
    """Dibuja y guarda los histogramas de distribución de cada columna PSRI.

    Args:
        df (pandas.DataFrame): DataFrame con las columnas PSRI.
        cols_psri_exist (list): Columnas PSRI presentes.
        output_dir (Path, optional): Directorio de salida de la figura.

    Returns:
        None
    """
    if len(cols_psri_exist) == 0:
        return
    n_cols = min(len(cols_psri_exist), 3)
    n_rows = (len(cols_psri_exist) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

    for i, col in enumerate(cols_psri_exist):
        ax = axes[i]
        data = df[col].dropna()
        sns.histplot(data, kde=True, ax=ax, bins=50)
        ax.axvline(data.mean(), color='red', linestyle='--', label=f'Media={data.mean():.3f}')
        ax.axvline(data.median(), color='green', linestyle='-.', label=f'Mediana={data.median():.3f}')
        ax.set_title(f'Distribución de {col}')
        ax.legend()
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = (output_dir / 'fig_psri_distribution.png') if output_dir else 'fig_psri_distribution.png'
    plt.savefig(path, dpi=300)
    plt.close(fig)
    writeLog("info", logger,f"\n✅ Figura '{path}' guardada.")


# ============================================================
# BLOQUE 2: CORRELACIÓN ENTRE COMPONENTES DEL PSRI
# ============================================================

def analyze_component_correlations(df, cols_psri_exist, output_dir=None):
    """Analiza y grafica la correlación entre los componentes del PSRI.

    Args:
        df (pandas.DataFrame): DataFrame con las columnas PSRI.
        cols_psri_exist (list): Columnas PSRI presentes.
        output_dir (Path, optional): Directorio de salida de la figura.

    Returns:
        None
    """
    if len(cols_psri_exist) < 2:
        return

    corr_matrix = df[cols_psri_exist].dropna().corr(method='pearson')
    writeLog("info", logger,"\n🔹 CORRELACIÓN ENTRE COMPONENTES:")
    writeLog("info", logger,corr_matrix.round(4))

    fig = sns.pairplot(df[cols_psri_exist].dropna(), diag_kind='kde')
    path = (output_dir / 'fig_psri_pairplot.png') if output_dir else 'fig_psri_pairplot.png'
    fig.savefig(path, dpi=300)
    plt.close('all')
    writeLog("info", logger,f"✅ Figura '{path}' guardada.")

    if 'PSRI_hr' in cols_psri_exist and 'PSRI_et' in cols_psri_exist:
        corr_hr_et = corr_matrix.loc['PSRI_hr', 'PSRI_et']
        writeLog("info", logger,f"\n🔍 Interpretación: Correlación HR-ET = {corr_hr_et:.3f}")
        if abs(corr_hr_et) < 0.1:
            writeLog("info", logger,"   (Ambas modalidades son prácticamente independientes)")

"""
# ============================================================
# BLOQUE 3: PSRI vs ETIQUETAS DURAS (Kruskal-Wallis)
# ============================================================

def _kruskal_by_hard_label(df, label_col, valid_values, value_col, title, filename, output_dir,
                            note_if_sig=None, note_if_ns=None):
    writeLog("info", logger,f"\n🔹 {value_col} SEGÚN ETIQUETA DURA ({title}):")
    if value_col not in df.columns or label_col not in df.columns:
        writeLog("info", logger, f"  Columnas no disponibles para {title}.")
        return None

    df_sub = df[[label_col, value_col]].dropna()
    df_sub = df_sub[df_sub[label_col].isin(valid_values)]
    if df_sub.empty:
        writeLog("info", logger,f"  No hay datos suficientes para {title}.")
        return None

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    sns.boxplot(data=df_sub, x=label_col, y=value_col, ax=ax)
    ax.set_title(f'{value_col} vs Hard Label ({title})')

    groups = [df_sub[df_sub[label_col] == v][value_col] for v in valid_values]
    stat, p_val = kruskal(*groups)
    writeLog("info", logger,f"  Kruskal-Wallis: estadístico={stat:.3f}, p-valor={p_val:.4f}")
    if p_val < 0.05 and note_if_sig:
        writeLog("info", logger,f"    ⚠️ Diferencia SIGNIFICATIVA ({note_if_sig})")
    elif p_val >= 0.05 and note_if_ns:
        writeLog("info", logger,f"    ✅ No hay diferencia significativa ({note_if_ns})")

    plt.tight_layout()
    path = (output_dir / filename) if output_dir else filename
    plt.savefig(path, dpi=300)
    plt.close(fig)
    writeLog("info", logger,f"✅ Figura '{path}' guardada.")
    return {'stat': stat, 'p_value': p_val}


def analyze_hard_labels(df, output_dir=None):
    results = {}
    results['task_21'] = _kruskal_by_hard_label(
        df, 'hard_21', ['YES', 'NO'], 'PSRI',
        'Task 2.1 - YES vs NO', 'fig_psri_vs_hardlabel.png', output_dir,
        note_if_sig='contradice a AI Wizards',
        note_if_ns='consistente con AI Wizards'
    )
    results['task_22'] = _kruskal_by_hard_label(
        df, 'hard_22', ['DIRECT', 'JUDGEMENTAL'], 'PSRI',
        'Task 2.2 - DIRECT vs JUDGEMENTAL', 'fig_psri_vs_intention.png', output_dir
    )
    return results

"""
# ============================================================
# BLOQUE 4: EXPERIMENTO 1 — PSRI/S_coher vs ENTROPÍA
# Con corrección por comparaciones múltiples (Bonferroni + FDR)
# ============================================================

def compute_correlation_experiment(df, metrics, tasks):
    """Calcula las correlaciones Pearson/Spearman de métricas vs entropías.

    Args:
        df (pandas.DataFrame): DataFrame con las métricas y entropías.
        metrics (list): Métricas PSRI a correlacionar.
        tasks (list): Lista de tuplas (nombre de tarea, columna de entropía).

    Returns:
        list: Filas con task, entropy_col, metric, n y coeficientes/p-valores.

    Raises:
        ValueError: Si el DataFrame tiene columnas duplicadas o si una métrica
            devuelve múltiples columnas tras la selección.
    """
    dup_cols = df.columns[df.columns.duplicated()].tolist()
    if dup_cols:
        raise ValueError(f"El DataFrame tiene columnas duplicadas: {dup_cols}.")

    rows = []
    for task, ent_col in tasks:
        if ent_col not in df.columns:
            continue
        df_temp = df[metrics + [ent_col]].dropna()
        if df_temp.empty:
            continue
        for metric in metrics:
            x = df_temp[metric]
            if isinstance(x, pd.DataFrame):
                raise ValueError(
                    f"La columna '{metric}' no es única tras la selección "
                    f"(devolvió {x.shape[1]} columnas) — revisa duplicados en 'metrics'."
                )
            r_pearson, p_pearson = pearsonr(x, df_temp[ent_col])
            r_spearman, p_spearman = spearmanr(x, df_temp[ent_col])
            rows.append({
                'task': task, 'entropy_col': ent_col, 'metric': metric, 'n': len(df_temp),
                'r_pearson': r_pearson, 'p_pearson': p_pearson,
                'r_spearman': r_spearman, 'p_spearman': p_spearman,
            })
    return rows


def apply_multiple_comparison_correction(rows, alpha=0.05):
    """Aplica Bonferroni y FDR (Benjamini-Hochberg) a los p-valores de Pearson.

    Args:
        rows (list): Filas de `compute_correlation_experiment`.
        alpha (float, optional): Nivel de significancia. Por defecto es 0.05.

    Returns:
        list: Las mismas filas con `p_bonferroni`, `reject_bonferroni`,
            `p_fdr` y `reject_fdr` añadidos.
    """
    if not rows:
        return rows

    p_values = [r['p_pearson'] for r in rows]

    reject_bonf, p_bonf, _, _ = multipletests(p_values, alpha=alpha, method='bonferroni')
    reject_fdr, p_fdr, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')

    for row, rb, pb, rf, pf in zip(rows, reject_bonf, p_bonf, reject_fdr, p_fdr):
        row['p_bonferroni'] = pb
        row['reject_bonferroni'] = bool(rb)
        row['p_fdr'] = pf
        row['reject_fdr'] = bool(rf)

    return rows


def print_correlation_experiment(rows):
    """Imprime por consola el resumen del experimento de correlación.

    Muestra cada test con sus p-valores (crudo, Bonferroni, FDR) y un
    veredicto final sobre si algún resultado sobrevive a la corrección.

    Args:
        rows (list): Filas ya corregidas de `compute_correlation_experiment`.

    Returns:
        None
    """
    writeLog("info", logger,"\n" + "=" * 90)
    writeLog("info", logger,"🧪 EXPERIMENTO 1 DEFINITIVO: PSRI (por sujeto) + S_coher vs ENTROPÍA")
    writeLog("info", logger,f"   Corrección por comparaciones múltiples sobre n={len(rows)} tests "
          f"(Bonferroni + FDR/Benjamini-Hochberg)")
    writeLog("info", logger,"=" * 90)

    for r in rows:
        sig_raw = "***" if r['p_pearson'] < 0.001 else "**" if r['p_pearson'] < 0.01 \
            else "*" if r['p_pearson'] < 0.05 else "ns"
        sig_bonf = "SIG" if r.get('reject_bonferroni') else "ns"
        sig_fdr = "SIG" if r.get('reject_fdr') else "ns"
        writeLog("info", logger,f"  {r['task']} | {r['metric']:10s} vs {r['entropy_col']}: "
              f"r={r['r_pearson']:+.4f} | p_crudo={r['p_pearson']:.4f} ({sig_raw}) | "
              f"p_bonf={r.get('p_bonferroni', float('nan')):.4f} ({sig_bonf}) | "
              f"p_fdr={r.get('p_fdr', float('nan')):.4f} ({sig_fdr})")

    n_sig_raw = sum(1 for r in rows if r['p_pearson'] < 0.05)
    n_sig_bonf = sum(1 for r in rows if r.get('reject_bonferroni'))
    n_sig_fdr = sum(1 for r in rows if r.get('reject_fdr'))

    writeLog("info", logger,f"\n🔍 Resumen: {n_sig_raw}/{len(rows)} significativos sin corregir | "
          f"{n_sig_bonf}/{len(rows)} tras Bonferroni | {n_sig_fdr}/{len(rows)} tras FDR")

    writeLog("info", logger,"\n" + "=" * 90)
    if n_sig_bonf == 0 and n_sig_fdr == 0:
        writeLog("info", logger,"✅ SANITY CHECK COMPLETADO.")
        writeLog("info", logger,"   Ninguna correlación sobrevive a corrección por comparaciones múltiples.")
        writeLog("info", logger,"   Las tres dimensiones de fiabilidad (estabilidad, coherencia, consenso)")
        writeLog("info", logger,"   no predicen el desacuerdo (Resultado Negativo Válido).")
    else:
        writeLog("info", logger,"⚠️ SANITY CHECK COMPLETADO — hay resultados que sobreviven a corrección.")
        writeLog("info", logger,"   Revisar signo y tamaño de efecto antes de interpretar como hallazgo real.")
    writeLog("info", logger,"=" * 90)

"""

def run_correlation_experiment(df, output_dir=None):
    #Orquesta cálculo + corrección + impresión, y exporta la tabla a CSV.
    metrics = ['PSRI'] + [c for c in ['PSRI_hr', 'PSRI_et', 'S_estab', 'S_coher', 'S_cond'] if c in df.columns]
    metrics = list(dict.fromkeys(metrics))  # dedupe defensivo, por si se repite en el futuro

    tasks = [('Task 2.1', 'entropy_21'), ('Task 2.2', 'entropy_22'), ('Task 2.3', 'entropy_23')]

    rows = compute_correlation_experiment(df, metrics, tasks)
    rows = apply_multiple_comparison_correction(rows, alpha=0.05)
    print_correlation_experiment(rows)

    df_results = pd.DataFrame(rows)
    path = (output_dir / 'correlation_experiment_results.csv') if output_dir else 'correlation_experiment_results.csv'
    df_results.to_csv(path, index=False)
    writeLog("info", logger, f"\n✅ Tabla de resultados exportada a '{path}' (para citar p_bonferroni/p_fdr en el paper).")

    return df_results
"""
def _kruskal_by_hard_label(df, label_col, valid_values, value_col, title, filename, output_dir):
    """Calcula Kruskal-Wallis y genera el boxplot para una etiqueta dura.

    NO imprime interpretación de significancia — eso se decide tras la
    corrección conjunta con las correlaciones.

    Args:
        df (pandas.DataFrame): DataFrame de datos.
        label_col (str): Columna de la etiqueta dura.
        valid_values (list): Valores válidos de la etiqueta.
        value_col (str): Columna del valor a comparar (p. ej. PSRI).
        title (str): Título del análisis.
        filename (str): Nombre del PNG de salida.
        output_dir (Path): Directorio de salida.

    Returns:
        dict: Con `test`, `label`, `p_value`, `stat`, `effect_size` (epsilon
            cuadrado) y `n`, o None si faltan columnas o datos.
    """
    if value_col not in df.columns or label_col not in df.columns:
        writeLog("info", logger, f"  Columnas no disponibles para {title}.")
        return None

    df_sub = df[[label_col, value_col]].dropna()
    df_sub = df_sub[df_sub[label_col].isin(valid_values)]
    if df_sub.empty:
        writeLog("info", logger, f"  No hay datos suficientes para {title}.")
        return None

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    sns.boxplot(data=df_sub, x=label_col, y=value_col, ax=ax)
    ax.set_title(f'{value_col} vs Hard Label ({title})')

    groups = [df_sub[df_sub[label_col] == v][value_col] for v in valid_values]
    stat, p_val = kruskal(*groups)
    n = len(df_sub)
    epsilon_sq = (stat - 2 + 1) / (n - 2) if n > 2 else float('nan')  # k=2 grupos siempre aquí

    writeLog("info", logger, f"\n🔹 {value_col} SEGÚN ETIQUETA DURA ({title}):")
    writeLog("info", logger, f"  Kruskal-Wallis: estadístico={stat:.3f}, p-valor_crudo={p_val:.4f}, "
                              f"epsilon²={epsilon_sq:.5f} (tamaño de efecto)")

    plt.tight_layout()
    path = (output_dir / filename) if output_dir else filename
    plt.savefig(path, dpi=300)
    plt.close(fig)
    writeLog("info", logger, f"✅ Figura '{path}' guardada.")

    return {'test': 'Kruskal-Wallis', 'label': f'{value_col} vs {label_col} ({title})',
            'p_value': p_val, 'stat': stat, 'effect_size': epsilon_sq, 'n': n}


def analyze_hard_labels(df, output_dir=None):
    """Aplica Kruskal-Wallis sobre las etiquetas duras de las tareas 2.1 y 2.2.

    Args:
        df (pandas.DataFrame): DataFrame con las columnas de etiquetas.
        output_dir (Path, optional): Directorio de salida de las figuras.

    Returns:
        list: Resultados de `_kruskal_by_hard_label` para cada tarea.
    """
    results = []
    r21 = _kruskal_by_hard_label(df, 'hard_21', ['YES', 'NO'], 'PSRI',
                                  'Task 2.1 - YES vs NO', 'fig_psri_vs_hardlabel.png', output_dir)
    if r21:
        results.append(r21)
    r22 = _kruskal_by_hard_label(df, 'hard_22', ['DIRECT', 'JUDGEMENTAL'], 'PSRI',
                                  'Task 2.2 - DIRECT vs JUDGEMENTAL', 'fig_psri_vs_intention.png', output_dir)
    if r22:
        results.append(r22)
    return results


def run_full_analysis_with_correction(df, output_dir=None, alpha=0.05):
    """Corrige por comparaciones múltiples TODO el pool de tests de forma unificada.

    Pool único formado por los Kruskal-Wallis de las etiquetas duras y las
    correlaciones Pearson contra las entropías, sobre el que se aplican
    Bonferroni y FDR. Sustituye a las dos correcciones separadas previas.

    Args:
        df (pandas.DataFrame): DataFrame con métricas y etiquetas.
        output_dir (Path, optional): Directorio de salida del CSV.
        alpha (float, optional): Nivel de significancia. Por defecto es 0.05.

    Returns:
        pandas.DataFrame: Tabla unificada con los p-valores corregidos,
            exportada a `unified_correction_results.csv`.
    """
    kw_results = analyze_hard_labels(df, output_dir)

    metrics = ['PSRI'] + [c for c in ['PSRI_hr', 'PSRI_et', 'S_estab', 'S_coher', 'S_cond'] if c in df.columns]
    metrics = list(dict.fromkeys(metrics))
    tasks = [('Task 2.1', 'entropy_21'), ('Task 2.2', 'entropy_22'), ('Task 2.3', 'entropy_23')]
    corr_rows = compute_correlation_experiment(df, metrics, tasks)

    unified = []
    for kw in kw_results:
        unified.append({'test': kw['test'], 'label': kw['label'], 'p_value': kw['p_value'],
                         'stat': kw['stat'], 'effect_size': kw['effect_size'], 'n': kw['n']})
    for row in corr_rows:
        unified.append({'test': 'Pearson', 'label': f"{row['task']} | {row['metric']} vs {row['entropy_col']}",
                         'p_value': row['p_pearson'], 'stat': row['r_pearson'],
                         'effect_size': row['r_pearson'] ** 2, 'n': row['n']})

    p_values = [u['p_value'] for u in unified]
    reject_bonf, p_bonf, _, _ = multipletests(p_values, alpha=alpha, method='bonferroni')
    reject_fdr, p_fdr, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    for u, rb, pb, rf, pf in zip(unified, reject_bonf, p_bonf, reject_fdr, p_fdr):
        u['p_bonferroni'] = pb
        u['reject_bonferroni'] = bool(rb)
        u['p_fdr'] = pf
        u['reject_fdr'] = bool(rf)

    writeLog("info", logger, "\n" + "=" * 100)
    writeLog("info", logger, f"🧪 CORRECCIÓN UNIFICADA POR COMPARACIONES MÚLTIPLES — n={len(unified)} tests totales "
                              f"({len(kw_results)} Kruskal-Wallis + {len(corr_rows)} Pearson)")
    writeLog("info", logger, "=" * 100)
    for u in unified:
        sig_bonf = "SIG" if u['reject_bonferroni'] else "ns"
        sig_fdr = "SIG" if u['reject_fdr'] else "ns"
        writeLog("info", logger, f"  [{u['test']:13s}] {u['label']:45s} | p_crudo={u['p_value']:.4f} | "
                                  f"p_bonf={u['p_bonferroni']:.4f} ({sig_bonf}) | p_fdr={u['p_fdr']:.4f} ({sig_fdr}) | "
                                  f"efecto={u['effect_size']:.5f}")

    n_sig_bonf = sum(1 for u in unified if u['reject_bonferroni'])
    n_sig_fdr = sum(1 for u in unified if u['reject_fdr'])
    writeLog("info", logger, f"\n🔍 Resumen: {n_sig_bonf}/{len(unified)} sobreviven Bonferroni | "
                              f"{n_sig_fdr}/{len(unified)} sobreviven FDR")
    if n_sig_bonf > 0:
        writeLog("info", logger, "⚠️ Hay resultado(s) que sobreviven a corrección — revisar tamaño de efecto "
                                  "antes de interpretar como relevante (ver columna 'efecto').")

    df_unified = pd.DataFrame(unified)
    path = (output_dir / 'unified_correction_results.csv') if output_dir else 'unified_correction_results.csv'
    df_unified.to_csv(path, index=False)
    writeLog("info", logger, f"\n✅ Tabla unificada exportada a '{path}'.")

    return df_unified
# ============================================================
# ORQUESTACIÓN
# ============================================================

def sanity_check(df_path, output_dir=None):
    """Ejecuta el sanity check completo sobre el CSV de `build_features.py`.

    Args:
        df_path (str o Path): Ruta al CSV generado por
            `build_psri_dataframe`.
        output_dir (Path, optional): Directorio de salida de figuras y CSV.

    Returns:
        None
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    df = load_and_prepare(df_path)

    has_s_coher = 'S_coher' in df.columns
    if has_s_coher:
        writeLog("info", logger,"📌 Columna 'S_coher' encontrada (coherencia cruzada HR-ET).")
    else:
        writeLog("info", logger,"⚠️ Columna 'S_coher' no encontrada. Se omitirá su análisis.")

    writeLog("info", logger,"=" * 70)
    writeLog("info", logger,"📊 PASO 0: SANITY CHECK DEL PSRI POR SUJETO + S_COHER")
    writeLog("info", logger,"=" * 70)

    cols_psri = ['PSRI', 'S_estab', 'PSRI_hr', 'PSRI_et']
    if has_s_coher:
        cols_psri.append('S_coher')
    if 'S_cond' in df.columns:
        cols_psri.append('S_cond')
    cols_psri_exist = [c for c in cols_psri if c in df.columns]

    print_descriptive_stats(df, cols_psri_exist)
    plot_distributions(df, cols_psri_exist, output_dir)
    analyze_component_correlations(df, cols_psri_exist, output_dir)

    run_full_analysis_with_correction(df, output_dir)


def process_sanity_check():
    """Función de entrada que usa `inicioModulo` y ejecuta `sanity_check`.

    Returns:
        None
    """
    input_dir, output_dir = inicioModulo("processSanityCheck")
    sanity_check(output_dir / "physio_with_psri_memes.csv", output_dir=output_dir)