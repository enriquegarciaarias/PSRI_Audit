# src/physionet/reporting.py
"""
Presentación (consola y figuras) de los resultados de validación del PSRI.

No calcula nada: recibe los dicts de métricas ya calculados (SRP).

Autor: Enrique
"""
import matplotlib.pyplot as plt


def print_metrics_block(title, m, extra_note=None):
    """Imprime un bloque de métricas ya calculado.

    Args:
        title (str): Título del bloque.
        m (dict): Métricas de `compute_classification_metrics`.
        extra_note (str, optional): Nota adicional a imprimir al final.

    Returns:
        None
    """
    print(f"\n=== {title} ===")
    if m['n'] == 0:
        print("  Sin datos.")
        return
    cm = m['cm']
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    print(f"Matriz de Confusión (Verdadero/Predicho):")
    print(f"               Pred. Acept | Pred. Inacept")
    print(f"Real Acept:      {tp:>7}    |    {fn:>7}")
    print(f"Real Inacept:    {fp:>7}    |    {tn:>7}")
    print(f"\nExactitud (Accuracy):       {m['acc']:.4f}")
    print(f"Sensibilidad (Aceptables):  {m['sens']:.4f}")
    print(f"Especificidad (Inacept):    {m['spec']:.4f}")
    print(f"** Media Geométrica (G-mean) = {m['gmean']:.4f} **")
    print(f"Área bajo la ROC (AUC): {m['auc']:.4f}")
    if extra_note:
        print(extra_note)


def print_validation_report(feature_method, umbral, fail_count, total, metrics_all, metrics_valid, diagnosis):
    """Imprime el reporte completo de una corrida de validación.

    Args:
        feature_method (str): Método de variabilidad usado.
        umbral (float): Umbral de decisión.
        fail_count (int): Número de fallos de extracción.
        total (int): Total de registros procesados.
        metrics_all (dict): Métricas sobre todos los registros.
        metrics_valid (dict): Métricas sobre la extracción válida.
        diagnosis (dict): Diagnóstico de la contribución de los fallos.

    Returns:
        None
    """
    print(f"\nMétodo de variabilidad: {feature_method} | Umbral: {umbral}")
    print(f"Registros procesados: {total}")
    print(f"  Fallos de extracción (detección de picos): {fail_count} "
          f"({fail_count / total * 100:.1f}%)" if total else "")

    print_metrics_block("RESULTADOS (TODOS LOS DATOS)", metrics_all)

    if metrics_valid['n'] > 0:
        print_metrics_block(
            f"RESULTADOS (SOLO EXTRACCIÓN VÁLIDA, excluyendo {fail_count} fallos)",
            metrics_valid
        )
        print(f"\n🔍 Impacto de los fallos de detección:")
        print(f"  AUC (todos) - AUC (válidos) = {diagnosis['diff_auc']:.4f}")
        print(f"  G-mean (todos) - G-mean (válidos) = {diagnosis['diff_gmean']:.4f}")
        if diagnosis['fails_drive_auc']:
            print("  ⚠️ Los fallos de detección contribuyen significativamente al AUC.")
            print("     La separación observada podría deberse en parte a que el detector de picos")
            print("     falla más en registros de mala calidad, no solo a la transformación PSRI.")
        else:
            print("  ✅ La contribución de los fallos de detección es pequeña.")
            print("     La validación refleja principalmente la capacidad de la transformación")
            print("     en U para separar calidad basada en variabilidad de ritmo.")


def plot_roc(feature_method, metrics_all, metrics_valid=None, save_path=None, show=True):
    """Dibuja la curva ROC del PSRI (todos y/o válidos) y la guarda.

    Args:
        feature_method (str): Método de variabilidad, usado en título y nombre.
        metrics_all (dict): Métricas sobre todos los registros.
        metrics_valid (dict, optional): Métricas sobre la extracción válida.
        save_path (str, optional): Ruta de guardado. Por defecto genera
            `roc_physionet_<method>.png`.
        show (bool, optional): Si mostrar la figura. Por defecto es True.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    plt.plot(metrics_all['fpr'], metrics_all['tpr'],
              label=f"All (AUC = {metrics_all['auc']:.3f})", lw=2)
    if metrics_valid is not None and metrics_valid['n'] > 0:
        plt.plot(metrics_valid['fpr'], metrics_valid['tpr'], '--',
                  label=f"Valid (AUC = {metrics_valid['auc']:.3f})", lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - PSRI (method: {feature_method})')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = save_path or f'roc_physionet_{feature_method}.png'
    plt.savefig(path, dpi=300)
    print(f"✅ Curva ROC guardada como '{path}'")
    if show:
        plt.show()
    plt.close()


def plot_boxplot_by_class(feature_method, y_true, scores, save_path=None, show=True):
    """Dibuja un boxplot de los scores PSRI separados por clase real.

    Args:
        feature_method (str): Método de variabilidad, usado en título y nombre.
        y_true (array-like): Etiquetas reales (0 inaceptable, 1 aceptable).
        scores (array-like): Scores PSRI.
        save_path (str, optional): Ruta de guardado. Por defecto genera
            `boxplot_psri_<method>.png`.
        show (bool, optional): Si mostrar la figura. Por defecto es True.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    data_to_plot = [scores[y_true == 0], scores[y_true == 1]]
    ax.boxplot(data_to_plot)
    ax.set_xticklabels(['Unacceptable', 'Acceptable'])
    ax.set_ylabel('PSRI score')
    ax.set_title(f'PSRI Distribution (method: {feature_method})')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = save_path or f'boxplot_psri_{feature_method}.png'
    plt.savefig(path, dpi=300)
    print(f"✅ Boxplot guardado como '{path}'")
    if show:
        plt.show()
    plt.close()


def print_comparison_table(results_by_method):
    """Imprime la tabla comparativa de métodos de variabilidad.

    Args:
        results_by_method (dict): {feature_method: resultado de
            `validate_psri(...)`}.

    Returns:
        None
    """
    print("\n" + "=" * 78)
    print("=== COMPARATIVA DE MÉTODOS DE VARIABILIDAD ===")
    print("=" * 78)
    header = f"{'Método':12s} | {'AUC(todos)':>10s} | {'AUC(válid)':>10s} | {'G-mean(t)':>10s} | {'G-mean(v)':>10s} | {'Fallos':>7s}"
    print(header)
    print("-" * len(header))
    for method, r in results_by_method.items():
        print(f"{method:12s} | {r['auc_all']:>10.4f} | {r['auc_valid']:>10.4f} | "
              f"{r['gmean_all']:>10.4f} | {r['gmean_valid']:>10.4f} | {r['fail_count']:>7d}")
    print("=" * 78)