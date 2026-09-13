# Análisis técnico proceso PhysioNet (Experimento 3 PSRI: validación del instrumento)

Este documento describe el flujo completo de procesamiento de la **validación del instrumento PSRI** (Estudio 3) contra el dataset **PhysioNet/CinC Challenge 2011**. Sigue un enfoque conceptual **Entradas → Proceso → Salidas**, centrado en el *qué entra*, el *qué se hace* y el *qué se produce*, más que en los detalles sintácticos de implementación.

---

## 1. Resumen conceptual

Los Estudios 1 y 2 (EXIST y K-EmoCon) usan el PSRI como predictor del desacuerdo entre anotadores. Este estudio es distinto en propósito: **valida el instrumento en sí** contra un ground truth de calidad de señal ECG etiquetado por expertos humanos. La pregunta es: ¿el PSRI (función en U gaussiana en log-espacio sobre la desviación estándar de la señal) separa ECGs de calidad aceptable de inaceptables, comparado con los índices de calidad de señal (SQI) de referencia de la literatura?

La hipótesis operativa es que la señal ECG de un registro de calidad **aceptable** tiene una variabilidad "típica" (actividad cardíaca genuina, QRS marcado), mientras que la de un registro **inaceptable** se aparta de esa típica en ambos extremos: señal plana/rota (varianza casi nula) o señal ruidosa/artefactada (varianza muy alta). Por eso la forma en U (penalizar tanto lo planó como lo ruidoso) es el mecanismo central, y esta es la única parte del framework donde puede contrastarse contra expertos humanos.

**Resultado validado (en el papel):** el PSRI alcanza **AUC = 0.887** (0.8868 exacto) sobre los 998 registros de `set-a`, superior a kSQI (curtosis, AUC 0.842) y a la Correlación inter-derivación (AUC 0.794). La ventaja sobre kSQI no es estadísticamente significativa (DeLong p = 0.066), pero sí lo es frente a la Correlación inter-derivación (p = 0.0003). El AUC se mantiene fuera de muestra (0.8860, IC95% [0.8545, 0.9195]), y el estudio de ablación demuestra que la transformación propuesta supera a sus alternativas no paramétricas (Percentil Empírico y Min-Max, p < 1e-4).

---

## 2. Entradas (inputs)

Todas las entradas viven bajo `results/input/PHYSIO/`. Se define por `inicioModulo("validate_psri")` a partir de `config.json` (`environment.input` + `args.proc`).

```
results/input/PHYSIO/
└── set-a/
    ├── RECORDS                  # lista completa de IDs de registros (998)
    ├── RECORDS-acceptable       # IDs etiquetados como calidad aceptable
    ├── RECORDS-unacceptable     # IDs etiquetados como calidad inaceptable
    └── <record_id>.{dat,hea,txt}  # señales ECG en formato WFDB (multi-derivación)
```

El ground truth se lee de los dos archivos `RECORDS-*` (`loader.load_ground_truth`): los IDs en `RECORDS-acceptable` → clase 1, los de `RECORDS-unacceptable` → clase 0. `RECORDS` da el orden de procesamiento. Cada registro se lee con la librería **`wfdb`** (`wfdb.rdrecord`), de la que se extrae la matriz de señales `p_signal` (N muestras × n derivaciones) y la frecuencia de muestreo `fs`.

---

## 3. Flujo de procesos (visión general)

La orquestación arranca en `main.py`:

```
python main.py --proc PHYSIO
```

```
main.py
  └── process_physio()                      sources/physionet/validator.py
        │
        ├─ [1] Comparativa de AUC            compare_psri_vs_reference_sqis()
        │        ├── PSRI (std_signal)       validate_psri(feature_method='std_signal')
        │        ├── kSQI (curtosis)         validate_reference_sqi(compute_ksqi)
        │        └── Correlación inter-deriv. validate_reference_sqi(compute_interlead_correlation_sqi)
        │
        ├─ [2] Significación estadística     compare_psri_vs_reference_sqis_with_significance()
        │        ├── Recalcular 3 scores por registro (intersección pareada)
        │        ├── PR-AUC (average precision) bajo desbalance 773/225
        │        ├── Test de DeLong (AUCs correlacionadas) + bootstrap pareado
        │        └── ROC superpuestas (png/pdf) + curvas con umbrales operativos
        │
        ├─ [3] Sensibilidad de calibración   check_psri_calibration_sensitivity()
        │        └── split-half out-of-sample (200 splits) → distribución de AUC
        │
        ├─ [4] Limitación de solapamiento    analyze_psri_overlap_limitation()
        │        └── zona P5-P95 de log(std_signal) por clase → ¿dónde caen los errores?
        │
        └─ [5] Ablación metodológica         run_ablation_study()
                 └── 5 variantes de la transformación sobre el MISMO std_signal
                      + DeLong/bootstrap + tabla LaTeX + ROC superpuesta
```

Módulos implicados:

| Módulo | Rol | Fase |
|---|---|---|
| `validator.py` | Orquesta las 5 validaciones | 1–5 |
| `loader.py` | Ground truth y listas de registros | todas |
| `features.py` | Extracción de variabilidad (`std_signal`, `std_window`, `rr_std`) | 1, 5 |
| `reference_sqi.py` | kSQI y Correlación inter-derivación | 1, 2 |
| `reporting.py` | Reportes de consola, ROC y boxplot por clase | 1 |
| `auc_comparison.py` | DeLong, bootstrap pareado, ROC comparativas | 2, 5 |
| `calibration_sensitivity.py` | Split-half fuera de muestra | 3 |
| `overlap_analysis.py` | Diagnóstico de la zona de solapamiento | 4 |
| `sources/psri/validation.py` | `compare_all_vs_valid` (todos vs extracción válida) | 1 |
| `sources/common/metrics.py` | Métricas de clasificación binaria (AUC, G-mean...) | 1 |

---

## 4. Proceso detallado

### 4.1. Extracción de variabilidad (`features.extract_variability`)

**Entrada:** ruta de un registro WFDB (`set-a/<record_id>`).

**Proceso:** `wfdb.rdrecord` → `p_signal` y `fs`. Tres métodos de variabilidad:
- `std_signal`: `std` por derivación de toda la señal, promediada (`np.mean(std_leads)`). **El método usado por defecto** (validado y comparado).
- `std_window`: `std` por ventanas de 1 segundo, mediana de las ventanas.
- `rr_std`: detección de picos QRS con `wfdb.processing.xqrs_detect` sobre la 2ª derivación (si existe), `std` de los intervalos RR. Si no hay ≥2 picos, devuelve `fail_flag=True` (y 0.0). Es el único método que puede **fallar** (detección de picos), y en `rr_std` el fallo se registra explícitamente.

**Salida:** `(valor, fail_flag)` por registro (con `return_fail_flag=True`). `std_signal` nunca falla (fail_flag siempre False).

### 4.2. Cálculo del score PSRI (`validator.validate_psri`)

**Entrada:** `variability_values` (std_signal por registro) + `y_true`.

**Proceso:**
1. `compute_psri_gaussian_log(variability_values, eps_hard=None)`: función en U en log-espacio — `eps_hard` = percentil 1 de los no-cero (señal plana → R=0), log-transform, z-score robusto mediana/MAD, `R = exp(−0.5·z²)`.
2. `compare_all_vs_valid(y_true, scores, valid_mask)`: métricas sobre **todos** los registros y sobre el subconjunto con **extracción válida** (sin fallos), con diagnóstico de si los fallos explican la separación (`fails_drive_auc`, umbral ΔAUC=0.05).
3. `compute_classification_metrics` (umbral 0.5): exactitud, sensibilidad, especificidad, **G-mean** = √(sens·spec) y **AUC** (curva ROC).

**Resultado (std_signal):** AUC = 0.8868 sobre los 998 registros (0 fallos). Reporte por consola de matrices de confusión, G-mean, y la comparación todos/válidos.

### 4.3. SQIs de referencia (`reference_sqi.py` + `validate_reference_sqi`)

**Entrada:** ruta de registro WFDB.

**Proceso:**
- `compute_ksqi`: curtosis (Fisher, sin sesgo) promediada por derivación. Si alguna derivación es plana (`std < 1e-6`) o la curtosis no es finita → **None** (señal degenerada → peor calidad).
- `compute_interlead_correlation_sqi`: correlación media de los pares de derivaciones (triángulo superior). Si hay <2 derivaciones → **`np.nan`** (legítimamente no evaluable). Si alguna derivación es plana → **None**.

**Manejo de casos degenerados en `validate_reference_sqi`:** los `None` se reasignan a un **sentinel** (valor finito por debajo de todos los scores reales: `min_finite − (|min_finite|·0.1 + 1)`); los `np.nan` se **descartan** de la evaluación (contados por clase). En set-a: **135 registros degenerados** reasignados a sentinel (6 aceptables, 129 inaceptables) tanto para kSQI como para la Correlación — sin descartes por no-evaluable.

**Resultados:** kSQI AUC=0.8421, G-mean=0.7067; Correlación inter-derivación AUC=0.7938, G-mean=0.6775.

### 4.4. Comparación pareada con significación (`auc_comparison.py`)

**Entrada:** los tres scores por `rec_id`.

**Proceso (comparación pareada):** se recalculan los 3 scores **por registro** y se construye la intersección donde los tres son válidos (DeLong exige correspondencia registro-a-registro; `validate_psri` no descarta registros pero `validate_reference_sqi` sí). En set-a: 998/998 con los tres scores (0 descartados).

1. **PR-AUC** (`average_precision_score`) bajo desbalance de clases (773 aceptables / 225 inaceptables ≈ 77/23): PSRI=0.9510, kSQI=0.9107, Correlación=0.8754.
2. **Test de DeLong** para AUCs correlacionadas (mismos registros): PSRI vs kSQI diff=+0.0446, p=0.0657 (no significativo); PSRI vs Correlación diff=+0.0930, p=0.0003 (significativo).
3. **Bootstrap pareado** (2000 réplicas, remuestreo de registros preservando el emparejamiento) como contraste independiente: PSRI vs kSQI diff=+0.0446, IC95%=[−0.0040, 0.0933], p=0.067; PSRI vs Correlación diff=+0.0930, IC95%=[0.0428, 0.1430], p=0.
4. **Figuras:** `fig_physionet_roc_comparison.{png,pdf}` (las 3 ROC superpuestas) y `fig_physionet_roc_thresholds.{png,pdf}` (curvas con cuartiles por método y umbral operativo: PSRI fijo en 0.5, kSQI/Correlación usan su mediana).

### 4.5. Sensibilidad de calibración (`calibration_sensitivity.py`)

**Motivación:** `compute_psri_gaussian_log` calibra su mediana/MAD/`eps_hard` sobre la **misma** población que puntúa (ventaja estructural frente a kSQI/Correlación, fórmulas cerradas sin calibración). Este chequeo cuantifica cuánto del AUC=0.887 depende de calibrar y evaluar sobre los mismos datos.

**Proceso (`split_half_sensitivity`):** 200 particiones aleatorias 50/50; en cada una se fija `eps_hard`/mediana/MAD con la mitad de **calibración** y se puntúa la mitad de **evaluación** (`compute_psri_scores_with_reference`). Se descartan los splits sin ambas clases en evaluación.

**Resultado:** AUC fuera de muestra media=**0.8860**, mediana=0.8851, std=0.0159, IC95%=[0.8545, 0.9195] (200/200 splits válidos), frente al AUC in-sample original 0.8870 → **la calibración no infla el resultado**.

### 4.6. Limitación estructural de solapamiento (`overlap_analysis.py`)

**Motivación:** reducir toda la señal a un único escalar (`std_signal`) no puede distinguir una señal limpia con QRS marcado (std alta genuina) de una ruidosa con artefactos de amplitud similar (std alta por perturbación) — ambas producen el mismo número de entrada.

**Proceso (`analyze_classification_overlap`):** en espacio `log(std_signal)`, se calcula el rango [P5, P95] de cada clase y su **intersección** = zona de solapamiento. Se comparan las tasas de mal/bien clasificados dentro de esa zona y se desglosa por tipo de error (FP vs FN), con baselines por clase.

**Resultado:** zona [0.0934, 0.4000] (std_signal); el 74.7% de los registros cae en la zona. **56.9%** de los mal clasificados (n=181) caen en la zona frente al **78.7%** de los bien clasificados (n=817): los errores del PSRI **no** se concentran desproporcionadamente en la zona de ambigüedad estructural (la tasa de bien clasificados en zona es mayor). El análisis `breakdown_by_error_type` distingue además FN (cola atípica de la clase aceptable, penalizada por el corte en U pese a ser legítima) de FP (corrupción moderada "que parece típica").

### 4.7. Ablación metodológica (`validator.run_ablation_study`)

**Entrada:** el MISMO array `std_signal` para las 5 variantes (solo difiere la transformación):

| Variante | Normalización | Forma funcional | Función |
|---|---|---|---|
| **Propuesta (PSRI)** | Log + Mediana/MAD | Gaussiana Invertida (U) | `compute_psri_gaussian_log` |
| Alt. A: Mono-Decay | Log + Mediana/MAD | Monótona Decreciente | `compute_psri_mono_decay` |
| Alt. B: Z-Score Clásico | Log + Media/Std | Gaussiana Invertida (U) | `compute_psri_classic_zscore` |
| Alt. C: Percentil Empírico | Ninguna (Ranking) | Escalonado No Paramétrico | `compute_psri_empirical_percentile` |
| Alt. D: Min-Max Crudo | Mínimo/Máximo | Lineal Acotada [0,1] | `compute_psri_minmax` |

**Proceso:** AUC (todos + solo válidos) por variante; frente a la Propuesta, **DeLong + bootstrap pareado** (AUCs correlacionadas, mismos registros). Genera la tabla por consola, la tabla **LaTeX** para el paper (`ablation_study_table.tex`) y la ROC superpuesta.

**Resultados (n=998):** Propuesta **0.8868**; Alt. A Mono-Decay 0.8868 (diff=0.0000, p=1 — el ranking es idéntico porque `1/(1+|z|)` también es monótona decreciente en |z|); Alt. B Z-Score Clásico 0.8681 (diff=+0.0187, p=0.096, ns); Alt. C Percentil Empírico 0.5174 (diff=+0.3694, p<1e-4); Alt. D Min-Max Crudo 0.5174 (diff=+0.3694, p<1e-4). La robustez (MAD) y la forma paramétrica en U superan claramente a las alternativas de ranking y normalización por extremos.

---

## 5. Salidas (outputs)

La mayoría se escribe en `results/output/PHYSIO/`:

| Archivo | Contenido |
|---|---|
| `ablation_study_table.tex` | Tabla LaTeX de la matriz de ablación (AUC por variante + p DeLong) |
| `fig_physionet_ablation_roc.{png,pdf}` | ROC superpuesta de las 5 variantes |
| `fig_physionet_roc_comparison.{png,pdf}` | ROC de PSRI vs kSQI vs Correlación (n=998) |
| `fig_physionet_calibration_sensitivity.{png,pdf}` | Histograma del AUC fuera de muestra (200 splits) |
| `fig_physionet_overlap_diagnostic.{png,pdf}` | Histogramas por clase + zona de solapamiento + errores |
| `outputs/fig_physionet_roc_thresholds.{png,pdf}` | ROC con cuartiles y umbrales operativos por método (**ruta relativa**, no en `results/output`) |

Nota: `validate_psri` también genera por método `roc_physionet_<method>.png` y `boxplot_psri_<method>.png` solo si se le piden (`plot_roc_curve`/`plot_box`); en el flujo activo no se generan. No se escriben CSVs en este estudio: los números se reportan por consola y en los PDFs/LaTeX.

---

## 6. Mapa Entradas → Salidas (resumen)

| Entrada | Proceso | Salida |
|---|---|---|
| `set-a/RECORDS{-acceptable,-unacceptable}` | `loader.load_ground_truth` + `load_record_ids` | `y_true` (1/0) por registro, orden de procesamiento |
| `set-a/<record_id>.{dat,hea}` | `features.extract_variability` (`std_signal`) | `variability_values` (n=998) |
| `variability_values` | `compute_psri_gaussian_log` + `compare_all_vs_valid` | Scores PSRI, AUC/G-mean (todos vs válidos) |
| `set-a/<record_id>` | `reference_sqi.compute_ksqi` / `compute_interlead_correlation_sqi` + sentinel | Scores kSQI / Correlación |
| 3 scores por registro | `compare_psri_vs_reference_sqis_with_significance` (DeLong + bootstrap + PR-AUC) | Comparativas de AUC + `fig_physionet_roc_comparison` / `_roc_thresholds` |
| `std_signal` + `y_true` | `split_half_sensitivity` (200 splits) | AUC out-of-sample + `fig_physionet_calibration_sensitivity` |
| `std_signal` + `y_true` + PSRI | `analyze_classification_overlap` | Diagnóstico de zona de solapamiento + `fig_physionet_overlap_diagnostic` |
| `std_signal` | `run_ablation_study` (5 variantes) | `ablation_study_table.tex` + `fig_physionet_ablation_roc` |

---

## 7. Notas clave y gotchas

- **Forma en U justificada por la ablación:** los peores competidores son los no paramétricos (Percentil Empírico y Min-Max, AUC ≈ 0.52, p<1e-4); la robustez MAD (Alt. B) apenas cambia el AUC (p=0.096). Es la evidencia que respalda la elección de Log + Mediana/MAD + Gaussiana invertida.
- **Alt. A Mono-Decay es indistinguible en AUC:** `1/(1+|z|)` y `exp(−0.5z²)` son ambas monótonas decrecientes en |z|, así que producen el **mismo ranking** y el mismo AUC (diff=0, p=1). Como ablación de la "necesidad de la forma en U" este brazo no discrimina — la diferencia entre ellas es de calibración/escala, no de orden. No reportar como "la alternativa da igual".
- **Comparación pareada obligatoria:** DeLong exige los mismos registros en ambos scores. `validate_psri` no descarta registros pero `validate_reference_sqi` sí descarta los no-evaluables (NaN), de ahí que la comparación con significación **recalcule** los 3 scores por `rec_id` y trabaje sobre la intersección (998/998 en set-a).
- **Sentinel vs NaN en los SQIs:** señal degenerada (derivación plana) → `None` → reasignado a un sentinel por debajo de todos los scores reales (peor calidad, 135 registros en set-a, mayoritariamente inaceptables); señal no evaluable (1 sola derivación, sin pares a correlacionar) → `np.nan` → descartado del análisis. No confundir los dos tratamientos.
- **Calibración in-sample no infla el AUC:** `compute_psri_gaussian_log` calibra mediana/MAD/eps_hard sobre la población que puntúa (ventaja estructural frente a fórmulas cerradas). El split-half demuestra que el AUC fuera de muestra (0.8860) es prácticamente idéntico al in-sample (0.8870): no es fuga, pero conviene reportarlo como limitación/robustez.
- **PR-AUC porque hay desbalance:** 773 aceptables / 225 inaceptables (~77/23). El ROC-AUC puede ser optimista bajo desbalance; el PR-AUC (average precision) es más sensible a los falsos positivos sobre la clase minoritaria. PSRI=0.9510.
- **`rr_std` es el único método que falla:** la detección de picos `xqrs_detect` puede no encontrar QRS válidos (fail_flag). `std_signal` nunca falla, por eso la comparación todos/válidos con `std_signal` es trivial (0 fallos). El patrón de fallos del detector de picos (más fallos en mala calidad) es el mismo fenómeno documentado en K-EmoCon (caída de IBI).
- **`process_physio` tiene código muerto tras el `return`:** el bloque `compare_feature_methods` (`std_signal` vs `rr_std`) quedó desactivado; el flujo activo son las 5 funciones de validación.
- **Figura con ruta relativa:** `plot_roc_comparison_with_thresholds` guarda en `outputs/fig_physionet_roc_thresholds.{png,pdf}` (relativo al CWD), no en `results/output/PHYSIO/` — incoherencia conocida con el resto de figuras.
- `config.json` contiene un token de HuggingFace — no se debe commitear.