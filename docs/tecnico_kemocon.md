# Análisis técnico proceso K-EmoCon (Experimento 2 PSRI)

Este documento describe el flujo completo de procesamiento del **Estudio 2** del framework PSRI sobre el dataset K-EmoCon. Sigue un enfoque conceptual **Entradas → Proceso → Salidas**, centrado en el *qué entra*, el *qué se hace* y el *qué se produce*, más que en los detalles sintácticos de implementación.

---

## 1. Resumen conceptual

El pipeline K-EmoCon construye, por **ventana de 5 segundos y por sujeto**, un índice de fiabilidad fisiológica multi-componente (**PSRI**) a partir de señales del wearable **Empatica E4**, señales **NeuroSky/Polar** y anotaciones emocionales (self, partner y raters externos). El objetivo del estudio es **validar si el PSRI predice el desacuerdo entre anotadores** de la emoción: la hipótesis es que cuando la señal fisiológica es de mala calidad (inestable, incoherente entre sistemas, o la conducta respecto a la tarea es anómala), los anotadores discrepan más.

El PSRI se compone de tres eslabones independientes, combinados con pesos iguales (1/3, 1/3, 1/3):

| Componente | Qué mide | Cómo se calcula |
|---|---|---|
| **S_estab** | Estabilidad intra-trial (varianza anómala = mala señal) | Función en U gaussiana en log-espacio (`compute_psri_gaussian_log`) sobre la std por ventana de BVP, EDA, HR, TEMP, IBI |
| **S_coher** | Coherencia entre dos sistemas fisiológicos distintos | Z-score respecto a baseline previo (`compute_z_score`) de HR y EDA, y acuerdo posterior (`compute_coherence`) |
| **S_cond** | Plausibilidad de la conducta respecto a la tarea | Caja de herramientas según la semántica del observable: monotónica cuando hay dirección unívoca (atención "más es mejor"), en U cuando ambos extremos son patológicos; aplicada a Attention (NeuroSky), movimiento (acc_std), Meditation y consistencia de la autoanotación |

---

## 2. Entradas (inputs)

Todas las entradas viven bajo `results/input/KEMOCON/` (raíz del dataset: `results/input/KEMOCON/k-emocon/`). Se define por `inicioModulo()` a partir de `config.json` (`environment.input` + `args.proc`).

```
results/input/KEMOCON/
└── k-emocon/
    ├── metadata/
    │   ├── subjects.csv            # catálogo de sujetos (id → edad/sexo/etc.)
    │   └── data_availability.csv   # qué modalidades/anotaciones tiene cada pid
    ├── e4_data/
    │   └── <pid>/E4_{ACC,BVP,EDA,HR,IBI,TEMP}.csv   # señales Empatica E4
    ├── neurosky_polar_data/
    │   └── <pid>/{Attention,BrainWave,Meditation,Polar_HR}.csv  # cobertura irregular
    ├── emotion_annotations/
    │   ├── self_annotations/P<pid>.self.csv           # autoanotación
    │   ├── partner_annotations/P<pid>.partner.csv     # anotación de pareja
    │   ├── external_annotations/P<pid>.R{1..5}.csv    # anotadores externos
    │   └── aggregated_external_annotations/P<pid>.external.csv
    └── data_quality_tables/
        ├── e4_completeness.csv    # fracción de señal completa por modalidad
        ├── e4_durations.csv       # duración por modalidad
        ├── e4_outliers.csv        # fracción de outliers por modalidad
        └── e4_zeros.csv           # fracción de ceros por modalidad
```

**Tablas de calidad (formato confirmado):** las tablas `data_quality_tables` **no tienen columna `pid`**; la fila N tras la cabecera corresponde al sujeto `pid=N`, y las celdas `n/a` indican que ese sujeto no tiene el archivo E4 correspondiente. Los valores pueden venir como número directo (`0.998`) o como texto `"count (fracción)"` (`"4007 (0.994)"`).

---

## 3. Flujo de procesos (visión general)

La orquestación arranca en `main.py`:

```
python main.py --proc KEMOCON
```

```
main.py
  └── process_kemocon()  ──►  sources/kemocon/run_pipeline.py::process_kemocon()
                                │
                                ├─ [1] Carga metadatos            loader.load_metadata()
                                │        └── sujeos elegibles     loader.eligible_subjects()
                                │              (E4 completo + self + partner + ≥3 raters externos)
                                │
                                ├─ [2] Tablas de calidad          sanity_check.load_quality_tables()
                                │        └── exclusión calidad    sanity_check.flag_low_quality_subjects()
                                │
                                ├─ [3] Por cada sujeto elegible:
                                │        ├── Carga E4 (6 modalidades)        loader.load_e4_subject()
                                │        ├── Carga NeuroSky/Polar (4 + 1)    loader.load_neurosky_polar_subject()
                                │        ├── Carga anotaciones               loader.load_annotations_subject()
                                │        └── Tabla ventanas sujeto           aggregator.build_subject_table()
                                │
                                ├─ [4] Fusión multi-sujeto        build_features.build_full_feature_table()
                                │        ├── S_estab  (población global)
                                │        ├── S_cond   (7 variantes en paralelo + 4 monotónicas)
                                │        ├── S_coher  (prev1 y prev5)
                                │        ├── PSRI compuesto prev1 / prev5
                                │        └── métricas de desacuerdo inter-anotador
                                │
                                ├─ [5] Filtrado de calidad        sanity_check.sanity_check_feature_table()
                                │
├─ [6] Validaciones + diagnósticos (correlación Spearman,
                                 │        bloque único de corrección múltiple Bonferroni/FDR,
                                 │        descomposición between/within LOSO, barrido de n_prev,
                                 │        independencia de componentes, task-validity pre/debate,
                                 │        confounds entre-sujetos de S_estab)
                                │
                                ├─ [7] Figuras                     plots.plot_scoher_baseline_sweep()
                                │        └──                       plots.plot_between_within_decomposition()
                                │
                                └─ [8] Salidas CSV + PNG/PDF a results/output/KEMOCON/
```

Módulos implicados:

| Módulo | Rol | Fase |
|---|---|---|
| `run_pipeline.py` | Orquesta todo el estudio | 1–8 |
| `loader.py` | Carga de datos crudos y elegibilidad | 1, 3 |
| `sanity_check.py` | Filtrado por calidad de señal | 2, 5 |
| `aggregator.py` | Ventaneo temporal y sincronización | 3 |
| `build_features.py` | Cálculo del PSRI y validaciones | 4, 6 |
| `task_validity.py` | Control de validez de tarea (reposo vs debate) | 6 |
| `sestab_confounds.py` | Control de confounds del efecto entre-sujetos de S_estab (robustez) | 6 |
| `plots.py` | Figuras del estudio | 7 |
| `experiment3.py` | Herramienta exploratoria LOSO-CV (fuera del paper) | aparte |

---

## 4. Proceso detallado

### 4.1. Selección de sujetos (`loader.py`)

**Entrada:** `metadata/subjects.csv`, `metadata/data_availability.csv`.

**Proceso:**
1. `load_metadata()` lee ambos CSV.
2. `eligible_subjects()` calcula el máscara de elegibilidad: **todos** los canales E4 disponibles, autoanotación presente, anotación de pareja presente y **≥ 3** anotadores externos (umbral configurable `require_external_min=3`).
3. Los timestamps se normalizan a **segundos** por umbral de magnitud: si la mediana de `timestamp` supera `1e12` se divide por 1000 (los CSVs de Empatica suelen estar en segundos; `Polar_HR.csv` está confirmado en **milisegundos**).

**Salida:** lista ordenada de `pid` elegibles.

### 4.2. Filtrado por calidad (`sanity_check.py`)

**Entrada:** tablas `data_quality_tables/`.

**Proceso:** `flag_low_quality_subjects()` excluye sujetos que incumplan **cualquiera** de estos criterios en **cualquier** modalidad:

| Criterio | Regla | Umbral por defecto |
|---|---|---|
| `completeness` | fracción de señal completa < `min_completeness` | 0.90 |
| `zeros` | fracción de ceros > `max_zero_frac` | 0.20 |
| `outliers` | fracción de outliers > `max_outlier_frac` | 0.10 |
| `duration_dropout` | `duración_min / duración_max` de las propias modalidades < `min_duration_ratio` | 0.50 |

El último criterio detecta **pérdida de señal específica de una modalidad** dentro de un sujeto por lo demás válido (p. ej. fallo del detector de picos IBI), no sesiones cortas. Con `verbose=True` se imprime el desglose por criterio (transparencia metodológica). Las celdas se parsean con el parser genérico `_extract_fraction` que entiende números directos, textos `"count (fracción)"` y `n/a`.

**Salida:** conjunto de sujetos excluidos.

### 4.3. Ventaneo y sincronización (`aggregator.py`)

**Entrada:** por sujeto, dicts de DataFrames (E4, NeuroSky/Polar, anotaciones) y `t0` = timestamp mínimo de todas las modalidades E4.

**Proceso (`window_signal`)**: cada señal se refiere a tiempo relativo (`t_rel = timestamp − t0`) y se colapsa en ventanas de **5 s** (`WINDOW_S = 5.0`), la resolución nativa de las anotaciones K-EmoCon:
- Canales univaluados (BVP, EDA, HR, TEMP, IBI, Polar_HR, Attention, Meditation) → `{prefix}_mean`, `{prefix}_std`, `{prefix}_n_samples`.
- Canal multi-eje (ACC: x, y, z) → solo `{prefix}_std` (media de las stds de los 3 ejes) y `{prefix}_n_samples`.

Las anotaciones se agregan con `groupby("win").mean()` (`window_annotation`) para garantizar **una sola fila por ventana** (si vinieran muestreadas a < 5 s, una fusión sin agregar duplicaría las filas fisiológicas).

**Regla crítica de fusión (`build_subject_table`):**
- Las fuentes **E4** definen la **población global de ventanas**: se fusionan entre sí con `outer` (todas cubren la misma sesión de debate).
- Las fuentes **auxiliares de otro dispositivo** (Polar_HR, Attention, Meditation) se fusionan con `left` **nunca con `outer`**: si un dispositivo tiene un rango de grabación distinto, un `outer` expande silenciosamente el conjunto de ventanas y contamina la población de referencia global de S_estab.
- Dos `assert` protegen esta invariante (nº de ventanas no debe cambiar y `win` no debe duplicarse).

**Salida:** tabla ventana × features por sujeto con prefijos por canal (`bvp_*`, `eda_*`, `hr_*`, `temp_*`, `ibi_*`, `acc_*`, `polar_hr_*`, `attention_*`, `meditation_*`) y columnas de anotación (`self_valence`, `self_arousal`, `R{i}_valence`, `R{i}_arousal`, `partner_valence`, ...).

### 4.4. Construcción del PSRI (`build_features.py`)

`build_full_feature_table()` concatena todas las tablas sujeto y añade, en orden:

1. **S_estab (`add_s_estab`)**: `compute_psri_gaussian_log` aplicado a la std por ventana de cada canal, con **población global** (todos los sujetos y ventanas juntos, no por sujeto). Se promedia la fiabilidad de los canales disponibles.
2. **S_cond (`add_s_cond_variants`)**: la caja de herramientas de la dimensión (conceptual_psri.md §1.3) aplicada a candidatas conductuales: **7 variantes en U** (observables con ambos extremos patológicos) calculadas en paralelo:
   - `S_cond_att_acc`: Attention (NeuroSky) + `acc_std` (la usada en el compuesto).
   - `S_cond_att_med`: Attention + Meditation.
   - `S_cond_self`: consistencia de autoanotación (`self_valence_local_std`, `self_arousal_local_std` por rolling de 5 ventanas) — la más próxima en espíritu al original de EXIST (conducta respecto a la tarea).
   - `S_cond_combined`: media de todos los `R_*` disponibles.
   - `S_cond_att`: solo Attention.
   - `S_cond_att_std`: Attention calculada sobre std en vez de mean.
   - `S_cond_att_med_std`: Attention + Meditation, ambas sobre std.

   Además se calculan **4 variantes monotónicas** (`_apply_monotonic_transform`) para los observables con dirección unívoca: `S_cond_att_mono` (Attention creciente), `S_cond_att_med_mono` (Attention+Meditation), `S_cond_att_mono_acc_hi` (movimiento "más es mejor") y `S_cond_att_mono_acc_lo` (movimiento "menos es mejor"). El contraste en-U vs. monotónica es la evidencia empírica de la regla de selección por semántica del observable (EstudioKemocon.md §3.9).
3. **S_coher (`add_s_coher`)**: **HR (E4) vs EDA**, dos sistemas fisiológicos genuinamente distintos (cardiovascular vs electrodérmico). Para ambas señales: Z-score con `baseline_prev` (media de los `n_prev` trials inmediatamente anteriores, por sujeto) y `subject_std` histórica completa del sujeto; después `compute_coherence(z_a, z_b) = exp(−|z_a − z_b|/2)`. Se genera en **dos variantes**: `S_coher_prev1` (n_prev=1) y `S_coher_prev5` (n_prev=5).
4. **PSRI compuesto (`add_psri_composite`)**: imputación por mediana de cada componente (`*_imputed`) y `compute_weighted_psri = w1·S_estab + w2·S_coher + w3·S_cond`, con `w = (1/3, 1/3, 1/3)`. Se genera `psri_composite_prev1` y `psri_composite_prev5` usando `S_cond_att_acc` como variante fija.
5. **Desacuerdo inter-anotador (`annotator_disagreement_features`)**: a partir de R1..R5 (patrón **explícito**, nunca glob `R*` — incluiría la autoanotación `R_self_*`):
   - `external_valence_mean/var/range`, `n_valid_raters_valence` (media y varianza sobre **los mismos raters**, skipna).
   - `external_arousal_mean/var`, `n_valid_raters_arousal`, rango de valence.
   - `self_partner_diff = |self − partner|`, `self_external_mean_diff = |self − media externos|`.

**Salida:** `kemocon_feature_table.csv` (una fila por par sujeto-ventana).

### 4.5. Validaciones y diagnósticos (`build_features.py` + `run_pipeline.py`)

- **`validate_psri_vs_disagreement_combined`** (resultado confirmatorio principal del paper): bloque **único** de corrección por comparaciones múltiples (Bonferroni + FDR) sobre los 10 tests — 2 variantes de PSRI (prev1/prev5) × 5 métricas de desacuerdo. Consistente con el criterio de corrección del Estudio 1. Se calcula tanto para el compuesto de **3 patas** (`psri_composite_prev1/prev5`) como para el de **2 patas** (`psri_composite_2leg_prev1/prev5`, sin S_cond) — la alternativa `PSRI_validated` de docs/dialogo.txt, que no diluye los dos componentes nucleares.
- **`validate_components_vs_disagreement`** (diagnóstico): desglose del compuesto en sus 14 columnas de componentes (S_estab, 7 variantes U de S_cond + 4 monotónicas, S_coher prev1/prev5) × 5 métricas, con su propio bloque de 70 tests.
- **`leverage_diagnostics`**: para un componente dado, descompone la correlación en `rho_full`, `rho_between_subject` (usando solo medias por sujeto — efecto "rasgo estable") y `rho_within_subject` (person-mean-centering — relación trial-a-trial). Añade análisis **leave-one-subject-out** (`loso_*`) para detectar si un único sujeto concentra el efecto. Se aplica a S_estab, S_coher prev1/prev5, las 7 variantes U de S_cond, al par S_estab–S_cond y a los compuestos de 2 y 3 patas (las 4 variantes monotónicas no pasan por este diagnóstico; sí lo hacen por el bloque de 70 tests y por task_validity).
- **`sweep_s_coher_baseline_window`**: barrido de `n_prev ∈ {1,2,3,5,8,10}` con descomposición between/within en cada punto, para comprobar si el efecto de S_coher decae suave y monótonamente con la ventana (señal genuina de corto plazo) o de forma errática (ruido de composición).
- **`cross_component_independence`**: Pearson entre los tres componentes (réplica de la Tabla 5 de EXIST); valores bajos justifican los pesos iguales.
- **`task_validity.run_task_validity`** (diagnóstico): contraste **reposo pre-debate vs. debate** como control de validez de tarea. Etiqueta cada ventana como `pre`/`debate` a partir de `initTime`/`startTime`/`endTime` (`assign_period`), y para cada componente/control ajusta un modelo mixto `col ~ C(period) + (1|subject_id)` (`fit_mixed_model`), un Wilcoxon pareado sobre medias por sujeto (`paired_wilcoxon`), cobertura por periodo, LOSO del modelo mixto (`loso_mixed`) y robustez a excluir las primeras k ventanas de cada periodo (`drop_first_windows`). Cobertura por sujeto: ventanas pre-debate entre 56 y 217.
- **`sestab_confounds.run_sestab_confounds`** (robustez): control de confounds del efecto **entre-sujetos** de S_estab. A nivel sujeto, correlación simple (réplica de `rho_between`) y **parcial de Spearman** controlando individualmente y en conjunto los rasgos `acc_std` (movimiento), `eda_mean`, `hr_mean`, `temp_mean`; además descomposición por canal (`{ch}_S_estab`) y OLS sujeto estandarizado. **Resultado:** el efecto entre-sujetos de S_estab sobre `external_valence_var` (ρ≈−0.28) se derrumba a ≈0.00 al controlar por `acc_std` (la cadena real es *movimiento → inestabilidad → desacuerdo*); sobrevive el control en `self_partner_diff`/`self_external_mean_diff` (ρ≈+0.27/+0.31) y la señal within-subject de `self_partner_diff` (ρ_within=+0.054, p≈0.004), que el movimiento no confunde. Ver EstudioKemocon.md §4.

### 4.6. Figuras (`plots.py`)

- `plot_scoher_baseline_sweep` → `fig1_scoher_baseline_sweep.{png,pdf}`.
- `plot_between_within_decomposition` → `fig2_between_within_decomposition.{png,pdf}` (barras de ρ between/within para S_estab y S_coher, marcando significancia con `*`).

Ambas toman directamente los DataFrames ya generados (no valores transcritos a mano). `reload_and_plot()` permite regenerar las figuras desde los CSV guardados sin re-ejecutar el pipeline.

### 4.7. Métricas de desacuerdo como features

El pipeline también imprime (diagnóstico) la cobertura real pre-imputación por componente y la distribución de `self_valence_local_std` / `self_arousal_local_std` (para investigar si `S_cond_self` es casi binario).

---

## 5. Salidas (outputs)

Todo se escribe en `results/output/KEMOCON/`:

| Archivo | Contenido |
|---|---|
| `kemocon_feature_table.csv` | Tabla ventana × features (componentes PSRI, imputados, compuestos y desacuerdo) |
| `kemocon_psri_validation_combined.csv` | Bloque confirmatorio de 10 tests (2 PSRI × 5 targets), p Bonferroni/FDR |
| `kemocon_psri_validation_combined_2leg.csv` | Ídem para el compuesto de 2 patas (S_estab+S_coher, sin S_cond) |
| `kemocon_psri_leverage_psri_composite_prev1.csv` / `_2leg_prev1.csv` | Leverage between/within/LOSO de ambos compuestos |
| `kemocon_psri_validation_components_diagnostic.csv` | Desglose 14 componentes (S_estab, 7 variantes U + 4 monotónicas de S_cond, S_coher prev1/prev5) × 5 targets (diagnóstico, incluye variantes monotónicas) |
| `kemocon_psri_validation_prev1_exploratory.csv` / `_prev5_exploratory.csv` | Exploratorio, no reportar en el paper |
| `kemocon_psri_leverage_diagnostic_estab.csv` | Leverage on S_estab (between/within/LOSO) |
| `kemocon_psri_leverage_diagnostic_coher_prev1.csv` / `_coher_prev5.csv` | Leverage on S_coher (HR-EDA) |
| `kemocon_psri_leverage_diagnostic_s_cond_{att,att_acc,att_med,att_std,att_med_std,self,combined}.csv` | Leverage por variante de S_cond (las 7 variantes U) |
| `kemocon_psri_leverage_estab_vs_cond.csv` | Descomposición between/within del par S_estab–S_cond |
| `kemocon_psri_scoher_baseline_sweep.csv` | Barrido de n_prev (rho_within por ventana) |
| `kemocon_psri_component_independence.csv` | Pearson 3×3 entre componentes |
| `kemocon_task_validity.csv` | Task-validity reposo vs. debate: modelo mixto, Wilcoxon pareado, cobertura, LOSO, robustez (por componente y control) |
| `kemocon_period_assignments.csv` | Nivel Q_window del marco de 3 niveles de validez: asignación pre/debate por ventana (auditoría) |
| `kemocon_sestab_subject_level.csv` | Tabla sujeto agregada (S_estab, canales, confounds, targets) |
| `kemocon_sestab_between_controls.csv` | rho simple y parcial por confound del efecto entre-sujetos de S_estab |
| `kemocon_sestab_channel_decomposition.csv` | Qué canal (bvp/eda/hr/temp/ibi) arrastra el efecto entre-sujetos |
| `kemocon_sestab_ols.csv` | OLS sujeto (estandarizado): target ~ S_estab + confounds |
| `fig1_scoher_baseline_sweep.png/pdf` | Figura del barrido de baseline |
| `fig2_between_within_decomposition.png/pdf` | Figura del desglose between/within sujetos |
| `experiment3/kemocon_experiment3_{per_fold,summary}.csv` | Salidas de la herramienta exploratoria |

---

## 6. Mapa Entradas → Salidas (resumen)

| Entrada | Proceso | Salida |
|---|---|---|
| `metadata/*.csv` | `loader.eligible_subjects` | Lista de sujetos elegibles |
| `data_quality_tables/*.csv` | `sanity_check.flag_low_quality_subjects` | Conjunto de sujetos excluidos |
| `e4_data/<pid>/E4_*.csv` | `aggregator.build_subject_table` (ventanas 5 s, fusión con invariantes) | Tabla ventana × features por sujeto |
| `neurosky_polar_data/<pid>/*.csv` | `aggregator.window_signal` (auxiliares con `left`) | Columnas `attention_*`, `meditation_*`, `polar_hr_*` |
| `emotion_annotations/**/*.csv` | `aggregator.window_annotation` (mean por ventana) | Columnas `self_*`, `partner_*`, `R{i}_*` |
| Tabla sujeto | `build_features.build_full_feature_table` | `kemocon_feature_table.csv` |
| Feature table (excluidos) | `sanity_check.sanity_check_feature_table` | Feature table final depurada |
| Feature table final | `build_features.validate_*`, `leverage_diagnostics`, `sweep_s_coher_baseline_window`, `cross_component_independence` | CSVs de validación/diagnóstico |
| Feature table final | `task_validity.run_task_validity`, `sestab_confounds.run_sestab_confounds` | CSVs de task-validity y confounds |
| CSVs de diagnóstico | `plots.plot_*` | `fig1_scoher_baseline_sweep.{png,pdf}`, `fig2_between_within_decomposition.{png,pdf}` |

---

## 7. Herramienta exploratoria separada: `experiment3.py`

**NO parte del paper** (decisión de diseño, jul 2026); se conserva como herramienta reusable. Evalúa si el PSRI usado como **filtro** o **peso** mejora operativamente una regresión continua frente a usar la señal cruda.

- **CV:** Leave-One-Subject-Out; métricas por fold: CCC (Lin 1989), RMSE, MAE.
- **Modelos:** RandomForest o HistGradientBoosting con hiperparámetros fijos. Cinco variantes:
  - M1 Base, M2 filtro por `psri > 0.6`, M3 ponderación por PSRI en `fit()`, M4 control por varianza del target (`1/(1+var)`), M5 control aleatorio (drop del mismo % que M2, seed fija).
- **Targets:** emoción continua (medias de raters externos) o desacuerdo externo (los mismos 5 targets del Estudio 2).
- **Significación:** Wilcoxon pareado + t pareado de los deltas M1 vs cada modelo a través de los folds.
- **Resultado empírico:** en todas las configuraciones el CCC por fold es ≈ 0 y ningún modelo PSRI supera al base; el control aleatorio lo iguala o supera.

Ejecución: `.venv/bin/python -m sources.kemocon.experiment3 [emotion|disagreement]`.

---

## 8. Notas clave y gotchas

- **Población global de S_estab:** la referencia (mediana/MAD en log) se calcula sobre TODOS los pares sujeto-ventana juntos. Solo las fuentes E4 pueden definir el rango de ventanas; las auxiliares de otro dispositivo se fusionan siempre con `left` (un `outer` contaminó números de S_estab que no dependían del dispositivo auxiliar).
- **Patrón R1..R5, nunca `R*`:** incluir los `R_self_*` en las métricas "externas" contaminaba la varianza con la autoanotación del propio sujeto (bug pre-existente, ya corregido).
- **Media y varianza externas sobre los mismos raters:** requerido por el Experimento 3 (Control M4).
- **Resolución temporal:** ventanas de 5 s = resolución nativa de las anotaciones.
- **Normalización de timestamps:** a segundos por umbral `1e12` (Polar_HR en ms, resto presumiblemente s).
- **Alineamiento temporal de las anotaciones:** en K-EmoCon, la columna `seconds` de las anotaciones es relativa al **inicio del debate** (`startTime`), mientras que la fisiología se ancla a `t0 = initTime` (inicio de grabación E4). `window_annotation`/`build_subject_table` aceptan `ann_offset_s = startTime/1000 − t0`, que expresa ambas series en el mismo reloj (ver EstudioKemocon.md, Sección 3.7).
- **S_cond y la validación a nivel de ventana de tarea (Q_window):** S_cond es la única dimensión con referente de tarea; validarla exige anclar la fisiología al tiempo de tarea (`initTime → startTime/endTime`, `task_validity.assign_period`). Ese mapeo materializa el nivel Q_window y separa calidad de señal (Q_physio) de condición de tarea. Documentado en EstudioKemocon.md §3.8 y conceptual_psri.md §1.3.2.
- **Dos bloques de corrección independientes** (compuesto de 10 tests vs diagnóstico de 70 tests): preguntas distintas, no se mezclan.
- **El efecto entre-sujetos de S_estab en valence es movimiento** (`sestab_confounds.py`): al controlar por el rasgo `acc_std`, la correlación sujeto S_estab↔valence_var cae de ≈−0.28 a ≈0.00 (S_estab–acc_std ρ=−0.43, acc_std↔valence_var ρ=+0.65 a nivel sujeto). No es colinearidad degenerada: el movimiento es un predictor más fuerte que la "calidad fisiológica" y lo explica. El claim defendible de S_estab queda en self_partner_diff/self_external_mean_diff (entre-sujetos) y en la señal within de self_partner_diff.
- **Duplicados en (subject_id, win):** `build_full_feature_table` avisa si los detecta, porque invalidan cualquier correlación posterior.
- `config.json` contiene un token de HuggingFace — no se debe commitear.