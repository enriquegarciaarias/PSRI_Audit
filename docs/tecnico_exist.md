# Análisis técnico proceso EXIST (Experimento 1 PSRI)

Este documento describe el flujo completo de procesamiento del **Estudio 1** del framework PSRI sobre el dataset EXIST 2026. Sigue un enfoque conceptual **Entradas → Proceso → Salidas**, centrado en el *qué entra*, el *qué se hace* y el *qué se produce*, más que en los detalles sintácticos de implementación.

---

## 1. Resumen conceptual

El pipeline EXIST construye, **por meme**, un índice de fiabilidad fisiológica multi-componente (**PSRI**) a partir de las señales de los sensores puestos a los sujetos (frecuencia cardíaca **Garmin** y eye-tracking **Tobii**). El objetivo del estudio es **validar si el PSRI predice el desacuerdo entre anotadores** sobre si un meme es sexista: la hipótesis es que cuando la señal fisiológica es de mala calidad (inestable, incoherente, o con conducta anómala respecto a la tarea), los anotadores discrepan más sobre la etiqueta dura del meme.

Diferencia estructural clave con K-EmoCon (Estudio 2): en EXIST la población que lleva los sensores (8 sujetos) y la población que anota el meme son **grupos disjuntos**, sin solapamiento. La única conexión posible entre la fisiología y el desacuerdo es una **causa común indirecta** (un meme ambiguo produce a la vez señal fisiológica más ruidosa y más desacuerdo). Esto hace que el test sea más débil que el de K-EmoCon, donde sujeto fisiológico y objeto del juicio son la misma persona.

El PSRI se compone de tres eslabones independientes, combinados con pesos iguales (1/3, 1/3, 1/3):

| Componente | Qué mide | Cómo se calcula |
|---|---|---|
| **S_estab** | Estabilidad intra-trial (varianza anómala = mala señal) | Función en U gaussiana en log-espacio (`compute_psri_gaussian_log`) sobre la std de HR y pupila por trial |
| **S_coher** | Coherencia entre dos sistemas fisiológicos distintos | Z-score respecto a baseline previo (`compute_z_score`) de HR y pupila, y acuerdo posterior (`compute_coherence`) |
| **S_cond** | Plausibilidad de la conducta respecto a la tarea | Caja de herramientas según la semántica del observable: transformación monotónica "menos es mejor" para el tiempo de reacción, función en U para el número de parpadeos (eye-tracking) |

**Resultado del estudio (nulo robusto):** ninguna de las 20 pruebas del pool unificado sobrevive a la corrección por comparaciones múltiples (Bonferroni ni FDR). Es un Resultado Negativo Válido, coherente con la desalineación poblacional del diseño. La comprobación final de cierre de fase (§8) confirma que el nulo se mantiene también frente a los reenfoques del framework, con UNA excepción robusta y aislada: el tiempo de reacción en versión monotónica (`S_cond_mono`), documentada como diagnóstico exploratorio.

---

## 2. Entradas (inputs)

Todas las entradas viven bajo `results/input/EXIST/`. Se define por `inicioModulo()` a partir de `config.json` (`environment.input` + `args.proc`).

```
results/input/EXIST/
├── HR_.xlsx                    # frecuencia cardíaca Garmin por trial (meme, username)
├── EEG_.xlsx                   # EEG — se carga pero queda EXCLUIDO del merge
├── ET_.xlsx                    # eye-tracking Tobii (pupila izq/der, fijaciones,
│                               #   sacadas, parpadeos, tiempo de reacción)
├── EXIST2026_training.json     # etiquetas: listas de anotadores por tarea (usado)
└── corpus_EXIST2026_training.json  # corpus bruto de memes (no usado en el cálculo)
```

Cada Excel contiene una fila por **trial** (par `meme_id`, `username`) con estadísticos ya agregados por trial (p. ej. `garmin_hr_mean`, `garmin_hr_std`, `garmin_hr_mean_baseline_prev`, `3d_eye_states_pupil diameter left [mm]_mean`, `reaction_time`, `blinks_count`). No hay series temporales crudas: cada trial es un único agregado.

El JSON de etiquetas mapea cada `meme_id` a las listas de respuestas de anotadores por tarea:
- `labels_task2_1` → lista `["YES","NO",...]` (¿es sexista?).
- `labels_task2_2` → lista `["DIRECT","JUDGEMENTAL",...]` (intención).
- `labels_task2_3` → lista de listas de categorías de sexismo (multiclase).

---

## 3. Flujo de procesos (visión general)

La orquestación arranca en `main.py`:

```
python main.py --proc EXIST
```

```
main.py
  └── process_exist()
        │
        ├─ [1] Construcción del dataframe     build_features.process_dataframe()
        │        └── build_psri_dataframe(input_dir, labels_path, filter_common=True)
        │              ├── Carga Excel HR/EEG/ET               loader.load_dataframes()
        │              ├── Diagnóstico de cobertura            loader.diagnose_coverage()
        │              ├── Filtro memes comunes HR∩ET          (EEG fuera del merge)
        │              ├── PSRI por sujeto (HR y ET)           aggregator.add_psri_subject()
        │              │       ├── PSRI_hr_subj (std HR)
        │              │       ├── PSRI_et_subj (std pupila)
        │              │       └── S_cond_subj (reaction_time + blinks)
        │              ├── Coherencia cruzada HR-ET por trial   compute_z_score + compute_coherence
        │              │       └── agregada por meme (media) → S_coher
        │              ├── Agregación por meme                  aggregator.aggregate_by_meme()
        │              │       ├── hr_* (medias y desviaciones)
        │              │       └── et_* (medias y desviaciones)
        │              ├── S_estab, S_coher, S_cond → PSRI     compute_weighted_psri
        │              └── Etiquetas + entropías                aggregator.merge_labels()
        │                    → physio_with_psri_memes.csv
        │
        └─ [2] Sanity check                    sanity_check.process_sanity_check()
                 ├── Carga y normalización de columnas   load_and_prepare()
                 ├── Estadísticos descriptivos y rangos  print_descriptive_stats()
                 ├── Distribuciones                      plot_distributions()
                 ├── Correlación entre componentes       analyze_component_correlations()
                 └── Pool unificado de corrección        run_full_analysis_with_correction()
                       (2 Kruskal-Wallis + 18 Pearson, Bonferroni + FDR)
                       → unified_correction_results.csv
```

Módulos implicados:

| Módulo | Rol | Fase |
|---|---|---|
| `loader.py` | Carga de los 3 Excel y diagnóstico de cobertura | 1 |
| `aggregator.py` | PSRI por sujeto, agregación por meme, etiquetas/entropías | 1 |
| `build_features.py` | Orquesta la construcción del dataframe final | 1 |
| `sanity_check.py` | Estadísticas, figuras y corrección unificada | 2 |

---

## 4. Proceso detallado

### 4.1. Carga y diagnóstico de cobertura (`loader.py`)

**Entrada:** `HR_.xlsx`, `EEG_.xlsx`, `ET_.xlsx`, `EXIST2026_training.json`.

**Proceso:**
1. `load_dataframes()` lee los tres Excel.
2. `diagnose_coverage()` normaliza `meme_id` a `str` y calcula el solapamiento entre fuentes a dos niveles:
   - **A nivel de par** `(meme_id, username)`: cuántos trials comparten HR↔EEG, HR↔ET, EEG↔ET y los tres.
   - **A nivel de meme** (sin usuario) y contra el conjunto de etiquetas del JSON.
3. El diagnóstico revela la decisión de diseño central: **EEG no tiene solapamiento con HR/ET a nivel de par** (0 pares comunes), pese a que a nivel de meme sí comparte memes.

**Salida:** dict de conjuntos (`hr_pairs`, `et_pairs`, `hr_memes`, `et_memes`, ...) usado para el filtrado posterior.

### 4.2. Filtro de población (`build_features.py`)

**Entrada:** DataFrames cargados y estadísticas de cobertura.

**Proceso (`filter_common=True`):** se conservan solo los memes que tienen **HR y ET** (`common_memes = hr_memes ∩ et_memes`). **EEG queda excluido del merge poblacional**: al tener población de sujetos disjunta (0 solapamiento de pares), mezclarlo por `meme_id` promediaría señales de grupos de sujetos distintos como si fueran el mismo sujeto multimodal, lo que es metodológicamente problemático.

**Salida:** `df_hr` y `df_et` restringidos a los memes comunes.

### 4.3. PSRI por sujeto (`aggregator.add_psri_subject`)

**Entrada:** `df_hr`, `df_et` (una fila por trial).

**Proceso (S_estab):** aplica `compute_psri_gaussian_log` (función en U: log-transform, estadísticos robustos mediana/MAD, campana gaussiana `exp(-0.5·z²)`) con `eps_hard` = **percentil 1 de los valores > 0** de cada modalidad:
- `PSRI_hr_subj` ← `garmin_hr_std` (estabilidad de la HR).
- `PSRI_et_subj` ← `3d_eye_states_pupil diameter left [mm]_std` (estabilidad de la pupila izquierda).

**Proceso (S_cond, caja de herramientas):** la forma se elige por la semántica del observable (conceptual_psri.md §1.3):
- `S_cond_rt` ← `reaction_time` con transformación **monotónica** "menos es mejor" (`S_cond_mono`): un RT más corto = procesamiento más fluido, no descuido. La función en U sobre este observable —que penaliza el RT rápido como "respuesta descuidada"— borra la señal direccional (ver §8).
- `S_cond_blink` ← `blinks_count` con la función en U (`compute_psri_gaussian_log`: tanto una tasa anómalamente baja como alta penalizan fiabilidad = fatiga).

**Salida:** `S_cond_subj = (S_cond_rt + S_cond_blink)/2`, más las columnas `PSRI_hr_subj` y `PSRI_et_subj` por trial.

### 4.4. Coherencia cruzada HR-ET (`S_coher`)

**Entrada:** `df_hr` y `df_et` unidos por **inner** en `(meme_id, username)` (misma población, HR=ET).

**Proceso:**
1. Por sujeto, std de HR y de pupila a través de todos sus trials (`hr_std_subj`, `pupil_std_subj`); los ceros se convierten en NaN para evitar división por cero.
2. Z-scores por fila con `compute_z_score(value, baseline_prev, subject_std)`:
   - `z_hr` ← `(garmin_hr_mean − garmin_hr_mean_baseline_prev) / hr_std_subj`.
   - `z_pupil` ← `(pupil_left_mean − pupil_left_mean_baseline_prev) / pupil_std_subj`.
3. `S_coher_trial ← compute_coherence(z_hr, z_pupil) = exp(−|z_hr − z_pupil|/2)` (dos sistemas fisiológicos genuinamente distintos: cardiovascular vs pupilar).
4. **Agregación por meme**: `S_coher = mean(S_coher_trial)` por `meme_id`; NaN → mediana global.

**Salida:** columna `S_coher` por meme.

### 4.5. Agregación por meme (`aggregator.aggregate_by_meme`)

**Entrada:** `df_hr`, `df_et` (trials).

**Proceso:** `groupby('meme_id')` calculando media y desviación de las métricas clave por modalidad, con prefijos `hr_` y `et_`:
- **HR:** `garmin_hr_mean/std/max/min`, baselines `prev`/`prev5`, `PSRI_hr_subj`.
- **ET:** pupila izquierda y derecha (mean/std), `fixations_duration_mean_ns`, `fixations_count`, `saccades_count`, `blinks_count`, `reaction_time`, `PSRI_et_subj`, `S_cond_subj`.

Fusión final `hr_agg.merge(et_agg, on='meme_id', how='inner')` (misma población HR=ET).

**Salida:** `df_merged` con una fila por meme.

### 4.6. Componentes y PSRI compuesto (`build_features.py`)

**Proceso:**
1. `S_estab = (PSRI_hr_subj_mean + PSRI_et_subj_mean)/2`, con imputación por mediana.
2. `S_cond = et_S_cond_subj_mean`, con imputación por mediana.
3. `S_coher` ya agregada, con imputación por mediana.
4. `PSRI = compute_weighted_psri(S_estab, S_coher, S_cond, w1=w2=w3=1/3)`; si hubiera NaN tras la imputación, fallback a la media de los tres componentes.
5. Renombrado por modalidad (`PSRI_hr_mean`, `PSRI_et_mean`) para el diagnóstico y el pool ampliado del sanity check.

**Salida:** columnas `S_estab`, `S_coher`, `S_cond`, `PSRI`, `PSRI_hr_mean`, `PSRI_et_mean` por meme.

### 4.7. Etiquetas y entropías (`aggregator.merge_labels`)

**Entrada:** `EXIST2026_training.json`.

**Proceso:** por meme, a partir de las listas de respuestas de anotadores:
- **Task 2.1** (¿sexista?): `entropy_21` = entropía binaria de las proporciones YES/NO; `hard_21 = 'YES'` si `p_yes ≥ 0.5`, sino `'NO'`; `soft_21_yes` = proporción de YES.
- **Task 2.2** (intención): `entropy_22` = entropía de Shannon sobre las frecuencias; `hard_22` = categoría modal.
- **Task 2.3** (categorías de sexismo, lista anidada): `entropy_23` = entropía de Shannon sobre la lista aplanada.

`inner` join con `df_merged` por `meme_id`. **Las entropías son las métricas de desacuerdo** (target continuo del estudio).

**Salida:** `df_final` → `physio_with_psri_memes.csv` (≈ 3984 memes).

### 4.8. Sanity check y corrección unificada (`sanity_check.py`)

**Entrada:** `physio_with_psri_memes.csv`.

**Proceso:**
1. `load_and_prepare()`: `meme_id` → str; renombra `PSRI_hr_mean → PSRI_hr`, `PSRI_et_mean → PSRI_et` (legacy de versiones previas).
2. `print_descriptive_stats()`: describe de los componentes, verificación de escala (media/mediana) y de que los rangos se mantienen en [0,1].
3. `plot_distributions()` → histogramas con media/mediana marcadas.
4. `analyze_component_correlations()`: matriz Pearson entre componentes + pairplot; interpreta la correlación HR-ET (≈ independiente si `|ρ| < 0.1`).
5. `run_full_analysis_with_correction()` — **pool unificado único** de comparaciones múltiples:
   - 2 tests Kruskal-Wallis (PSRI vs `hard_21` YES/NO y PSRI vs `hard_22` DIRECT/JUDGEMENTAL) con ε² como tamaño de efecto.
   - 18 correlaciones Pearson (6 métricas: PSRI, PSRI_hr, PSRI_et, S_estab, S_coher, S_cond × 3 entropías `entropy_21/22/23`).
   - `multipletests(..., method='bonferroni')` y `method='fdr_bh')` sobre los **20 p-valores juntos** (no bloques separados).

**Resultado:** ninguna de las 20 pruebas sobrevive a Bonferroni ni a FDR (la única p cruda < 0.05, PSRI vs hard_21 con p ≈ 0.0035, no sobrevive la corrección). Es el Resultado Negativo Válido del estudio.

**Salida:** `unified_correction_results.csv` con p cruda, Bonferroni, FDR y tamaño de efecto por test.

---

## 5. Salidas (outputs)

Todo se escribe en `results/output/EXIST/`:

| Archivo | Contenido |
|---|---|
| `physio_with_psri_memes.csv` | Una fila por meme (≈3984): features agregadas `hr_*`/`et_*`, componentes `S_estab`/`S_coher`/`S_cond`, `PSRI`, `PSRI_hr_mean`/`PSRI_et_mean`, etiquetas `hard_*` y entropías |
| `unified_correction_results.csv` | Pool unificado de 20 tests (2 Kruskal-Wallis + 18 Pearson) con p Bonferroni/FDR y tamaño de efecto |
| `fig_psri_distribution.png` | Distribuciones de PSRI y componentes con media/mediana |
| `fig_psri_pairplot.png` | Pairplot de los componentes (correlaciones y densidades) |
| `fig_psri_vs_hardlabel.png` | Boxplot PSRI vs `hard_21` (Task 2.1: YES vs NO) |
| `fig_psri_vs_intention.png` | Boxplot PSRI vs `hard_22` (Task 2.2: DIRECT vs JUDGEMENTAL) |

---

## 6. Mapa Entradas → Salidas (resumen)

| Entrada | Proceso | Salida |
|---|---|---|
| `HR_.xlsx`, `EEG_.xlsx`, `ET_.xlsx` | `loader.load_dataframes` + `diagnose_coverage` | DataFrames + dict de cobertura (conjuntos de memes/pares) |
| Memes HR∩ET | `build_features.build_psri_dataframe` (filtro `filter_common`) | `df_hr`, `df_et` comunes |
| `df_hr`, `df_et` (trials) | `aggregator.add_psri_subject` | `PSRI_hr_subj`, `PSRI_et_subj`, `S_cond_subj` |
| `df_hr` × `df_et` (pares) | `compute_z_score` + `compute_coherence` + media por meme | `S_coher` |
| Trials por sujeto | `aggregator.aggregate_by_meme` | `df_merged` (medias/desv por meme) |
| `df_merged` | `build_features` (S_estab, S_cond, `compute_weighted_psri`) | `PSRI`, `S_estab`, `S_coher`, `S_cond`, `PSRI_hr/et` |
| `EXIST2026_training.json` | `aggregator.merge_labels` | `hard_*`, `entropy_*`, `soft_21_yes` |
| DataFrame final | `build_features.process_dataframe` | `physio_with_psri_memes.csv` |
| `physio_with_psri_memes.csv` | `sanity_check.sanity_check` | `unified_correction_results.csv` + figuras |

---

## 7. Notas clave y gotchas

- **Fusión multimodal SOLO HR+ET:** EEG se carga y diagnostica, pero queda excluido del merge poblacional porque su población de sujetos no está alineada con HR/ET (0 solapamiento a nivel de par `(meme, username)`). Mezclar por `meme_id` promediaría señales de grupos disjuntos como si fueran el mismo sujeto multimodal.
- **Desalineación poblacional = test débil:** los 8 sujetos que llevan sensores y los anotadores son poblaciones disjuntas; la única conexión es una causa común indirecta. El nulo robusto de EXIST es coherente con esto y fue el motivo de diseñar K-EmoCon (Estudio 2) donde la misma persona es fuente fisiológica y objeto del juicio.
- **S_coher a nivel de trial, agregada por meme:** la coherencia HR-ET se calcula por par `(meme_id, username)` con baselines `prev`/`prev5`, y solo después se promedia por meme. No se calcula directamente sobre agregados de meme.
- **Imputación por mediana:** `S_estab`, `S_coher` y `S_cond` rellenan NaN con la mediana global; si quedara algún NaN, `PSRI` degenera a la media de los tres componentes (fallback explícito).
- **`eps_hard` por modalidad:** percentil 1 de los valores > 0 de cada variable (HR, pupila, reaction_time, blinks), no un umbral fijo.
- **Un único pool de corrección:** a diferencia de K-EmoCon (dos bloques independientes), EXIST corrige Kruskal-Wallis + Pearson en **un solo pool de 20 tests** (Bonferroni + FDR/BH) sobre la misma pregunta (¿predice el PSRI el desacuerdo?).
- **Resultado nulo robusto:** la única p cruda significativa (PSRI vs `hard_21`, p ≈ 0.0035) no sobrevive Bonferroni ni FDR — reportar como Resultado Negativo Válido, no como hallazgo.
- **Renombrado legacy en sanity_check:** si las columnas llegan como `PSRI_hr_mean`/`PSRI_et_mean`, se renombran a `PSRI_hr`/`PSRI_et` al cargar; el análisis trabaja con la versión renombrada.
- **Nombres de columna heredados de Excel:** p. ej. `3d_eye_states_pupil diameter left [mm]_mean` — largos, con espacios y corchetes; no "sanear" los nombres en `build_features` porque `aggregator` y `sanity_check` dependen de ellos.
- `config.json` contiene un token de HuggingFace — no se debe commitear.

---

## 8. Comprobación final de cierre de fase: reenfoques del framework reflejados en EXIST (diagnóstico exploratorio)

Para cerrar el círculo de la fase, se verificó si alguno de los reenfoques explorados a fondo en K-EmoCon tiene reflejo en los **dos objetos de estudio de EXIST**: (a) el **desacuerdo entre anotadores** (`entropy_21/22/23` y la proporción continua `soft_21_yes`) y (b) la **propia valoración de sexismo** (`hard_21` YES/NO, `hard_22` DIRECT/JUDGEMENTAL). Módulo standalone: `sources/exist/final_check.py` (solo lee `physio_with_psri_memes.csv` y los Excel de HR/ET; salidas en `results/output/EXIST/final_check/`).

**Reenfoques probados y su operacionalización:**

| Reenfoque (de K-EmoCon) | Reflejo en EXIST |
|---|---|
| Descomposición por componente | Kruskal-Wallis de cada eslabón (PSRI, PSRI_hr, PSRI_et, S_estab, S_coher, S_cond) contra `hard_21` y `hard_22` (el sanity check solo probaba PSRI) |
| Compuesto de 2 patas (`PSRI_validated`) | `PSRI_2leg = 0.5·S_estab + 0.5·S_coher`, contra entropías, `soft_21_yes` y etiquetas duras |
| Valoración continua | Correlaciones de todos los componentes contra `soft_21_yes` (target continuo más fino que el corte binario) |
| "La variabilidad de la fiabilidad lleva señal" | Dispersión entre-sujetos de la fiabilidad por meme (`PSRI_hr_std`, `PSRI_et_std`, `et_S_cond_subj_std`) |
| Barrido de baseline de S_coher (prev1 vs prev5) | Recomputed de `S_coher_prev5` a nivel de trial (columnas `*_baseline_prev5` de los Excel) y re-agregado por meme |
| Caja de herramientas de S_cond (forma por semántica del observable) | `S_cond_mono`: tiempo de reacción con la transformación monotónica "menos es mejor" que su semántica unívoca prescribe, a nivel de trial, re-agregado por meme |

Todas las pruebas (46 correlaciones Spearman + 20 Kruskal-Wallis = 66) se corrigieron como **un único bloque** (Bonferroni + FDR), consistente con la filosofía de pool único de EXIST.

**Resultado — la única señal robusta es el tiempo de reacción monotónico:**

- **`S_cond_mono` vs desacuerdo (negativo):** `entropy_22` ρ=−0.173 (p≈4e-28), `entropy_23` ρ=−0.172, `entropy_21` ρ=−0.086 — **todas sobreviven Bonferroni** en el pool de 66 tests.
- **`S_cond_mono` vs valoración de sexismo:** `soft_21_yes` ρ=−0.138 (p≈2e-18, sobrevive); Kruskal vs `hard_21` ε²=0.0255 (p<1e-4) y vs `hard_22` ε²=0.0081 (p≈1e-4) — **ambos sobreviven**.
- **Verificado sobre el tiempo de reacción crudo** (no solo sobre la transformación): media de RT por meme vs `entropy_22` ρ=+0.196 (p≈7e-36), robusto a winsorización al 1%, a exclusión de extremos (ρ=+0.184) y presente por separado en los subgrupos EN (ρ=+0.183) y ES (ρ=+0.199). Mediana de RT: memes NO=11651 ms vs YES=14005 ms; JUDGEMENTAL=15822 ms vs DIRECT=13566 ms.

**Lectura:** los memes que los sujetos de sensores procesan con **más fluidez** (tiempo de reacción más corto) se asocian con **menos desacuerdo anotador** y con valoración **menos sexista**; los memes más sexistas y ambiguos se procesan más despacio. La interpretación causal es la causa común indirecta típica de EXIST (poblaciones disjuntas): el contenido del meme produce a la vez procesamiento más lento en los 8 sujetos de sensores y más desacuerdo / más YES en los anotadores. La señal vive en el **canal conductual** (RT), no en los fisiológicos (HR, pupila), que siguen nulos.

**Por qué esto cierra el círculo con el framework:** S_cond no presupone una forma funcional única; la herramienta se elige por la semántica del observable (conceptual_psri.md §1.3). El RT tiene dirección unívoca ("menos es mejor"), de modo que su herramienta natural es la monotónica; la función en U sobre el mismo observable —la herramienta inadecuada para esa semántica— penaliza el RT rápido como "respuesta descuidada" y **borra esta señal direccional** (todas sus correlaciones son nulas). La caja de herramientas la recupera: es el ÚNICO lugar donde el framework tiene reflejo robusto en EXIST, y encaja con la lectura S_obs de observabilidad (conceptual_psri.md §1.3.1): un estímulo más fluido de procesar es más observable y genera más consenso. El resultado nace de la comprobación de cierre de fase, es a nivel de meme y con el diseño débil de causa común; se documenta como diagnóstico exploratorio, no como resultado confirmatorio del paper. El nulo confirmatorio de los 20 tests (incluidos todos los componentes fisiológicos y la forma en U de S_cond) se mantiene intacto.

**Resto de reenfoques:** sin reflejo. El compuesto de 2 patas, la dispersión entre-sujetos y `S_coher_prev5` no sobreviven la corrección (todas p≥0.1 tras FDR). **Latencia de sensores:** no comprobable en EXIST — no hay series temporales crudas (cada trial es un agregado) y el par S_coher es cross-device (Garmin × Tobii), exactamente el caso donde el framework exige verificar la alineación (conceptual_psri.md §1.2.1); se documenta como limitación estructural.