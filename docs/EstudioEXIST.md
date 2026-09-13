# Estudio 1 EXIST

> Documento regenerado a partir de `EstudioEXIST.pdf`/`.odt` y alineado con la implementación actual del pipeline (`sources/exist/`) y sus salidas en `results/output/EXIST/` (agosto 2026). Se incorporan dos elementos respecto al documento fuente: (1) la **comprobación final de cierre de fase** (`sources/exist/final_check.py`), que verifica si los reenfoques del framework explorados en K-EmoCon tienen reflejo en EXIST tanto sobre el desacuerdo como sobre la valoración de sexismo — con un hallazgo robusto aislado en el tiempo de reacción monotónico; y (2) la cifra correcta del máximo de `PSRI` (0.9813, no 1.0) y las tablas completas de componentes.

## 1. Introducción

El laboratorio EXIST 2026 (Task 2) proporciona, junto con memes y sus etiquetas de sexismo (yes/no, intención y categorización), los registros fisiológicos de los sujetos que los vieron: frecuencia cardíaca (HR, pulsera Garmin), electroencefalografía (EEG) y seguimiento ocular (ET, pupilómetro Tobii). Los trabajos previos del propio shared task (AI Wizards, Cloud-17, SPECTRA, BioSentinel, Aegis Athena) muestran que integrar estas señales como características de entrada para la clasificación produce resultados inconsistentes y, a menudo, **empeora** el rendimiento respecto a usar solo texto e imagen.

La hipótesis de este estudio es distinta: el valor de las señales fisiológicas no reside en predecir la etiqueta de sexismo directamente, sino en **caracterizar la fiabilidad de la señal** (estabilidad, coherencia entre sistemas y compromiso conductual). La pregunta de investigación es si un meme cuya señal fisiológica es de *baja calidad* (inestable, descoordinada, con conducta anómala respecto a la tarea) se asocia con **más desacuerdo entre anotadores** sobre su etiqueta — la hipótesis LeWiDi (less widely disputed) extendida a la fisiología: un estímulo que produce señal ruidosa debería ser también un estímulo sobre el que los anotadores humanos discrepan más.

Este es el **Estudio 1** del framework PSRI y tiene una relación distinta con el resto de estudios: el **Estudio 3 (PhysioNet)** valida el instrumento en sí (¿la función en U discrimina calidad de señal etiquetada por expertos? → sí, AUC 0.887); los Estudios 1 y 2 usan ese instrumento para preguntar lo que el instrumento mide *importa* (¿la fiabilidad fisiológica predice el desacuerdo humano?). EXIST es, además, el estudio donde se define la **arquitectura de tres eslabones** (S_estab, S_coher, S_cond) que K-EmoCon hereda y adapta.

## 2. El dataset EXIST 2026

EXIST 2026 (Task 2) consiste en detectar si un meme es sexista y con qué intención. Los datos incluyen registros por par `(meme_id, username)` para HR, EEG y ET, además de un JSON con las etiquetas de las tareas 2.1 (¿sexista?: YES/NO), 2.2 (intención: DIRECT/JUDGEMENTAL) y 2.3 (categorías de sexismo, multiclase) por `meme_id`.

### 2.1. Cobertura y la desalineación poblacional

Se realizó un análisis de cobertura para entender el solapamiento entre las fuentes y, sobre todo, entre las poblaciones de sujetos:

| Aspecto | HR | EEG | ET |
|---|---|---|---|
| Usuarios únicos | 8 | 8 | 8 |
| Usuarios comunes a las tres | – | – | 5 |
| Memes cubiertos | 4044 | 4042 | 4044 |
| Pares `(meme, usuario)` | 7782 | — | 7782 |

- **HR y ET comparten los mismos 8 usuarios** (EN1-4, ES1-4) con un solapamiento del 100% a nivel de par `(meme, usuario)` (7782 pares idénticos).
- **EEG proviene de un conjunto distinto de sujetos**: 5 de sus 8 usuarios aparecen también en HR/ET (EN1, EN3, ES1, ES3, ES4), pero **0 pares `(meme, usuario)` se solapan** entre EEG y HR/ET — es decir, ni un solo trial tiene a la vez señal EEG y HR/ET del mismo sujeto viendo el mismo meme. El solapamiento de *memes* (sin usuario) sí existe (4042 memes), pero a nivel de trial es nulo.
- **Los anotadores son un tercer grupo independiente**, con etiquetas para 3984 memes.

**Implicación metodológica y decisión de diseño.** Los datos fisiológicos provienen de dos conjuntos de sujetos con solapamiento nulo a nivel de trial (Set A = HR+ET, Set B = EEG). Promediar EEG con HR/ET por `meme_id` mezclaría señales de grupos disjuntos como si fueran el mismo sujeto multimodal — metodológicamente problemático. La fusión multimodal del PSRI se limita a **HR y ET** (misma población de 8 sujetos); el **EEG queda excluido del merge** por su desalineación poblacional, aunque se carga y diagnostica en el pipeline.

**La desalineación poblacional es el rasgo estructural que define este estudio.** La población que genera la señal fisiológica (8 sujetos con sensores) y la población que anota el sexismo del meme son grupos **disjuntos**: el sujeto A ve el meme y su corazón reacciona; el anotador B, que no lleva sensores, juzga de forma independiente si ese meme es sexista. La única conexión posible entre la fisiología de A y el juicio de B es una **hipótesis de causa común indirecta**: un meme ambiguo produce a la vez una reacción fisiológica más ruidosa en quien lo ve y más desacuerdo en quien lo juzga. Es una cadena causal de dos eslabones con ruido acumulándose en ambos extremos — un test estructuralmente débil, como se confirmará en los resultados, y el motivo por el que se diseñó el Estudio 2 (K-EmoCon) con la misma persona como fuente fisiológica y objeto del juicio. En la Sección 3.8 esta desconexión se **mide** con los propios datos del PSRI (fiabilidad del agregado por meme ≈ 0), sin depender de las etiquetas.

### 2.2. Variables fisiológicas seleccionadas

Para cada modalidad se seleccionaron las métricas representativas de la respuesta cognitiva:

| Modalidad | Media | Desviación intra-trial | Línea base | Otras |
|---|---|---|---|---|
| HR | `garmin_hr_mean` | `garmin_hr_std` | `garmin_hr_mean_baseline_prev` | — |
| ET | pupila izquierda `[mm]_mean` | pupila izquierda `[mm]_std` | `[mm]_mean_baseline_prev` | `reaction_time`, `blinks_count` |
| EEG | `eeg_alpha_mean` (y otras bandas) | No disponible | No disponible | — (excluido de la fusión) |

No hay series temporales crudas: cada trial es un único agregado estadístico por `(meme, username)`.

## 3. Construir el pipeline de EXIST

El pipeline replica la arquitectura loader → aggregator → build_features → sanity_check reutilizando `psri/calculator.py` sin cambios. La orquestación arranca en `main.py` (`python main.py --proc EXIST`):

```
main.py
  └── process_exist()
        ├─ [1] build_features.process_dataframe()
        │        └── build_psri_dataframe(input_dir, labels_path, filter_common=True)
        │              ├── loader.load_dataframes + diagnose_coverage
        │              ├── Filtro de memes comunes HR∩ET (EEG fuera)
        │              ├── aggregator.add_psri_subject → PSRI_hr_subj, PSRI_et_subj, S_cond_subj
        │              ├── S_coher por trial (z-scores con baseline_prev) → media por meme
        │              ├── aggregator.aggregate_by_meme → df_merged (medias/desv por meme)
        │              ├── S_estab, S_coher, S_cond → compute_weighted_psri → PSRI
        │              └── aggregator.merge_labels → hard_*, entropy_*, soft_21_yes
        │                    → physio_with_psri_memes.csv (3984 memes)
        └─ [2] sanity_check.process_sanity_check()
                 ├── load_and_prepare + descriptivos + distribuciones + pairplot
                 └── run_full_analysis_with_correction → unified_correction_results.csv
                       (pool único: 2 Kruskal-Wallis + 18 Pearson, Bonferroni + FDR)
```

### 3.1. Filtro de población

Con `filter_common=True` se conservan solo los memes con **HR y ET** (`common_memes = hr_memes ∩ et_memes` = 4044). El EEG se excluye del merge poblacional por la desalineación a nivel de trial (Sección 2.1). Tras la unión `inner` con las etiquetas, el dataframe de trabajo queda en **3984 memes**.

### 3.2. Los tres componentes del PSRI

**S_estab — estabilidad intra-trial (`aggregator.add_psri_subject`).** Se aplica `compute_psri_gaussian_log` (función en U: log-transform, mediana/MAD robusta de población global, campana gaussiana `exp(-0.5·z²)`, con `eps_hard` = percentil 1 de los valores > 0) por trial:
- `PSRI_hr_subj` ← `garmin_hr_std` (variabilidad intra-trial de la HR).
- `PSRI_et_subj` ← `3d_eye_states_pupil diameter left [mm]_std` (variabilidad intra-trial de la pupila izquierda).

A nivel de meme: `S_estab = (mean(PSRI_hr_subj) + mean(PSRI_et_subj)) / 2`, con imputación por mediana.

**S_coher — coherencia cruzada HR-ET (`compute_z_score` + `compute_coherence`).** A nivel de trial, para cada par `(meme_id, username)`:
1. Por sujeto, se calcula la std de las medias de HR y de pupila a través de todos sus trials (`hr_std_subj`, `pupil_std_subj`); los ceros se convierten en NaN para evitar división por cero. Este denominador **entre-trial** del sujeto normaliza la reacción contra la variabilidad típica de ese sujeto (no se penaliza a quien tiene respuesta autonómica naturalmente alta) — se eligió deliberadamente en lugar de la std intra-trial (`garmin_hr_std`).
2. Z-scores con `compute_z_score(valor, baseline_prev, subject_std)`: `z_hr = (garmin_hr_mean − garmin_hr_mean_baseline_prev) / hr_std_subj`, y análogo para la pupila. La línea base es la media del trial inmediatamente anterior del mismo sujeto (no la media de la sesión).
3. `S_coher_trial = compute_coherence(z_hr, z_pupil) = exp(−|z_hr − z_pupil|/2)` — dos sistemas fisiológicos genuinamente distintos (cardiovascular vs pupilar), con decaimiento exponencial que penaliza suavemente la descoordinación.
4. `S_coher = mean(S_coher_trial)` por `meme_id`; NaN → mediana global.

**S_cond — consistencia conductual.** Los dos observables de ET ejercitan la caja de herramientas de S_cond (conceptual_psri.md §1.3):
- `S_cond_rt` ← `reaction_time` (dirección unívoca "menos es mejor": un RT más corto = procesamiento más fluido; la herramienta natural es la monotónica, `S_cond_mono`, §3.7).
- `S_cond_blink` ← `blinks_count` (tasa anómala en ambos extremos = fatiga; herramienta en U).

`S_cond_subj = (S_cond_rt + S_cond_blink) / 2` por trial; `S_cond = mean(S_cond_subj)` por meme. La comprobación de cierre de fase (§3.7) aplicó al RT la herramienta monotónica que su semántica prescribe y confirmó la regla de selección. Este es el **observable conductual directo** del estudio — el caso "ideal" del framework, ausente en K-EmoCon.

### 3.3. Agregación por meme y PSRI compuesto

`aggregator.aggregate_by_meme` agrupa por `meme_id` calculando media y desviación de las métricas clave con prefijos `hr_`/`et_`, y fusiona HR y ET por `inner` (misma población). `build_features` construye los componentes y el compuesto:

- `S_estab`, `S_coher`, `S_cond` con imputación por mediana de la propia dimensión.
- `PSRI = compute_weighted_psri(S_estab, S_coher, S_cond, w1=w2=w3=1/3)` — pesos iguales, réplica de la decisión de diseño de los otros estudios.
- Renombrado `PSRI_hr_mean`→`PSRI_hr`, `PSRI_et_mean`→`PSRI_et` para el diagnóstico por modalidad.

### 3.4. Etiquetas y entropías (`aggregator.merge_labels`)

Desde el JSON de etiquetas, por meme:
- **Task 2.1**: `entropy_21` = entropía binaria de las proporciones YES/NO; `hard_21 = 'YES'` si `p_yes ≥ 0.5`; `soft_21_yes` = proporción de YES (target continuo).
- **Task 2.2**: `entropy_22` = entropía de Shannon sobre las frecuencias de categorías; `hard_22` = categoría modal.
- **Task 2.3**: `entropy_23` = entropía de Shannon sobre la lista aplanada de categorías de sexismo.

Las **entropías son las métricas de desacuerdo** (target continuo del estudio); las **hard labels son la valoración de sexismo** (binaria / de intención). `inner` join con `df_merged` por `meme_id`.

### 3.5. Sanity check y corrección unificada

`sanity_check.py` genera descriptivos, distribuciones, pairplot de componentes y un **pool único de 20 tests**:
- **2 Kruskal-Wallis**: PSRI vs `hard_21` (YES/NO) y PSRI vs `hard_22` (DIRECT/JUDGEMENTAL), con ε² como tamaño de efecto — réplica del análisis de AI Wizards.
- **18 correlaciones de Pearson**: 6 métricas (PSRI, PSRI_hr, PSRI_et, S_estab, S_coher, S_cond) × 3 entropías (`entropy_21/22/23`).
- `multipletests(..., method='bonferroni')` y `method='fdr_bh')` sobre los **20 p-valores juntos** (no bloques separados).

### 3.6. Resultados del sanity check

#### 3.6.1. Estadísticas descriptivas (n=3984 memes)

| Variable | Media | Desv. Est. | Mínimo | 25% | 50% | 75% | Máximo |
|---|---|---|---|---|---|---|---|
| PSRI | 0.7618 | 0.0828 | 0.4227 | 0.7103 | 0.7684 | 0.8213 | 0.9813 |
| PSRI_hr | 0.7339 | 0.1865 | 0.0000 | 0.6172 | 0.7601 | 0.8839 | 1.0000 |
| PSRI_et | 0.7163 | 0.2015 | 0.0000 | 0.5740 | 0.7422 | 0.8838 | 1.0000 |
| S_estab | 0.7251 | 0.1394 | 0.1524 | 0.6390 | 0.7388 | 0.8247 | 1.0000 |
| S_coher | 0.7471 | 0.1391 | 0.1334 | 0.6623 | 0.7605 | 0.8512 | 0.9998 |
| S_cond | 0.8133 | 0.1269 | 0.2147 | 0.7437 | 0.8386 | 0.9070 | 0.9999 |

Observaciones: las medias de PSRI_hr y PSRI_et son prácticamente idénticas (~0.72–0.73), confirmando que la normalización MAD y la campana gaussiana han eliminado el efecto de escala entre modalidades; los mínimos de PSRI_hr/PSRI_et son 0.0, lo que indica que la rama de "señal plana" de la función en U se ha activado correctamente (corte duro `eps_hard`); S_coher tiene varianza real entre memes (rango [0.133, 1.0]).

#### 3.6.2. Correlación entre componentes (Pearson)

| | PSRI | S_estab | PSRI_hr | PSRI_et | S_coher | S_cond |
|---|---|---|---|---|---|---|
| PSRI | 1.0000 | 0.6598 | 0.4966 | 0.4535 | 0.6213 | 0.5524 |
| S_estab | 0.6598 | 1.0000 | 0.6916 | 0.7437 | 0.1094 | 0.0738 |
| PSRI_hr | 0.4966 | 0.6916 | 1.0000 | 0.0315 | 0.1020 | 0.1009 |
| PSRI_et | 0.4535 | 0.7437 | 0.0315 | 1.0000 | 0.0570 | 0.0087 |
| S_coher | 0.6213 | 0.1094 | 0.1020 | 0.0570 | 1.0000 | 0.0004 |
| S_cond | 0.5524 | 0.0738 | 0.1009 | 0.0087 | 0.0004 | 1.0000 |

Lectura (coherente con la Tabla 5 citada en `conceptual_psri.md`): **las tres dimensiones son prácticamente independientes entre sí** (S_estab–S_coher r=0.109, S_estab–S_cond r=0.074, S_coher–S_cond r≈0.0004) — justifica los pesos iguales del compuesto y la lectura de "tres eslabones independientes". Además, **PSRI_hr y PSRI_et son casi ortogonales (r=0.032)** a pesar de provenir de los mismos sujetos: HR y pupila capturan aspectos distintos de la respuesta fisiológica (HR → procesos autonómicos/arousal; pupila → atención/carga cognitiva).

#### 3.6.3. Relación con la valoración de sexismo (Kruskal-Wallis)

- **Task 2.1 (YES vs NO)**: estadístico = 8.501, p_crudo = 0.0035, ε² = 0.00188. Mediana PSRI: YES = 0.7710, NO = 0.7647.
- **Task 2.2 (DIRECT vs JUDGEMENTAL)**: estadístico = 2.572, p_crudo = 0.1088, ε² = 0.00088. Mediana PSRI: DIRECT = 0.7718, JUDGEMENTAL = 0.7651.

Tras la corrección unificada (umbral Bonferroni efectivo 0.0025 para 20 tests), el de Task 2.1 **no es significativo** (0.0035 > 0.0025): el PSRI no separa las clases de sexismo de forma estadísticamente significativa tras el ajuste.

#### 3.6.4. Correlación con el desacuerdo (entropías, Pearson)

| Tarea | PSRI | PSRI_hr | PSRI_et | S_estab | S_coher | S_cond |
|---|---|---|---|---|---|---|
| Task 2.1 | +0.013 (0.403) | +0.001 (0.929) | +0.016 (0.301) | +0.013 (0.419) | +0.005 (0.776) | +0.007 (0.661) |
| Task 2.2 | +0.026 (0.100) | +0.015 (0.335) | +0.016 (0.324) | +0.022 (0.174) | +0.010 (0.535) | +0.017 (0.296) |
| Task 2.3 | +0.026 (0.102) | +0.035 (0.029*) | +0.015 (0.355) | +0.034 (0.033*) | +0.010 (0.523) | +0.003 (0.868) |

Los valores son r de Pearson (p-valor entre paréntesis). * p < 0.05 crudo.

**Resultado nulo robusto.** Ninguna de las 20 pruebas del pool unificado sobrevive a la corrección por comparaciones múltiples (ni Bonferroni ni FDR). Las únicas correlaciones con significancia cruda (PSRI_hr y S_estab en Task 2.3) colapsan al corregir. Los tamaños de efecto son triviales en todas las dimensiones: el máximo r² encontrado es 0.00120 — la fiabilidad de la señal explica **menos del 0.12% de la varianza del desacuerdo entre anotadores**. La hipótesis central (la fiabilidad fisiológica predice el desacuerdo) queda **refutada** en EXIST con un resultado negativo metodológicamente sólido (n≈3984 da potencia suficiente para detectar incluso correlaciones débiles r≈0.05; la ausencia de correlación no se debe a falta de potencia).

#### 3.6.5. Interpretación del nulo

Tres dimensiones independientes de fiabilidad (estabilidad intra-trial, coherencia HR-ET, consistencia conductual), probadas contra tres métricas de desacuerdo y dos valoraciones duras de sexismo, no muestran relación tras corrección. Esto es coherente con el **diseño débil de causa común**: los 8 sujetos de sensores y los anotadores son poblaciones disjuntas, la cadena causal es indirecta y el ruido se acumula en ambos extremos. El resultado no indica que el instrumento falle (el Estudio 3 lo valida con AUC 0.887) ni que la hipótesis sea falsa en general: indica que **este diseño no podía detectarla fiablemente** — el motivo exacto por el que se diseñó K-EmoCon (Estudio 2) con población alineada.

### 3.7. Comprobación final de cierre de fase: reenfoques del framework reflejados en EXIST

Para cerrar el círculo de la fase se verificó si alguno de los reenfoques explorados a fondo en K-EmoCon tiene reflejo en los **dos objetos de estudio de EXIST** — el desacuerdo (`entropy_21/22/23`, `soft_21_yes`) y la valoración de sexismo (`hard_21`, `hard_22`) — mediante `sources/exist/final_check.py` (66 tests corregidos como un único bloque, Bonferroni + FDR; salidas en `results/output/EXIST/final_check/`). Detalle completo en `tecnico_exist.md` §8.

**Reenfoques probados y su reflejo:** descomposición por componente vs etiquetas duras (el sanity check solo probaba PSRI); compuesto de 2 patas (`PSRI_2leg = 0.5·S_estab + 0.5·S_coher`); valoración continua (`soft_21_yes`); dispersión entre-sujetos de la fiabilidad por meme (`PSRI_hr_std`, `PSRI_et_std`, `et_S_cond_subj_std`); `S_coher_prev5` recomputado a nivel de trial con las columnas `*_baseline_prev5`; y `S_cond_mono` (tiempo de reacción con la transformación **monotónica** "menos es mejor" que la regla de selección de S_cond prescribe para un observable con dirección unívoca, a nivel de trial, re-agregado por meme).

**Resultado — la única señal robusta es el tiempo de reacción monotónico:**

- **`S_cond_mono` vs desacuerdo (negativo):** `entropy_22` ρ=−0.173 (p≈4e-28), `entropy_23` ρ=−0.172, `entropy_21` ρ=−0.086 — **todas sobreviven Bonferroni** en el pool de 66 tests.
- **`S_cond_mono` vs valoración de sexismo:** `soft_21_yes` ρ=−0.138 (p≈2e-18, sobrevive); Kruskal vs `hard_21` ε²=0.0255 (p<1e-4) y vs `hard_22` ε²=0.0081 (p≈1e-4) — **ambos sobreviven**.
- **Verificado sobre el RT crudo** (no solo sobre la transformación): media de RT por meme vs `entropy_22` ρ=+0.196 (p≈7e-36), robusto a winsorización al 1% (ρ=+0.196), a exclusión de extremos (ρ=+0.184, n=3904) y presente por separado en los subgrupos de sujetos EN (ρ=+0.183) y ES (ρ=+0.199). Mediana de RT: memes NO = 11651 ms vs YES = 14005 ms; JUDGEMENTAL = 15822 ms vs DIRECT = 13566 ms.

**Lectura.** Los memes que los sujetos de sensores procesan con **más fluidez** (tiempo de reacción más corto) se asocian con **menos desacuerdo anotador** y con valoración **menos sexista**; los memes más sexistas y ambiguos se procesan más despacio. La interpretación causal es la causa común indirecta típica de EXIST (poblaciones disjuntas): el contenido del meme produce a la vez procesamiento más lento en los 8 sujetos de sensores y más desacuerdo / más YES en los anotadores. La señal vive en el **canal conductual** (RT), no en los fisiológicos (HR, pupila), que siguen nulos.

**Por qué esto cierra el círculo con el framework.** S_cond no presupone una forma funcional única: la herramienta se elige por la semántica del observable (conceptual_psri.md §1.3). El RT tiene dirección unívoca ("menos es mejor" = procesamiento más fluido), de modo que su herramienta natural es la monotónica; la función en U sobre el mismo observable —la herramienta inadecuada para esa semántica— penaliza el RT rápido como "respuesta descuidada" y borra la señal (todas sus correlaciones nulas). La caja de herramientas es la razón de que la señal direccional del único eslabón con señal en EXIST sea recuperable, y es el **único lugar donde el framework tiene reflejo robusto en EXIST**; encaja con la lectura S_obs de observabilidad (`conceptual_psri.md` §1.3.1): un estímulo más fluido de procesar es más observable y genera más consenso.

**Resto de reenfoques: sin reflejo.** El compuesto de 2 patas, la dispersión entre-sujetos y `S_coher_prev5` no sobreviven la corrección (todas p≥0.1 tras FDR). **Latencia de sensores: no comprobable** en EXIST — no hay series temporales crudas y el par S_coher es cross-device (Garmin × Tobii), exactamente el caso donde el framework exige verificar la alineación (`conceptual_psri.md` §1.2.1); se documenta como limitación estructural.

### 3.8. La desalineación medida con los datos del propio PSRI: fiabilidad del agregado por meme (diagnóstico exploratorio)

La explicación estructural del nulo (Sección 2.1) afirma que las poblaciones son disjuntas y que el único puente entre fisiología y desacuerdo es un agregado por meme sobre pocos sujetos. Esa afirmación se puede **medir con los propios datos del PSRI, sin etiquetas**: si el agregado por meme no tiene señal reproducible, la correlación con el desacuerdo es imposible por diseño, exista o no la cadena causal. Módulo: `sources/exist/meme_reliability.py` (solo lee los Excel de trial y el JSON de etiquetas para la sonda final; salidas en `results/output/EXIST/meme_reliability/`).

**Estructura verificada:** cada meme es visto por **como máximo 2 sujetos** (4044 memes: 3738 con 2 viewers, 306 con 1; k media 1.92; 12 pares de sujetos distintos). El agregado por meme es la media sobre 1–2 sujetos.

**Cheques y resultados (todos diagnóstico, sin corrección por comparaciones múltiples):**

1. **Descomposición de varianza (ICC).** Fracción de varianza atribuible a cada nivel (ICC_meme_adj = meme neta de rasgo de sujeto; fiabilidad de la media = Spearman-Brown con k medio; atenuación = √fiab):

| Métrica | ICC_subject | ICC_meme_adj | Fiab. media (SB) | Atenuación √fiab |
|---|---|---|---|---|
| `PSRI_trial` (compuesto) | 0.096 | 0.033 | 0.058 | 0.24 |
| `PSRI_hr_subj` / `PSRI_et_subj` | 0.005 / 0.110 | ≈0 / ≈0 | ≈0 | ≈0 |
| `garmin_hr_mean` (crudo) | **0.797** | 0.0001 | 0.0002 | 0.01 |
| pupila `_mean` (cruda) | **0.646** | 0.043 | 0.075 | 0.27 |
| `reaction_time` (conductual) | 0.073 | **0.268** | 0.414 | 0.64 |
| `blinks_count` (conductual) | 0.056 | 0.180 | 0.221 | 0.47 |

Los **niveles fisiológicos crudos son ~65–80% rasgo de sujeto y 0% señal de meme**: un meme no induce respuesta reproducible entre viewers. Los observables **conductuales** (RT, parpadeos) son los únicos con señal de meme.

2. **Fiabilidad entre viewers (neta de rasgo de sujeto).** Correlación entre los 2 viewers de cada meme (valores centrados por sujeto), mediana de los 12 pares: `PSRI_trial` r=+0.039, HR media r=−0.025, pupila r=+0.053 — **≈0**. `reaction_time` r=+0.454 (fiabilidad de la media SB=0.62). El único canal con agregado fiable es el conductual.

3. **Cota de potencia.** N_eff ≈ 3900–4200 memes → el diseño detectaría |r| ≥ 0.045 *si el agregado fuera fiable*. Pero con fiabilidad de la media ≈0.06 (`PSRI_trial`), el efecto **verdadero** mínimo detectable es ≈0.19 — y la fiabilidad medida es ≈0. El nulo queda acotado: no es "no hay nada", es "este diseño no puede resolver señal fisiológica a nivel de meme".

4. **Sonda del eslabón (necesita etiquetas).** Dispersión entre viewers por meme vs desacuerdo: todo ≈0 en los canales fisiológicos (máximo |ρ|=0.04, sin consistencia entre targets); solo `reaction_time` y `blinks_count` muestran pequeñas correlaciones positivas (ρ≈0.05–0.10), coherentes con el hallazgo de `S_cond_mono` (§3.7).

**Lectura.** La desalineación deja de ser solo un argumento conceptual: queda **medida**. Los canales fisiológicos no tienen señal reproducible a nivel de meme (ICC_meme_adj≈0, fiabilidad entre viewers≈0), el agregado es ruido promediado sobre 1–2 sujetos, y por eso ninguna correlación con el desacuerdo es posible — exactamente el fallo de arquitectura que motiva K-EmoCon. El contraste interno demuestra que el diagnóstico discrimina: el único canal con señal de meme reproducible es el **conductual** (RT), y es precisamente el único que en el pool corregido de §3.7 produce hallazgo (`S_cond_mono`). El PSRI es un método, no un índice suelto: aplicar el método obliga a auditar los pre-requisitos de su claim (que la señal fisiológica a nivel de unidad de análisis sea reproducible); del mismo modo, la verificación de latencia de sensores que exige S_coher en K-EmoCon es otro pre-requisito del método.

**Frontera de la afirmación.** Este diagnóstico es sobre el **canal fisiológico agregado** y el diseño, no sobre la conducta de anotación ni sobre el compromiso del sujeto. No dice que el sujeto no vea el meme ni que no reaccione —puede hacerlo—, ni que no anote (los anotadores son un grupo independiente, §2.1). Dice que no existe una **firma fisiológica reproducible a nivel de meme** que un agregado de 1–2 viewers pueda recuperar, de modo que el regresor fisiológico no es utilizable para la inferencia a nivel de meme.

### 3.9. Transferencia de utilidad downstream (diagnóstico exploratorio)

La auditoría de agregación (§3.8) predice que el canal fisiológico, sin señal reproducible a nivel de meme, no debe aportar utilidad predictiva, mientras que el canal conductual (RT, parpadeos), que sí la tiene, debería aportarla. Módulo: `sources/exist/downstream_transfer.py` (salidas en `results/output/EXIST/downstream_transfer/`). Unidad = meme; features de **contenido** provistas por el corpus (texto OCR con TF-IDF + 64 dims de imagen), conductuales y fisiológicas; Ridge/logística con CV de 5 particiones (aleatoria y agrupada por viewers) y métrica out-of-fold; control con target permutado.

| Bloque | entropy_22 (ρ) | entropy_23 (ρ) | soft_21_yes (ρ) | hard_21 (AUC) |
|---|---|---|---|---|
| content | 0.210 | 0.287 | 0.453 | 0.736 |
| content + conductual | **0.222** | **0.300** | **0.458** | **0.742** |
| content + fisiológico | 0.211 | 0.286 | 0.452 | 0.736 |
| content + ambos | 0.223 | 0.298 | 0.456 | 0.740 |
| control permutado | ≈0 | ≈0 | ≈0 | ≈0.5 |

(valores pooled OOF, CV aleatoria; la agrupada por viewers coincide.)

**Lectura.** El contenido es un baseline fuerte (ρ≈0.45 en la proporción de YES; AUC≈0.74). Añadir el canal conductual mejora de forma pequeña pero **consistente** en los cuatro objetivos; añadir el fisiológico **no cambia nada**, y el combinado iguala al conductual. La auditoría de agregación **anticipa** qué canal aporta utilidad downstream: solo el que tiene fiabilidad de meme reproducible. Esto convierte el diagnóstico de descriptivo en predictivo y responde directamente a la crítica de "downstream gains not demonstrated". **Frontera:** la ganancia conductual es modesta y los modelos son lineales; no se usan `img_score`/`img_cluster` del corpus para evitar cualquier duda de fuga.

## 4. Limitaciones

- **Datos agregados (sin series temporales).** Solo se dispone de estadísticos resumen (media, std, min, max) por trial, no de la serie completa. Esto impide analizar la dinámica temporal de la respuesta y calcular correlaciones temporales reales entre HR y pupila — y, en particular, impide verificar la **latencia cross-device** del par HR (Garmin) × pupila (Tobii), el caso donde el framework exige comprobarla.
- **Conjuntos de sujetos disjuntos (la limitación estructural).** HR y ET provienen de un grupo de 8 sujetos; EEG de otro grupo con 0 solapamiento a nivel de trial; los anotadores, de un tercer grupo. La única conexión entre fisiología y desacuerdo es una causa común indirecta. El nulo del sanity check es coherente con esto y fue el motivo de diseñar K-EmoCon (Estudio 2) con población alineada. La desconexión se mide en §3.8 con los datos del propio PSRI (cada meme lo ven ≤2 sujetos; el agregado por meme tiene fiabilidad ≈0).
- **EEG sin métrica intra-trial equivalente y excluido de la fusión.** No hay std intra-trial de EEG comparable a `garmin_hr_std`/pupila `_std`, y aunque hubiera un PSRI_EEG, fusionarlo por `meme_id` mezclaría poblaciones disjuntas.
- **Normalización poblacional frente a personalización.** El PSRI usa referencia global (mediana/MAD de toda la población) en lugar de normalización por sujeto. Protege contra fallos sistémicos de sensores (un sensor permanentemente degradado pasaría desapercibido en una MAD individual), pero penaliza respuestas legítimamente atípicas pero estables en un individuo (p. ej. un sujeto con arritmia sinusal con HR_std consistentemente alta). Una normalización híbrida requeriría series temporales crudas para modelar artefactos intra-trial, no disponibles en el formato agregado de EXIST.
- **La señal conductual de RT solo aparece con la herramienta que la semántica del observable prescribe.** `S_cond_mono` es el RT con la transformación monotónica "menos es mejor" que la regla de selección de S_cond (conceptual_psri.md §1.3) prescribe para un observable con dirección unívoca; la forma en U borraba esa señal. El hallazgo nace de la comprobación de cierre de fase, es a nivel de meme (no dentro-de-sujeto) y se apoya en el diseño débil de causa común — se documenta como diagnóstico exploratorio, no como resultado confirmatorio del paper. El nulo confirmatorio de los 20 tests se mantiene intacto.
- **El instrumento es más fuerte sobre onda cruda que sobre magnitud derivada.** La validación de PhysioNet (Estudio 3) muestra que el PSRI discrimina bien sobre `std_signal` (AUC 0.887) pero mucho más débilmente sobre una magnitud derivada (RR/HR, AUC 0.671 al aislar el sesgo de detección) — lo que explica la debilidad relativa de `PSRI_hr` en EXIST, que opera sobre HR ya derivada.

## 5. Síntesis (puntos de partida)

1. Las tres dimensiones de fiabilidad **no predicen el desacuerdo** en EXIST: se probaron la estabilidad intra-sujeto (función en U), la coherencia entre modalidades (transformación exponencial) y la consistencia conductual (U sobre RT y parpadeos). Ninguna mostró relación con la entropía de las anotaciones, incluso tras la corrección por comparaciones múltiples (pool único de 20 tests).
2. El nulo también se extiende a la **valoración de sexismo** (hard labels): el Kruskal-Wallis de PSRI vs YES/NO (p_crudo=0.0035) no sobrevive la corrección, y PSRI vs DIRECT/JUDGEMENTAL es nulo (p=0.109).
3. **HR y ET son prácticamente independientes** (r=0.032) pese a venir de los mismos sujetos: HR refleja procesos autonómicos (arousal, estrés) y la pupila, atención y carga cognitiva — capturan aspectos distintos de la respuesta.
4. El resultado negativo es **metodológicamente sólido**: con n≈3984 hay potencia para detectar correlaciones débiles (r≈0.05); la ausencia de correlación no se debe a falta de potencia, sino al diseño de causa común con poblaciones disjuntas.
5. La comprobación final de cierre de fase confirma el nulo para todos los componentes fisiológicos, el compuesto de 2 patas, la dispersión entre-sujetos y `S_coher_prev5`, con **UNA excepción robusta y aislada**: `S_cond_mono` (tiempo de reacción monotónico "menos es mejor"). Los memes procesados con más fluidez se asocian con menos desacuerdo (ρ≈−0.17) y valoración menos sexista (soft_21_yes ρ≈−0.14), de forma robusta (winsorizado, sin extremos, EN/ES por separado).
6. El hallazgo de RT es el reflejo en EXIST de la **caja de herramientas de S_cond**: la forma se elige por la semántica del observable, y el RT (dirección unívoca "menos es mejor") exige la transformación monotónica; la U penalizaba el RT rápido como "descuidado" y borraba la señal direccional. Encaja con la lectura S_obs (más fluidez de procesamiento = más observabilidad = más consenso). Es de diseño débil (causa común, poblaciones disjuntas) → diagnóstico, no confirmación.
7. La **desalineación poblacional** es la lección estructural: la conexión indirecta entre fisiología y desacuerdo es demasiado débil para detectarse en EXIST incluso con ~4000 memes — el estudio que resuelve esta limitación es K-EmoCon (Estudio 2), donde sujeto fisiológico y objeto del juicio coinciden. La desconexión queda **medida** con datos del propio PSRI (§3.8): el agregado por meme no tiene señal reproducible (fiabilidad ≈0, ICC_meme≈0), por lo que el nulo es esperado por diseño.

**En una frase**: EXIST es el estudio donde se define y somete a un primer nulo robusto la arquitectura de tres eslabones del PSRI; su desalineación poblacional explica el nulo y motiva K-EmoCon, y su comprobación final revela que el único eslabón con señal —el tiempo de reacción— solo la muestra con la herramienta que el framework prescribe para un observable con dirección unívoca (la transformación monotónica "menos es mejor").

## 6. Tabla de campos, significado y transformación

Misma lógica que las tablas de los Estudios 2 y 3, con la peculiaridad de que EXIST es donde se define la arquitectura de tres eslabones y donde el ground truth son las etiquetas de sexismo (que juegan un rol distinto al de PhysioNet — ver nota final).

| Campo | Significado | Transformación |
|---|---|---|
| `garmin_hr_std` | Variabilidad intra-trial de la HR (magnitud ya derivada de la señal cardíaca) | `compute_psri_gaussian_log` (log + mediana/MAD + campana, `eps_hard` percentil 1) → `PSRI_hr_subj` |
| `3d_eye_states_pupil diameter left [mm]_std` | Variabilidad intra-trial de la pupila izquierda | `compute_psri_gaussian_log` → `PSRI_et_subj` |
| `PSRI_hr_subj`, `PSRI_et_subj` | Fiabilidad de estabilidad por modalidad y trial, en [0,1] | Media por meme → `S_estab` (media de las medias por modalidad) |
| `garmin_hr_mean`, `garmin_hr_mean_baseline_prev` | Media de HR del trial actual y media del trial inmediatamente anterior (línea base) | `compute_z_score` con `subject_std` (std entre-trial del sujeto) → `z_hr` |
| pupila `[mm]_mean`, `[mm]_mean_baseline_prev` | Media de la pupila del trial y su línea base previa | `compute_z_score` con `subject_std` → `z_pupil` |
| `z_hr`, `z_pupil` | Desviaciones estandarizadas respecto al estado basal del propio sujeto | `compute_coherence(z_hr, z_pupil) = exp(-|z_hr - z_pupil|/2)` → `S_coher_trial` → media por meme → `S_coher` |
| `reaction_time`, `blinks_count` | Tiempo de reacción y parpadeos por trial (conducta respecto a la tarea) | Caja de herramientas de S_cond: U para `blinks_count` (ambos extremos patológicos), monotónica "menos es mejor" para `reaction_time` → `S_cond_rt`/`S_cond_blink` → media → `S_cond_subj` → media por meme → `S_cond`; `S_cond_mono` (monotónica) en el cierre de fase |
| `S_estab`, `S_coher`, `S_cond` | Los tres eslabones, imputados por su mediana si falta algún valor | `compute_weighted_psri(w1=w2=w3=1/3)` → `PSRI` |
| `reaction_time` (versión monotónica) | Idem, con transformación "menos es mejor" (no-U) | `R = 1 − exp(−z_clip/2)` con `z = −(rt − mediana)/(1.4826·MAD)` → `S_cond_mono` (diagnóstico de cierre de fase) |
| `hard_21`, `soft_21_yes`, `entropy_21` | Valoración de sexismo (YES/NO), proporción continua de YES y entropía binaria (desacuerdo) | No entran en el cálculo del PSRI — son los targets contra los que se correlaciona/compara |
| `hard_22`, `entropy_22`, `entropy_23` | Intención (DIRECT/JUDGEMENTAL) y entropías de desacuerdo de las tareas 2.2 y 2.3 | Idem |

**Nota final — el rol del ground truth (y su diferencia con PhysioNet).** En PhysioNet (Estudio 3), `RECORDS-acceptable/unacceptable` es un **ground truth de calidad de señal** etiquetado por expertos contra el que se valida el instrumento (¿funciona la fórmula?). Aquí, las etiquetas de sexismo y las entropías **no son** un ground truth de calidad del PSRI — el instrumento ya se validó en PhysioNet — sino las variables de interés científico (¿la fiabilidad fisiológica predice el desacuerdo humano y cómo se relaciona con la valoración de sexismo?). Es el mismo cambio de rol que se documenta en K-EmoCon (Estudio 2, Sección 6.5): en PhysioNet el ground truth pregunta "¿el instrumento mide bien?"; en EXIST la etiqueta de sexismo pregunta "¿lo que el instrumento mide, importa para el juicio humano?".

---

## Anexo. Métodos de validación: en qué consisten y su enfoque metodológico

Como en los Estudios 2 y 3, cada comprobación está diseñada para descartar una amenaza concreta a la validez de la conclusión.

### A.1. Test de Kruskal-Wallis (etiquetas duras)

**En qué consiste.** Prueba no paramétrica (equivalente al ANOVA sin asumir normalidad) que determina si dos grupos independientes —memes etiquetados YES frente a NO, DIRECT frente a JUDGEMENTAL— presentan diferencias significativas en sus distribuciones de PSRI. Responde a: "¿es estadísticamente diferente el PSRI de los memes sexistas frente a los que no lo son, sin importar la forma exacta de sus curvas?". Se reporta con el tamaño de efecto ε² (epsilon cuadrado).

**Qué amenaza controla.** Replica el análisis de AI Wizards para contrastar si el PSRI separa las clases de sexismo de forma robusta, y no solo como diferencia descriptiva.

**Cómo se aplica aquí.** PSRI vs `hard_21`: estadístico=8.501, p_crudo=0.0035, ε²=0.00188 (mediana YES=0.7710, NO=0.7647); PSRI vs `hard_22`: estadístico=2.572, p=0.1088, ε²=0.00088 (DIRECT=0.7718, JUDGEMENTAL=0.7651). Ninguno sobrevive la corrección unificada (Sección 3.6.3).

### A.2. Experimento de correlación (PSRI vs entropías)

**En qué consiste.** Para cada una de las 6 métricas y cada una de las 3 entropías (desacuerdo), se calcula el coeficiente de correlación de Pearson entre el valor de la métrica por meme y la entropía de las soft labels por meme.

**Qué amenaza controla.** La hipótesis central: si la fiabilidad de la señal predice el desacuerdo, esperaríamos una correlación (idealmente negativa: más fiabilidad → menos desacuerdo) sistemática entre algún componente del PSRI y la entropía. La entropía es la medida matemática del desacuerdo: 0 = consenso total, alta = meme muy subjetivo.

**Cómo se aplica aquí.** Máximo r² observado = 0.00120 (menos del 0.12% de la varianza). Ninguna correlación sobrevive la corrección. Con n≈3984, el estudio tiene potencia para detectar r≈0.05, así que el nulo no es un artefacto de potencia (Sección 3.6.4).

### A.3. Corrección por comparaciones múltiples (Bonferroni y FDR)

**En qué consiste.** Con 20 tests a la vez, la probabilidad de al menos un falso positivo crece con el número de tests. Se usan dos criterios complementarios sobre un **pool único**: Bonferroni (divide el umbral entre el nº de tests; el más estricto, controla la tasa de error por familia) y FDR/Benjamini-Hochberg (controla la proporción esperada de falsos positivos entre los declarados significativos; más permisivo).

**Qué amenaza controla.** La inflación de falsos positivos por probar muchas hipótesis a la vez. Corregir todo el pool junto (2 Kruskal + 18 Pearson) impide que tests relacionados se cuenten como independientes o que un gran número de tests diluya el criterio.

**Cómo se aplica aquí.** El umbral Bonferroni efectivo es 0.05/20 = 0.0025. Ni la correlación bruta significativa de la Task 2.3 (p=0.030) ni el Kruskal de la Task 2.1 (p=0.0035) lo superan: **ningún resultado es significativo tras el ajuste** (Sección 3.6.3-3.6.4).

### A.4. Comprobación final de cierre de fase (pool ampliado)

**En qué consiste.** `sources/exist/final_check.py` amplía el pool a **66 tests** (46 correlaciones Spearman + 20 Kruskal) que combinan los reenfoques del framework: descomposición por componente, compuesto de 2 patas, `soft_21_yes` continuo, dispersión entre-sujetos, `S_coher_prev5` y `S_cond_mono`. Todo se corrige como un único bloque Bonferroni + FDR.

**Qué amenaza controla.** Que la selección de la forma funcional (la caja de herramientas de S_cond, que en K-EmoCon reveló señales específicas según la semántica del observable) pudiera tener reflejo en EXIST y que el nulo del sanity check fuera un artefacto de haber aplicado la herramienta en-U al RT, un observable con dirección unívoca.

**Cómo se aplica aquí.** 6/66 sobreviven Bonferroni y 7/66 FDR, todos ellos asociados a `S_cond_mono` (salvo el PSRI vs `hard_21` que sobrevive FDR): el tiempo de reacción monotónico es el único reflejo robusto, verificado además sobre el RT crudo con winsorización, exclusión de extremos y subgrupos EN/ES (Sección 3.7).

### A.5. Independencia entre componentes

**En qué consiste.** Se cuantifica la correlación de Pearson entre cada par de los tres eslabones (S_estab, S_coher, S_cond) y entre modalidades (PSRI_hr vs PSRI_et).

**Qué amenaza controla.** La redundancia entre dimensiones: si dos componentes estuvieran muy correlacionados, el compuesto doblaría información y los pesos iguales no estarían justificados. La independencia entre modalidades descarta que HR y pupila midan lo mismo.

**Cómo se aplica aquí.** S_estab–S_coher r=0.109, S_estab–S_cond r=0.074, S_coher–S_cond r≈0.0004, PSRI_hr–PSRI_et r=0.032 — los tres eslabones y las dos modalidades son prácticamente independientes (Sección 3.6.2), justificando la arquitectura y los pesos.

### Resumen del enfoque general

Los métodos forman una cadena de defensa en capas: el Kruskal ataca la separación de clases de sexismo, la correlación con entropías ataca la hipótesis de desacuerdo, la corrección múltiple ataca la inflación de falsos positivos, la comprobación final ataca la arbitrariedad de la operacionalización (¿algún reenfoque revela señal que la formulación original escondía?) y la independencia ataca la redundancia. Conjunto: el nulo de EXIST es robusto frente a las amenazas identificables en este diseño, con la única excepción exploratoria del tiempo de reacción monotónico.
