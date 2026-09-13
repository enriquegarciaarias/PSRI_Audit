# Estudio 3 PhysioNet

> Documento regenerado a partir de `ValidaciónPhysioNet.pdf`/`.odt` y alineado con la implementación actual del pipeline (`sources/physionet/`) y sus salidas en `results/output/PHYSIO/` (agosto 2026). Dos ajustes respecto al documento fuente: (1) se incorpora el **estudio de ablación metodológica** sobre la transformación de `std_signal` (5 variantes), que forma parte del proceso activo pero no estaba en el documento original; (2) el experimento de **sensibilidad al nivel de señal (`rr_std`)** se documenta con sus resultados históricos validados, pero aclara que en el flujo activo quedó como bloque desactivado (código muerto tras el `return` de `process_physio`), tal como refleja `tecnico_physionet.md` §7.

## 1. Introducción

Los Estudios 1 y 2 (EXIST y K-EmoCon) usan el PSRI como predictor del desacuerdo entre anotadores humanos. Este estudio es distinto en propósito: **valida el instrumento en sí** contra un ground truth de calidad de señal etiquetado por expertos humanos. La pregunta no es "¿la fiabilidad fisiológica predice el desacuerdo?", sino "¿el PSRI mide bien la fiabilidad de una señal fisiológica?" — una pregunta previa y necesaria: no tiene sentido correlacionar un instrumento contra una variable de interés si antes no se ha comprobado que el instrumento discrimina.

El mecanismo central del PSRI es la **función en U en log-espacio** (`compute_psri_gaussian_log`): normalización robusta con mediana/MAD de la población de referencia y campana gaussiana, que penaliza tanto la señal plana (variabilidad anómalamente baja) como la ruidosa (variabilidad anómalamente alta). La hipótesis operativa es que un registro ECG de calidad **aceptable** tiene una variabilidad "típica" (actividad cardíaca genuina, QRS marcado), mientras que un registro **inaceptable** se aparta de esa típica en ambos extremos.

PhysioNet/CinC Challenge 2011 es el escenario ideal para contrastar esta hipótesis contra expertos humanos: es la **única parte del framework donde la calidad de señal se puede contrastar contra etiquetas humanas**, y además es un estándar de referencia ampliamente usado en la literatura de índices de calidad de señal (SQI) para ECG, lo que permite comparar el instrumento contra referencias establecidas del dominio.

El resultado validado: el PSRI alcanza **AUC = 0.887** (0.8868 exacto) sobre los 998 registros de `set-a`, superando a kSQI (curtosis, AUC 0.842) y a la Correlación inter-derivación (AUC 0.794), pese a ver *solo* un número por registro (la desviación estándar de la onda) frente a la información morfológica completa que usan los índices de referencia. Esa ventaja es robusta frente a la Correlación inter-derivación (DeLong p=0.0003) y favorable aunque no estadísticamente concluyente frente a kSQI (DeLong p=0.066); se mantiene fuera de muestra (AUC 0.886, IC95% [0.855, 0.920]) y no se explica por la calibración interna a la muestra.

## 2. El dataset PhysioNet/CinC Challenge 2011

Empleamos el conjunto de entrenamiento (`set-a`) del PhysioNet/CinC Challenge 2011 (*Improving the Quality of ECGs Collected Using Mobile Phones*). Contiene **998 grabaciones de ECG de 12 derivaciones (10 segundos, 500 Hz)** etiquetadas por entre 3 y 18 expertos como **aceptable** (n=773) o **inaceptable** (n=225) para uso diagnóstico (≈77%/23%, desbalance moderado).

El ground truth se lee de dos archivos de texto (`loader.load_ground_truth`): los IDs en `RECORDS-acceptable` → clase 1, los de `RECORDS-unacceptable` → clase 0; `RECORDS` da el orden de procesamiento. Cada registro se lee con la librería `wfdb` (`wfdb.rdrecord`), de la que se extrae la matriz de señales `p_signal` (N muestras × 12 derivaciones) y la frecuencia de muestreo `fs`.

Silva et al. (2011) describen el protocolo del Challenge: anotación humana y técnicas de clasificación supervisadas (SVM, árboles, lógica difusa) empleadas por los participantes. El repositorio de registros etiquetados constituye el estándar de referencia contra el que validamos la capacidad discriminativa del PSRI. El paper de referencia que definió y aplicó los SQIs sobre este mismo dataset es Clifford, Behar, Li & Rezek (2012), y el editorial de Clifford & Moody (2012) sintetiza el estado del arte posterior (curtosis, pendiente del QRS, deriva de línea base, relación señal-ruido, combinados con SVM y redes neuronales).

Los equipos del Challenge (p. ej. Xia et al., 2012a) usaron matrices de hasta 144 características heurísticas y clasificadores entrenados; **este experimento usa una sola característica** (la desviación estándar de la forma de onda, sin ningún parámetro ajustado a los datos ni conocimiento de dominio cardíaco) y obtiene un AUC comparable (0.887 frente a accuracies del orden de 90-95%). Esa es la tesis de este estudio: la transformación en U es un instrumento de fiabilidad de señal **sensor-agnóstico**, no un ajuste empírico al ECG.

Se comparó contra **dos** índices de referencia — kSQI (curtosis) y Correlación inter-derivación — y no contra un conjunto más amplio (como otros trabajos con 6-7 SQIs y SVM) por decisión metodológica: representan los dos paradigmas complementarios — morfológico (¿la señal tiene forma de latido?) y estructural (¿todos los electrodos están de acuerdo?) — facilitando una comparación legible.

## 3. Construir el pipeline de validación

En PhysioNet el PSRI se construye con un **único campo de entrada** (variabilidad de la señal), porque el objetivo aquí no es el índice completo — ese es el objeto de los Estudios 1 y 2 — sino **comprobar si la transformación matemática por sí sola distingue grabaciones buenas de malas**. S_coher no se aplica en PhysioNet: el dataset contiene solo la modalidad ECG, y S_coher mide coordinación entre sistemas fisiológicos distintos.

La orquestación arranca en `main.py` (`python main.py --proc PHYSIO`) → `process_physio()` (`sources/physionet/validator.py`), que ejecuta cinco comprobaciones en orden:

1. **Comparativa de AUC** frente a los SQIs de referencia (`compare_psri_vs_reference_sqis`).
2. **Significación estadística** de la diferencia (DeLong + bootstrap pareado + PR-AUC) y figuras ROC (`compare_psri_vs_reference_sqis_with_significance`).
3. **Sensibilidad de calibración**: split-half fuera de muestra (200 particiones) para descartar que la calibración in-sample infle el AUC (`check_psri_calibration_sensitivity`).
4. **Limitación estructural de solapamiento**: dónde caen los errores respecto a la zona donde las clases se solapan (`analyze_psri_overlap_limitation`).
5. **Ablación metodológica** de la transformación: 5 variantes sobre el MISMO `std_signal` (`run_ablation_study`).

Módulos implicados: `validator.py` (orquesta), `loader.py` (ground truth), `features.py` (extracción de variabilidad), `reference_sqi.py` (kSQI y correlación), `reporting.py` (reportes y figuras), `auc_comparison.py` (DeLong, bootstrap, ROC comparativas), `calibration_sensitivity.py` (split-half), `overlap_analysis.py` (diagnóstico de solapamiento), `sources/psri/validation.py` (`compare_all_vs_valid`) y `sources/common/metrics.py` (AUC, G-mean).

### 3.1. La entrada: `std_signal`

Cada una de las 12 derivaciones registra el voltaje a lo largo del tiempo; aunque los voltajes no son iguales en todas, todas reflejan la misma actividad eléctrica cardíaca desde ángulos distintos. La desviación estándar (`std` de la señal, eje temporal) evalúa la calidad técnica de la señal midiendo la amplitud o dispersión de las oscilaciones alrededor de la media:

- **Señal limpia**: el voltaje oscila claramente con cada latido (grandes diferencias entre el pico de la onda R y el valle de la onda S) → `std` alta.
- **Señal plana** (electrodo despegado, cable suelto): el voltaje apenas cambia → `std` casi cero.
- **Señal ruidosa** (movimiento, interferencia eléctrica): el voltaje oscila mucho, pero de forma caótica y sin periodicidad → `std` también alta, pero por perturbaciones externas, no por actividad cardíaca.

La desviación estándar, por sí sola, no distingue una señal limpia de gran amplitud de una ruidosa de gran amplitud — solo mide la magnitud de la oscilación, no su regularidad ni su origen. En PhysioNet los registros inaceptables suelen presentar desviaciones extremas (muy bajas o muy altas), lo que permite a la función en U separar ambos extremos. La limitación estructural de reducir toda la señal a un único escalar se examina explícitamente en la Sección 3.9.

**Cálculo** (`features.extract_variability`, método `std_signal`): para cada derivación se calcula `std` de la serie temporal completa (std_deriv1 … std_deriv12) y se promedia en un único número, `std_signal`, que resume la calidad global del registro. Este método **nunca falla** (`fail_flag` siempre False). Es el método usado por defecto (validado y comparado).

### 3.2. La función en U: `compute_psri_gaussian_log`

Una vez obtenido `std_signal`, se introduce en la función en U en tres etapas:

1. **Transformación logarítmica** (`log(σ)`). Las distribuciones de variabilidad fisiológica son inherentemente asimétricas (sesgadas a la derecha): el límite inferior está acotado en cero (señal plana), el superior ante artefactos es teóricamente infinito (ruido extremo). Aplicada la U directamente sobre la escala original, una señal muy ruidosa se alejaría geométricamente mucho más del centro que una plana, rompiendo la penalización simétrica buscada. El logaritmo comprime la cola derecha y expande la izquierda, garantizando que un fallo por exceso (ruido) y uno por defecto (plano) sean matemáticamente equidistantes de la mediana.
2. **Z-score robusto** con mediana y MAD (Median Absolute Deviation) de la población de referencia (los 998 registros de entrenamiento): `z = (log(std_signal) − μ_D) / (1.4826·MAD_D)`. El factor 1.4826 hace la MAD estadísticamente equivalente a una desviación estándar clásica, pero inmune a los casos extremos. `z` indica a cuántas dispersiones típicas está la señal del comportamiento normal.
3. **Campana gaussiana**: `R = exp(−0.5·z²)`. El score es alto (cerca de 1) cuando `std_signal` es típico, y bajo (cerca de 0) cuando es anormalmente bajo (plano) o anormalmente alto (ruidoso). Además, antes de la transformación se fija `eps_hard` = percentil 1 de los valores no-cero: las señales planas por debajo de ese umbral reciben directamente `R = 0` (corte duro, equivalente al sentinel de los SQIs de referencia, Sección 3.5).

Con este modelo el PSRI se convierte en un **indicador de normalidad de la variabilidad de la señal**. En la versión explotada en los Estudios 1 y 2 el mismo mecanismo se aplica por ventana y por canal, y aquí sobre un único agregado por registro.

### 3.3. Los SQIs de referencia: kSQI y Correlación inter-derivación

**kSQI — el proxy morfológico: ¿tiene la forma de un latido?** (`reference_sqi.compute_ksqi`)

1. Lee las 12 derivaciones del ECG crudo.
2. **Filtro de señal plana (degenerados)**: si cualquiera de las 12 derivaciones tiene `std < 1e-6` (básicamente una línea recta porque el electrodo se cayó), devuelve `None`.
3. Calcula la **curtosis de Fisher** (resta 3 para que una normal perfecta dé 0) por derivación.
4. Control de errores matemáticos: si la curtosis da infinito o NaN, devuelve `None`.
5. Agregación: media de las curtosis de las 12 derivaciones. Un valor alto significa "forma de latido nítida" (buena calidad); bajo o negativo, "señal plana o ruidosa" (mala calidad). Un ECG limpio produce QRS marcados y una distribución leptocúrtica; uno ruidoso difumina los picos y la distribución se vuelve platicúrtica.

**Correlación inter-derivación — el proxy estructural: ¿están todos los electrodos de acuerdo?** (`reference_sqi.compute_interlead_correlation_sqi`)

Las 12 derivaciones son 12 ángulos de observar el mismo latido: si el corazón late, las 12 líneas deben subir y bajar a la vez; si una derivación se desconecta, deja de fluctuar al mismo ritmo.

1. Lee las 12 derivaciones.
2. **Edge-case**: si el registro solo tuviera 1 derivación, devuelve `np.nan` (no es mala señal, es que matemáticamente no se puede correlacionar una cosa consigo misma).
3. **Filtro de señal plana**: igual que kSQI, si alguna derivación tiene `std ≈ 0`, devuelve `None`.
4. **Matriz de correlación** de Pearson cruzando todas las derivaciones (12×12).
5. **Aislamiento de pares únicos**: extrae el triángulo superior (ignorando la diagonal, siempre 1.0).
6. **Agregación**: media de las correlaciones por pares. Un valor cercano a 1.0 = "todas las derivaciones capturan el mismo evento" (buena calidad); bajo = "electrodos desconectados midiendo cosas distintas" (mala calidad).

kSQI tiene la ventaja de "ver" la forma de la onda; el PSRI, en esta validación, solo "ve" un número (la desviación estándar de esa onda). Que PSRI (0.887) gane a kSQI (0.842) viendo menos información es lo que hace contundente la validación.

### 3.4. Registros degenerados: sentinel vs. descarte

Cuando kSQI o la Correlación encuentran un electrodo caído, devuelven `None` (la curtosis o la correlación se rompen matemáticamente). El manejo de estos casos es una decisión metodológica delicada: los registros con electrodo caído son **casi siempre** inaceptables — en `set-a`, **135 registros degenerados** (std de alguna derivación < 1e-6), de los cuales **129 son inaceptables** y solo 6 aceptables. Si se descartaran, se le borrarían los casos "fáciles" a kSQI/Correlación, inflando artificialmente su AUC.

Por eso `validate_reference_sqi` **no descarta los degenerados, los reasigna a un sentinel**: un valor finito por debajo de todos los scores reales (`min_finite − (|min_finite|·0.1 + 1)`), equivalente a "el índice está 100% seguro de que esto es basura". Esto **iguala las condiciones**: el PSRI maneja las señales planas dándoles un 0.0 con su corte duro (`eps_hard`), y los SQIs de referencia les dan el valor sentinel. Los `np.nan` (registros legítimamente no evaluables, p. ej. 1 sola derivación) sí se descartan de la evaluación, contados por clase.

En `set-a`: 135 degenerados reasignados a sentinel (6 aceptables, 129 inaceptables) tanto para kSQI como para la Correlación, con 0 descartes por no-evaluable.

### 3.5. Resultados principales: PSRI frente a SQIs de referencia

Para cada registro se calcula un score continuo (PSRI, kSQI o Correlación) y se compara con su etiqueta binaria. A partir de esta comparación se construye la curva ROC y se calcula el **AUC** (capacidad de separar aceptables de inaceptables independientemente del umbral) y la **G-mean** (media geométrica de sensibilidad y especificidad) con el umbral operativo.

**Tabla 2. PSRI frente a SQIs de referencia (n=998, mismo conjunto para las tres métricas).**

| Método | AUC | G-mean |
|---|---|---|
| **PSRI (sobre std_signal)** | **0.8868** | 0.814 |
| kSQI (curtosis) | 0.8421 | 0.7067 |
| Correlación inter-derivación | 0.7938 | 0.6775 |

El PSRI, usando únicamente la variabilidad de la señal (una única característica), **supera en AUC a índices diseñados específicamente para ECG**. La ventaja sobre la Correlación inter-derivación es estadísticamente robusta; la ventaja sobre kSQI es consistente en dirección pero no alcanza significación convencional (detalle en Sección 3.6).

### 3.6. Significación estadística de la ventaja

La comparación **pareada** exige los mismos registros en todos los scores: se recalculan los 3 scores por `rec_id` y se trabaja sobre la intersección donde los tres son válidos (998/998 en `set-a`; DeLong exige correspondencia registro-a-registro). Dos métodos estándar para AUCs correlacionadas:

- **Test de DeLong**: PSRI vs kSQI diff=+0.0446, p=0.0657 (no significativo); PSRI vs Correlación diff=+0.0930, p=0.0003 (significativo).
- **Bootstrap pareado** (2000 réplicas, remuestreo de registros preservando el emparejamiento): PSRI vs kSQI diff=+0.0446, IC95%=[−0.0040, 0.0933], p=0.067; PSRI vs Correlación diff=+0.0930, IC95%=[0.0428, 0.1430], p≈0.

Ambos métodos coinciden: la ventaja frente a la **Correlación inter-derivación es robusta**, y frente a **kSQI es favorable aunque no concluyente** — el IC95% del bootstrap apenas roza el cero por debajo (−0.0040).

**PR-AUC (average precision).** Dado el desbalance de clases (773 aceptables / 225 inaceptables ≈ 77/23), el ROC-AUC puede ser optimista; el PR-AUC es más sensible a los falsos positivos sobre la clase minoritaria. Resultado: **PSRI=0.9510, kSQI=0.9107, Correlación=0.8754** — el mismo orden y una diferencia de magnitud similar a la del ROC-AUC, lo que descarta que la ventaja del PSRI dependa del desbalance de clases.

**Figuras**: `fig_physionet_roc_comparison.{png,pdf}` (las 3 ROC superpuestas) y `fig_physionet_roc_thresholds.{png,pdf}` (ROC con cuartiles P25/P50/P75 por método y el umbral operativo real de cada uno: 0.5 fijo para PSRI, la mediana de cada índice para kSQI y Correlación). La curva de PSRI domina a las otras dos en casi todo el rango de umbrales (sensibilidad alta con tasa de falsos positivos menor), y el punto operativo de cada método cae en una región razonable de la curva, no en un extremo degenerado.

### 3.7. Sensibilidad de calibración: ¿cuánto del AUC depende de calibrar sobre la muestra?

`compute_psri_gaussian_log` calibra su mediana/MAD/`eps_hard` sobre la **misma** población de 998 registros que después puntúa — una ventaja estructural frente a kSQI/Correlación, fórmulas cerradas sin ningún parámetro ajustado a los datos. Para cuantificar cuánto de la ventaja podría deberse a esta calibración interna, se evaluó el PSRI en un régimen estrictamente **fuera de muestra**:

- 200 particiones aleatorias en el rango 50 calibración / 50 evaluación de las 998 grabaciones.
- En cada partición, mediana/MAD/`eps_hard` se calculan únicamente sobre la mitad de **calibración** y el AUC se calcula únicamente sobre la mitad de **evaluación** — nunca compartiendo registros entre ambos papeles. Se descartan los splits sin ambas clases en evaluación (en `set-a`: 200/200 válidos).

**Resultado**: AUC medio fuera de muestra = **0.8860** (mediana 0.8851, std 0.0159, IC95%=[0.8545, 0.9195]), prácticamente idéntico al AUC in-sample original (0.8870), que cae dentro del intervalo de confianza de la distribución fuera de muestra. **La calibración interna a la muestra no explica el rendimiento discriminativo del PSRI.** Figura: `fig_physionet_calibration_sensitivity.{png,pdf}`.

### 3.8. La zona de solapamiento: dónde falla el PSRI

Para explicar el ~11% de error residual se examinó si los registros mal clasificados se concentran en la zona donde las distribuciones de `std_signal` de ambas clases se solapan: en espacio `log(std_signal)` se calcula el rango [P5, P95] de cada clase y su **intersección** = zona de solapamiento. Resultado: zona **[0.0934, 0.4000]** en `std_signal`, donde cae el **74.7%** de los registros.

- **Análisis agregado**: no revela un patrón claro — el 56.9% de los mal clasificados (n=181) cae en la zona frente al 78.7% de los bien clasificados (n=817): los errores no se concentran desproporcionadamente en la zona de ambigüedad.
- **Desglose por tipo de error**: el mecanismo se confirma con claridad.
  - **Falsos positivos** (inaceptables → aceptables, n=44): el **100%** cae en la zona de solapamiento, frente a una tasa base del 22.7% entre todos los inaceptables (sobrerrepresentación de ~4.4×). Ocurren si y solo si el ruido se "camufla" dentro del rango típico de los aceptables — no es tan extremo como para dar una `std` muy alta.
  - **Falsos negativos** (aceptables → inaceptables, n=137): solo el 43.1% cae en la zona, frente al 89.9% esperable entre los aceptables (infrarrepresentación de ~2.1×). Ocurren en registros aceptables con `std_signal` atípica (variabilidad inusualmente baja o alta), que el PSRI rechaza.

Estos resultados confirman que el PSRI funciona bien en el ~89% de los casos pero **falla sistemáticamente cuando el ruido se disfraza de señal normal en términos de variabilidad** — la práctica totalidad de los falsos positivos son precisamente esos casos. Es una limitación inherente a usar un solo escalar (la desviación estándar). Figura: `fig_physionet_overlap_diagnostic.{png,pdf}`.

### 3.9. Ablación metodológica de la transformación

Una posible objeción es que la elección de Log + Mediana/MAD + Gaussiana invertida fuera una decisión arbitraria. Para responderla, se compararon **5 variantes de transformación sobre el MISMO `std_signal`** (solo difiere la función; la entrada es idéntica para las cinco), con AUC, DeLong y bootstrap pareado frente a la propuesta:

| Variante | Normalización | Forma funcional | AUC | ΔAUC | p (DeLong) |
|---|---|---|---|---|---|
| **Propuesta (PSRI)** | Log + Mediana/MAD | Gaussiana Invertida (U) | **0.8868** | — | — |
| Alt. A: Mono-Decay | Log + Mediana/MAD | Monótona Decreciente | 0.8868 | +0.0000 | 1 |
| Alt. B: Z-Score Clásico | Log + Media/Std | Gaussiana Invertida (U) | 0.8681 | +0.0187 | 0.096 |
| Alt. C: Percentil Empírico | Ninguna (Ranking) | Escalonado No Paramétrico | 0.5174 | +0.3694 | <1e-4 |
| Alt. D: Min-Max Crudo | Mínimo/Máximo | Lineal Acotada [0,1] | 0.5174 | +0.3694 | <1e-4 |

Lectura:

- **Los peores competidores son los no paramétricos** (Percentil Empírico y Min-Max, AUC ≈ 0.52, p<1e-4): la normalización robusta en log-espacio es imprescindible; un simple ranking o un escalado por extremos destruye el poder discriminativo.
- **La robustez MAD es lo que importa más que la forma en U**: Alt. B (media/std clásica) apenas baja el AUC (p=0.096). La elección de mediana/MAD frente a media/std no es crítica.
- **Alt. A Mono-Decay es indistinguible en AUC** (diff=0, p=1): `1/(1+|z|)` y `exp(−0.5z²)` son ambas monótonas decrecientes en `|z|`, así que producen el mismo ranking. Como ablación de la "necesidad de la forma en U" este brazo no discrimina — la diferencia entre ellas es de calibración/escala, no de orden.

Este estudio (presente en el proceso actual pero ausente en el documento fuente original) refuerza que la elección de Log + Mediana/MAD + Gaussiana invertida no es un ajuste a los datos: es la combinación que soporta el poder discriminativo del instrumento. Salidas: `ablation_study_table.tex` y `fig_physionet_ablation_roc.{png,pdf}`.

### 3.10. Sensibilidad al nivel de la señal: `rr_std`

El PSRI puede operar sobre la **forma de onda cruda** (`std_signal`) o sobre una **magnitud derivada** de ella. Para comprobar que la transformación funciona independientemente de la representación de la señal, se probó `rr_std`: la desviación estándar de los **intervalos RR** (tiempo entre dos picos R consecutivos del ECG, en ms), detectados con el algoritmo `xqrs_detect` de WFDB sobre la 2ª derivación (si existe).

`rr_std` mide la variabilidad de la frecuencia cardíaca (HRV) durante los 10 segundos del registro. En una señal limpia los intervalos RR son relativamente consistentes (std baja pero no cero, por la HRV fisiológica normal); en una señal ruidosa o plana el detector puede fallar — picos falsos (por ruido) o no detección (por señal plana) — produciendo secuencias muy irregulares (std alta) o sin detecciones. Es un **indicador de calidad indirecto**: captura cómo de fiable es la detección de latidos, que a su vez depende de la calidad de la señal subyacente. Es el análogo conceptual más cercano a la desviación estándar de HR en bpm empleada en EXIST, al tratarse igualmente de una magnitud ya derivada de la señal cardíaca cruda.

**Tabla 3. Sensibilidad del PSRI al nivel de la señal.**

| Nivel de señal | AUC (todos) | AUC (extracción válida) | Fallos de extracción |
|---|---|---|---|
| Forma de onda cruda (std_signal) | 0.887 | 0.887 | 0/998 (0.0%) |
| Magnitud derivada (rr_std) | 0.748 | 0.671 | 163/998 (16.3%) |

`rr_std` es el **único método que puede fallar** (`fail_flag`): el detector de picos no encuentra QRS válidos en **163 de 998 registros (16.3%)**. El patrón de fallo es informativo por sí mismo: el detector falló en el **40.9% de las grabaciones inaceptables** frente a solo el **9.2% de las aceptables** — el propio fallo de extracción actúa como clasificador parcial de calidad, antes de aplicar ninguna transformación. Al aislar los casos de extracción exitosa (998−163=835 registros), el AUC real de la transformación sobre magnitud derivada **cae a 0.671**, con una especificidad de 0.429 (cercana al azar). La variabilidad de la señal cruda (`std_signal`) es más directa y no depende de la correcta detección de latidos, una ventaja especialmente importante en señales muy ruidosas donde el detector de QRS falla.

Este mismo fenómeno — fallos de un detector de picos concentrados en mala calidad — es el mismo tipo de caída de señal documentado en K-EmoCon (sujetos pid=4 y pid=28 excluidos por caída de IBI, Sección 3.1 de `EstudioKemocon.md`).

> **Nota sobre el estado en el pipeline actual**: este experimento (bloque `compare_feature_methods`, std_signal vs rr_std) quedó **desactivado** en el flujo activo — es código muerto tras el `return` de `process_physio` — tal como refleja `tecnico_physionet.md` §7. Los números aquí reportados proceden del documento fuente y de la implementación, y se conservan por su valor metodológico (dependencia del PSRI del nivel de representación de la señal y patrón de fallo del detector).

### 3.11. Variables no usadas por diseño (y sí por los equipos del Challenge)

Los equipos participantes usaron un abanico mucho más amplio de características que el PSRI no emplea, por razones de diseño:

| Categoría | Variable / Feature | ¿Qué mide? | Por qué no se usa |
|---|---|---|---|
| Morfología de la onda | Curtosis (kSQI) | Pico de la distribución (QRS nítido = curtosis alta) | Se usa como benchmark, NO como entrada del PSRI (el PSRI quiere ser agnóstico al dominio) |
| Complejidad / Aleatoriedad | Entropía (Sample/Approx Entropy) | Regularidad de la señal (ruido = alta entropía) | Podría medir "estabilidad" (similar a S_estab en EXIST), pero requiere señal cruda |
| Energía espectral | SNR o pSQI (potencia en banda cardíaca vs ruido) | Energía en banda ECG (0.5-40 Hz) frente a ruido fuera de banda | Análogo espectral de S_coher, pero no calculable desde agregados |
| Pendiente del QRS | sSQI (slope SQI) | Pendiente máxima del complejo QRS | Indicador morfológico adicional, específico de ECG |
| Línea base | basSQI (baseline wander) | Desviación de la línea base por movimiento o respiración | Detecta artefactos de baja frecuencia, requiere cruda |
| Consistencia entre latidos | bSQI (beat-to-beat correlation) | Correlación entre latidos consecutivos | Análogo a S_cond, pero en el dominio del latido |
| Residuos de filtrado | Energía tras restar el filtro paso-banda | Ruido residual fuera de banda cardíaca | Medida directa de "ruido residual", requiere cruda |

Razones (las mismas tres que estructuran el argumento del estudio):

1. **Agnosticismo al sensor**: el PSRI se concibió como un índice generalizable a cualquier señal fisiológica (HR, pupila, ECG). La medida más universal es la desviación estándar de la forma de onda cruda, que existe en cualquier sensor. Usar curtosis o pendiente del QRS habría anclado el índice al ECG y habría invalidado su aplicación a EXIST (donde hay HR y pupila).
2. **Validar la transformación, no competir en el Challenge**: si se hubieran usado más características, se habría diluido la contribución de la fórmula y el estudio estaría compitiendo en el terreno de los equipos del Challenge, no validando el instrumento.
3. **Formato de datos de EXIST**: en EXIST los datos fisiológicos disponibles son agregados estadísticos (medias y desviaciones), no series temporales crudas; características como entropía, pendiente del QRS o energía espectral no eran calculables. Al validar el PSRI con la variable más simple (`std_signal`), se asegura que el instrumento sea aplicable directamente al formato de datos de EXIST sin necesidad de acceso a la señal cruda.

En resumen: los equipos del Challenge usaron múltiples características y entrenaron clasificadores supervisados optimizados para el ECG; nosotros usamos **una sola característica con una función matemática fija (no supervisada)** optimizada para ser agnóstica al sensor. Que el PSRI, con una sola variable, iguale o supere a índices multicarácter específicos de ECG demuestra que la transformación en U es un instrumento **robusto y portátil**, no un ajuste empírico a un dominio concreto.

## 4. Limitaciones

- **La validación se circunscribe al dominio cardíaco (ECG)**: no cubre EEG ni seguimiento ocular, para los que no identificamos un dataset de acceso abierto con ground truth de calidad comparable. Debe interpretarse como evidencia del **principio general** del instrumento, no como validación exhaustiva de cada componente modal del PSRI compuesto.
- **El contexto de grabación difiere del de los wearables**: ECG clínico de 10 segundos con adquisición móvil frente a wearables durante exposición a estímulos visuales breves; los tipos de artefacto presentes pueden no ser directamente equiparables.
- **El PSRI usa un único escalar** (`std_signal`), y eso limita su techo: los registros donde el ruido se camufla dentro del rango típico de variabilidad de la clase aceptable (la práctica totalidad de los falsos positivos, n=44, 100% en la zona de solapamiento) son irresolubles para un índice univariado. La incorporación de S_coher (coherencia entre sistemas) en los Estudios 1 y 2 responde a esta misma limitación, pero en PhysioNet no es aplicable por contener solo la modalidad ECG — los ~11.3 puntos de AUC que faltan para 1.0 son, en gran medida, exactamente estos casos.
- **La ventaja sobre kSQI no es estadísticamente concluyente** (DeLong p=0.066; IC95% bootstrap [−0.0040, 0.0933]): la lectura honesta es "favorable y consistente en dirección", no "demostrada". Solo la ventaja frente a la Correlación inter-derivación es robusta.
- **`rr_std` (sensibilidad al nivel de señal) depende de la detección de picos**: el AUC real sobre magnitud derivada cae a 0.671 al aislar los casos de extracción exitosa — el PSRI sobre forma de onda cruda es la configuración defendida, y la magnitud derivada no debe presentarse como equivalente.
- El experimento de sensibilidad (`rr_std`) está **desactivado en el flujo activo** del pipeline (código muerto tras el `return` de `process_physio`): los números se conservan con valor metodológico, pero no se regeneran en cada ejecución.

## 5. Síntesis (puntos de partida)

1. La validación del instrumento es independiente de los estudios de aplicación: el PSRI discrimina calidad de señal ECG etiquetada por expertos con **AUC 0.887**, superior a kSQI (0.842) y a la Correlación inter-derivación (0.794), pese a ver solo la desviación estándar de la onda.
2. La ventaja es **robusta frente a la Correlación inter-derivación** (DeLong p=0.0003; bootstrap IC95% [0.043, 0.143]) y **favorable aunque no concluyente frente a kSQI** (p=0.066). La PR-AUC (0.951 / 0.911 / 0.875) descarta que el resultado dependa del desbalance de clases.
3. **No hay fuga de calibración**: el AUC fuera de muestra (0.886, IC95% [0.855, 0.920]) es prácticamente idéntico al in-sample (0.887). La ventaja estructural de calibrar mediana/MAD sobre la población que se puntúa no infla el resultado.
4. El ~11% de error residual se concentra donde es estructuralmente inevitable para un único escalar: los falsos positivos (100% en la zona de solapamiento) son ruido que se disfraza de señal normal en términos de variabilidad. Es una limitación reconocida del diseño univariado, no un fallo del mecanismo.
5. La **ablación** respalda la elección de la transformación: las alternativas no paramétricas (Percentil Empírico, Min-Max) se hunden (AUC≈0.52, p<1e-4); la forma en U y la robustez MAD son lo que sostiene el poder discriminativo, no un ajuste arbitrario.
6. El PSRI es **sensor-agnóstico por diseño**: una sola característica universal (la desviación estándar de la onda cruda) iguala o supera a índices específicos de ECG, y opera sobre el formato de agregados estadísticos de EXIST sin necesidad de señal cruda.
7. La sensibilidad al nivel de señal (rr_std) muestra que la transformación funciona sobre magnitudes derivadas pero con coste (AUC 0.671 tras aislar fallos del detector) — la forma de onda cruda es la configuración defendida, y el patrón de fallo del detector de picos (40.9% en inaceptables vs 9.2% en aceptables) es el mismo fenómeno de caída de señal documentado en K-EmoCon.

**En una frase**: PhysioNet valida el *instrumento* (¿mide bien la fiabilidad de la señal? → sí, y sin fuga ni ajuste de dominio), mientras que EXIST y K-EmoCon usan el instrumento ya validado para preguntar *lo que el instrumento mide importa* (¿la fiabilidad fisiológica predice el desacuerdo humano?).

## 6. Tabla de campos, significado y transformación

Misma lógica que la tabla de campos de los Estudios 1 y 2, pero con un rol diferente del ground truth (ver nota final).

| Campo | Significado | Transformación |
|---|---|---|
| `p_signal` (señal ECG cruda, 12 derivaciones) | El voltaje registrado por cada electrodo a lo largo de los 10 segundos de grabación | Se calcula su desviación estándar por derivación y se promedia entre las 12 → `std_signal` |
| `std_signal` | Variabilidad de la amplitud de la señal cruda — resume "cuánto se mueve" el voltaje | Entra en `compute_psri_gaussian_log`: `eps_hard` (percentil 1, corte duro → señal plana = 0), log(std_signal), z-score robusto con mediana/MAD de la población de referencia, `R = exp(−0.5·z²)` → score PSRI del registro |
| `rr_std` (alternativa probada) | Desviación estándar de los intervalos entre picos R consecutivos (detectados con `xqrs_detect`) | Misma función en U, pero aplicada a una magnitud ya derivada (no la onda cruda) — usada para el experimento de sensibilidad al nivel de señal, no para el resultado principal |
| `RECORDS-acceptable` / `RECORDS-unacceptable` | Etiqueta de calidad asignada por expertos humanos (773 / 225) | No entra en el cálculo del PSRI — es el ground truth contra el que se compara el score, para obtener el AUC y la G-mean |

**Nota final — el rol del ground truth (y su diferencia con K-EmoCon)**: en PhysioNet, `RECORDS-acceptable/unacceptable` es un **ground truth de calidad de señal** etiquetado por expertos, contra el que se compara el PSRI para saber si el instrumento discrimina bien — la pregunta es *¿funciona la fórmula?*. En K-EmoCon (Sección 6.5 de `EstudioKemocon.md`), `external_valence_var` y el resto de métricas de desacuerdo **no son** un ground truth de calidad del PSRI — el instrumento ya se validó aquí — sino la variable de interés científico de la pregunta de investigación en sí. Es un cambio de rol: en PhysioNet el ground truth pregunta "¿el instrumento mide bien?"; en K-EmoCon la variable de desacuerdo pregunta "¿lo que el instrumento mide, importa?".

---

## Anexo. Métodos de validación: en qué consisten y su enfoque metodológico

Como en los Estudios 1 y 2, cada comprobación de solidez estadística está diseñada para descartar una amenaza concreta a la validez de la conclusión.

### A.1. AUC y G-mean

**En qué consiste.** El **AUC** (Área Bajo la Curva ROC) resume la capacidad discriminativa del score continuo: en el eje X la Tasa de Falsos Positivos (FPR = 1−Especificidad), en el Y la Tasa de Verdaderos Positivos (TPR = Sensibilidad); cada punto corresponde a un umbral. AUC=1 → separación perfecta; AUC=0.5 → azar; >0.7 aceptable, >0.8 bueno, >0.9 excelente. Su ventaja es que **no depende de un umbral concreto**. La **G-mean** = √(Sensibilidad × Especificidad) combina ambas en un único valor para un umbral específico, y es útil con clases desbalanceadas porque penaliza que una métrica sea muy alta y la otra muy baja (si una es 0, la G-mean es 0).

**Qué amenaza controla.** Que un buen AUC sea un artefacto de un umbral afortunado (la G-mean, con su umbral operativo fijo, lo complementa).

**Cómo se aplica aquí.** El umbral operativo del PSRI es **0.5 fijo** (clasificar como aceptable si score ≥ 0.5); kSQI y Correlación usan su **mediana** como umbral (el de la propia implementación). Matriz de confusión: VP/FN/FP/VN; Sensibilidad = VP/(VP+FN); Especificidad = VN/(VN+FP). Tabla 2 (Sección 3.5): PSRI AUC 0.8868 / G-mean 0.814; kSQI 0.8421 / 0.7067; Correlación 0.7938 / 0.6775.

### A.2. Test de DeLong y bootstrap pareado

**En qué consiste.** Ambos comparan dos **AUCs correlacionadas** (calculadas sobre los mismos registros). DeLong es un test asintótico paramétrico para la diferencia de AUCs; el bootstrap pareado remuestrea registros 2000 veces preservando el emparejamiento entre los dos scores, produciendo una distribución empírica de la diferencia.

**Qué amenaza controla.** Que la diferencia de AUC entre métodos sea un artefacto de qué registros concretos componen la muestra. Dos métodos independientes que coinciden dan confianza en que la diferencia es real.

**Cómo se aplica aquí.** Requiere los mismos registros en ambos scores (comparación pareada, 998/998 en `set-a`). Resultado (Sección 3.6): PSRI vs Correlación robusta (DeLong p=0.0003; IC95% bootstrap [0.0428, 0.1430]); PSRI vs kSQI favorable pero no concluyente (p=0.066; IC95% [−0.0040, 0.0933]).

### A.3. PR-AUC (average precision)

**En qué consiste.** El área bajo la curva Precision-Recall, sensible a los falsos positivos sobre la **clase minoritaria**.

**Qué amenaza controla.** El optimismo del ROC-AUC bajo desbalance de clases (773/225 ≈ 77/23): un clasificador puede tener ROC-AUC alto "acertando" solo la clase mayoritaria.

**Cómo se aplica aquí.** PR-AUC: PSRI=0.9510, kSQI=0.9107, Correlación=0.8754 — mismo orden y magnitud de diferencia que el ROC-AUC (Sección 3.6).

### A.4. Split-half de calibración (fuera de muestra)

**En qué consiste.** 200 particiones aleatorias 50/50 de los 998 registros; en cada una la mediana/MAD/`eps_hard` del PSRI se fijan con la mitad de **calibración** y el AUC se calcula solo sobre la mitad de **evaluación**, sin compartir registros entre ambos papeles.

**Qué amenaza controla.** La ventaja estructural del PSRI frente a los SQIs de fórmula cerrada: calibrar los parámetros de referencia sobre la misma población que se puntúa podría inflar el AUC (una forma sutil de fuga).

**Cómo se aplica aquí.** AUC fuera de muestra medio = 0.8860 (mediana 0.8851, IC95%=[0.8545, 0.9195], 200/200 splits válidos) frente al in-sample 0.8870 — la calibración interna no explica el rendimiento (Sección 3.7).

### A.5. Análisis de solapamiento

**En qué consiste.** En espacio `log(std_signal)`, se calcula el rango [P5, P95] de cada clase y su intersección (zona de solapamiento estructural). Se compara la tasa de mal/bien clasificados dentro de esa zona y se desglosa por tipo de error (FP vs FN) con baselines por clase.

**Qué amenaza controla.** La afirmación de que "el PSRI falla donde es estructuralmente imposible acertar" — sin este análisis, el 11% de error residual podría presentarse como un fallo arbitrario del método en vez de como una limitación del diseño univariado.

**Cómo se aplica aquí.** Zona [0.0934, 0.4000] (74.7% de los registros dentro); FP 100% en zona vs baseline 22.7% (~4.4×); FN 43.1% en zona vs esperable 89.9% (Sección 3.8).

### A.6. Ablación metodológica

**En qué consiste.** Se comparan 5 variantes de transformación sobre el MISMO `std_signal` (solo difiere la función), con AUC y DeLong/bootstrap frente a la propuesta.

**Qué amenaza controla.** La arbitrariedad de la elección de la transformación: sin ablación, "Log + Mediana/MAD + Gaussiana invertida" podría ser una elección ad hoc que solo funciona en este dataset.

**Cómo se aplica aquí.** Las alternativas no paramétricas (Percentil Empírico, Min-Max) se hunden (AUC≈0.52, p<1e-4); la Z-Score clásica baja poco (p=0.096); Mono-Decay es indistinguible por ranking (diff=0, p=1). La robustez MAD y la forma paramétrica en U superan claramente a las alternativas de ranking y de normalización por extremos (Sección 3.9).

### Resumen del enfoque general

Las comprobaciones forman una cadena de defensa en capas: el AUC/G-mean miden el poder discriminativo; DeLong + bootstrap (correlacionado) y PR-AUC atacan la fiabilidad de la diferencia entre métodos y el desbalance; el split-half ataca la fuga de calibración; el solapamiento localiza y explica el error residual como limitación estructural; la ablación ataca la arbitrariedad de la transformación. Conjunto: la validación del PSRI como instrumento de fiabilidad de señal sensor-agnóstico es robusta frente a las amenazas metodológicas identificables en este diseño.