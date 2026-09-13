# Análisis conceptual índice PSRI

## 1. Razonamiento: tres dimensiones

Afirmar que un trial fisiológico es "poco fiable" es ambiguo y precisa definir en qué punto de la cadena de medición ha fallado algo. Para abordar este problema, se definen conceptualmente tres dimensiones distintas e independientes entre sí en esta cadena de medición:

- **Instrumento — S_estab (estabilidad intra-trial):** *"¿el sensor capturó algo físicamente plausible?"*. Se calcula tomando la desviación típica de una señal (por ejemplo, cuánto varía la frecuencia cardíaca durante una ventana de tiempo) y pasándola por una transformación en forma de U (normalización robusta en log-espacio + campana gaussiana), de modo que tanto una señal demasiado plana (electrodo desconectado) como demasiado ruidosa reciben una puntuación de fiabilidad baja; solo la variabilidad "típica" (ni mucho ni poco, comparada con el resto de la muestra) recibe puntuación alta. Es, conceptualmente, un índice de calidad de señal (*Signal Quality Index*), el mismo concepto que ya usa la ingeniería biomédica.

- **Organismo — S_coher (coherencia multimodal):** *"¿lo que registraron los sensores es una respuesta fisiológica genuina y coordinada, o es ruido de un canal aislado?"*. El cuerpo humano responde a un estímulo activador de forma sistémica: si hay *arousal* genuino, se espera que dos sistemas fisiológicos distintos (por ejemplo, corazón y piel) se desvíen juntos de su línea base. Si uno se mueve y el otro no, lo más natural es pensar que ese movimiento aislado es un artefacto. A diferencia de S_estab (que usa una forma de U para encontrar un punto dulce de variabilidad), S_coher usa un decaimiento exponencial: se busca que la discrepancia entre sistemas sea cero (coordinación perfecta), no un valor intermedio.

- **Conducta — S_cond (consistencia conductual):** *"¿estaba la persona genuinamente comprometida con la tarea?"*. Un sensor puede funcionar perfectamente y el cuerpo puede reaccionar de forma coordinada, y aun así la persona puede no estar realmente prestando atención (el equivalente a "respuestas descuidadas" en un cuestionario, tiempos de reacción imposibles, parpadeo excesivo, etc.). Su forma funcional no está fijada de antemano: se elige por la semántica del observable (§1.3).

Si la separación en tres dimensiones fuera artificial, si midieran lo mismo con distinto nombre, esperaríamos correlaciones altas entre ellas. Las correlaciones observadas son consistentes con que efectivamente capturan fuentes de varianza independientes, constituyendo un modelo de tres dimensiones. En EXIST las correlaciones son (S_estab–S_coher: r≈0.110; S_estab–S_cond: r≈0.075; S_coher–S_cond: r≈0.001) y en K-EmoCon el par S_estab–S_cond muestra una correlación *pooled* más alta que en EXIST (r=0.193 vs. 0.075), pero la descomposición between/within-subject revela que esta se debe casi enteramente a covariación entre-sujetos (ρ_between=0.373), mientras que la relación a nivel de trial es prácticamente nula (ρ_within=0.029, r²<0.001).

| Par | EXIST (Tabla 5) | K-EmoCon |
|---|---|---|
| S_estab–S_coher | 0.110 | 0.075 |
| S_estab–S_cond | 0.075 | 0.193 |
| S_coher–S_cond | 0.001 | 0.013 |

Un fallo en cualquiera de las tres dimensiones aporta razones cualitativamente distintas; una dimensión puede fallar sin que las otras dos lo hagan. Esa es la base lógica de por qué el índice no puede ser una única métrica: si lo fuera, estaría mezclando causas de invalidez que no tienen la misma naturaleza.

El enfoque adoptado aborda la falta de un índice de fiabilidad de la señal, PSRI, que permita:

1. Filtrar a los sujetos con señales ruidosas o inconsistentes, mejorando la relación señal-ruido.
2. Ponderar las anotaciones o las predicciones de los modelos en función de la fiabilidad fisiológica del sujeto.
3. Validar si las señales de alta fiabilidad correlacionan con un menor desacuerdo entre anotadores (LeWiDi) y una menor carga cognitiva aparente, aportando una métrica de evaluación centrada en el ser humano.

## 1.1. Estabilidad Intra-trial (S_estab)

*¿El sensor capturó algo físicamente plausible?*

Este es el nivel más básico con un antecedente directo y consolidado en la literatura: es, conceptualmente, un *Signal Quality Index* (SQI), el mismo concepto que ya usa la ingeniería de señales biomédicas para ECG, EEG y EDA. Un electrodo desconectado o un sensor saturado son fallos del instrumento, no del organismo ni de la conducta: ocurrirían igual aunque el sujeto estuviera perfectamente atento y su sistema nervioso respondiera con normalidad. Este nivel se valida de forma independiente y externa con el dataset PhysioNet, que ofrece un ground truth objetivo posible (expertos etiquetando "grabación técnicamente utilizable sí/no").

Sin S_estab, una señal plana por sensor caído no se distinguiría de una señal plana por relajación genuina: el índice no podría diferenciar "no hay dato" de "hay un dato válido de baja activación".

### Evolución del modelo

La formulación teórica inicial de S_estab, planteada como coeficiente de variación normalizado, presentaba dos problemas al aplicarse a los datos reales de EXIST: (i) medias (µ valor promedio) cercanas a cero en algunas variables producían coeficientes de variación extremos y no acotados, y (ii) las tres modalidades operan en escalas físicas incomparables (bpm, mm, µV²), lo que distorsionaba cualquier combinación directa entre ellas.

Se sustituyó el coeficiente de variación por una transformación en dos fases (log-MAD + gaussiana), aplicada directamente a cada valor de la desviación estándar intra-trial en cada modalidad o tipo de señal fisiológica, independientemente del dispositivo. En el caso de EXIST son dos modalidades: a) `garmin_hr_std` para HR y b) la desviación estándar del diámetro pupilar izquierdo para ET (`3d_eye_states_pupil diameter left [mm]_std`).

**Fase 0: Desviación estándar intra-trial.**

La desviación estándar intra-trial mide la variabilidad de la señal en el intervalo. En EXIST ya vienen calculadas en el dataset (`garmin_hr_std` y `3d_eye_states_pupil diameter left [mm]_std`); en PhysioNet y en K-EmoCon se calculan a partir de las señales.

Alternativas consideradas:

- **Rango intercuartílico (IQR):** mide la dispersión del 50% central de los datos e ignora el 25% superior e inferior. Nosotros aprovechamos todas las muestras del intervalo; la robustez frente a atípicos se maneja después, en la normalización.
- **Media de la desviación absoluta:** es menos sensible a valores extremos pues penaliza las fluctuaciones grandes, pero es menos habitual en análisis de señales fisiológicas. Somos más sensibles a aumentos de variabilidad, lo que es útil en señales ruidosas; la robustez se trata después adecuadamente los valores atípicos.
- **Entropía:** cuantifica el grado de complejidad o irregularidad; es útil si el objetivo es analizar la complejidad dinámica de una señal fisiológica, pero nosotros queremos medir variabilidad. Mide la impredictibilidad de los patrones.

**Fase 1: Normalización robusta en espacio logarítmico.**

Alternativas transformadas consideradas:

- **Transformación de raíz cuadrada:** reduce la dispersión de los valores de forma más suave que el logaritmo, aunque suele ser menos eficaz cuando la distribución presenta una fuerte asimetría.
- **Transformación Box-Cox:** ajusta la transformación a los datos mediante un parámetro, pero requiere estimar dicho parámetro para cada conjunto de datos.
- **Transformación Yeo-Johnson:** similar a Box-Cox, con la ventaja de admitir valores nulos o negativos, aunque también requiere la estimación de un parámetro.

La **transformación logarítmica** ofrece un equilibrio adecuado entre eficacia y generalidad. Reduce la asimetría de la distribución, atenúa la influencia de valores extremadamente altos y no introduce parámetros que deban ajustarse para cada conjunto de datos. Estas características la convierten en una etapa adecuada para preparar las medidas de variabilidad antes de la normalización robusta y facilitar la comparación entre sensores con escalas de variabilidad heterogéneas. Modifica la forma de la distribución de la medida de variabilidad comprimiendo los valores elevados y reduciendo la asimetría, preparando los datos para una normalización robusta posterior.

**Normalización por MAD y mediana.**

Se realiza una normalización robusta mediante la **mediana** y la **desviación absoluta mediana (MAD)**. El objetivo es estandarizar (centrar y escalar) las medidas de variabilidad minimizando la influencia de valores atípicos y permitiendo la comparación entre sensores con distribuciones y escalas diferentes. Se utiliza la mediana porque es robusta frente a valores extremos, y la MAD porque mide la dispersión de forma resistente a valores atípicos.

Alternativas consideradas:

- **Normalización por media y desviación estándar (z-score):** es común, pero sensible a valores extremos, lo que puede distorsionar la normalización con señales fisiológicas ruidosas.
- **Normalización min-max:** transforma los datos a un intervalo fijo, pero depende completamente de los valores mínimo y máximo, siendo muy sensible a un único valor extremo.
- **Escalado por percentiles:** no proporciona una medida explícita de dispersión equivalente a la MAD ni es tan habitual cuando se busca una puntuación robusta y comparable entre múltiples sensores.

En conjunto, la mediana y la MAD ofrecen una normalización resistente a artefactos, especialmente adecuada para señales fisiológicas. Esta elección es coherente con el resto del diseño del índice: primero se usa la desviación estándar para capturar toda la variabilidad de la señal, después el logaritmo para estabilizar la distribución y, finalmente, la mediana y la MAD para hacer la normalización robusta. Cada paso cumple una función distinta y complementaria.

Para una modalidad (m) y desviación intra-trial σ, la transformación es:

```
log(σ + ε)
z = (log(σ + ε) - mediana(log)) / MAD(log)
```

donde ε es una constante pequeña para evitar log(0), y existe un umbral "duro" `eps_hard`: si la señal es menor que 1e-6, no se hace el logaritmo y se asigna directamente una puntuación de PSRI de 0.0.

El uso de mediana y MAD en lugar de media y desviación estándar evita que un subconjunto de sensores defectuosos desplace el centro de referencia, y el trabajo en espacio logarítmico hace comparables las desviaciones "demasiado plana" y "demasiado ruidosa" en una escala simétrica.

**Fase 2: Función en U (campana gaussiana).**

Transforma cada valor de variabilidad normalizado en una puntuación de calidad entre 0 y 1. La fiabilidad se define como:

```
R = exp(-0.5 * z²)
```

Esta forma funcional, a diferencia de una transformación monótona, penaliza simétricamente ambos extremos: una señal excesivamente plana (potencialmente un electrodo desconectado) recibe una fiabilidad tan baja como una señal excesivamente ruidosa, en lugar de recibir la fiabilidad máxima como ocurriría con una transformación monótona ingenua.

En el caso degenerado, para σ por debajo de un umbral físico (`eps_hard`, percentil 1 de la distribución de σ para esa modalidad), se asigna R=0 directamente, sin depender de que la curva gaussiana lo capture por sí sola.

En el caso de EXIST, esta misma función (`compute_psri_gaussian_log`) se aplica de forma independiente a `garmin_hr_std` (obteniendo `PSRI_hr_subj`) y a la desviación estándar del diámetro pupilar (obteniendo `PSRI_et_subj`) por cada par (sujeto, meme), y se calcula el valor de S_estab a nivel de meme como la media entre los sujetos que vieron cada meme.

## 1.2. Coherencia Multimodal (S_coher)

*Dado que el sensor funciona, ¿lo que registró es una respuesta fisiológica genuina y coordinada, o es ruido específico de un canal que casualmente parece señal?*

Un sensor puede funcionar perfectamente (sin fallo técnico, sin caída de impedancia) y aun así registrar una variación que no tiene origen cognitivo: un reflejo pupilar a un cambio de luminancia del propio meme, un movimiento del sujeto que altera transitoriamente la HR sin relación con el contenido. S_estab, midiendo un solo canal, es ciego a esta distinción: una desviación "típica" en magnitud pasa como fiable, tenga o no origen cognitivo real.

La justificación biológica de por qué la coherencia entre canales sirve como filtro es un principio bien establecido en psicofisiología: la respuesta ante un estímulo activador no es un evento aislado en un órgano, es una respuesta sistémica orquestada por el sistema nervioso autónomo. Ante *arousal* genuino, se espera covariación entre las modalidades (por ejemplo, pupila y HR en EXIST). Si un canal se mueve y el otro no, lo más lógico es que el canal que se movió lo haya hecho por una causa no compartida (artefacto, reflejo local), y no por procesamiento cognitivo del estímulo.

Sin S_coher, cualquier desviación "de magnitud normal" en un único canal se contaría como señal fiable, aunque su origen no tuviera nada que ver con la percepción del meme. Bradley y Lang sobre emoción y motivación aportan la base empírica clásica de por qué HR, conductancia de la piel y pupila covarían como índices convergentes de activación motivacional/*arousal*.

Tal como señalan Gambarotta et al. (2016) en su revisión sobre calidad de señal cardiorrespiratoria, la mayoría de los métodos existentes se diseñan para contextos clínicos muy específicos (UCI, telemedicina) y dependen de características propias de la señal, como la morfología del QRS, la energía espectral o la detección de picos mediante algoritmos supervisados (Clifford et al., 2012; Jekova et al., 2012). Esta dependencia del dominio limita su aplicabilidad a entornos con sensores heterogéneos, como los wearables o los estudios de neurociencia afectiva. En contraposición, nuestro PSRI propone un índice de fiabilidad sensor-agnóstico, validado externamente sobre ECG (PhysioNet) y aplicado a señales de HR y pupila (EXIST) así como en K-EmoCon, que no requiere entrenamiento ni conocimiento previo de la morfología de la señal, sino que se fundamenta en un principio estadístico universal: la transformación en U de la variabilidad respecto a una referencia robusta (mediana/MAD).

La construcción de este índice se basa en que, para cada trial (sujeto *i*, meme *j*), se calcula la desviación de la media de HR y del diámetro pupilar respecto a sus líneas base individuales, normalizadas por la desviación estándar de las medias de ese sujeto a través de todos sus trials.

Sea M<sub>i,j</sub> la media de la señal del sujeto *i* en el trial *j*, B<sub>i</sub> su línea base previa, y σ<sub>i</sub> la desviación estándar de las medias de ese sujeto a través de todos sus trials. El Z-score de desviación respecto al estado basal se define como:

```
z = (M_i,j - B_i) / σ_i
```

donde N es el número total de trials que realizó el sujeto *i* y M̄<sub>i</sub> es la media global de ese sujeto en toda la sesión.

Las equivalencias de los campos por modalidad y dataset son:

| Dataset | M_{i,j} (Media del trial actual) | B_i (Línea base previa) | σ_i (Desv. estándar de los trials) |
|---|---|---|---|
| EXIST (HR) | `garmin_hr_mean` | `garmin_hr_mean_baseline_prev` | Desv. estándar de `garmin_hr_mean` por `username` |
| EXIST (ET) | `3d_eye_states_pupil diameter left [mm]_mean` | `3d_eye_states_pupil diameter left [mm]_mean_baseline_prev` | Desv. estándar de la media pupilar por `username` |
| K-EmoCon (HR) | `hr_mean` (Agregado de E4) | `baseline_prev` (o `baseline_prev5`) | `subject_std` (calculado por pid) |
| K-EmoCon (EDA) | `eda_mean` (Agregado de E4) | `baseline_prev` (o `baseline_prev5`) | `subject_std` (calculado por pid) |

La coherencia se define como una exponencial negativa de la discrepancia entre ambos Z-scores:

```
S_coher = exp(-|z_a - z_b| / scale)
```

Esta forma, a diferencia de una versión lineal inicialmente considerada, garantiza un rango acotado en [0,1] sin producir valores negativos cuando la discrepancia entre Z-scores es grande, y decae suavemente en lugar de truncarse.

A diferencia de S_estab (que utiliza una función en U o campana gaussiana porque busca un "punto adecuado" de variabilidad), S_coher utiliza un decaimiento exponencial monótono. Para medir la coherencia no buscamos un valor "típico"; buscamos que la discrepancia sea exactamente cero. Si la discrepancia es 0, la exponencial da 1 (coherencia máxima). Si los sistemas se descoordinan, la exponencial decae suavemente hacia 0 sin producir valores negativos, garantizando un rango acotado en [0,1].

La diferencia de cálculo entre los dos índices reside en:

- **S_estab** utiliza una campana gaussiana: el cuadrado genera una forma de "U" que castiga simétricamente los extremos; busca un punto dulce de variabilidad y aplasta a cero cualquier señal que se desvíe drásticamente del centro (ruido severo).
- **S_coher** utiliza un decaimiento exponencial: al omitir el cuadrado y usar la distancia lineal absoluta, genera una forma de "V" con un pico ideal en el cero (coordinación perfecta) que decae de forma suave y tolerante. Esto es intencional: dos sistemas fisiológicos distintos (por ejemplo, cardiovascular y simpático) no responden en milisegundos exactos de sincronía. Una caída exponencial permite penalizar la descoordinación sin tratar una pequeña asincronía fisiológica natural como si fuera un fallo catastrófico de sensor.

El valor a nivel de meme se obtiene promediando S_coher entre los sujetos que vieron ese meme.

### 1.2.1. Latencia de sensores: ¿hay que modelarla en S_coher?

S_coher compara dos sistemas fisiológicos *en la misma ventana temporal*. Eso presupone que las dos señales están alineadas en el tiempo: si un canal comenzara su medición real con una latencia no contemplada (o llevara un reloj propio desviado), se estarían comparando sistemas sobre rejillas desplazadas y S_coher mediría desalineación, no coordinación. La cuestión conceptual tiene dos fuentes de "latencia" bien distintas, que conviene no confundir:

1. **Desfase de reloj/dispositivo (hardware).** Ocurre cuando las dos señales provienen de **dispositivos independientes** (en EXIST, HR de Garmin y pupilómetro; en K-EmoCon, E4 frente a NeuroSky/Polar). Ahí sí existe un offset real entre relojes y debe verificarse (o corregirse) antes de computar S_coher.
2. **Latencia fisiológica de respuesta.** El SNA orquesta la respuesta sistémica, pero cada efector tiene su propia dinámica: la respuesta de conductancia de la piel (EDA) es más lenta que la cardiovascular. Es asincronía *de los sistemas*, no de los instrumentos, y está dentro del propio concepto de "coordinación con tolerancia".

El diseño de S_coher ya incorpora una decisión explícita sobre el punto 2: el decaimiento exponencial en V (discrepancia en valor absoluto, sin cuadrado) penaliza suavemente una pequeña asincronía fisiológica natural en lugar de tratarla como un fallo catastrófico (§1.2, nota sobre el decaimiento exponencial). Un desfase pequeño constante entre dos sistemas bien sincronizados degrada la coherencia de forma acotada y sin invertir su signo; solo un desfase *sistemático y grande* (del orden de la ventana, 5 s en K-EmoCon) podría confundir la interpretación.

La cuestión empírica (¿existe un desfase sistemático sin corregir?) se validó en K-EmoCon con un diagnóstico dedicado (`sources/kemocon/sensor_latency.py`, cuatro chequeos: latencia de arranque por canal, alineación de rejillas de muestreo, cross-correlación HR-EDA y sensibilidad al lag del resultado confirmatorio; resumen en EstudioKemocon.md §3.11). Conclusiones:

- **Dentro del mismo dispositivo no hay offset persistente.** En el E4, los canales comparten un único reloj: HR y EDA muestrean sobre la misma rejilla temporal (100% de coincidencia exacta de timestamps) y las únicas demoras de arranque (HR +10 s, IBI variable) son calentamiento de canales *derivados* (BVP→HR) que cae en el tramo pre-debate, fuera de las ventanas de análisis.
- **El efecto confirmatorio de S_coher es robusto a la latencia plausible.** El pico de la asociación within-de-sujeto S_coher↔desacuerdo está en el lag 0 y sobrevive desplazamientos de ±1 ventana (±5 s), decayendo solo a ±2-3.
- **El riesgo real es cross-device.** Las señales NeuroSky/Polar empiezan 0-396 s después del E4 (mediana ~112 s): cualquier par S_coher *entre dispositivos* exige verificar la alineación (cross-correlación o grid-check) antes de interpretar la coherencia. Es el caso de EXIST (HR Garmin × pupilómetro), donde esta verificación es parte del protocolo de integración.

Lectura conceptual: **modelar la latencia no es un requisito universal del índice, sino una verificación obligatoria por pares de sensores** — automática cuando el par comparte reloj (K-EmoCon HR-EDA), y explícita cuando los sistemas provienen de dispositivos independientes. El diagnóstico queda como la comprobación de latencia de la **auditoría PSRI**, reutilizable en cualquier nueva instanciación de S_coher, no como una corrección aplicada a los resultados actuales.

## 1.3. Consistencia Conductual (S_cond)

*Dado que el sensor funciona y la respuesta fisiológica es internamente coherente, ¿el sujeto estaba realmente comprometido con la tarea de ver y procesar el meme?*

Esta es la dimensión más "externa" a la fisiología; es, de hecho, un criterio de validez conductual de la sesión, análogo al concepto establecido en metodología experimental y encuestas de "respuesta poco cuidadosa" o *insufficient effort responding* (Meade & Craig, 2012): el mismo principio detrás de por qué se descartan respuestas *straight-liner* en cuestionarios, o tiempos de reacción imposiblemente cortos en tareas cognitivas. Un sujeto puede tener sensores intactos y una respuesta cardiopupilar internamente coherente, y aun así no haber estado realmente atendiendo al estímulo, por ejemplo, si respondió por puro reflejo motor sin mirar la imagen. En ese caso, S_estab y S_coher podrían salir altos por pura casualidad estadística, y solo un indicador conductual (tiempo de reacción, tasa de parpadeo) puede detectarlo.

Un sujeto genuinamente comprometido produce valores de sus observables conductuales coherentes con la tarea. Sin S_cond, un trial "técnicamente limpio y fisiológicamente coordinado" pero conductualmente inválido (el sujeto ni siquiera procesó el estímulo) pasaría como plenamente fiable.

A diferencia de S_estab, **S_cond no presupone una forma funcional única**: la forma es una herramienta que el método selecciona según la semántica del observable.

- **Observable con interpretación unívoca → transformación monotónica.** Cuando un solo extremo es deseable, la fiabilidad es monótona en esa dirección: un tiempo de reacción más corto indica procesamiento más fluido, no descuido ("menos es mejor"); una métrica de atención más alta indica más compromiso ("más es mejor"). Penalizar también el extremo deseable destruiría señal real.
- **Observable con ambos extremos patológicos → función en U.** Cuando exceso y defecto son igualmente inválidos (una señal plana como un electrodo caído, una tasa de parpadeo anómalamente alta como fatiga), se usa la campana gaussiana en log-espacio (`compute_psri_gaussian_log`), que penaliza simétricamente los dos lados.

La regla se decide a priori por la semántica del observable, no por el resultado estadístico. La U queda así anclada a su dominio de validación externa (S_estab, PhysioNet, §1.1); S_cond opera sobre la caja de herramientas con el criterio que cada contexto dicta. El método es el objeto y la pregunta ("¿estaba la persona comprometida con la tarea?"); la fórmula es subordinada a ambos.

En EXIST, para cada sujeto *i* en el trial *j*, se computaron ambas herramientas sobre las dos métricas de eye tracking disponibles:

- `reaction_time` (tiempo de reacción en la tarea) → herramienta monotónica "menos es mejor" (`S_cond_mono`).
- `blinks_count` (conteo de parpadeos durante el trial) → herramienta en U.

Obteniendo dos puntuaciones de fiabilidad conductual parciales. El componente de consistencia conductual a nivel de sujeto se define como la media aritmética de ambas puntuaciones:

```
S_cond = (S_cond_rt + S_cond_blink) / 2
```

Al carecer de seguimiento ocular en K-EmoCon, la operativa de S_cond requirió buscar sustitutos conductuales entre los sensores disponibles (dispositivo NeuroSky de un canal y acelerómetro de la pulsera E4). Aplicando la misma regla de selección a los sustitutos disponibles, se probaron inicialmente cuatro variantes:

- **Variante 1 (S_cond_att_acc):** `Attention.value` (índice propietario de atención del EEG) y `acc_std` (desviación estándar del acelerómetro, como proxy de inestabilidad motora).
- **Variante 2 (S_cond_att_med):** `Attention.value` y `Meditation.value` (ambas de NeuroSky, midiendo compromiso activo frente a desconexión pasiva).
- **Variante 3 (S_cond_self):** `self_valence_local_std` y `self_arousal_local_std` (variabilidad local de las auto-anotaciones emocionales en una ventana rodante).
- **Variante 4 (S_cond_combined):** media de todas las métricas R disponibles para el sujeto.

Posteriormente se añadieron tres variantes de control durante el análisis exploratorio: `S_cond_att` (solo Attention), `S_cond_att_std` (Attention sobre std en vez de mean) y `S_cond_att_med_std` (Attention + Meditation, ambas sobre std).

Como se detalla en el estudio K-EmoCon (EstudioKemocon.md §3.4 y §3.9), el resultado confirma la regla y sus consecuencias: la Variante 3 (`S_cond_self`) produjo correlaciones artificialmente enormes pero inválidas (variable casi binaria); las variantes en forma U sobre atención/movimiento aportan señal pooled (`S_cond_att_acc` ρ≈−0.12 sobre `external_valence_var`) y, sobre todo, **sostienen la validez de tarea** (distinguen reposo de debate), mientras que las variantes monotónicas recuperan señal within-de-sujeto que la U enmascara (ρ_within≈−0.050, p=0.009, dirección de la hipótesis) aunque no robusta a LOSO y a costa de perder esa validez de tarea. Ninguna variante de S_cond sustenta un efecto within-de-sujeto robusto sobre el desacuerdo en K-EmoCon. La razón es estructural y el framework la nombra sin ambigüedad: K-EmoCon carece de observables conductuales directos (no hay seguimiento ocular ni tiempos de reacción de la tarea), la condición exacta que la caja de herramientas necesita para operar con su herramienta más natural. S_cond se documenta en K-EmoCon como limitación de disponibilidad de sensores, no como fallo de la dimensión.

Finalmente, análogamente a los demás componentes, el valor a nivel de meme se obtiene promediando entre los sujetos que vieron ese meme.

**La regla se confirma en EXIST (comprobación de cierre de fase — tecnico_exist.md §8):** el tiempo de reacción es el observable conductual directo del diseño y, por su semántica unívoca ("menos es mejor"), su herramienta natural es la monotónica. Operativizado así (`S_cond_mono`, tiempo de reacción más corto = procesamiento más fluido), la media de RT por meme correlaciona fuertemente con el desacuerdo anotador (ρ=−0.17 frente a `entropy_22/23`, p≈1e-28, sobrevive Bonferroni en un pool de 66 tests) y con la valoración de sexismo (ρ=−0.14 frente a `soft_21_yes`; memes sexistas se procesan más despacio: mediana de RT 14.0 s en YES vs 11.7 s en NO). La función en U sobre el mismo observable —la herramienta inadecuada para una variable con dirección unívoca— penaliza el RT rápido como "respuesta descuidada" y borra la señal (todas las correlaciones nulas). Elegir la herramienta por la semántica del observable no es, por tanto, un refinamiento cosmético: es lo que separa la detección de la señal real de su borrado. La lectura causal es la causa común indirecta del diseño EXIST (el contenido del meme produce a la vez procesamiento lento y desacuerdo/YES), y el hallazgo encaja con la interpretación S_obs (§1.3.1): un estímulo más fluido de procesar es más observable y genera más consenso. El resultado nace de una comprobación de cierre de fase y se documenta como diagnóstico, no como confirmatorio del bloque principal.

### 1.3.1. La alternativa S_obs (observabilidad humana) — docs/dialogo.txt

docs/dialogo.txt planteó una reorganización conceptual de la tercera pata: en vez de `Instrumento → Organismo → Conducta`, cabía `Instrumento → Organismo → Percepción`. Bajo esta lectura, la tercera dimensión sería **S_obs (human-observability / perceptual interpretability)**: *hasta qué punto la respuesta del sujeto es observable e interpretable por terceros*. La justificación es directa: si una señal fisiológica inestable o descoordinada se asocia con MÁS desacuerdo externo, eso puede leerse como "una señal menos interpretable genera menos consenso", es decir, el desacuerdo anotador actúa como la medida inversa de la observabilidad del estado emocional.

Esta alternativa NO se implementó como componente computable independiente, por dos razones metodológicas explícitas:

1. **Coincide operativamente con la relación ya medida.** En K-EmoCon, la "observabilidad" no tiene un canal propio: la única evidencia disponible es la asociación entre S_estab/S_coher y el desacuerdo externo (`external_valence_var`), que es exactamente el resultado confirmatorio del estudio (Sección 3.6 de EstudioKemocon.md). Convertir esa asociación en un tercer componente sería circular — se validaría S_obs contra la misma métrica que la define.
2. **El desacuerdo es la variable de resultado, no un insumo.** El framework usa `external_valence_var` como target (¿predice la fiabilidad el desacuerdo?), y el diálogo recomendó explícitamente mantenerlo así: "usa external_valence_var como resultado de observabilidad/desacuerdo humano, no como componente conductual".

Por tanto, S_obs queda documentada como **interpretación conceptual del hallazgo**, no como una cuarta pata: la asociación convergente entre menor fiabilidad fisiológica y mayor desacuerdo humano es consistente con que la fiabilidad sea un determinante de la *observabilidad perceptual* del sujeto. Esa lectura refuerza la contribución (calidad de señal → interpretabilidad humana), sin añadir un componente que duplicaría la señal ya medida.

### 1.3.2. S_cond y la validación a nivel de ventana de tarea (Q_window)

La orientación conceptual de S_cond tiene una consecuencia metodológica que conviene hacer explícita: por ser la **única dimensión con referente de tarea** (no de señal), su validación no puede hacerse a nivel de señal (Q_physio), sino a nivel de ventana de tarea — **Q_window** en el marco de tres niveles de validez (metodologia_esqueleto.md §6.1). Para saber si una métrica conductual (atención, compromiso) es válida hay que saber *cuándo* el sujeto está realizando la tarea.

La feature table ancla la ventana 0 al inicio de grabación del sensor (`initTime`); etiquetar cada ventana como reposo o tarea requiere los eventos de tarea (`startTime`/`endTime`) de la metadata del dataset. K-EmoCon los proporciona en `metadata/subjects.csv`, de modo que la asignación pre/debate por ventana es directa (EstudioKemocon.md §3.7 y §3.8). Esta separación entre calidad de señal (Q_physio) y condición de tarea (Q_window) evita que una sesión con buena señal se lea como una sesión de alta atención.

## 1.4. PSRI Compuesto: Combinación Ponderada

El PSRI compuesto se define como la suma ponderada de las tres dimensiones:

```
PSRI = w1 * S_estab + w2 * S_coher + w3 * S_cond
```

En este trabajo se emplean **pesos iguales** (1/3, 1/3, 1/3) como elección de diseño explícita, no derivada empíricamente ni ajustada mediante validación cruzada. Esta decisión se apoya en la evidencia de que las correlaciones cruzadas entre S_estab, S_coher y S_cond son bajas (≤0.110), lo que indica que ninguna dimensión es redundante con las otras dos y que, en ausencia de un criterio empírico de ponderación, una combinación equitativa es una decisión razonable.

En el caso de EXIST, el EEG queda excluido de esta combinación: no dispone de métrica de estabilidad intra-trial y no comparte sujeto ni trial con HR/ET, por lo que no puede contribuir a S_coher.

A continuación exponemos por qué otras opciones razonables no entraron en nuestro modelo:

- **Consenso entre sujetos (la Versión 1 del PSRI, ya descartada):** mide si la gente se parece entre sí, no si un sujeto concreto está bien medido. Es un error de categoría: confunde heterogeneidad poblacional genuina (personas que reaccionan distinto de forma legítima) con mala calidad de instrumento. Se descartó precisamente por esto.
- **Magnitud absoluta de la respuesta ("más dilatación pupilar = mejor señal"):** se rechazó porque confundiría intensidad de *arousal* con fiabilidad. Una respuesta pequeña pero consistente es tan fiable como una grande; un pico enorme puede ser exactamente el artefacto que queremos detectar, no la señal deseable.
- **Un único índice global de SNR:** técnicamente posible, pero destruiría el valor diagnóstico del marco. Si el objetivo final incluye filtrar o ponderar sujetos/trials, necesitamos saber *por qué* un trial es poco fiable —¿sensor roto? ¿respuesta incoherente? ¿desatención?— porque cada causa sugiere una acción distinta (descartar el sujeto entero, revisar el trial concreto, o simplemente no confiar en ese meme). Colapsar todo en un número perdería esa capacidad diagnóstica.
- **Calidad espectral/temporal fina (por ejemplo, artefactos de banda específica en EEG):** conceptualmente válida, pero inviable con los datos agregados de EXIST (no hay series temporales crudas). Superar esta limitación nos ha permitido la construcción de un índice sensor-agnóstico.

Resumen de los campos utilizados:

| Componente | EXIST | PhysioNet | K-EmoCon |
|---|---|---|---|
| **S_estab** | `garmin_hr_std`, `3d_eye_states_pupil diameter left [mm]_std` | `std_signal` (y `rr_std` como sensibilidad) | `hr_std`, `eda_std`, `bvp_std`, etc. |
| **S_coher** | `garmin_hr_mean_baseline_prev` × `pupil_mean_baseline_prev` | (no aplica — validación de un solo componente) | HR (E4) × EDA |
| **S_cond** | `reaction_time`, `blinks_count` | (no aplica) | – |

## 1.5. La auditoría PSRI: fiabilidad de la unidad de análisis (el caso EXIST)

El PSRI es un **método, no un índice suelto**: cada vez que se instancia activa la **auditoría PSRI**, que exige comprobar los pre-requisitos de su claim. Cuando la unidad de análisis (el trial, la ventana, el meme) es un **agregado sobre pocos sujetos**, el pre-requisito crítico es que ese agregado tenga **señal reproducible propia**. Si el agregado es solo ruido promediado sobre una muestra mínima, cualquier correlación con una variable de esa unidad es imposible por diseño — exista o no la relación real entre fiabilidad fisiológica y el constructo de interés. Esta es la segunda comprobación de la auditoría PSRI, junto a la ya documentada **latencia de sensores** en S_coher (§1.2.1, cuándo dos señales comparten inicio temporal). La nueva es la de **agregación**: ¿el agregado por unidad de análisis transporta señal, o solo identidad de sujeto y ruido?

En EXIST la auditoría de agregación es la que convierte la *desalineación poblacional* (un argumento conceptual: generadores de señal y jueces son disjuntos) en un hecho **medido con los propios datos del PSRI, sin etiquetas** (`sources/exist/meme_reliability.py`, resumen en EstudioEXIST.md §3.8). Resultados:

1. **Estructura**: cada meme es visto por como máximo 2 sujetos (media 1.9). El agregado por meme es la media de 1–2 viewers.
2. **No hay señal de meme en la fisiología**: la descomposición de varianza da ICC_meme≈0 neta de rasgo de sujeto (HR cruda es ~80% rasgo de sujeto y 0% meme; pupila ~65%/4%; la fiabilidad de la media del `PSRI_trial`, Spearman-Brown, es ≈0.06). La correlación entre los 2 viewers del mismo meme es ≈0 para todo lo fisiológico.
3. **El contraste que valida el diagnóstico**: los observables **conductuales** (tiempo de reacción, r=+0.45 entre viewers) sí tienen señal de meme reproducible — y son exactamente los que producen el único hallazgo robusto de EXIST (`S_cond_mono`). El diagnóstico discrimina: marca como "desconectado" un diseño cuya fisiología no transporta señal a nivel de meme, y "conectado" un canal que sí la transporta.
4. **La potencia nominal engaña**: N_eff≈4000 daría |r|≥0.045 detectable, pero al descontar la atenuación por la baja fiabilidad del agregado, el efecto *verdadero* mínimo detectable es ≈0.19 — y la fiabilidad medida es ≈0.

La lección conceptual es la misma que en la otra auditoría: **los controles emergen de la lógica del método, no de haber sospechado el fallo concreto**. La auditoría de agregación no se construyó para "probar la desalineación", sino porque aplicar el PSRI a una unidad de análisis agregada exige comprobarlo; el resultado (fiabilidad ≈0) transforma el nulo de EXIST en una propiedad de diseño medida, y refuerza la decisión de diseñar K-EmoCon con la población alineada.

**Frontera:** esto es una propiedad del **canal fisiológico agregado**, no de la conducta de anotación (grupo independiente) ni del compromiso del sujeto; no afirma que el sujeto no vea o no reaccione al meme, sino que su reacción no es reproducible entre viewers.

## 2. Evolución del Modelo

La formulación aquí presentada es el resultado de un proceso iterativo de corrección, no la primera implementación ensayada. En particular: (i) una primera versión de S_estab basada en coeficiente de variación entre sujetos —en lugar de estabilidad intra-sujeto— medía consenso poblacional, no fiabilidad de señal, y fue descartada; (ii) la transformación monótona asignaba fiabilidad máxima a señales planas, y fue sustituida por la función en U; (iii) la primera formulación de S_coher, calculada entre trials de un mismo sujeto, carecía de varianza real entre memes y fue sustituida por la versión a nivel de trial. Estas correcciones, varias de ellas motivadas por la validación externa descrita en la Validación PhysioNet, se documentan con mayor detalle en el material suplementario de este trabajo.

El PSRI indica: *"¿cuánto se parece la variabilidad de esta señal a la variabilidad típica de la población de referencia?"*.

- **PSRI = 1** → señal típica, dentro del rango esperado.
- **PSRI = 0** → señal muy atípica (por exceso o defecto de variabilidad).

Por tanto, el PSRI no es un indicador de "bueno" o "malo" en sí mismo, sino de "normal" o "anómalo" respecto a la referencia.

## 2.1. Definición del umbral

De cara a definir el umbral de "anormalidad", en PhysioNet existe ground truth pues se disponen las etiquetas (acceptable/unacceptable) dadas por expertos. En aplicaciones donde no tenemos etiquetas de calidad (por ejemplo, en un sistema en producción), se presentan varias opciones:

- **Umbral por percentil:** fijar, por ejemplo, el percentil 5 o 10 de la distribución del PSRI en la población utilizada. Se consideran "atípicas" las señales que caen por debajo de ese percentil. Esto no es arbitrario si se elige un percentil basado en la literatura o en requisitos operativos (por ejemplo, descartar el 5% de las señales más ruidosas).
- **Uso continuo, sin umbral:** en muchos casos no es preciso un umbral binario. Puede usarse el PSRI como un peso o una confianza. Por ejemplo, en EXIST, si se quisiera ponderar las anotaciones, se usaría el PSRI directamente como peso, sin necesidad de decidir qué es "bueno" o "malo".

Hay que tener en cuenta que la referencia debe ser representativa del dominio donde va a aplicarse el índice, y el umbral debe calibrarse con datos del mismo dominio, no con una referencia genérica. Si cambia el dominio (por ejemplo, se pasa de ECG clínico a señales de pulsera deportiva), la referencia y el umbral deben recalcularse.

En la práctica, el umbral no es el centro de la contribución. En este trabajo el PSRI se usa de varias formas:

- **Como score continuo para correlacionar con entropía:** no hay umbral, solo correlación.
- **Como filtro en clasificación:** pueden probarse varios umbrales y ver cuál da mejor rendimiento. El umbral se convierte en un hiperparámetro que se optimiza en validación.
- **Como peso para ponderar anotaciones:** no es preciso umbral, se usa el PSRI directamente.

El umbral binario solo es necesario si se quiere tomar una decisión categórica ("aceptar" o "rechazar" una señal). Y en ese caso debe justificarse con base en el ground truth o en un criterio operativo. No es arbitrario si se optimiza sobre datos etiquetados.

## 2.2. Justificación del modelo

La elección de una transformación en forma de U mediante una campana gaussiana no es arbitraria, sino que responde a dos principios matemáticos consolidados en la literatura:

**Desde la estadística robusta:** la formulación `exp(-0.5 z²)` es matemáticamente equivalente a la inversa de la función de pérdida de Welsch (o Leclerc), ampliamente utilizada en M-estimadores robustos (Maronna et al., 2006). En este contexto, el PSRI actúa como un "peso de Welsch": en lugar de descartar muestras atípicas de forma dura (umbral binario), asigna un peso continuo que decae suavemente, garantizando que una señal degenerada no aporte fiabilidad sin introducir discontinuidades en el índice.

**Desde la lógica difusa (Fuzzy Logic):** el PSRI puede interpretarse como una función de membresía gaussiana (Klir & Yuan, 1995) que mide el grado de pertenencia de una señal al conjunto difuso "señal de calidad típica". El valor z=0 (variabilidad igual a la mediana de la población) representa el prototipo ideal (membresía = 1), mientras que las desviaciones extremas (plana o ruidosa) reducen el grado de pertenencia sin alcanzar un corte abrupto de tipo "verdadero/falso".

Esta doble fundamentación justifica que, a diferencia de un índice de SNR (Relación Señal-Ruido) lineal o un simple filtro de umbrales, el PSRI maneje la incertidumbre de la calidad de la señal de forma continua y acotada en [0, 1].

## Referencias

- Bradley, M. M., & Lang, P. J. (2007). Emotion and motivation. En *Handbook of Psychophysiology* (3rd ed., pp. 581–607). Cambridge University Press. https://doi.org/10.1017/CBO9780511546396.025
- Gambarotta, N., Aletti, F., Baselli, G., & Ferrario, M. (2016). A review of methods for the signal quality assessment to improve reliability of heart rate and blood pressures derived parameters. *Medical & Biological Engineering & Computing*, 54, 1025–1035. https://doi.org/10.1007/s11517-016-1453-5
- Meade, A. W., & Craig, S. B. (2012). Identifying careless responses in survey data. *Psychological Methods*, 17(3), 437–455. https://doi.org/10.1037/a0028085
- Maronna, R. A., Martin, R. D., & Yohai, V. J. (2006). *Robust statistics: theory and methods*. John Wiley & Sons.
- Klir, G. J., & Yuan, B. (1995). *Fuzzy sets and fuzzy logic: theory and applications*. Prentice Hall.
