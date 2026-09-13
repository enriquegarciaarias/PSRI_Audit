# Estudio 2 K-EmoCon

> Documento regenerado a partir de `EstudioKemocon.pdf` y corregido para alinear cifras y descripciones con la implementación actual del pipeline (`sources/kemocon/`) y sus salidas en `results/output/KEMOCON/` (agosto 2026).

> **Nota de anclaje temporal**: en K-EmoCon las anotaciones (columna `seconds`) son relativas al inicio del debate (`startTime`) y la fisiología E4 al inicio de grabación (`initTime`). El pipeline aplica el desplazamiento entre ambos relojes antes de fusionarlas; el detalle y su verificación están en la **Sección 3.7**.

## 1. Introducción

El Estudio 1 (EXIST 2026) partía de una pregunta sencilla: si un sujeto ve un meme con sensores fisiológicos puestos (frecuencia cardíaca, pupila, EEG), ¿la fiabilidad de esa señal fisiológica predice si los anotadores humanos discreparán sobre si ese meme es sexista?

El resultado fue un nulo robusto: ninguna de las 20 pruebas de correlación sobrevivió a la corrección por comparaciones múltiples, ni con Bonferroni (el criterio más estricto) ni con FDR (el más permisivo). Esto no se debió a que el instrumento (el PSRI) fuera defectuoso — se validó de forma independiente en el Estudio 3 contra un dataset de ECG con expertos humanos etiquetando calidad (PhysioNet/CinC 2011), donde alcanzó un AUC de 0.887, superando a los índices de calidad de referencia de la literatura biomédica (kSQI, correlación inter-derivación). El instrumento funcionaba. El problema estaba en otro sitio.

Ese "otro sitio" es una característica estructural del propio dataset EXIST: los 8 sujetos que llevan los sensores y la población de anotadores que juzga el sexismo del meme son grupos completamente distintos, reclutados para roles distintos, que nunca se solapan. El sujeto A ve el meme y su corazón reacciona; el anotador B que no lleva sensores juzga, de forma independiente, si ese mismo meme es sexista. La única conexión posible entre la fisiología de A y el juicio de B es una hipótesis de causa común indirecta: "un meme ambiguo produce tanto una reacción fisiológica más ruidosa en quien lo ve, como más desacuerdo en quien lo juzga". Es una cadena causal de dos eslabones, con ruido acumulándose en ambos extremos, y el resultado nulo de EXIST es coherente con que esa cadena sea, en la práctica, demasiado débil para detectarse incluso con ~4000 memes de muestra.

La pregunta que motiva este estudio es: ¿el nulo de EXIST refleja que la hipótesis central es falsa (la fiabilidad fisiológica NO predice el desacuerdo), o refleja que el diseño de EXIST no podía haberla detectado aunque fuera cierta, por la desalineación poblacional? Para responder, hace falta un dataset donde la misma persona sea, a la vez, la fuente de la señal fisiológica y el objeto del juicio de desacuerdo.

## 2. El dataset K-EmoCon

El dataset K-EmoCon (Korean Emotional Conversation) ofrece exactamente el escenario necesario para poner a prueba la hipótesis que EXIST no pudo evaluar. En K-EmoCon, parejas de sujetos debaten temas controvertidos durante 10-30 minutos mientras llevan sensores fisiológicos puestos (pulsera Empatica E4, banda Polar y diadema NeuroSky). La diferencia estructural con EXIST es total: el participante que genera la señal fisiológica es exactamente la persona sobre la que se emiten juicios.

Tras cada debate, el estado emocional del sujeto es evaluado desde tres perspectivas: por él mismo (self), por su pareja de debate (partner) y por cinco anotadores externos (R1-R5) que evalúan la grabación. K-EmoCon mide la emoción en un espacio continuo de Valencia (positivo/negativo) y Activación/Arousal (calma/excitación). Esta estructura permite calcular métricas de desacuerdo continuas (ej. la varianza de las puntuaciones de R1-R5 en una ventana de tiempo específica) y alinearlas milimétricamente con la señal fisiológica de esa misma persona en ese mismo instante. Ya no hay una cadena causal indirecta: el vínculo entre la fiabilidad de la señal y el desacuerdo es directo.

Desde el punto de vista técnico para el cálculo del PSRI, K-EmoCon corrige las tres limitaciones estructurales de EXIST:

1. **Series temporales crudas**: Se dispone de las señales biomédicas milisegundo a milisegundo (BVP, EDA, Temperatura, Acelerómetro), lo que permite calcular la estabilidad de la señal en ventanas temporales finas (en este estudio, ventanas consecutivas de 5 segundos, sin solapamiento) en lugar de depender de un único agregado estadístico por ensayo.
2. **Líneas base relativas**: El protocolo de K-EmoCon incluye fases de reposo explícitas por sesión. En la implementación efectiva, sin embargo, la referencia de S_coher no se extrae de esas fases de reposo: se usa como baseline la media de los trials inmediatamente anteriores del propio sujeto (`baseline_prev`) junto con su variabilidad histórica — es decir, un proxy entre ensayos. Las fases de reposo quedan disponibles en el material bruto como refinamiento futuro, pero conviene no presentar el cálculo como una desviación frente al reposo.
3. **Múltiples sistemas fisiológicos**: La disponibilidad simultánea de frecuencia cardíaca (derivada de BVP/E4) y EDA (conductancia de la piel) permite calcular una coherencia cruzada (S_coher) entre dos sistemas fisiológicos genuinamente distintos (cardiovascular y simpático), superando la limitación de EXIST donde solo se disponía de una magnitud derivada (HR).

## 3. Construir el pipeline de K-EmoCon

El plan inicial era replicar la arquitectura del Estudio 1 (loader → aggregator → build_features → sanity_check), reutilizando `psri/calculator.py` tal cual. En la práctica, cada paso requirió ajustes según se fue conociendo el esquema real de los datos (algo habitual en investigación con datos reales — el plan sobre el papel casi nunca coincide con el CSV real):

- Las señales de la pulsera E4 (E4_ACC, E4_BVP, E4_EDA, E4_HR, E4_IBI, E4_TEMP) resultaron venir ya en formato tabular limpio (columnas `timestamp`, `pid`, `value` o `x,y,z`), no en el formato crudo típico de Empatica (con cabecera de frecuencia de muestreo) que se había asumido inicialmente.
- Los timestamp de E4 y de la fuente Polar/NeuroSky venían en milisegundos, no en segundos — un detalle que, de pasar desapercibido, habría desalineado silenciosamente toda la sincronización entre señales fisiológicas y anotaciones. Se implementó una detección automática por magnitud (umbral en 1e12) para normalizar a segundos de forma robusta, en vez de asumir la unidad.
- La fuente adicional (`neurosky_polar_data/<pid>/{Attention, BrainWave, Meditation, Polar_HR}.csv`) tiene cobertura muy irregular por sujeto: algunos solo tienen BrainWave, otros carecen de Polar_HR pero sí tienen Attention/Meditation, etc. Esto obligó a que cada componente del PSRI manejara de forma explícita qué fracción de las ventanas tiene un valor realmente calculado frente a imputado.
- Las tablas de calidad ya calculadas (`data_quality_tables/`) no tenían columna de identificador de sujeto — la fila N tras la cabecera correspondía al sujeto `pid=N`, y las filas `n/a` marcaban sujetos sin ese archivo. El formato de las celdas tampoco era uniforme: `e4_completeness.csv` traía fracciones numéricas directas, pero `e4_zeros.csv` y `e4_outliers.csv` venían como texto `"count (fracción)"` (p. ej. `"4007 (0.994)"`), y `e4_durations.csv` eran segundos. Cada uno necesitó su propio parser.

### 3.1. Sanity check: qué sujetos se excluyeron y por qué

Con las tablas de calidad ya interpretadas correctamente, se aplicaron cuatro criterios de exclusión, cada uno documentado y con un umbral explícito:

1. Completeness (< 90% en cualquier modalidad)
2. Fracción de ceros (> 20% en cualquier modalidad)
3. Fracción de outliers (> 10% en cualquier modalidad)
4. Ratio de duración entre modalidades (< 50% de la duración máxima del propio sujeto) — este criterio detecta caídas de señal específicas de una modalidad dentro de un sujeto por lo demás válido, no sesiones cortas en general. Por ejemplo, el sujeto pid=4 tiene ACC=984s pero IBI=54s — no es que su sesión fuera corta, es que el detector de picos de su señal IBI falló casi por completo (el mismo tipo de fallo del detector de QRS que se documentó en la validación de PhysioNet del Estudio 1).

De los 27 sujetos elegibles (con las seis modalidades E4 y anotaciones self/partner/≥3 externas disponibles), cuatro quedaron excluidos: pid 4 y pid 28 por caída de señal IBI específica; pid 17 y pid 20 por EDA prácticamente degenerada (completeness ≈ 0.0002 y 0.0). Quedan 23 sujetos y, tras el filtrado, 5528 ventanas de 5 segundos.

### 3.2. Proceso de programa

#### 3.2.1. Colisión de columnas al fusionar más de dos modalidades

La función que agrega cada señal fisiológica por ventana de 5 segundos devolvía columnas `mean`, `std` y `n_samples` con nombres genéricos, sin prefijo de canal. Al fusionar solo dos tablas, pandas resolvía la colisión de nombres añadiendo sufijos automáticos (`_x`, `_y`); al añadir una tercera tabla, esos sufijos ya no bastaban y el proceso fallaba con un error de fusión. Se corrigió prefijando cada estadístico por canal (`bvp_std`, `hr_n_samples`, etc.) dentro de la propia función de ventaneo, en vez de renombrar después.

#### 3.2.2. Duplicación silenciosa de filas por desajuste de resolución en las anotaciones

Este era más grave porque no producía ningún error visible — simplemente contaminaba los números sin avisar. La función que ventanea las anotaciones asignaba un índice de ventana de 5 segundos a cada muestra anotada, pero no agregaba las muestras dentro de esa ventana (a diferencia de la función equivalente para las señales fisiológicas, que sí lo hacía). Si la anotación bruta venía muestreada más fina que 5 segundos — algo habitual en anotación continua tipo RankTrace —, cada ventana podía tener varias filas de anotación en lugar de una sola. Al fusionar esas anotaciones con la tabla fisiológica, cada fila fisiológica se duplicaba tantas veces como muestras de anotación cayeran en esa ventana.

Este bug llevaba presente desde la primera ejecución exitosa del pipeline. Se detectó por una vía indirecta: en una ejecución posterior a añadir una fuente de datos nueva (Meditation), la correlación de S_estab con una de las métricas de desacuerdo cambió de signo y cuadruplicó su magnitud, sin que el código que calcula S_estab se hubiera tocado en absoluto. Ante un cambio así de grande sin cambio de código correspondiente, la explicación más probable es que el conjunto de filas sobre el que se calcula haya cambiado de forma no controlada — y efectivamente, al corregir la agregación de anotaciones, el número de ventanas brutas bajó de 6574 a 6524 (50 filas duplicadas eliminadas).

Curiosamente, una vez corregido, se comprobó que esas 50 filas duplicadas no eran la causa del cambio de signo que las había hecho sospechosas en primer lugar — los valores de S_estab quedaron exactamente iguales antes y después de esta corrección. La verdadera causa de aquel cambio de signo se explica en la Sección 6.4. Esto ilustra otro principio útil: encontrar y arreglar un bug real no garantiza que sea el bug que se estaba buscando; conviene seguir investigando hasta que la explicación cuadre numéricamente, no solo cualitativamente.

### 3.3. Iterando sobre S_coher: de HR-HR a HR-EDA

La primera implementación de S_coher en K-EmoCon emparejó la frecuencia cardíaca medida por la pulsera E4 con la frecuencia cardíaca medida por la banda de pecho Polar. El resultado fue un nulo limpio: ninguna correlación con las métricas de desacuerdo sobrevivió a la corrección por comparaciones múltiples.

Este resultado, sin embargo, era esperable por una razón conceptual, no solo estadística: HR (E4) y HR (Polar) son la misma magnitud fisiológica medida por dos instrumentos distintos, no dos sistemas fisiológicos distintos. S_coher está diseñado para detectar si el cuerpo responde de forma coordinada entre sistemas (p. ej. corazón y piel, o corazón y pupila, como en el Estudio 1) — comparar un instrumento contra otro que mide lo mismo es, en el mejor de los casos, un chequeo de acuerdo entre sensores (redundancia instrumental), y una eventual discrepancia reflejaría ruido de medición o idiosincrasias de colocación del sensor, no descoordinación fisiológica ante el estímulo.

Se sustituyó por completo el emparejamiento por HR (E4) frente a EDA — frecuencia cardíaca (sistema cardiovascular) frente a actividad electrodérmica (rama simpática pura), dos sistemas fisiológicos genuinamente distintos, análogo conceptual al par HR-pupila del Estudio 1. El resultado cambió radicalmente: S_coher pasó a mostrar una de las señales más fuertes de todo el análisis, con signo **positivo** (se detalla en la Sección 3.6.1).

Se consideró también usar Attention (NeuroSky, derivada de EEG) como pareja para S_coher, por ser un sistema fisiológico aún más distinto (cortical, frente a autonómico). Se descartó por dos motivos: Attention ya se estaba usando en S_cond, y reutilizarla también en S_coher habría hecho que las dos dimensiones dejaran de ser independientes por construcción — justo el tipo de redundancia que el diseño de tres componentes independientes pretende evitar. Además, la cobertura de NeuroSky por sujeto es más irregular que la de Polar_HR.

### 3.4. Iterando sobre S_cond: cuatro intentos, ninguno concluyente

S_cond mide compromiso conductual con la tarea. En el Estudio 1, se utiliza reaction_time y blinks_count — dos señales del eye-tracker que no existen en K-EmoCon (que no tiene seguimiento ocular). Hubo que buscar sustitutos con los datos disponibles, y aquí es donde el análisis exploratorio fue más largo y menos concluyente.

S_cond mide el compromiso del sujeto fisiológico (el participante del debate que lleva los sensores) con la tarea de debatir/vivir la conversación — no el compromiso del anotador externo con la tarea de anotar. Son personas distintas en K-EmoCon (a diferencia de EXIST, donde coincidían), y el PSRI, tal como está formulado, solo puede hablar de la fiabilidad de quien genera la señal fisiológica.

Se probaron siete variantes en paralelo:

1. **S_cond_att_acc** (Attention + inestabilidad de movimiento del acelerómetro): la implementación original, análoga en espíritu a "atención + ausencia de fatiga motora". Resultado: señal débil e inconsistente a nivel within-subject; gana significación pooled sobre `external_valence_var` (ρ ≈ −0.12).
2. **S_cond_att_med** (Attention + Meditation, ambas del mismo dispositivo NeuroSky): compromiso activo frente a desconexión pasiva. Resultado: débil, sin señal limpia.
3. **S_cond_self** (consistencia de la autoanotación: variabilidad local de self_valence/self_arousal en una ventana rodante de 5 trials): la más próxima en espíritu al original de EXIST, porque mide conducta respecto a una tarea (autoevaluarse), no fisiología pasiva. Resultado: una correlación estadísticamente enorme (ρ entre -0.45 y -0.61, p tan bajo como 1e-292) — pero inválida como evidencia, por dos razones que se descubrieron al investigar por qué era tan grande:
   - **Estructural**: el 73.8% (valence) y el 68.3% (arousal) de los valores de "variabilidad local" son exactamente cero — es decir, la variable no es continua, es casi binaria ("la autoanotación cambió en esta ventana" frente a "no cambió"). La función de transformación en U asigna automáticamente fiabilidad cero a cualquier valor por debajo de un umbral duro, así que ese ~70% de ventanas recibe la misma puntuación mínima por un corte binario, no por una campana matizada. Correlacionar una variable así de concentrada en un único valor extremo con una variable continua produce coeficientes de correlación de rango artificialmente grandes, sin que eso refleje necesariamente una relación de esa magnitud.
   - **Conceptual, y más importante**: a diferencia de un tiempo de reacción imposiblemente corto (que inequívocamente señala que la persona no procesó el estímulo), que la autoevaluación emocional de alguien no cambie durante 25 segundos es ambiguo. Puede significar desconexión de la tarea (lo que S_cond pretende detectar) — pero puede significar igual de bien que la persona sintió genuinamente lo mismo durante ese tramo, lo cual no es un fallo de nada. El supuesto "plano = poco fiable" que funciona sin ambigüedad para tiempo de reacción no está justificado de la misma forma para consistencia emocional.
4. **S_cond_combined** (media de todos los R_* disponibles): hereda los mismos problemas de S_cond_self por incluirla en la media, con magnitudes intermedias (ρ entre -0.35 y -0.46).
5. **S_cond_att** (solo Attention, sin el acelerómetro): variante de control para aislar la contribución de la atención pura. Resultado: sin señal significativa limpia.
6. **S_cond_att_std** (std de Attention en lugar de mean): probada por si la variabilidad momentánea de la atención capturara mejor el descompromiso que su nivel. Resultado: sin soporte within-subject.
7. **S_cond_att_med_std** (std de Attention + std de Meditation): misma lógica aplicada al par NeuroSky. Resultado: sin soporte within-subject.

**Conclusión sobre S_cond**: ninguna variante aporta evidencia limpia y defendible de un efecto trial-a-trial genuino; la única señal que sobrevive es la pooled de S_cond_att_acc. S_cond sigue siendo la dimensión más débil y menos resuelta del PSRI en K-EmoCon — la recomendación es no usar S_cond_self/combined en el compuesto y tratar S_cond como una limitación reconocida del estudio, no forzar una narrativa de éxito donde no la hay.

### 3.5. El método de descomposición between/within-subject (explicado desde cero)

Antes de interpretar cualquier correlación en un dataset donde varias observaciones vienen del mismo sujeto (aquí, cientos de ventanas de 5 segundos por persona), hay una trampa estadística clásica que conviene explicar con un ejemplo simple.

Imagina que mides la fiabilidad fisiológica y el nivel de desacuerdo en 1000 ventanas repartidas entre 20 personas (50 ventanas cada una). Supón que, por pura coincidencia de qué tipo de personas participaron, las 10 personas con fisiología más estable en promedio también resultan ser, como grupo, las que generan menos desacuerdo en general — quizá porque son personas con expresiones emocionales más claras en general, sin que eso tenga nada que ver con la variabilidad trial a trial de su fisiología. Si calculas la correlación mezclando las 1000 ventanas sin más, vas a encontrar una correlación fuerte — pero esa correlación puede estar reflejando enteramente una diferencia entre personas (una es "el tipo de persona con fisiología estable y expresión clara", otra no lo es), no una relación real de que "en los momentos en que la fisiología de una persona concreta es más estable, hay menos desacuerdo sobre ella". Esto último — la relación dentro de cada persona, momento a momento — es la que el PSRI pretende capturar conceptualmente, porque S_estab/S_coher se definen como propiedades del trial, no como rasgos estables del sujeto.

La forma de separar ambas cosas es sencilla una vez se entiende el problema:

- **Correlación "entre sujetos" (rho_between)**: se calcula la media de cada variable por persona (una fila por persona, en vez de una fila por ventana) y se correlacionan esas medias. Esto captura exclusivamente "¿las personas con fisiología en promedio más estable tienden a generar, en promedio, menos desacuerdo?" — es la pregunta sobre rasgos, no sobre momentos.
- **Correlación "dentro de sujeto" (rho_within)**: a cada valor se le resta la media de esa misma persona (un procedimiento llamado centrado por sujeto), y se correlacionan esos residuos. Al restar la media de cada persona, cualquier diferencia estable entre personas desaparece por construcción — lo que queda es exclusivamente la fluctuación de cada persona alrededor de su propia media, momento a momento. Esta es la pregunta que de verdad interesa: "cuando la fisiología de una persona concreta está, en un momento dado, más estable de lo habitual para ella, ¿hay menos desacuerdo en ese momento concreto?".

Un efecto que sobrevive al centrado por sujeto (rho_within significativo) es mucho más fuerte como evidencia que uno que solo aparece en la versión sin centrar, porque descarta la explicación alternativa más simple ("son solo diferencias estables entre personas").

Además de esta descomposición, se aplicó una segunda comprobación de robustez: leave-one-subject-out (LOSO) — recalcular la correlación excluyendo, una por una, a cada una de las 23 personas, para comprobar que el resultado no depende de un único sujeto con valores extremos que arrastre todo el efecto.

### 3.6. Resultados finales

#### 3.6.1. El hallazgo central: S_coher positivo dentro-de-sujeto

`external_valence_var` (la varianza entre los cinco anotadores externos al puntuar cuánta valencia positiva/negativa tenía la emoción de esa persona en esa ventana) muestra una correlación **positiva** y significativa dentro de sujeto con S_coher, y **nula** con S_estab:

- **S_estab**: rho_within ≈ +0.01, **no significativo** (p ≈ 0.56) — no predice el desacuerdo dentro-de-sujeto.
- **S_coher (HR-EDA)**: rho_within ≈ +0.06, significativo (p ≈ 0.001 en prev1; p ≈ 0.001 en prev5).

Interpretado en palabras: en los momentos en que la señal fisiológica de una persona concreta es, para ella, más **coordinada** entre sistemas (S_coher alto), los cinco anotadores externos tienden a **discrepar más** entre sí sobre cuál era su estado emocional. El efecto es el **opuesto** al predicho por la hipótesis original, y S_estab no contribuye a nivel within-subject.

Cabría esperar que dos mecanismos conceptualmente independientes — la estabilidad de un solo canal fisiológico y la coordinación entre dos sistemas fisiológicos distintos — apuntaran a la misma conclusión por caminos separados. Los datos no muestran esa convergencia negativa: S_estab es nulo dentro-de-sujeto y S_coher apunta en dirección positiva.

Un chequeo adicional confirmó que el signo positivo de S_coher no es casualidad de haber elegido una ventana de baseline concreta: se repitió el cálculo con ventanas de baseline de 1, 2, 3, 5, 8 y 10 trials anteriores, y el efecto se mantiene **positivo y significativo en las cinco primeras** (rho_within entre +0.066 con n_prev=2 y +0.046 con n_prev=8, p entre 0.0004 y 0.014), con atenuación marginal en n_prev=10 (+0.036, p=0.053). El patrón es el esperable para un efecto genuino de corto/medio plazo, pero de signo opuesto al hipotetizado.

#### 3.6.2. Un matiz: valence_range

`external_valence_range` (el rango máximo-mínimo entre los cinco anotadores, en vez de la varianza) sigue el mismo patrón que valence_var: es significativo en el bloque combinado del compuesto (pooled), y dentro de sujeto muestra el mismo **signo positivo** en S_coher (rho_within ≈ +0.03 a +0.05, significativo solo en las ventanas de baseline cortas: p=0.005 en prev1, p=0.068 en prev5), mientras que en S_estab es nulo (rho_within ≈ 0.00, p≈0.99).

Por ser una medida más sensible a valores extremos y estadísticamente menos eficiente que la varianza, conviene mantener `valence_var` como el hallazgo y presentar `valence_range` como réplica de apoyo — pero con el mismo caveat de signo invertido respecto a la hipótesis original.

#### 3.6.3. self_partner_diff y self_external_mean_diff: qué se sostiene dentro-de-sujeto

S_estab muestra una relación positiva con la discrepancia entre la autoevaluación de la persona y la de su pareja de debate (self_partner_diff) que **sobrevive al centrado por sujeto**:

- **S_estab vs `self_partner_diff`**: la correlación pooled es positiva (ρ_full ≈ +0.10, p≈0.004 tras centrado) y **sobrevive al centrado por sujeto** (rho_within ≈ +0.05, p ≈ 0.004). No es un simple artefacto entre-sujetos.
- **S_estab vs `self_external_mean_diff`**: es principalmente entre-sujetos (rho_within ≈ −0.01, p≈0.72) — la correlación pooled (ρ ≈ +0.07) refleja diferencias estables entre personas.
- **S_coher vs `self_external_mean_diff`**: conserva un efecto within **negativo** y marginal (rho_within ≈ −0.05, p ≈ 0.012 en prev1; −0.036, p=0.057 en prev5).

Esto muestra que las métricas de desacuerdo no se comportan de forma uniforme: cada componente se asocia selectivamente con unas u otras.

#### 3.6.4. arousal_var: significativo pooled, nulo dentro-de-sujeto

`external_arousal_var` muestra un patrón que conviene separar por nivel:

- **Compuesto pooled**: `external_arousal_var` es significativo (rho ≈ +0.101 en prev1, p≈3.9e-8, sobrevive Bonferroni). El compuesto se correlaciona positivamente con el desacuerdo en arousal a nivel global.
- **Dentro de sujeto**: es nulo en los componentes (S_estab rho_within ≈ +0.01, p≈0.48; S_coher prev1 ≈ +0.03, p≈0.07). El efecto pooled está dominado por la variación entre sujetos.

Es decir: a nivel global el compuesto predice el desacuerdo en arousal (positivo), pero ese vínculo no se sostiene momento-a-momento dentro de cada persona. Debe reportarse solo con esa distinción explícita, no como un hallazgo within-subject.

#### 3.6.5. S_cond: señal pooled, sin soporte within

Como se detalla en la Sección 3.4, ninguna de las siete variantes de S_cond aporta evidencia limpia a nivel within-subject. La variante **S_cond_att_acc (la usada en el compuesto)** **muestra significación a nivel pooled** sobre `external_valence_var` (rho ≈ −0.120, p≈0.006 tras corrección de 50 tests) y `external_valence_range` (−0.099). Sin embargo, **dentro-de-sujeto sigue nulo** (rho_within ≈ −0.02, p≈0.25 para valence_var): el efecto de S_cond_att_acc es entre-sujetos, no trial-a-trial. Sigue siendo la dimensión menos resuelta del PSRI.

#### 3.6.6. Comprobaciones de robustez que no cambiaron ninguna conclusión

- **Leave-one-subject-out**: en ningún componente ni target un solo sujeto domina de forma catastrófica. Los deltas máximos de rho al excluir un sujeto están entre ~0.02 y ~0.086, y el **signo del efecto dentro-de-sujeto de S_coher nunca se invierte** (rho_within se mantiene positivo en el rango [+0.046, +0.066] en el barrido).
- **Cobertura real vs. imputada**: S_estab y S_cond (variante att_acc del compuesto) tienen cobertura de ≈100% (99.96%, 5526/5528 ventanas — siempre hay E4 disponible por diseño del filtro de sujetos elegibles, salvo 2 ventanas marginales); S_coher (HR-EDA) tiene 98.8% de cobertura real — el resultado de S_coher no depende de la imputación.
- **Corrección por comparaciones múltiples**: el resultado principal (compuesto, 10 tests: 2 variantes de baseline × 5 métricas de desacuerdo) se corrigió como un único bloque, con el mismo criterio de rigor que los 20 tests del Estudio 1 — no se relajó el estándar al pasar de un estudio a otro. Sobreviven a Bonferroni en ambas variantes `external_valence_var`, `external_valence_range` y `external_arousal_var` (este último solo en prev1).

#### 3.6.7. Las figuras

En las Figuras 1 y 2 (Secciones 6.5) se muestran, respectivamente, la robustez del efecto de S_coher frente a la ventana de baseline y la descomposición entre/dentro-de-sujeto. Los valores representados corresponden a los resultados del bloque confirmatorio.

### 3.7. Anclaje temporal de las anotaciones y de la fisiología

#### 3.7.1. El anclaje en K-EmoCon

En K-EmoCon las dos series temporales que se fusionan tienen orígenes distintos, y ambos están declarados en el propio dataset:

- **Fisiología (E4)**: los timestamps son absolutos y la feature table ancla la ventana 0 a `t0 = initTime` (inicio de la grabación de la pulsera).
- **Anotaciones**: la columna `seconds` es **relativa al inicio del debate** (`startTime`). El máximo de `seconds` coincide con la duración del debate (`endTime − startTime`, ratio mediana 0.99, rango 0.86–1.18) y no con la duración de la sesión completa (`endTime − initTime`, ratio mediana 0.55, rango 0.39–0.83).
- **Metadata**: `metadata/subjects.csv` proporciona `initTime`, `startTime` y `endTime` (ms) por sujeto, de modo que el desplazamiento entre ambos relojes es recuperable de forma exacta: `ann_offset_s = startTime − t0`.

El intervalo `initTime → startTime` (mediana 8.6 min, rango 4.3–18.1 min) es un tramo pre-debate en el que no hay interacción ni anotaciones. `window_annotation` (aggregator.py) suma `ann_offset_s` al tiempo de anotación antes de ventanear, de modo que las anotaciones quedan alineadas con sus ventanas fisiológicas reales.

#### 3.7.2. Verificación del anclaje

1. **Anclaje al debate, no a la sesión**: self/partner/external abarcan la duración del debate (ratio mediana 0.99), no la de la sesión (0.55). Los anotadores externos (R1–R5) evalúan la **grabación del debate** — no pueden cubrir el pre-debate, porque no lo ven.
2. **Borde izquierdo**: aplicando `ann_offset_s = startTime − t0`, **0 anotaciones** caen antes del inicio del debate en los 23 sujetos (mediana de la fracción dentro del debate = 1.000; rango 0.844–1.000).
3. **Post-debate inocuo**: unos pocos sujetos (19, 20, 25, 26) tienen anotaciones self que se extienden un poco más allá de `endTime` (self > debate), pero para esos mismos sujetos los raters externos cubren exactamente el debate, y las anotaciones extra caen en ventanas que E4 no graba (E4 acaba en `endTime`) — se descartan limpiamente en el merge sin contaminar.

#### 3.7.3. Implementación

`window_annotation` recibe el parámetro `offset_s` (segundos a sumar al tiempo de anotación antes de ventanear), propagado desde `build_subject_table` vía `ann_offset_s`, calculado en `run_pipeline.py` como `startTime/1000 − t0`. La feature table resultante tiene 5528 ventanas y 23 sujetos.

#### 3.7.4. Análisis de lag (retraso perceptivo de las anotaciones)

Para descartar que el signo positivo dependiera de un desfase fino entre señal fisiológica y anotación (la anotación es un proceso manual, con posible retraso perceptivo-motor de 1–2 ventanas), se repitió la correlación dentro-de-sujeto de cada componente contra el desacuerdo desplazado ±2 ventanas:

- **S_coher vs `external_valence_var`**: efecto **plano y positivo en todos los lags** (k=−2: +0.026 p=0.16; k=−1: +0.048 p=0.01; k=0: +0.060 p=0.001; k=+1: +0.064 p=0.001; k=+2: +0.056 p=0.002). El pico está en k=0/+1 — compatible con un retraso perceptivo mínimo, pero la magnitud apenas cambia en todo el rango.
- **S_estab vs `external_valence_var`**: nulo en todos los lags (ρ_within ≈ 0.005–0.025, p≥0.18) — el efecto no aparece en ningún desplazamiento.
- **S_coher vs `external_valence_range`**: positivo y significativo para k=−1..+2, atenuándose en los extremos.
- **S_coher vs `external_arousal_var`**: marginalmente significativo solo en k=−2/−1 (p≤0.04), decae a cero en k≥0.

Conclusión del lag: el resultado positivo de S_coher no es un artefacto de elegir un desfase concreto; es un efecto de baja frecuencia temporal que persiste en todo el rango de ±2 ventanas. No hay evidencia de que exista un offset perceptivo sistemático grande. El análisis de lag se incluye como comprobación de robustez, no como corrección.

### 3.8. Task-validity: reposo pre-debate (control negativo) vs. debate

La lectura de los resultados exige saber si los componentes del PSRI son sensibles a la *condición* (reposo vs. conversación) o solo responden a la *calidad* fisiológica. El diseño de K-EmoCon permite un contraste casi experimental: el tramo **pre-debate** (inicio de grabación E4 → inicio del debate, sin interacción, gap mediano ≈ 8.6 min) actúa como reposo, y el **debate** como tarea conversacional. Se comparó cada componente y unos controles fisiológicos simples entre ambos periodos con un modelo mixto `componente ~ C(period) + (1|subject_id)` (β₁ = diferencia media debate − pre, dentro-de-sujeto), Wilcoxon pareado sobre medias por sujeto, LOSO y robustez a excluir las 1–2 primeras ventanas de cada periodo. Módulo: `sources/kemocon/task_validity.py`; salida: `kemocon_task_validity.csv`.

**Resultados principales (n=23 sujetos):**

- **S_cond (atención NeuroSky) sube de forma fuerte y consistente durante el debate.** Todas las variantes muestran β₁>0 altamente significativo en el modelo mixto y en el Wilcoxon pareado: `S_cond_att` β₁=+0.158 (p≈1.4e-63; 20/22 sujetos a favor del debate, Wilcoxon p≈2.1e-4), `S_cond_att_acc` β₁=+0.063 (p≈9.4e-38; 21/23), `S_cond_att_med` β₁=+0.143 (p≈1.1e-83; 21/22). Robustos a LOSO (β nunca cruza a cero) y a descartar las primeras 1–2 ventanas.
- **S_cond_combined es la única variante que BAJA en debate** (β₁=−0.144, p≈3.5e-187; 22/23 sujetos en contra del debate). Es esperable: el combinado mezcla las dimensiones de atención con la autoanotación (`S_cond_self`), que en reposo no existe (cobertura pre-debate 0%), así que el signo del combinado no es informativo de la atención — refuerza la decisión de no usar `S_cond_self` (cobertura 0% en pre-debate, ~70% ceros en debate).
- **S_estab no reacciona a la condición** (β₁=−0.016, p≈4.1e-6 en el modelo mixto pero Wilcoxon pareado p=0.30 con 10/23 sujetos a favor) — magnitud despreciable y sin consistencia por sujeto. Es el control esperado: la estabilidad intra-canal no debe ser sensible a la tarea.
- **S_coher baja levemente en debate** (β₁=−0.013 y −0.027 para prev1/prev5, significativo en el mixto pero Wilcoxon marginal/NS: p≈0.05 y p≈0.014) — efecto pequeño y de magnitud muy inferior al de S_cond.
- **Controles fisiológicos** (para calibrar qué es "esperable" en el contraste): movimiento `acc_std` (β₁=+1.74), EDA (β₁=+0.46) y HR (β₁=+5.15) suben en debate (p<1e-29) — coherente con que conversar activa la fisiología. `bvp_mean` es el único control plano (β₁≈0, NS), útil como control nulo.

**Lectura metodológica**: S_cond es el único componente del PSRI que distingue con claridad reposo de conversación dentro-de-sujeto, de forma robusta y consistente (20–21 de 22–23 sujetos), lo que respalda su validez como módulo de *task-validity*: detecta que el participante está realizando una tarea con carga atencional. El hallazgo es coherente con que S_cond gane señal pooled sobre el desacuerdo (Sección 3.6.5): un módulo sensible a la condición es un candidato plausible de mediador del desacuerdo, aunque no se sostenga dentro-de-sujeto. S_estab se comporta como control (insensible a la condición), y S_coher apenas baja en debate. Este contraste refuerza además la verificación del anclaje (Sección 3.7.2): las anotaciones solo existen en las ventanas de debate y el tramo pre-debate queda limpio para servir de reposo.

Este módulo materializa el nivel **Q_window** del marco de tres niveles de validez (formalizado en `metodologia_esqueleto.md` §6.1): **Q_session** (exclusiones de sujeto de `sanity_check`), **Q_window** (etiqueta pre/debate por ventana, exportada en `kemocon_period_assignments.csv`) y **Q_physio** (S_estab+S_coher, núcleo confirmatorio). La separación evita que una sesión con buena señal se lea como sesión de alta atención.

### 3.9. S_cond: la caja de herramientas (formas monotónicas vs. en U) — respuesta a docs/dialogo.txt §2

El marco conceptual de S_cond (conceptual_psri.md §1.3) establece que la dimensión **no presupone una forma funcional única**: la herramienta se elige por la semántica del observable — monotónica cuando el observable tiene interpretación unívoca (`Attention`/`Meditation`: el extremo alto es deseable, no ruido; el movimiento, según la dirección), en U cuando ambos extremos son patológicos. docs/dialogo.txt §2 señaló precisamente que la función en U (`compute_psri_gaussian_log`) **no es adecuada para variables con interpretación unívoca** porque penaliza el extremo deseable como si fuera ruido. Para contrastar la regla en K-EmoCon se añadieron variantes MONOTÓNICAS (`_apply_monotonic_transform`: cola alta mapeada a fiabilidad en (0,1] con mediana/MAD de población global) calculadas en paralelo con las en-U:

- `S_cond_att_mono` (solo Attention, creciente), `S_cond_att_med_mono` (Attention+Meditation), `S_cond_att_mono_acc_hi` (movimiento "más es mejor"), `S_cond_att_mono_acc_lo` (movimiento "menos es mejor").

**Resultados (diagnóstico):**

- **`S_cond_att_mono` es la variante que CAPTURA señal within-de-sujeto que la U enmascara**: ρ_within=−0.050 (p=0.009) sobre `external_valence_var` y ρ_within=−0.041 (p=0.032) sobre `external_valence_range` — la versión en-U (`S_cond_att`) era nula dentro-de-sujeto (ρ_within≈−0.006, NS). El signo negativo es **en la dirección de la hipótesis original** (menos atención → más desacuerdo). La señal no es robusta a LOSO para valence_var (ρ_within ∈ [−0.059, +0.017], cruza a cero), aunque sí lo es para `external_arousal_var` (ρ_within=+0.062, p≈0.001, LOSO [+0.056, +0.120]).
- **`S_cond_att_mono_acc_lo`** (movimiento "menos es mejor"): señal pooled negativa robusta sobre `external_valence_var` (ρ=−0.109, p≈2e-9, LOSO [−0.138, −0.078]) y within negativa (ρ_within=−0.037, p=0.047); `external_arousal_var` pooled +0.198 (p≈3e-27) y within +0.114 (p≈7e-10).
- **`S_cond_att_mono_acc_hi`** (movimiento "más es mejor"): señal pooled POSITIVA sobre valencia (ρ=+0.186) — es decir, más movimiento+atención se asocia con MÁS desacuerdo — pero puramente entre-sujetos (ρ_between=+0.556, ρ_within≈−0.01 NS). Confirma empíricamente el argumento del diálogo: la dirección del movimiento es ambigua y solo una de las dos direcciones (menos movimiento) produce señal coherente con la hipótesis.
- **Task-validity (reposo vs debate)**: ninguna variante monotónica sube en debate como las en-U — `S_cond_att_mono` β₁=−0.025 (Wilcoxon NS, 11/22) y `S_cond_att_mono_acc_lo` β₁=−0.036 (Wilcoxon p=0.002, pero solo 6/23 sujetos a favor del debate). Es decir: la transformación monotónica pierde la validez de tarea que la U capturaba.

**Lectura**: la regla de selección se confirma y, con ella, se acota el dominio de validez de cada herramienta. La monotónica recupera señal within-de-sujeto sobre el desacuerdo que la en-U enmascara (dirección de la hipótesis), porque penalizar el extremo alto de la atención borra información real; pero esa herramienta pierde la validez de tarea que la en-U captura y su señal within no es robusta a LOSO. Es exactamente el comportamiento que la caja de herramientas predice: la forma no es neutra —cada observable y cada propósito de validación (predecir desacuerdo frente a auditar la condición) tienen su herramienta—. La conclusión operativa es firme: en K-EmoCon, sin observables conductuales directos, ninguna variante sustenta un efecto within-de-sujeto robusto; la dimensión se valida como módulo de task-validity (Sección 3.8) y la caja de herramientas queda lista para datasets con observables conductuales directos, donde EXIST ya la confirmó (EstudioEXIST.md §3.7, tecnico_exist.md §8). Columnas en la feature table: `R_attention_mono`, `R_meditation_mono`, `R_acc_mono_hi`, `R_acc_mono_lo`, `S_cond_att_mono`, `S_cond_att_med_mono`, `S_cond_att_mono_acc_hi`, `S_cond_att_mono_acc_lo`.

### 3.10. Hipótesis derivada: la U de S_estab enmascara señal dentro-de-sujeto en la región de alta variabilidad (diagnóstico exploratorio)

Reflexión de diseño (docs/dialogo.txt): la función en U de S_estab (`compute_psri_gaussian_log`) penaliza tanto el extremo "bajo" como el "alto" de la variabilidad como "no fiable". Para la pregunta de subjetividad (desacuerdo), el extremo alto podría llevar señal, pero no con las variabilidades extremas: debería existir un umbral a partir del cual las altas son relevantes, y el nivel de coordinación (S_coher) podría actuar como regulador ("las altas interesan si la coordinación no está descontrolada"). Para contrastarlo se construyó un diagnóstico exploratorio standalone: `sources/kemocon/experiment_estab_highvar.py` (solo lee la feature table; salidas en `results/output/KEMOCON/experiment_estab_highvar/`).

**Método**: `z_var` = media sobre los canales E4 (`bvp/eda/hr/temp/ibi_std`) del z-score robusto de `log-sigma` (mediana/MAD 1.4826, la misma estadística de `compute_psri_gaussian_log`). Región alta = `z_var > 0` (2612 ventanas, 23 sujetos, de 5457 con `z_var` y `S_coher`). Tres análisis: (1) ρ within-de-sujeto `z_var`↔desacuerdo en la región alta, con y sin control por movimiento (`acc_std`); (2) barrido de umbral sobre cuantiles q∈{0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99}; (3) S_coher como moderador (split por mediana, 0.935) dentro de la región alta.

**Resultados (diagnóstico):**

- **En la región alta, el nivel de variabilidad SÍ lleva señal within-de-sujeto que la U enmascara**: `external_valence_var` ρ_within=−0.096 (p=0.0002), **robusto al control por movimiento** (ρ_within_ctrl_acc=−0.119, p<0.0001); `external_valence_range` −0.082 (p=0.001; ctrl −0.106); `self_partner_diff` −0.063 (p=0.014); `self_external_mean_diff` +0.057 (p=0.028). Es decir: dentro-de-sujeto, más variabilidad alta → menos desacuerdo externo en valencia. `external_arousal_var` nulo (se explica por el movimiento: −0.048→−0.020 NS al controlar).
- **El barrido de umbral confirma la intuición de "no con las muy altas"** (`external_valence_var`, ρ_within): q0=−0.053, q0.5=−0.088, q0.75=**−0.102**, q0.9=−0.030, q0.95=**+0.064** (patrón análogo controlando `acc_std`: −0.137 en q0.75, +0.044 en q0.95). La asociación negativa crece hasta el cuantil 0.75, colapsa en 0.90 y **se invierte en el extremo (q0.95)**. Hay un rango "dulce" de variabilidad moderadamente alta que es informativo; las muy altas invierten la señal.
- **S_coher como regulador — matiz respecto a la intuición original**: el efecto se concentra en las ventanas altas **descoordinadas** (S_coher < mediana): `external_valence_var` ρ_within=−0.117 (p=0.0009; ctrl −0.157), `external_valence_range` −0.102 (p=0.004; ctrl −0.137); en las coordinadas el efecto es n.s. (−0.047/−0.044). La señal vive donde la coordinación está *baja*, no donde "la cosa va bien".

**Lectura**: es una **hipótesis derivada exploratoria, no un hallazgo confirmatorio**. No estaba pre-registrada (nació de una reflexión a posteriori), las p no están corregidas por comparaciones múltiples (5 targets × 2 versiones × 7 umbrales) y su dirección contradice la lógica de la U (el paper lee alta variabilidad = no fiable → más desacuerdo; aquí es lo contrario dentro de la región alta). Se documenta al mismo nivel que la interpretación invertida de S_coher (§6.3) y la caja de herramientas de S_cond (§3.9): si un dataset con observables conductuales directos lo replica, justificaría tratar la subjetividad con una variante monótona (no-U) para el extremo alto de variabilidad. No toca el bloque confirmatorio del paper.

### 3.11. Latencia de sensores: ¿S_coher compara HR-EDA sobre el mismo inicio real? (diagnóstico exploratorio)

Reflexión de diseño: al construir S_coher, HR (E4) y EDA (E4) se comparan por ventana suponiendo que ambas señales comparten el mismo inicio temporal (el pipeline ancla todo a `t0` = primer timestamp mínimo de las modalidades E4; `aggregator.window_signal`). Si un canal empezara su medición *real* con una latencia no contemplada, las dos series se compararían sobre rejillas desplazadas y S_coher mediría desalineación, no coordinación. Para contrastarlo se construyó un diagnóstico exploratorio standalone: `sources/kemocon/sensor_latency.py` (lee solo los datos crudos + la feature table; salidas en `results/output/KEMOCON/sensor_latency/`).

**Método** — cuatro chequeos independientes:

1. **Startup latency**: offset del primer sample de cada canal respecto a `t0`, por sujeto, para E4 y para las fuentes cross-device (NeuroSky/Polar).
2. **Grid alignment**: ¿cada muestra HR (1 Hz, timestamps enteros) cae exactamente sobre el grid de EDA (4 Hz, cuartos de segundo)? 100% de coincidencia exacta = mismo reloj de dispositivo, sin offset persistente intra-E4 tras el arranque.
3. **Cross-correlación** de medias por ventana (5 s) HR-EDA por sujeto, lags −6..+6 ventanas: un lag fisiológico/de sensor sistemático daría un pico agudo y consistente en un lag pequeño no nulo; un perfil plano o monótono = sin lag sistemático.
4. **Sensibilidad al lag** del resultado confirmatorio: recalcular `S_coher_prev1` desplazando `hr_mean` −3..+3 ventanas por sujeto y correlacionar contra las métricas de desacuerdo (pooled y within-subject). Si el efecto within sobrevive en un rango de latencia plausible (±1-2 ventanas = ±5-10 s), la conclusión del paper no es un artefacto de la sincronización asumida.

**Resultados (n=27 sujetos elegibles):**

- **Startup latency E4**: ACC, BVP, EDA y TEMP empiezan exactamente en `t0` en los 27 sujetos (offset 0). HR empieza a **exactamente +10 s** (determinista: canal derivado que necesita ~10 s de BVP para el primer valor) e IBI a **+11.98..+41.14 s** (mediana +15.9 s). Ambas son demoras de *arranque* del canal derivado, confinadas al tramo pre-debate (el debate empieza ~8.6 min después de `t0`): **no hay latencia de arranque dentro de las ventanas de debate**, donde se calcula el S_coher confirmatorio.
- **Grid alignment**: 100% de las muestras HR coinciden exactamente con el grid de EDA (distancia máxima 0 ms) en todos los sujetos. Es decir, HR y EDA comparten reloj y rejilla temporal: **no existe un offset persistente intra-dispositivo**; la comparación "mismo inicio" es físicamente correcta para HR-EDA.
- **Cross-correlación HR-EDA**: perfil monótono decreciente de ρ ≈ +0.18 (lag −6) a ≈ +0.13 (lag +6) sin pico en un lag pequeño; la distribución del lag de |ρ| máximo por sujeto es bimodal en los extremos (−6 y +6) — el clásico artefacto de tendencia (ambas series suben en el debate), no un desfase fisiológico sistemático.
- **Sensibilidad del efecto confirmatorio** (S_coher_prev1 vs `external_valence_var`, ρ_within): shift 0 = **+0.060 (p=0.001)**; ±1 ventana = +0.057/+0.036 (p=0.002/0.050); ±2 = +0.026/+0.011 (NS); ±3 ≈ 0. El efecto within-de-sujeto tiene el pico en el lag 0 y solo decae a ±2-3 ventanas (patrón idéntico al análisis de lag de anotaciones, §3.7.4). `external_valence_range` replica el patrón (pico en 0/+1, decae a ±2); `external_arousal_var` es marginalmente positivo en todos los lags.

**Lectura**: la asunción de latencia **tiene sentido conceptualmente**, pero **los datos la descartan para el par actual de S_coher (HR-EDA, mismo dispositivo E4)**: el "mismo inicio" no es una asunción, es el hardware (un solo reloj de dispositivo, grids alineados al 100%). La latencia observable de arranque (HR +10 s, IBI variable) es un comportamiento determinista de canales derivados y cae fuera del debate. El resultado confirmatorio positivo within-de-sujeto de S_coher es robusto a correcciones de latencia de hasta ±5 s (shift ±1), coherente con un efecto de baja frecuencia temporal y no con un desfase sin corregir. **Donde la latencia es real es en los pares cross-device**: las fuentes NeuroSky/Polar empiezan 0-396 s después del E4 (mediana ~112 s), lo que afecta a la cobertura de S_cond (atención) cerca del inicio del debate y sería un riesgo genuino si S_coher se extendiera a un par entre dispositivos (en EXIST, HR Garmin vs pupilómetro ya son cross-device — ahí sí procedería una corrección/verificación de alineación). El módulo queda como herramienta reutilizable; no toca el bloque confirmatorio del paper.

### 3.12. Réplica del efecto de S_coher en otros pares cross-system: ¿es específico de HR-EDA? (diagnóstico exploratorio, respuesta a revisión externa)

Reflexión de diseño: el efecto confirmatorio dentro-de-sujeto de S_coher (ρ_within≈+0.06 sobre `external_valence_var`, §3.6.1) se estableció con el par **HR (E4) vs EDA**. Una revisión externa (docs/ECIR2027/review.md, pregunta 8) preguntó si el efecto se replica en otros pares entre sistemas fisiológicos distintos (p. ej. BVP-EDA), aunque fuera más débil, o si es un artefacto específico de esa pareja de sensores. Para responder se construyó un diagnóstico exploratorio standalone: `sources/kemocon/experiment_coher_pairs.py` (solo lee la feature table; salidas en `results/output/KEMOCON/experiment_coher_pairs/`).

**Método**: se recalcula `S_coher_prev1` con la fórmula **idéntica** a la del par de referencia (baseline rolling prev1 por sujeto, std histórica del sujeto, `compute_z_score` + `compute_coherence`) para cada par entre los canales E4 {bvp, eda, hr, temp, ibi}. Familias: (a) **pares cross-system de réplica** (cardiovascular × simpático × termorregulador): bvp-eda, hr-temp, bvp-temp, eda-temp, ibi-eda, ibi-temp; (b) **pares intra-sistema de control** (redundancia, análogo al HR-HR descartado en §3.3): bvp-hr, hr-ibi, bvp-ibi. El par hr-eda se recalcula como sanity check (debe reproducir ρ_within=+0.060, p=0.001). Para cada par × target (los 5 de desacuerdo) se aplica `leverage_diagnostics` (pooled/between/within + LOSO) y todos los `p_within` se corrigen como **un único bloque** (Bonferroni + FDR), coherente con la política de corrección del proyecto.

**Resultados** (ρ_within sobre `external_valence_var`; familia 10 pares × 5 targets = 50 tests):

| Par | Sistemas | Rol | ρ_within | p (cruda) | Bonferroni |
|---|---|---|---|---|---|
| hr-eda | cardiovascular × simpático | referencia | +0.060 | 0.001 | 0.057 (marginal) |
| **hr-temp** | cardiovascular × termorregulador | réplica | **+0.062** | 0.001 | **0.040 (sobrevive)** |
| bvp-eda | cardiovascular(BVP) × simpático | cross-system sin HR | ≈0 | n.s. | — |
| bvp-temp | cardiovascular(BVP) × termorregulador | cross-system sin HR | ≈0 | n.s. | — |
| eda-temp | simpático × termorregulador | sin HR | ≈0 | n.s. | — |
| bvp-hr | intra-sistema (derivados del mismo BVP) | control | ≈0 | n.s. | — |
| bvp-ibi | intra-sistema | control | ≈0 | n.s. | — |
| ibi-eda | cardiovascular(IBI) × simpático | con IBI | −0.042 | 0.023 | no |
| ibi-temp | cardiovascular(IBI) × termorregulador | con IBI | −0.038 | 0.042 | no |
| hr-ibi | intra-sistema (derivados del mismo BVP) | con IBI | −0.051 | 0.006 | no |

**Lectura**: el efecto invertido (mayor coordinación → mayor desacuerdo) **no es un artefacto de la pareja HR-EDA concreta**: se replica con el par **HR-TEMP** a magnitud casi idéntica (+0.062) y sobrevive la corrección Bonferroni de toda la familia. La réplica tiene una frontera clara: el efecto positivo vive en la pata cardiovascular medida por **HR** (el índice derivado limpio); es nulo cuando la pata cardiovascular es la onda cruda BVP o cuando el par no incluye HR (eda-temp); es nulo en los controles intra-sistema BVP-HR/BVP-IBI; y se **invierte de signo** en los pares con IBI (derivado por detección de latidos), coherente con que la varianza de IBI arrastra ruido de detección (§3.3). Implicación para el paper: el claim dentro-de-sujeto de S_coher se presenta como efecto de coordinación cardiovascular (HR) × otro sistema, **replicado en dos pares**, no como propiedad de "cualquier par cross-system". Es un diagnóstico exploratorio; no toca el bloque confirmatorio (fijado en HR-EDA).

## 4. Limitaciones

- El vínculo entre fiabilidad fisiológica y desacuerdo es modesto y no reproduce la dirección hipotetizada: el efecto dentro-de-sujeto de S_coher sobre el desacuerdo en valencia es **positivo** (ρ_within ≈ +0.05 a +0.07), el de S_estab es **nulo**, y la única señal de S_cond (att_acc) es a nivel pooled, no dentro-de-sujeto. La magnitud es pequeña en términos absolutos.
- S_cond carece de observables conductuales directos en K-EmoCon (no hay seguimiento ocular ni tiempos de reacción de la tarea), la condición que la caja de herramientas de la dimensión necesita para operar con su herramienta más natural; ninguna variante (en-U ni monotónica) sustenta un efecto within-de-sujeto robusto, aunque gana señal pooled (`S_cond_att_acc`) y la monotónica recupera señal within no robusta a LOSO (Sección 3.9). Se documenta como limitación de disponibilidad de sensores, no como fallo de la dimensión.
- Solo 23 sujetos tras el filtro de calidad; los efectos que dependen en parte de variación entre sujetos deben interpretarse con la cautela propia de un n de ese tamaño.
- La sustitución de S_coher de HR-Polar_HR a HR-EDA, y las variantes probadas para S_cond, son decisiones tomadas durante el análisis exploratorio a la luz de resultados intermedios — hay que ser explícitos en el paper sobre qué se decidió a priori (la arquitectura general, heredada de EXIST) y qué se decidió a posteriori por motivos conceptuales explícitos (no por ajuste ciego a los datos), para no dar la impresión de p-hacking. La defensa metodológica es que cada cambio tuvo una justificación conceptual clara antes de mirar si mejoraba el resultado (HR-HR es redundancia instrumental, no coherencia entre sistemas — se cambió por ese motivo, y solo después se comprobó el efecto).
- **El componente entre-sujetos del efecto de S_estab sobre la valencia está confundido por el movimiento** (análisis de robustez, `sestab_confounds.py`): a nivel sujeto, la asociación S_estab↔`external_valence_var` (ρ≈−0.28) se derrumba a ≈0.00 al controlar por el rasgo de movimiento (`acc_std`; ρ_sujeto S_estab–acc_std=−0.43, mientras que acc_std↔valence_var=+0.65). La cadena real es *movimiento → inestabilidad de señal → desacuerdo*, no calidad fisiológica independiente. Lo que S_estab sí conserva sin confundir: `self_partner_diff`/`self_external_mean_diff` entre-sujetos (ρ≈+0.27/+0.31 tras controlar movimiento) y la señal within-subject de `self_partner_diff` (ρ_within=+0.054, p≈0.004), a nivel de trial. Implica que el claim entre-sujetos de S_estab en valencia debe presentarse como proxy del movimiento o, alternativamente, acotarse a las métricas self/partner y al nivel within.

## 5. Síntesis (puntos de partida)

1. El diseño de K-EmoCon (sujeto fisiológico = objeto del juicio) resuelve la limitación estructural identificada en el Estudio 1.
2. Con la población alineada, el PSRI compuesto muestra un efecto pooled robusto sobre el desacuerdo en valencia (ρ ≈ −0.19 a −0.21, p≈0, sobrevive Bonferroni en ambas variantes de baseline). Sin embargo, la descomposición dentro-de-sujeto **no** sigue la dirección hipotetizada: **S_coher (HR-EDA) se correlaciona positivamente** con `external_valence_var` (ρ_within ≈ +0.06, p≈0.001) y **S_estab no predice dentro-de-sujeto** (ρ_within ≈ +0.01, NS).
3. El efecto pooled es robusto: sobrevive corrección estricta por comparaciones múltiples y leave-one-subject-out; dentro-de-sujeto, el signo de S_coher se mantiene positivo en todo el barrido de baseline y en el análisis de lag ±2 ventanas. Matiz: su componente entre-sujetos en valencia está explicado por el rasgo de movimiento del sujeto (análisis de confounds, §4) — la defensa del efecto pooled es sobre el desacuerdo en valencia, no sobre un mecanismo de calidad fisiológica entre-sujetos independiente del movimiento.
4. Las métricas de desacuerdo se comportan de forma selectiva: `external_arousal_var` es significativo pooled (+0.101, p≈3.9e-8 en prev1) aunque nulo dentro-de-sujeto; `self_partner_diff`/`self_external_mean_diff` no son significativos pooled, pero `self_partner_diff` gana significación within en S_estab (+0.054, p≈0.004).
5. S_cond (att_acc) gana señal pooled (−0.120 sobre valence_var), pero sigue sin soporte within-subject — una limitación parcial, más que una oportunidad cerrada.
6. El diagnóstico de task-valididad (Sección 3.8) respalda la interpretación de S_cond como módulo de *task-validity*: es el único componente que distingue de forma robusta y consistente el reposo pre-debate del debate dentro-de-sujeto (S_cond_att β₁=+0.158, p≈1.4e-63; 20/22 sujetos), mientras S_estab (control) no reacciona a la condición. Esto es compatible con que S_cond gane señal pooled sobre el desacuerdo.
7. El compuesto de **dos patas** (S_estab + S_coher, sin S_cond) — la alternativa `PSRI_validated` de docs/dialogo.txt — no pierde potencia sobre `external_valence_var`/`external_valence_range` y **recupera señal que la tercera pata enmascaraba**: `self_partner_diff` vuelve a ser significativo pooled (ρ≈+0.100, p≈9e-8 prev1) **con señal within-de-sujeto** (ρ_within=+0.049, p=0.009), y `self_external_mean_diff` recupera significación pooled. El efecto principal sobre valencia sigue siendo entre-sujetos en ambas versiones. Esto refuerza la recomendación del diálogo: el núcleo confirmatorio del paper son dos patas, y S_cond se presenta como módulo de task-validity separado.
8. La **caja de herramientas** de S_cond (Sección 3.9, formalizada en conceptual_psri.md §1.3) se confirma: la forma se elige por la semántica del observable. `S_cond_att_mono` captura señal within-de-sujeto sobre `external_valence_var` que la versión en-U enmascara (ρ_within=−0.050, p=0.009, dirección de la hipótesis original), pero pierde la validez de tarea (reposo vs debate) que la U capturaba y su señal within de valencia no es robusta a LOSO. En K-EmoCon, sin observables conductuales directos, ninguna variante sustenta un efecto within robusto; la caja de herramientas queda lista para datasets con observables directos, donde EXIST ya la confirmó con el RT monotónico (`S_cond_mono`, EstudioEXIST.md §3.7).
9. La alternativa **S_obs** (observabilidad humana, docs/dialogo.txt) no se implementa como cuarta pata: coincide operativamente con la asociación S_estab/S_coher↔desacuerdo ya medida, y usar `external_valence_var` como insumo la haría circular. Se documenta como interpretación conceptual del hallazgo (la fiabilidad fisiológica como determinante de la observabilidad perceptual del sujeto), detalle en `conceptual_psri.md` §1.3.1.
10. **Hipótesis derivada exploratoria (Sección 3.10, NO confirmatoria)**: dentro de la región de alta variabilidad de S_estab (`z_var>0`), el nivel de variabilidad correlaciona negativamente dentro-de-sujeto con el desacuerdo en valencia (ρ_within=−0.096, p=0.0002; robusto al control por movimiento −0.119), con un patrón de umbral que respalda la intuición de "no con las muy altas" (crece hasta q0.75, se invierte en q0.95) y que se concentra en las ventanas descoordinadas (S_coher < mediana). Sugiere que la U penaliza indebidamente el extremo alto para la pregunta de subjetividad, pero requiere réplica en un dataset con observables directos antes de tocar el diseño del índice.
11. **El efecto within-de-sujeto de S_coher se replica en otro par cross-system (Sección 3.12, diagnóstico)**: el par HR-TEMP reproduce la asociación positiva con el desacuerdo a magnitud casi idéntica (ρ_within=+0.062, p=0.001, sobrevive Bonferroni en la familia de 10 pares × 5 targets), lo que descarta que sea un artefacto de la pareja HR-EDA concreta; la frontera es HR-específica (nula con BVP y en pares sin HR, invertida con IBI). El claim se presenta como coordinación cardiovascular (HR) × otro sistema, replicada en dos pares.

---

La hipótesis central del PSRI, tal como estaba formulada — "cuando la señal fisiológica se vuelve inestable (S_estab) o descoordinada (S_coher), los observadores pierden consenso" — **no se confirma en su versión negativa dentro-de-sujeto**:

- **El efecto dentro-de-sujeto de S_coher es positivo**: en los momentos en que los sistemas fisiológicos de una persona están más coordinados entre sí, los anotadores tienden a **discrepar más** (ρ_within ≈ +0.06, p≈0.001). Es el **opuesto** al efecto hipotetizado.
- **S_estab no predice dentro-de-sujeto** (ρ_within ≈ +0.01, NS).
- **El compuesto pooled sí predice el desacuerdo en valencia** (ρ ≈ −0.19 a −0.21) y es además significativo sobre `external_arousal_var`, pero ese efecto pooled mezcla variación entre-sujetos y dentro-de-sujeto, y dentro-de-sujeto no se sostiene en la dirección predicha.

**Cuantificando los valores**:

- **Compuesto pooled vs `external_valence_var`**: ρ ≈ −0.205 (prev1) / −0.190 (prev5), p≈0, sobrevive Bonferroni.
- **S_coher (HR-EDA) within vs `external_valence_var`**: ρ_within ≈ +0.06 (p ≈ 0.001), positivo y estable en el barrido n_prev 1–10 (entre ≈ +0.036 y ≈ +0.066) y en el análisis de lag ±2.
- **S_estab within vs `external_valence_var`**: ρ_within ≈ +0.01 (NS) — efecto desaparecido.
- **S_cond_att_acc pooled vs `external_valence_var`**: ρ ≈ −0.12 (sig), sin soporte within.

**Solidez Estadística**

El efecto pooled y el signo positivo de S_coher dentro-de-sujeto son robustos:

- Corrección por comparaciones múltiples (Bonferroni estricto): `external_valence_var`, `external_valence_range` y `external_arousal_var` sobreviven en el compuesto.
- Descomposición Between/Within-subject: revela que el efecto dentro-de-sujeto de S_coher es positivo y el de S_estab nulo — no soportan la hipótesis original de signo negativo.
- Leave-One-Subject-Out: el signo de S_coher dentro-de-sujeto nunca se invierte (ρ_within en [+0.046, +0.066]).
- Barrido sistemático de ventanas de línea base y análisis de lag ±2: el efecto positivo se mantiene estable.

Para el cálculo de S_coher, emparejar HR(Polar) con HR(Empatica) dio nulo (es solo redundancia de sensores). El cambio a HR vs. EDA (sistemas fisiológicos distintos) fue el detonante del hallazgo, que resultó tener signo positivo.
`external_arousal_var` es significativo pooled (no es un artefacto puro), mientras que `self_partner_diff`/`self_external_mean_diff` no lo son a nivel pooled; y `self_partner_diff` aparece significativo dentro-de-sujeto en S_estab.
En el caso de S_cond, la variante att_acc gana señal pooled pero ninguna variante sostiene un efecto dentro-de-sujeto. Se mantiene como limitación parcial.

Con este experimento no podemos afirmar la versión original de la hipótesis (señal inestable ⇒ más desacuerdo). La lectura robusta del estudio es que **la fiabilidad fisiológica se asocia con el desacuerdo humano en K-EmoCon en una dirección y a un nivel (pooled vs. within) que no coinciden con la hipótesis original**. Cualquier sistema de IA Afectiva que use el PSRI como ponderación debe tener en cuenta estos resultados.

## 6. Tabla de campos, significado y transformación

Misma lógica que la tabla de PhysioNet del Estudio 1 (validación del instrumento), pero aquí aplicada a la aplicación del instrumento ya validado, no a su validación — por eso al final hay una diferencia importante respecto a PhysioNet que se explica en la nota final de esta sección.

### 6.1. S_estab — estabilidad intra-canal (un bloque por cada señal E4)

| Campo | Significado | Transformación |
|---|---|---|
| `E4_BVP.value` | Volumen de pulso sanguíneo crudo (fotopletismografía), muestreado por la pulsera E4 | Se agrega por ventana de 5s → `bvp_mean`, `bvp_std` |
| `bvp_std` | Variabilidad del BVP dentro de la ventana de 5s | Entra en `compute_psri_gaussian_log`, con la población de referencia = todas las ventanas de todos los sujetos (no solo las del mismo sujeto) → `bvp_S_estab` |
| `E4_EDA.value` | Actividad electrodérmica cruda (conductancia de la piel, rama simpática) | Mismo patrón → `eda_std` → `eda_S_estab` |
| `E4_HR.value` | Frecuencia cardíaca derivada internamente por el dispositivo E4 a partir del BVP | Mismo patrón → `hr_std` → `hr_S_estab` |
| `E4_TEMP.value` | Temperatura de la piel | Mismo patrón → `temp_std` → `temp_S_estab` |
| `E4_IBI.value` | Intervalo entre latidos consecutivos | Mismo patrón → `ibi_std` → `ibi_S_estab` |
| `bvp_S_estab`, `eda_S_estab`, `hr_S_estab`, `temp_S_estab`, `ibi_S_estab` | Fiabilidad de estabilidad por canal, cada una en [0,1] | Media aritmética de las disponibles → `S_estab` (compuesto) |

> **Nota**: el acelerómetro (`E4_ACC.x/.y/.z`) **no** forma parte de S_estab. Su estadístico `acc_std` (std de cada eje promediada entre los 3, sin `acc_mean`) se calcula y se reutiliza exclusivamente en S_cond como `R_acc_movement` (inestabilidad de movimiento).

### 6.2. S_coher — coherencia cruzada HR(E4)–EDA

| Campo | Significado | Transformación |
|---|---|---|
| `hr_mean` | Media de HR (E4) en la ventana actual | Entra en `compute_z_score(valor, baseline, std_sujeto)` |
| `eda_mean` | Media de EDA en la ventana actual | Entra en `compute_z_score(valor, baseline, std_sujeto)` |
| `baseline_prev (n=1)` / `baseline_prev5 (n=5)` | Media de `hr_mean`/`eda_mean` del/los trial(s) inmediatamente anteriores del mismo sujeto (ventana rodante, no la media de toda la sesión) | Es el término "baseline" del Z-score — dos variantes probadas en paralelo |
| `subject_std` (histórico) | Desviación típica de `hr_mean`/`eda_mean` de ESE sujeto a lo largo de todos sus trials | Es el término normalizador del Z-score (distinto de `hr_std`/`eda_std`, que es variabilidad dentro de una sola ventana) |
| `Z_hr`, `Z_eda` | Cuántas desviaciones típicas (propias del sujeto) se ha desviado cada señal de su reposo reciente | `compute_coherence(Z_hr, Z_eda) = exp(-|Z_hr - Z_eda|/2)` → `S_coher` |

> **Nota**: la primera versión de S_coher emparejaba HR(E4) con HR(Polar) — misma magnitud, dos instrumentos — y dio nulo; se sustituyó por HR-EDA (dos sistemas fisiológicos distintos) tras razonar que la redundancia instrumental no es lo que S_coher está diseñado para detectar (Sección 3.3).

### 6.3. S_cond — consistencia conductual

7 variantes probadas; ninguna adoptada como definitiva a nivel within-subject (S_cond_att_acc conserva señal pooled).

| Campo | Significado | Transformación |
|---|---|---|
| `Attention.value` (NeuroSky) | Índice propietario de atención derivado de EEG de un canal | `compute_psri_gaussian_log` (población global) → `R_attention` |
| `acc_std` | Inestabilidad de movimiento (ya calculada, reutilizada aquí — NO en S_estab) | `compute_psri_gaussian_log` → `R_acc_movement` |
| `Meditation.value` (NeuroSky) | Índice propietario de relajación/meditación derivado de EEG | `compute_psri_gaussian_log` → `R_meditation` |
| `self_valence_local_std`, `self_arousal_local_std` | Desviación típica de la autoanotación (`self_valence`/`self_arousal`) en una ventana rodante de 5 trials centrada en la ventana actual | `compute_psri_gaussian_log` → `R_self_valence`, `R_self_arousal` — descartado: 73.8%/68.3% de los valores son exactamente 0 (variable casi binaria, no continua) |
| `S_cond_att_acc` | = media(`R_attention`, `R_acc_movement`) | Variante del compuesto — señal pooled (ρ≈−0.12 sobre valence_var), sin soporte within |
| `S_cond_att_med` | = media(`R_attention`, `R_meditation`) | Variante alternativa, sin señal limpia |
| `S_cond_self` | = media(`R_self_valence`, `R_self_arousal`) | Estadísticamente enorme pero inválida (ver Sección 3.4) |
| `S_cond_combined` | = media de todos los `R_*` disponibles | Hereda el problema de `S_cond_self` |
| `S_cond_att` | = solo `R_attention` | Variante de control — sin señal limpia |
| `S_cond_att_std` | = `R_attention` calculado sobre std de Attention | Variante de control — sin soporte within |
| `S_cond_att_med_std` | = media(`R_attention_std`, `R_meditation_std`) | Variante de control — sin soporte within |
| `S_cond_att_mono` | = `R_attention_mono` (transformación MONOTÓNICA creciente, no en U) | Variante no-U — **captura señal within-de-sujeto que la U enmascara** (ver nota) |
| `S_cond_att_med_mono` | = media(`R_attention_mono`, `R_meditation_mono`) | Variante no-U |
| `S_cond_att_mono_acc_hi` | = media(`R_attention_mono`, `R_acc_mono_hi`), movimiento "más es mejor" | Variante no-U — señal pooled positiva (entre-sujetos) |
| `S_cond_att_mono_acc_lo` | = media(`R_attention_mono`, `R_acc_mono_lo`), movimiento "menos es mejor" | Variante no-U — señal pooled y within |

### 6.4. Combinación en el compuesto

| Campo | Significado | Transformación |
|---|---|---|
| `S_estab`, `S_coher_prev1`/`S_coher_prev5`, `S_cond_att_acc` | Los tres componentes, con imputación por mediana de la propia dimensión si falta algún valor | `compute_weighted_psri(S_estab, S_coher, S_cond, w1=w2=w3=1/3)` → `psri_composite_prev1` / `psri_composite_prev5` |
| `S_estab`, `S_coher_prev1`/`S_coher_prev5` (sin S_cond) | Compuesto de DOS patas — alternativa `PSRI_validated` (docs/dialogo.txt): no diluir los dos componentes nucleares con una tercera pata débil | `0.5·S_estab + 0.5·S_coher` (tras imputación por mediana) → `psri_composite_2leg_prev1` / `psri_composite_2leg_prev5` |

> Aquí (Sección 3.2.2): el cambio de signo observado tras añadir Meditation resultó ser un artefacto entre-sujetos y no el bug de duplicación de anotaciones — ver Sección 3.6.4 para el caso general de arousal_var.

**Compuesto de 2 patas (comparación confirmatoria, agosto 2026).** Al dejar la tercera pata fuera del compuesto (recomendación de docs/dialogo.txt), el resultado **no pierde potencia y recupera señal que la tercera pata enmascaraba**:

- **`external_valence_var`**: ρ ≈ −0.209 (prev1) / −0.181 (prev5) — comparable al de 3 patas (−0.205/−0.190), sobrevive Bonferroni.
- **`external_valence_range`**: ρ ≈ −0.205 / −0.187 — comparable (−0.191/−0.184), sobrevive Bonferroni.
- **`external_arousal_var`**: ρ ≈ +0.113 / +0.096 — igualmente significativo.
- **`self_partner_diff`**: ρ pooled ≈ +0.100 (prev1) / +0.080 (prev5) — **recupera significación** (p≈9e-8 prev1, sobrevive Bonferroni) que el compuesto de 3 patas había perdido (ρ≈+0.027, NS); **y además aparece señal within-de-sujeto** (ρ_within=+0.049, p=0.009 en prev1), ausente en 3 patas.
- **`self_external_mean_diff`**: ρ pooled ≈ +0.055 (prev1, p FDR≈0.004) — recupera significación (en 3 patas era NS).

La descomposición between/within muestra que el efecto principal sobre `external_valence_var` sigue siendo entre-sujetos (ρ_between≈−0.34, ρ_within≈+0.02, NS) tanto en 2 como en 3 patas — el compuesto de 2 patas no altera esa lectura. La ganancia es específica: **la tercera pata (S_cond_att_acc) diluía la señal pooled de `self_partner_diff` y `self_external_mean_diff`**. Salidas: `kemocon_psri_validation_combined_2leg.csv` y `kemocon_psri_leverage_psri_composite_{prev1,2leg_prev1}.csv`.

### 6.5. Variables de desacuerdo

Papel análogo a `RECORDS-acceptable/unacceptable` de PhysioNet — con una diferencia importante (ver nota final).

| Campo | Significado | Uso |
|---|---|---|
| `self_valence`, `self_arousal` | Autoanotación del propio participante sobre su emoción en esa ventana | Insumo para `self_partner_diff`, `self_external_mean_diff` y para `S_cond_self` |
| `partner_valence`, `partner_arousal` | Anotación de la pareja de debate sobre la emoción del participante | Insumo para `self_partner_diff` |
| `R1_valence` … `R5_valence` (y `_arousal`) | Anotación de cada uno de los 5 anotadores externos que ven la grabación | Insumo para `external_valence_var`, `external_valence_range`, `external_arousal_var` |
| `external_valence_var` | Varianza de R1..R5 en valence para esa ventana | **Variable objetivo principal** — significativo en pooled (ρ≈−0.19/−0.21) y dentro-de-sujeto en S_coher (ρ_within≈+0.06, p≈0.001, signo positivo) |
| `external_valence_range` | Máximo − mínimo de R1..R5 en valence | Target secundario — significativo pooled; dentro-de-sujeto sigue a valence_var (positivo en S_coher, nulo en S_estab) |
| `external_arousal_var` | Varianza de R1..R5 en arousal | **Significativo a nivel pooled** (ρ≈+0.10, p≈3.9e-8 en prev1), pero nulo dentro-de-sujeto — reportar solo a nivel pooled |
| `self_partner_diff` | `\|self_valence − partner_valence\|` | No significativo pooled; **significativo within en S_estab** (ρ_within≈+0.05, p≈0.004) |
| `self_external_mean_diff` | `\|self_valence − media(R1..R5 valence)\|` | No significativo pooled; efecto within residual negativo en S_coher (ρ_within≈−0.05, p≈0.012 prev1) |

**Nota final — la diferencia clave con la tabla de PhysioNet**: en PhysioNet, `RECORDS-acceptable/unacceptable` es un ground truth de calidad de señal etiquetado por expertos, contra el que se compara el PSRI para saber si el instrumento discrimina bien (esa es la pregunta: ¿funciona la fórmula?). Aquí, `external_valence_var` y el resto de métricas de desacuerdo no son un ground truth de calidad del PSRI — el PSRI ya se validó como instrumento en PhysioNet. Son la variable de interés científico de la pregunta de investigación en sí (¿la fiabilidad fisiológica predice el desacuerdo humano?), así que aquí el PSRI no se evalúa contra ellas, se correlaciona con ellas para poner a prueba la hipótesis. Es un cambio de rol: en PhysioNet el ground truth pregunta "¿el instrumento mide bien?"; en K-EmoCon la variable de desacuerdo pregunta "¿lo que el instrumento mide, importa?".

Para descartar que el efecto de S_coher dependiera de una elección arbitraria de la ventana de baseline, se repitió el cálculo de la correlación dentro de sujeto variando sistemáticamente n_prev de 1 a 10 trials previos. La Figura 1 muestra que `external_valence_var` mantiene un efecto **positivo** y significativo en las cinco primeras ventanas (ρ_within entre ≈ +0.066 y ≈ +0.046, p entre 0.0004 y 0.014), con atenuación marginal en n_prev=10 (+0.036, p=0.053); `external_valence_range` es significativo solo en las ventanas cortas (p=0.005 en prev1, p=0.068 en prev5).

**Figura 1**: Robustez del efecto de S_coher (HR–EDA) frente al tamaño de la ventana de baseline (n_prev = 1 a 10 trials previos). Círculos rellenos = p<0.05; marcadores vacíos = no significativo. `external_valence_var` (círculos) permanece significativo en las cinco primeras ventanas con ρ_within positivo alrededor de +0.05/+0.07, atenuándose a +0.036 (p=0.053) en n_prev=10.

**Figura 2**: Correlación de Spearman entre S_estab/S_coher y cinco métricas de desacuerdo, descompuesta en componente entre-sujetos (gris) y dentro-de-sujeto (azul). El asterisco marca p<0.05 en la componente within-subject. Dentro-de-sujeto, solo `external_valence_var`/`range` (en S_coher, **con signo positivo**) y `self_partner_diff` (en S_estab, positivo) son significativos; `external_arousal_var`, `self_external_mean_diff` y S_estab sobre valencia colapsan a ~0 al centrar por sujeto.

---

## Anexo. Métodos de validación: en qué consisten y su enfoque metodológico

Este anexo describe el propósito y el enfoque de cada comprobación de solidez estadística utilizada en el estudio, como soporte metodológico para la interpretación de los resultados. Ninguno de estos métodos es un "extra" cosmético: cada uno está diseñado para descartar una amenaza concreta a la validez de la conclusión. Se presentan en orden de menor a mayor especificidad respecto al diseño del estudio.

### A.1. Descomposición between/within-subject (centrado por sujeto)

**En qué consiste.** Una correlación calculada sobre todas las ventanas mezcladas mezcla dos niveles de variación: la que existe *entre* personas (algunas personas son, como rasgo, más estables o más discordantes que otras) y la que existe *dentro* de cada persona, momento a momento. La descomposición separa ambos niveles en dos correlaciones complementarias:

- **rho_between**: se reduce cada persona a un único punto (la media de cada variable en esa persona) y se correlacionan esas medias. Responde a: "¿las personas con fisiología en promedio más estable generan, en promedio, menos desacuerdo?".
- **rho_within**: a cada ventana se le resta la media de su propia persona (centrado por sujeto) y se correlacionan los residuos. Al restar la media, cualquier diferencia estable entre personas desaparece por construcción. Responde a la pregunta que el PSRI pretende contestar: "cuando la señal de una persona concreta es, para ella, más inestable de lo habitual, ¿aumenta el desacuerdo sobre ella en ese momento?".

**Qué amenaza controla.** La confusión entre "rasgo estable del sujeto" y "estado momento a momento". Un efecto que solo aparece en rho_full pero colapsa en rho_within no es evidencia trial-a-trial: es un artefacto de composición de la muestra (cuáles personas entraron). Un efecto que sobrevive al centrado es evidencia mucho más fuerte porque descarta la explicación alternativa más simple.

**Cómo se aplica aquí.** La descomposición se aplica **a cada índice por separado** — S_estab, S_coher (prev1 y prev5) y cada variante de S_cond — contra cada métrica de desacuerdo. Este fue el filtro que reveló que el efecto dentro-de-sujeto de S_coher sobre `external_valence_var` es **positivo** (ρ_within ≈ +0.06, p≈0.001), que S_estab no predice dentro-de-sujeto (ρ_within ≈ +0.01, NS), y que `self_partner_diff` gana significación within en S_estab.

### A.2. Leave-one-subject-out (LOSO)

**En qué consiste.** Se recalcula la correlación completa excluyendo un sujeto cada vez (23 réplicas), y se examina cuánto se mueve el coeficiente. Reporta el rango de rho obtenido y el cambio máximo absoluto (con qué sujeto).

**Qué amenaza controla.** Que el efecto esté concentrado en un único sujeto atípico (por ejemplo, una persona con valores extremos que por sí sola arrastra la correlación). Si al excluir al sujeto más influyente la correlación cambia poco y nunca invierte de signo, el resultado es una propiedad del conjunto, no de un caso.

**Cómo se aplica aquí.** Sobre cada componente y cada target. El resultado: los deltas máximos están entre ~0.02 y ~0.086, y el signo del efecto dentro-de-sujeto de S_coher nunca se invierte (p. ej. valence_var se mantiene positivo, entre +0.046 y +0.066 al excluir sujetos; arousal_var pooled entre +0.081 y +0.132).

### A.3. Corrección por comparaciones múltiples (Bonferroni y FDR)

**En qué consiste.** Cuando se ejecutan muchos tests a la vez, la probabilidad de obtener al menos un falso positivo crece con el número de tests (con 10 tests a α=0.05, ~40% de probabilidad de algún positivo espurio). La corrección reajusta los umbrales de significación. Se usan dos criterios complementarios:

- **Bonferroni**: divide el umbral entre el número de tests (el más estricto; controla la tasa de error *por familia*). Un resultado que sobrevive a Bonferroni es robusto al peor caso.
- **FDR (Benjamini-Hochberg)**: controla la proporción esperada de falsos positivos *entre los resultados declarados significativos* (más permisivo y con más potencia).

**Qué amenaza controla.** La inflación de falsos positivos por "pesca" en muchas hipótesis a la vez. Además, la forma de corregir importa: corregir cada familia de tests con su propio bloque (en lugar de un test suelto por bloque) es el estándar que impide que un gran número de tests diluya o, al revés, que tests relacionados se cuenten como independientes.

**Qué es exactamente cada test.** Cada test del bloque es una **correlación de Spearman** entre un valor del PSRI y una métrica de desacuerdo, calculada sobre todas las ventanas agrupadas (las 5528 ventanas de los 23 sujetos juntas, sin descomponer por sujeto — es decir, son los `rho_full`). El bloque principal es el producto cartesiano de dos dimensiones:

- **2 variantes de PSRI**: `psri_composite_prev1` y `psri_composite_prev5` (difieren solo en la ventana de baseline de S_coher: 1 vs 5 trials previos).
- **5 métricas de desacuerdo**: `external_valence_var`, `external_valence_range`, `external_arousal_var`, `self_partner_diff`, `self_external_mean_diff`.

Cada test responde a: "¿el PSRI compuesto se correlaciona con esta métrica de desacuerdo?", y produce un `rho` (magnitud y signo) y un `p` (probabilidad de observar ese rho si no hubiera relación real). La corrección opera *a posteriori* sobre los 10 p-valores ya calculados, como un único bloque: Bonferroni multiplica cada p por el nº de tests (estricto, controla la tasa de error de toda la familia); FDR (Benjamini-Hochberg) ordena los p-valores y los ajusta contra la proporción esperada de falsos positivos (más potencia). Sobreviven a Bonferroni en ambas variantes `external_valence_var` y `external_valence_range` (p_bonferroni < 1e-6), y `external_arousal_var` en `prev1` (ρ≈+0.101, p_bonferroni ≈ 3.9e-7); las discrepancias self-* no sobreviven.

**Cómo se aplica aquí.** El resultado principal se corrigió como **un único bloque de 10 tests** (2 variantes de baseline × 5 métricas de desacuerdo), con Bonferroni y FDR, con el mismo criterio de rigor que los 20 tests del Estudio 1 — el estándar no se relajó al pasar de un estudio a otro. El bloque diagnóstico de componentes (10 × 5 = 50 tests) tiene su propio bloque de corrección, por ser una pregunta distinta (de dónde viene el efecto), no mezclado con el confirmatorio. Existe además un bloque exploratorio de 5 tests por variante (`prev1` y `prev5` por separado), sin corregir como bloque oficial — no reportar en el paper.

Una sutileza clave: al estar agrupados entre sujetos, estos tests mezclan efecto between y within, así que el hallazgo no se apoya solo en ellos. El resultado pooled (compuesto vs `external_valence_var`, ρ≈−0.19/−0.21) es robusto a la corrección, pero **dentro-de-sujeto el signo de S_coher es el opuesto** (positivo) y S_estab es nulo (Sec. 3.6.1) — por eso la interpretación final depende de la descomposición, no solo del bloque pooled.

### A.4. Barrido de la ventana de baseline (n_prev)

**En qué consiste.** S_coher depende de un parámetro elegido por el investigador: cuántos trials inmediatamente anteriores se usan como baseline (`n_prev`). En lugar de fijar un único valor y reportarlo, se repite el cálculo de rho_within para una rejilla de valores (1, 2, 3, 5, 8 y 10) y se examina cómo responde el efecto.

**Qué amenaza controla.** La arbitrariedad de la elección de parámetro (p-hacking implícito por ajuste fino). Si el efecto es genuino, se espera un patrón característico: significativo en todo el rango, con magnitud que se atenúa de forma suave al ampliar la ventana (el efecto es de corto/medio plazo). Si fuera ruido de composición, el patrón sería errático o desaparecería en ventanas concretas.

**Cómo se aplica aquí.** Solo tiene sentido sobre S_coher (es el único índice con parámetro de baseline). Resultado: `external_valence_var` permanece significativo en las cinco primeras ventanas con ρ_within **positivo** (entre ≈ +0.066 y ≈ +0.046, p entre 0.0004 y 0.014), atenuándose a +0.036 (p=0.053) en la ventana más larga — patrón de efecto genuino pero de signo opuesto al hipotetizado. `external_valence_range` es significativo solo en las ventanas cortas (p=0.005 prev1, 0.068 prev5).

### A.5. Cobertura real frente a imputación

**En qué consiste.** El compuesto imputa con la mediana de la dimensión los valores faltantes antes de combinar. La cobertura mide qué fracción de ventanas tiene un valor *realmente calculado* (no imputado) en cada índice.

**Qué amenaza controla.** Que una correlación esté sostenida por valores imputados (fabricados por el procedimiento) en lugar de por datos reales. Si un índice tuviera baja cobertura real, su correlación podría ser un artefacto de la imputación, no de la señal.

**Cómo se aplica aquí.** S_estab y S_cond (att_acc) tienen ~100% de cobertura real (99.96%; siempre hay E4 disponible por el filtro de elegibilidad); S_coher tiene 98.8% (las ventanas sin baseline al inicio de cada sesión). El hallazgo de S_coher no depende de la imputación.

### A.6. Independencia entre componentes

**En qué consiste.** El compuesto asume que los tres eslabones (S_estab, S_coher, S_cond) son fuentes de información independientes y por eso usa pesos iguales. La comprobación cuantifica la correlación de Pearson entre cada par de componentes.

**Qué amenaza controla.** La redundancia entre dimensiones: si dos componentes estuvieran altamente correlacionados, el compuesto estaría doblemente contando la misma información y el "efecto convergente en dos mecanismos independientes" no sería tal. Correlaciones bajas entre pares justifican tanto los pesos iguales como la lectura de convergencia independiente.

**Cómo se aplica aquí.** Réplica directa de la Tabla 5 del Estudio 1 (mismo estadístico, Pearson, para ser comparable). Los valores bajos obtenidos refuerzan que el principio de "tres eslabones independientes" no es específico de la instanciación concreta (HR-pupila/reaction_time en EXIST vs. HR-EDA/Attention+acc en K-EmoCon).

### Resumen del enfoque general

Los seis métodos forman una cadena de defensa en capas, cada una contra una amenaza distinta: la descomposición between/within ataca la confusión por rasgo-sujeto, LOSO ataca la dependencia de un caso atípico, la corrección múltiple ataca la inflación de falsos positivos, el barrido de baseline ataca la arbitrariedad de parámetros, la cobertura ataca la fabricación por imputación, y la independencia entre componentes ataca la redundancia de la información. La robustez estadística se mantiene para el resultado pooled y para el signo positivo de S_coher dentro-de-sujeto; la interpretación sustantiva (dirección del efecto) es la opuesta a la hipótesis original.