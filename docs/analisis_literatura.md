# Análisis de Literatura — PSRI

Síntesis de `docs/analisis_literatura_ANNOTATE.odt` (análisis individual de ~30 papers + síntesis ejecutiva + deep-dives de calidad de señal). Objetivo: servir de base para el SOTA del paper y para las justificaciones metodológicas.

---

## 1. Posicionamiento en ECIR

El PSRI no es un paper de ingeniería biomédica: es un paper de IR. Encaja en tres pilares que ECIR busca explícitamente:

| Pilar | Encaje |
|---|---|
| **Evaluation research** (el más fuerte) | Crítica metodológica a cómo se evalúan los sistemas multimodales con señales fisiológicas. El "desacople poblacional" de EXIST que invalida las métricas es una contribución de evaluación pura. |
| **Societally-motivated IR** (gran oportunidad) | Enmarcar el trabajo como "cómo el ruido de los sensores introduce sesgo y distorsiona la evaluación humana en detección de discurso de odio/sexismo" → billete directo al track IR-for-Good. |
| **User aspects / Multimodal IR** | Modelar estado del usuario (carga cognitiva, atención, arousal) con señales fisiológicas como contexto para la IR. |

**La objeción que pondrá un revisor de ECIR** (primera línea de la revisión):

> "Muy bonito el índice PSRI y muy interesante el nulo en EXIST. Pero, al final del día, ¿mejora esto la precisión de un buscador o un clasificador? Si filtro por PSRI, ¿sube el F1? ¿Baja la latencia?"

- En IR, el oro son métricas de rendimiento (nDCG, MAP, F1, Precision, Recall). La correlación con la entropía no basta.
- PERO: EXIST usa una población disjunta (sujetos-sensores ≠ anotadores), por lo que **no se puede validar la mejora operativa en EXIST**. Hay que repensar la validación.
- Hay tradición fuerte en ECIR de papers que no suben el estado del arte, sino que **destapan fallos metodológicos graves** en cómo la comunidad evalúa. Este paper es exactamente eso: crítica de evaluación con propuesta de métrica de calidad (PSRI).

---

## 2. El gap central (la contribución del PSRI)

### Hallazgo principal de la revisión

> **Ninguno de los ~30 papers analizados trata la calidad de la señal como variable de diseño** (medible, cuantificable, utilizable para filtrar/ponderar anotaciones). Solo 6 los mencionan como problema a resolver con preprocesamiento (filtros, ICA, MARA), nunca como métrica. Abbey (2017) aborda la *calidad de respuesta* en encuestas (attention checks binarios) e Ibrahim et al. (2025) la *fiabilidad del anotador* (matrices de confusión), pero ambos sin señales fisiológicas → siguen sin cubrir el gap.

| Paper | ¿Menciona calidad de señal? | Cómo lo argumenta | ¿Propone un índice? |
|---|---|---|---|
| Mohr et al. (2017) | Sí | Como desafío del campo (variabilidad, artefactos, incertidumbre) | No |
| Dzedzickis et al. (2020) | Sí | Como problema a resolver con filtrado/preprocesamiento | No |
| Mukhopadhyay (2015) | Sí | Como fuente de falsas alarmas, necesidad de algoritmos robustos | No |
| Oguz et al. (2023) | Sí | Como ruido a eliminar con filtros estándar | No |
| Hollenstein et al. (2021) | Sí | Como necesidad de preprocesamiento para eliminar artefactos | No |
| Zhang et al. (2024) | Sí | Como paso previo (preprocesamiento con MARA) | No |
| Resto | No | No mencionan calidad de señal | No |

**Conclusión**: este es el gap más grande. El PSRI llena un vacío que la literatura no ha abordado de forma sistemática.

---

## 3. Papers de calidad de señal (núcleo del SOTA)

Estos 10 papers son los que más alinean con la premisa del PSRI y justifican su necesidad. Todos comparten la premisa "la señal fisiológica no es ground truth sin control de calidad", pero difieren en método (supervisado vs. no supervisado, sensor-específico vs. agnóstico). Sirven tanto para fundamentar la premisa como para diferenciar la contribución.

### 3.1 Nahmias & Kontson (2021) — Quantifying signal quality from unimodal and multimodal sources (ML)

- **Premisa idéntica al PSRI**: "quality is defined by how much of the acquired signal is from the source of interest and not noise from external or internal sources". Critica los SQI clásicos: "previous works have generally used a limited number of quantitative features, have not used noise sources directly, and/or have characterized signal quality into no more than three discrete categories".
- **Método**: 30 features cuantitativas (varianza, desviación estándar, energía, complejidad de Lempel-Ziv) + KDE para modelar distribuciones limpio/ruidoso + valor crítico de decisión bayesiana (νf*) → puntuación continua Q∈[0,1].
- **Alineación con PSRI**:
  - S_estab: su KDE bayesiano limpio-vs-ruidoso es conceptualmente equivalente a la campana gaussiana sobre z-scores del PSRI. Valida el uso de desviación estándar como feature discriminativa.
  - S_coher: para datos multimodales (EEG+IMU) usan DCNN para clasificar limpio/ruidoso; mismo principio (contrastar modalidades para inferir fiabilidad), distinto método (DL vs. decaimiento exponencial en V).
  - S_cond: reconocen que artefactos conductuales (parpadeo, EMG, movimiento) corrompen la señal — base de S_cond — pero no lo abordan como dimensión.
  - Agregación: promedian sub-scores de múltiples modalidades (media) — idéntico a la media de modalidades del PSRI.
- **Limitaciones que el PSRI supera**: requiere ground truth etiquetado; sensor-específico (features EEG); sin dimensión conductual.
- **Citas útiles**: definición operativa de calidad/fiabilidad; "Creating metrics to determine the quality of non-invasive electrophysiological recordings would inform those using the data how representative it is of the desired physiological source"; carencia de métodos para comparar algoritmos de eliminación de artefactos.

```bibtex
@article{nahmias2021quantifying,
  author={Nahmias, David O and Kontson, Kimberly L},
  title={Quantifying Signal Quality From Unimodal and Multimodal Sources: Application to EEG With Ocular and Motion Artifacts},
  journal={Frontiers in Neuroscience}, volume={15}, pages={566004}, year={2021},
  doi={10.3389/fnins.2021.566004}
}
```

### 3.2 Boulanger (2026) — Multi-modal DL robust arrhythmia screening

- **Premisa**: los artefactos de movimiento generan falsas alarmas: "85–99% of arrhythmia alerts are false positives, primarily caused by patient movement". Critica los SQI tradicionales: "aggressive quality thresholds reduce false positives but risk missing genuine arrhythmia that occurs during movement" → justifica un índice continuo como PSRI en vez de umbrales binarios.
- **Método**: fusión ECG + acelerómetro con attention gates aprendidos g∈[0,1] que ponderan cada modalidad según fiabilidad contextual; entrenamiento multi-SNR (24→-6 dB) para representaciones noise-invariant.
- **Alineación con PSRI**:
  - S_estab: la desviación estándar del acelerómetro σ|a| correlaciona linealmente con la tasa de falsos positivos (r=0.982) → **valida que la variabilidad de la señal es proxy fiable de su calidad**.
  - S_coher: es el núcleo del paper. La coherencia ECG-acelerómetro (ratio g_ECG/g_ACC decrece de 1.87 a 1.40 al degradar SNR) reduce falsos positivos un 67% → **evidencia empírica del principio de coherencia inter-modal**. Paralelismo directo con K-EmoCon: HR (cardiovascular) + EDA (simpático).
  - S_cond: clasifica intensidad de movimiento (Rest/Light/Moderate/Vigorous) — conducta física, pero **no atención/compromiso cognitivo** (dimensión que el PSRI añade).
- **Limitaciones**: requiere entrenamiento supervisado con datos etiquetados; sensor-específico (ECG+ACC); salida binaria (normal/arritmia), no índice de fiabilidad explícito.

```bibtex
@article{Boulanger2026,
  author={Boulanger, Pierre},
  title={From Motion Artifacts to Clinical Insight: Multi-Modal Deep Learning for Robust Arrhythmia Screening in Ambulatory ECG Monitoring},
  journal={Sensors}, volume={26}, number={4}, pages={1135}, year={2026},
  doi={10.3390/s26041135}
}
```

### 3.3 Hyun et al. (2023) — Unsupervised seq2seq signal quality assessment (CVS)

- **Premisa**: "labeling a large amount of CVS data is extremely costly... prone to inevitable human errors... the perfect annotation is almost impossible". → **valida el enfoque no supervisado del PSRI**.
- **Método**: auto-encoder punto-a-punto y ciclo-a-ciclo sobre cardiac volume signal; calidad = residual de reconstrucción; umbral τ con regla de 2-sigmas (sin etiquetas).
- **Alineación**: S_estab muy fuerte (variabilidad/contexto temporal como clave para distinguir señal de artefacto; la regla 2-sigma ≈ uso de σ). Resultado: ACC 0.957, AUC 0.948 — el enfoque no supervisado **compite con el supervisado** (ACC 0.967) → tesis directa del PSRI.
- **Limitaciones**: una sola modalidad; requiere entrenar LSTM-VAE (costoso); no cubre coherencia multimodal ni conducta. Sugiere además que el PSRI podría usarse como pseudo-etiquetado/filtro previo en sistemas supervisados.

```bibtex
@article{Hyun2023unsupervised,
  author={Hyun, Chang Min and Kim, Tae-Geun and Lee, Kyounghun},
  title={Unsupervised sequence-to-sequence learning for automatic signal quality assessment in multi-channel electrical impedance-based hemodynamic monitoring},
  journal={Biomedical Signal Processing and Control}, volume={86}, pages={105273}, year={2023},
  doi={10.1016/j.bspc.2023.105273}
}
```

### 3.4 Gupta, Khomami Abadi et al. (2016) — Quality-adaptive multimodal affect recognition (ICMR)

- **Premisa**: "the signals from the above mentioned modalities are often contaminated with various sources of noise, that significantly hinders the task of affect recognition". Los sistemas multimodales no pueden asumir ground truth válido; la fusión multimodal sin control de calidad puede **empeorar** el rendimiento.
- **Método**: Signal Quality Estimators (SQEs) supervisados por modalidad (EEG, ECG, GSR, head-pose) con features mayoritariamente estadísticas; fusión adaptativa q_i·t_i con q_i∈{0,1} binario. Validación cross-subject.
- **Alineación**: features estadísticas puras (media, mediana, skew, kurtosis) ≈ S_estab; la fusión adaptativa pondera menos las modalidades de baja calidad ≈ idea del PSRI como peso continuo; head-pose como conducta (sigue el ritmo) ≈ S_cond. Reducción de tasa de rechazo de muestras (de 22.58% uni-modal a 0% multimodal) → la adaptación a calidad mejora el rendimiento.
- **Limitaciones clave para diferenciarse**: requiere anotadores expertos (κ≥0.73); calidad **binaria** (buena/mala), pierde granularidad; no calcula coherencia entre modalidades; no aborda atención/compromiso. → El PSRI es no supervisado, continuo, con S_coher y S_cond.

```bibtex
@inproceedings{Abadi2016quality,
  author={Gupta, Rishabh and Khomami Abadi, Mojtaba and Cárdenes Cabré, Jesús Alejandro and Morreale, Fabio and Falk, Tiago H. and Sebe, Nicu},
  title={A Quality Adaptive Multimodal Affect Recognition System for User-Centric Multimedia Indexing},
  booktitle={Proceedings of the 2016 ACM on International Conference on Multimedia Retrieval (ICMR)}, pages={317--320}, year={2016},
  doi={10.1145/2911996.2912059}
}
```

### 3.5 Liu et al. (2024) — Taxonomy & real-time classification of ECG artifacts

- **Aporta** una taxonomía exhaustiva de artefactos de ECG y 11 features estadísticas (mean, median, std, MAD, variance, RMS, max, min, IQR, kurtosis, skewness) — casi idénticas a la base de S_estab → **valida el uso de features estadísticas puras** en vez de morfológicas.
- **Limitación que justifica el PSRI**: los artefactos de movimiento (TA-M, TA-S) son difíciles de clasificar con una sola modalidad → refuerza la necesidad de S_coher (multimodal) para validar la coherencia entre sensores.
- Nota: la variabilidad intra-sujeto puede ser tan alta que un índice no supervisado como S_estab solo no basta → respalda la adición de S_coher y S_cond.

```bibtex
@article{Liu2024taxonomy,
  author={Liu, Hui and Zhang, Shiyao and Gamboa, Hugo and Xue, Tingting and Zhou, Congcong and Schultz, Tanja},
  title={Taxonomy and Real-Time Classification of Artifacts During Biosignal Acquisition: A Starter Study and Dataset of ECG},
  journal={IEEE Sensors Journal}, volume={24}, number={6}, pages={9162--9171}, year={2024},
  doi={10.1109/JSEN.2024.3356651}
}
```

### 3.6 Ronca et al. (2026) — Real-world benchmarking of wearable EEG

- **Premisa**: los dispositivos consumer-grade (Muse S, Emotiv EPOC X, Mindtooth Touch) **no son fiables por defecto**; su fiabilidad en tareas ecológicas "remains largely unexplored". Miden % de artefactos y estabilidad espectral (PSD, correlación primera/segunda mitad) — análogos a S_estab.
- **Hallazgo**: los dispositivos de mayor calidad de señal correlacionan mejor con la experiencia subjetiva del usuario → **la calidad de señal afecta a la validez de los resultados** (exactamente la premisa del PSRI aplicada a dispositivos portátiles como los de EXIST).
- Neurométricas (vigilancia, atención, carga mental) como proxies de estado cognitivo ≈ S_cond, pero no miden conducta.
- Uso en el paper: evidencia empírica de que la calidad de señal no es un problema menor y justifica un filtro de calidad previo a cualquier análisis.

```bibtex
@article{Ronca2026beyond,
  author={Ronca, Vincenzo and Cecchetti, Marianna and Capotorto, Rossella and Di Flumeri, Gianluca and Giorgi, Andrea and Germano, Daniele and Borghini, Gianluca and Babiloni, Fabio and Arico, Pietro},
  title={Beyond the lab: real-world benchmarking of wearable EEGs for passive brain-computer interfaces},
  journal={Brain Informatics}, volume={13}, pages={3}, year={2026},
  doi={10.1186/s40708-025-00290-x}
}
```

### 3.7 Gao et al. (2021) — Reliability of self-report data (UbiComp/ISWC)

- **Premisa**: los autoinformes no son ground truth fiable por defecto; están sujetos a sesgo, subjetividad y falta de atención.
- **Hallazgo clave**: "participants with similar physiological patterns may report very different perceived engagement and participants with similar self-report annotations may also have very different physiological patterns" → la señal fisiológica tampoco es ground truth perfecto → respalda la necesidad de evaluar su fiabilidad antes de usarla.
- **Alineación fuerte con S_cond**: usan confianza auto-reportada y tiempo de finalización del cuestionario como indicadores de fiabilidad de respuestas → análogo directo a S_cond (tiempos de reacción, parpadeo), pero el PSRI usa métricas objetivas en vez de autoinformes (también subjetivos).
- Cita: "it will be interesting to use survey completion time as an indicator of survey reliability and assign appropriate weights to self-report responses" → exactamente lo que el PSRI hace, pero aplicado a señales fisiológicas.
- Uso en el paper: evidencia de que el problema de fiabilidad del ground truth es general, no específico de EXIST.

```bibtex
@inproceedings{Gao2021reliability,
  author={Gao, Nan and Rahaman, Mohammad Saiedur and Shao, Wei and Salim, Flora D.},
  title={Investigating the Reliability of Self-report Data in the Wild: The Quest for Ground Truth},
  booktitle={Adjunct Proceedings of the 2021 ACM International Joint Conference on Pervasive and Ubiquitous Computing (UbiComp-ISWC '21 Adjunct)}, pages={1--6}, year={2021},
  doi={10.1145/3460418.3479338}
}
```

### 3.8 Chatzaki & Tsiknakis (2025) — Stress analysis: systematic review of open datasets

Revisión sistemática de datasets abiertos de estrés. Es el único paper del conjunto que propone **prácticas concretas de fiabilidad** en el pipeline completo:
- **Preprocesamiento**: filtros, min-max, ventanas deslizantes (60 s, 50% overlap); ICA + DSWT para artefactos EEG; segmentación con ventanas cortas/largas (descubre que las largas son mejores para EEG, las cortas para señales periféricas).
- **Sincronización multimodal**: timestamps desde el PC de estímulos, dispositivo maestro (Empatica E4 como reloj), picos de señal extremos (doble toque en WESAD), software ad-hoc.
- **Protocolo**: entornos controlados, bloques de línea base/relajación para calibrar frente a estado neutro.
- **Etiquetado**: enfoques híbridos (SAM/PANAS/STAI + anotación externa + validación biométrica como cortisol).
- **Desbalanceo**: SMOTE, ADASYN, weighted loss.
- **Evaluación**: LOSO y Leave-One-Trial-Out para testear generalización a nuevos sujetos/ensayos.
- Cita clave: "por muy avanzados que sean los modelos computacionales, la fiabilidad de los resultados sigue dependiendo en gran medida de la calidad de las señales de entrada".

```bibtex
@article{Chatzaki2025,
  author={Chatzaki, C. and Tsiknakis, M.},
  title={An Overview of Stress Analysis Based on Physiological Signals: Systematic Review of Open Datasets and Current Trends},
  journal={Sensors}, volume={25}, number={23}, pages={7108}, year={2025},
  doi={10.3390/s25237108}
}
```

### 3.9 Del Pup & Atzori (2023) — Self-supervised learning for biomedical signals (survey)

- Revisión de SSL aplicado a biosensores. Señala que las señales fisiológicas tienen alta variabilidad y artefactos (interferencias oculares/musculares en EEG, deriva de electrodos, ruido ambiental, heterogeneidad de protocolos).
- Aporta la cita que identifica el gap de forma explícita:
  > "If these data are to be effectively applied and correctly interpreted, it is important to understand the quality of data being recorded."
- Validez metodológica: el SSL puede aprender representaciones robustas, pero su eficacia depende de distinguir patrones fisiológicos genuinos de degradaciones espurias → refuerza la necesidad de una métrica de calidad previa (PSRI).

```bibtex
@article{delpup2023applications,
  author={Del Pup, Federico and Atzori, Manfredo},
  title={Applications of Self-Supervised Learning to Biomedical Signals: a Survey},
  journal={IEEE Access}, volume={11}, pages={144680--144704}, year={2023},
  doi={10.1109/ACCESS.2023.3344531}
}
```

### 3.10 Adams (2026) — High-reliability signal quality validation via sensor fusion + software indices (Sensors)

- **Premisa**: la calidad de una señal biomédica no se puede validar con un único índice; propone un **marco híbrido en dos etapas**: (I) gating de integridad del hardware (IMU + impedancia de electrodos) y (II) SQIs de software (plausibilidad RR, DTW morfológico frente a plantillas adaptativas, SNR por frecuencia, baseline wander), con salida **ordinal 1–6**.
- **Método**: índices sin supervisión + umbrales calibrados en dataset independiente (τ_motion=0.05 m²/s⁴, Z_min=10 kΩ, RR 240–1200 ms, DTW k=2.0, θ_SNR=10 dB, θ_baseline=0.75).
- **Validación**: 8644 latidos ECG, 20 participantes → **98.13% acc, 98.81% sens, 96.70% spec**; el **DTW morfológico es el discriminador individual más fuerte (90.4%, +7.89 pp)**; supera a Orphanidou, Behar, Liu, Zhou, Fu y Fotsing Kuetche.
- **Alineación con PSRI**:
  - S_estab: la morfología estable (DTW vs. plantilla) como mejor proxy de calidad → respalda features morfológicas/estabilidad.
  - S_coher: la **fusión sensor (IMU+impedancia) + software** reduce falsos positivos → valida el gating multimodal; HR consistente entre ECG/ICG/PPG como señal de coherencia.
  - S_cond: el IMU captura movimiento físico (conducta) → base de S_cond.
- **Limitaciones que el PSRI supera**: requiere hardware sincronizado (IMU/impedancia) no disponible en wearables como el E4; calibración de umbrales supervisada; solo ECG → el PSRI es software-only, agnóstico al sensor y sin calibrar.

```bibtex
@article{Adams2026signalquality,
  author={Adams, Basel},
  title={High-Reliability Signal Quality Validation for Biosignals Using Sensor Fusion and Software Indices},
  journal={Sensors}, volume={26}, number={11}, pages={3478}, year={2026},
  doi={10.3390/s26113478}
}
```

---

## 4. Papers de señales fisiológicas en NLP/IR (contexto y lecciones metodológicas)

Síntesis de los análisis individuales restantes, ordenados por relevancia. Ninguno reenfoca el planteamiento; todos validan que las señales fisiológicas contienen información cognitiva/subjetiva y que la fusión multimodal es beneficiosa.

### 4.1 Datasets con anotación subjetiva (validan la hipótesis)

**GAZE4HATE** — Detección de hate speech con gaze (ET). 43 participantes, 90 frases sobre mujeres (3870 instancias). Anotaciones: hatefulness 1-7, confianza, racionales. Alemán. Abierto CC-BY-NC (gitlab.ub.uni-bielefeld.de/clause/gaze4hate). Modelo MEANION integra gaze con LMs.
- **Relevancia**: el sujeto que genera la señal ET es el mismo que anota → permite probar directamente la hipótesis original (señal fisiológica → subjetividad de anotación) que EXIST no permitía por el desacople poblacional.
- Lección: las métricas de fijación (no solo pupila) son predictores robustos de la subjetividad → candidatas a incluir en S_estab/S_cond.

**eyeStyliency** — Saliencia de estilo textual con ET (cortesía, sentimiento). 20 participantes, 90 items, EyeLink 1000 Plus a 1000 Hz. Abierto (github.com/minnesotanlp/eyeStyliency).
- Lecciones: **dwell time es la métrica ET más robusta**; la normalización **z-score a nivel de participante** mejora la señal (alternativa/complemento a la MAD del PSRI); el ET captura información (verbos, adverbios) que los humanos no seleccionan (adjetivos) → el ET complementa, no sustituye, la anotación.

**ZuCo** (Hollenstein et al., 2018/2021) — EEG + ET simultáneos en lectura naturalista. 12-18 participantes, ~400-350 oraciones/tarea. Anotación: sentiment (SST), relation detection. Abierto (osf.io/q3zws, osf.io/2urht).
- **Único corpus público con EEG+ET simultáneos** → validar S_coher (EEG+ET) en dominio lingüístico. No tiene desacuerdo entre anotadores → no sirve para la hipótesis central.
- Lecciones: las **bandas de frecuencia** (theta, alpha, beta, gamma) son más informativas que la señal completa; la **fusión late** supera a la early; el EEG aporta más cuando hay pocos datos (ablation study); el EEG mejora más tareas simples (sentiment) que complejas (relation detection).

**CEAP-360VR** (Xue et al., 2023) — Dataset de anotación emocional continua en VR 360°. 32 participantes, 8 clips 360° de 60 s, Vive Pro Eye (ET 120 Hz) + Empatica E4 (HR, EDA, SKT, BVP, IBI). Anotación continua de valencia-arousal dentro del HMD (joystick) + SAM within-VR. Abierto (github.com/cwi-dis/CEAP-360VR-Dataset, CC-BY-NC 4.0).
- **Relevancia**: protocolo de anotación continua casi idéntico a K-EmoCon; **ablación explícita**: conducta + fisiológica > cada una por separado → valida S_coher; **normalización z-score de la pupila a nivel de participante con corrección de luminancia** → respalda S_estab; consistencia de saliencia HM/EM entre sujetos (CC>0.8) → coherencia conductual inter-sujeto.
- ICC de anotaciones: SAM V ICC=0.984, A ICC=0.951; acuerdo continua↔SAM: V ICC=0.882, A ICC=0.714.
- Limitación: sujeto=anotador sin desacuerdo → no sirve para la hipótesis central.

### 4.2 Metodológicos (enseñan cómo integrar señales en modelos)

| Paper | Lección clave para el PSRI |
|---|---|
| **Sood et al. (2020)** — gaze-guided neural attention (TSM) | Integrar señales cognitivas en la **atención** del modelo (no solo como feature); **joint training** adapta la señal a la tarea; modelos híbridos (cognitivo E-Z Reader + data-driven) superan a los puramente data-driven; usar **predicciones** de la señal (no solo mediciones) funciona. → PSRI como señal de regularización/atención, no solo filtro. |
| **Vazquez-Rodriguez et al. (2022)** — Transformer SSL para emociones (ECG) | **Pre-entrenamiento autosupervisado** (masked prediction) supera la escasez de datos; transformadores superan a CNN/LSTM en señales fisiológicas; CLS token para representación global. → pre-entrenar modelos de fiabilidad con datos no etiquetados. |
| **Zhang et al. (2024)** — LLM + EEG + ET para relevancia de palabras (ZuCo) | **Los LLMs (GPT-3.5/4) pueden generar ground truth** cuando faltan anotaciones humanas → plan B para generar etiquetas de subjetividad/dificultad; la integración EEG+ET supera a EEG solo; alineación fixation-locked EEG es crucial; preprocesamiento MARA/EOG esencial. |
| **Das et al. (2017)** — VQA-HAT | La atención humana es dependiente de la tarea (diferentes preguntas → diferentes regiones). **No usa ET real** (interfaz de desenfoque) → no relevante para PSRI. Solo validación conceptual débil. |
| **Barrett et al.** — Weakly supervised PoS tagging con ET | **La agregación a nivel de tipo (promedio sobre todas las ocurrencias de una palabra) reduce ruido** → valida el promediado entre sujetos de EXIST y cualquier agregación de señales. Corpus Dundee restringido y sin anotación subjetiva → baja prioridad. |
| **Yang et al. (2023)** — BIOT, biosignal transformer | Tokeniza cada canal por separado (segmentos + embeddings de canal/posición) → maneja **canales no coincidentes, longitudes variables y valores faltantes**; normalización por **percentil 95** (robusta a outliers); pre-entrenamiento autosupervisado mejora downstream (+4pp). → respalda la normalización robusta del PSRI y la filosofía de descartar tokens corruptos en vez de imputar. |
| **Kumar et al. (2024)** — AER multimodal ECG+EDA (CASE/WESAD) | Multimodal (EDA+ECG) > unimodal (CASE: 86.66% vs 84.58% vs 81.04%); **cvxEDA** separa tónica/fásica → analogía con separar componente estable de respuesta; CWT > STFT > MFC. CASE es dataset de anotación continua más para el catálogo. |
| **Dar et al. (2020)** — CNN+LSTM ECG/EEG/GSR (AMIGOS/DREAMER) | Señales mal clasificadas tenían **ruido residual que el preprocesamiento estándar no eliminó** → evidencia directa del gap; fusión por majority voting eleva ACC pero **añadir una modalidad débil (GSR) la degrada** → lección de ponderación por calidad (el PSRI descontaría GSR de baja calidad). |

### 4.3 Survey / contextuales

| Paper | Valor para el paper |
|---|---|
| **Barrett & Hollenstein (2020)** — Survey ET en NLP | Estado del arte + catálogo de corpora (Dundee, GECO, Provo, ZuCo, CFILT). Valida que el ET es útil en tareas **subjetivas** (sarcasmo, sentimiento, hate speech); agreación type-level y multi-task learning como estrategias para usar ET sin tenerlo en test. |
| **Mohr et al. (2017)** — Personal sensing | Marco conceptual jerárquico (raw → features → behavioral markers → clinical targets) análogo al PSRI. "The Curse of Variability": variabilidad entre dispositivos/personas/entornos como desafío central → refuerza la normalización MAD robusta. El más cercano al gap, pero sin métrica. |
| **Dzedzickis et al. (2020)** — Revisión sensores para emociones | Tabla 14: ECG, EEG, GSR son las técnicas más precisas; la **fusión multimodal** es práctica recomendada; ECG/EEG sensibles a artefactos de movimiento → justifica validación en PhysioNet y S_estab. |
| **Mukhopadhyay (2015)** — Wearables | Survey tecnológico; artefactos de movimiento como fuente de falsas alarmas; necesidad de algoritmos robustos con datos incompletos. Valor puramente contextual. |
| **Oguz et al. (2023)** — Emoción con ECG + feature engineering | Feature engineering automatizada mejora la clasificación (→ optimizar pesos del compuesto); BiLSTM captura dependencias temporales; tres tipos de ruido en ECG (baseline drift, power line, HF). MAHNOB-HCI sin desacuerdo de anotadores → no central. |
| **Survey ET-NLP (Mathias et al., 2020)** — gaze behaviour | Catálogo de corpora multilingüe; scanpath complexity y regresiones ≈ S_cond; el gaze es una señal distribuida → justifica el compuesto PSRI; señales fisiológicas como "fortuitous data" (Plank, 2016). |
| **Abbey & Meloy (2017)** — Attention checks (JOM) | Filtrar por atención mejora la **validez de constructos** (85–91% de ajustes, p<0.01) pero **no** la significación experimental (44–61% n.s.); pérdida media de muestra **35.79%** (hasta 69%) → riesgo de Type II. → **predice el resultado nulo del experimento3** y obliga a reportar el coste del filtrado por calidad. |
| **Ibrahim et al. (2025)** — Crowdsourcing noisy labels (IEEE SPM) | Marco formal del desacuerdo: matrices de confusión por anotador (Dawid-Skene, EM, crowdlayer E2E); error decae exponencialmente con el nº de anotadores. Los métodos son **agnósticos al dato físico** → el PSRI aporta la dimensión fisiológica que estos modelos ignoran (complementariedad directa, no competencia). |
| **EEG+ChatGPT (revisión conceptual)** | Propuesta EEG+NLP para psicoterapia virtual; sin datos ni métodos. Solo mención introductoria; cita DEAP/DREAMER/SEED (no relevantes). |

---

## 5. Catálogo de datasets consolidado

| Dataset | Señales | Anotación | Desacuerdo entre anotadores | Relevancia |
|---|---|---|---|---|
| **K-EmoCon** | EEG, BVP, EDA, temperatura, HR | Auto, pareja, observadores (tripartita) | ✅ Sí (Krippendorff's alpha) | **CRÍTICO** — hipótesis central |
| **PhysioNet/CinC 2011** | ECG | Calidad técnica (acceptable/unacceptable) | ❌ No (ground truth objetivo) | **CRÍTICO** — validar S_estab |
| **ZuCo** | EEG + ET simultáneos | Sentiment, relations | ❌ No | **RELEVANTE** — validar S_coher (EEG+ET) |
| **GAZE4HATE** | ET | Hate speech (subjetivo) | ❌ No (sí subjetivo) | **INTERESANTE** — S_estab en hate speech, sujeto=anotador |
| **eyeStyliency** | ET (1000 Hz) | Saliencia de estilo | ❌ No | **COMPLEMENTARIO** — S_estab en estilo |
| **CEAP-360VR** | ET, pupila, HR, EDA, SKT (VR 360°) | Valence/arousal continuo (joystick) + SAM | ❌ No (sujeto=anotador) | **COMPLEMENTARIO** — S_estab (pupila z-score) y S_coher (ablación conducta+fisiológica) |
| **CASE** | ECG, PPG, EDA, RSP, SKT, EMG | 4 emociones, anotación continua joystick | ❌ No | Complementario — S_estab/S_coher |
| **WESAD** | ECG, EDA, BVP, temp, resp (RespiBAN+E4) | Amusement/neutral/stress | ❌ No | Complementario — S_estab/S_coher |
| **CITL/Scansam** | ET | Sentimiento (subjetivo) | — | Complementario (subjetividad) |
| **MAHNOB-HCI** | EEG, ECG, GSR, temp, resp, ET | Valence/arousal/dominance (auto-reporte) | ❌ No | Sólo S_estab/S_coher; sin desacuerdo |
| DEAP, DREAMER, AMIGOS, ASCERTAIN | EEG, ECG... | Valence/arousal (auto-reporte) | ❌ No | NO RELEVANTES para la hipótesis central; útiles solo para S_estab/S_coher |
| Dundee, GECO, Provo | ET | PoS, dependencias, NER | ❌ No | NO RELEVANTES (sin subjetividad) |
| StudentLife, MONARCA, CrossCheck | Smartphone | Salud mental | ❌ No | NO RELEVANTES |

**Ruta de validación clara**: PhysioNet → S_estab (calidad técnica); K-EmoCon → hipótesis central (PSRI → desacuerdo); ZuCo → S_coher (EEG+ET) en dominio lingüístico; GAZE4HATE/eyeStyliency como complementos.

---

## 6. Lecciones metodológicas reutilizables

| Lección | Aplicación al PSRI |
|---|---|
| **Agregación a nivel de tipo reduce ruido** (Barrett et al.; surveys ET) | Valida el promediado entre sujetos de EXIST; aplicar a estímulos repetidos si existen |
| **Dwell time es la métrica ET más robusta** (eyeStyliency) | Complemento/alternativa a σ de pupila en S_estab |
| **Normalización z-score a nivel de participante** (eyeStyliency) | Alternativa/complemento a la MAD del PSRI |
| **El gaze es una señal distribuida** (survey ET) | Justifica el compuesto S_estab+S_coher+S_cond |
| **Fusión multimodal (late) mejora la precisión** (Dzedzickis; Hollenstein) | Valida S_coher y el enfoque multimodal |
| **Bandas de frecuencia > señal completa** (Hollenstein) | Para EEG crudo futuro, usar bandas theta/alpha/beta/gamma |
| **Joint training / señal en la atención** (Sood et al.) | PSRI como regularización/atención, no solo filtro |
| **Pre-entrenamiento autosupervisado** (Vazquez-Rodriguez; Del Pup) | Pre-entrenar modelos de fiabilidad con datos sin etiquetar |
| **LLMs como generadores de ground truth** (Zhang et al.) | Plan B para etiquetas de subjetividad cuando faltan anotaciones humanas |
| **Regla 2-sigma / campana gaussiana** (Hyun et al.) | Respaldan la transformación robusta de S_estab |
| **LOSO como validación cross-user** (Abadi et al.; Chatzaki et al.) | Respaldan la exigencia de generalización a sujetos no vistos |
| **Variabilidad entre dispositivos/entornos** (Mohr et al.) | Refuerza la normalización MAD robusta y el filtrado por calidad |
| **Normalización por percentil 95** (BIOT) | Alternativa robusta a outliers para S_estab |
| **Descarte de tokens corruptos sin imputar** (BIOT) | Filosofía del PSRI como filtro: eliminar señal de mala calidad |
| **Filtrar por calidad mejora la medida pero no la predicción** (Abbey) | Contextualiza el nulo del experimento3; el PSRI es índice de fiabilidad de medida |
| **Pérdida de muestra al filtrar (~36%)** (Abbey) | Reportar el coste del filtrado por calidad como límite del PSRI |
| **Ruido residual que el preprocesamiento no elimina** (Dar) | Evidencia empírica del gap: los filtros estándar no bastan |
| **Añadir una modalidad débil degrada la fusión** (Dar) | El PSRI debe ponderar/descontar modalidades por su calidad (S_estab/S_coher/S_cond) |
| **Matrices de confusión por anotador** (Ibrahim et al.) | Marco formal del desacuerdo; PSRI como prior fisiológico en modelos DS/EM/crowdlayer |

---

## 7. Implicaciones para el paper / hoja de ruta

1. **PhysioNet** — validar S_estab (calidad técnica). ✅ Hecho (AUC 0.887).
2. **K-EmoCon** — validar hipótesis central (PSRI → desacuerdo). 🔴 Urgente, principal objetivo.
3. **ZuCo / GAZE4HATE / eyeStyliency** — validaciones complementarias (S_coher EEG+ET; S_estab en subjetividad).
4. **LLMs para ground truth** — plan B si faltan anotaciones.
5. **Joint training** — extensión futura (PSRI como regularización en modelos NLP).

**Mensajes clave para el paper**:
- Ninguno de los ~30 papers analiza la calidad de señal como variable de diseño → el PSRI es único y necesario.
- K-EmoCon es el dataset crítico (anotación tripartita con desacuerdo documentado).
- Los papers complementan con técnicas (fusión multimodal, joint training, SSL, LLM-ground-truth), pero **ninguno reenfoca** el planteamiento porque la fiabilidad de señal no está cubierta.
- Abbey (2017) e Ibrahim et al. (2025) confirman que filtrar/ponderar por fiabilidad es un problema reconocido en encuestas y crowdsourcing, pero **sin señales fisiológicas** → el PSRI es el puente entre ese marco y la calidad de señal.
- Adams (2026) es el más cercano a un índice de calidad reutilizable, pero exige hardware (IMU/impedancia) y calibración supervisada → el PSRI sigue diferenciado por ser software-only, agnóstico y sin calibrar.

---

## 8. Referencias BibTeX completas

```bibtex
% Calidad de señal (núcleo SOTA)
@article{nahmias2021quantifying,
  author={Nahmias, David O and Kontson, Kimberly L},
  title={Quantifying Signal Quality From Unimodal and Multimodal Sources: Application to EEG With Ocular and Motion Artifacts},
  journal={Frontiers in Neuroscience}, volume={15}, pages={566004}, year={2021},
  doi={10.3389/fnins.2021.566004}
}
@article{Boulanger2026,
  author={Boulanger, Pierre},
  title={From Motion Artifacts to Clinical Insight: Multi-Modal Deep Learning for Robust Arrhythmia Screening in Ambulatory ECG Monitoring},
  journal={Sensors}, volume={26}, number={4}, pages={1135}, year={2026},
  doi={10.3390/s26041135}
}
@article{Hyun2023unsupervised,
  author={Hyun, Chang Min and Kim, Tae-Geun and Lee, Kyounghun},
  title={Unsupervised sequence-to-sequence learning for automatic signal quality assessment in multi-channel electrical impedance-based hemodynamic monitoring},
  journal={Biomedical Signal Processing and Control}, volume={86}, pages={105273}, year={2023},
  doi={10.1016/j.bspc.2023.105273}
}
@inproceedings{Abadi2016quality,
  author={Gupta, Rishabh and Khomami Abadi, Mojtaba and Cárdenes Cabré, Jesús Alejandro and Morreale, Fabio and Falk, Tiago H. and Sebe, Nicu},
  title={A Quality Adaptive Multimodal Affect Recognition System for User-Centric Multimedia Indexing},
  booktitle={Proceedings of the 2016 ACM on International Conference on Multimedia Retrieval (ICMR)}, pages={317--320}, year={2016},
  doi={10.1145/2911996.2912059}
}
@article{Liu2024taxonomy,
  author={Liu, Hui and Zhang, Shiyao and Gamboa, Hugo and Xue, Tingting and Zhou, Congcong and Schultz, Tanja},
  title={Taxonomy and Real-Time Classification of Artifacts During Biosignal Acquisition: A Starter Study and Dataset of ECG},
  journal={IEEE Sensors Journal}, volume={24}, number={6}, pages={9162--9171}, year={2024},
  doi={10.1109/JSEN.2024.3356651}
}
@article{Ronca2026beyond,
  author={Ronca, Vincenzo and Cecchetti, Marianna and Capotorto, Rossella and Di Flumeri, Gianluca and Giorgi, Andrea and Germano, Daniele and Borghini, Gianluca and Babiloni, Fabio and Arico, Pietro},
  title={Beyond the lab: real-world benchmarking of wearable EEGs for passive brain-computer interfaces},
  journal={Brain Informatics}, volume={13}, pages={3}, year={2026},
  doi={10.1186/s40708-025-00290-x}
}
@inproceedings{Gao2021reliability,
  author={Gao, Nan and Rahaman, Mohammad Saiedur and Shao, Wei and Salim, Flora D.},
  title={Investigating the Reliability of Self-report Data in the Wild: The Quest for Ground Truth},
  booktitle={Adjunct Proceedings of the 2021 ACM International Joint Conference on Pervasive and Ubiquitous Computing (UbiComp-ISWC '21 Adjunct)}, pages={1--6}, year={2021},
  doi={10.1145/3460418.3479338}
}
@article{Chatzaki2025,
  author={Chatzaki, C. and Tsiknakis, M.},
  title={An Overview of Stress Analysis Based on Physiological Signals: Systematic Review of Open Datasets and Current Trends},
  journal={Sensors}, volume={25}, number={23}, pages={7108}, year={2025},
  doi={10.3390/s25237108}
}
@article{delpup2023applications,
  author={Del Pup, Federico and Atzori, Manfredo},
  title={Applications of Self-Supervised Learning to Biomedical Signals: a Survey},
  journal={IEEE Access}, volume={11}, pages={144680--144704}, year={2023},
  doi={10.1109/ACCESS.2023.3344531}
}
@article{Adams2026signalquality,
  author={Adams, Basel},
  title={High-Reliability Signal Quality Validation for Biosignals Using Sensor Fusion and Software Indices},
  journal={Sensors}, volume={26}, number={11}, pages={3478}, year={2026},
  doi={10.3390/s26113478}
}
@article{Diachenko2022improved,
  author={Diachenko, Marina and Houtman, Simon J and Juarez-Martinez, Erika L and Ramautar, Jennifer R and Weiler, Robin and Mansvelder, Huibert D and Bruining, Hilgo and Bloem, Peter and Linkenkaer-Hansen, Klaus},
  title={Improved Manual Annotation of EEG Signals through Convolutional Neural Network Guidance},
  journal={eNeuro}, volume={9}, number={5}, pages={ENEURO.0160--22.2022}, year={2022},
  doi={10.1523/ENEURO.0160-22.2022}
}

% Metodológicos / señales en NLP
@inproceedings{Sood2020improving,
  author={Sood, Ekta and Tannert, Simon and M{\"u}ller, Philipp and Bulling, Andreas},
  title={Improving Natural Language Processing Tasks with Human Gaze-Guided Neural Attention},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, volume={33}, pages={1--15}, year={2020}
}
@inproceedings{VazquezRodriguez2022transformer,
  author={Vazquez-Rodriguez, Juan and Lefebvre, Gr{\'e}goire and Cumin, Julien and Crowley, James L.},
  title={Transformer-Based Self-Supervised Learning for Emotion Recognition},
  booktitle={Proceedings of the 26th International Conference on Pattern Recognition (ICPR)}, year={2022},
  note={Also available as arXiv:2204.05103}
}
@misc{Zhang2024readingembedding,
  author={Zhang, Yuhong and Yang, Shilai and Cauwenberghs, Gert and Jung, Tzyy-Ping},
  title={From Word Embedding to Reading Embedding Using Large Language Model, EEG and Eye-tracking},
  year={2024}, eprint={2401.15681}, archivePrefix={arXiv}, primaryClass={cs.HC},
  note={Versión publicada (IEEE TNSRE 32:3465--3475, 2024, doi 10.1109/TNSRE.2024.3435460): "Integrating Large Language Model, EEG, and Eye-Tracking for Word-Level Neural State Classification in Reading Comprehension"}
}
@inproceedings{Barrett2016weakly,
  author={Barrett, Maria and Bingel, Joachim and Keller, Frank and S{\o}gaard, Anders},
  title={Weakly Supervised Part-of-speech Tagging Using Eye-tracking Data},
  booktitle={Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL)}, pages={579--584}, year={2016}, address={Berlin, Germany},
  doi={10.18653/v1/P16-2094}
}
@article{Barrett2020sequence,
  author={Barrett, Maria and Hollenstein, Nora},
  title={Sequence labelling and sequence classification with gaze: Novel uses of eye-tracking data for Natural Language Processing},
  journal={Language and Linguistics Compass}, volume={14}, number={11}, pages={1--16}, year={2020},
  doi={10.1111/lnc3.12396}
}
@inproceedings{Das2016human,
  author={Das, Abhishek and Agrawal, Harsh and Zitnick, C. Lawrence and Parikh, Devi and Batra, Dhruv},
  title={Human Attention in Visual Question Answering: Do Humans and Deep Networks look at the same regions?},
  booktitle={Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP)}, pages={932--937}, year={2016}, address={Austin, Texas},
  doi={10.18653/v1/D16-1092}
}
@article{Mohr2017personal,
  author={Mohr, David C. and Zhang, Mi and Schueller, Stephen M.},
  title={Personal Sensing: Understanding Mental Health Using Ubiquitous Sensors and Machine Learning},
  journal={Annual Review of Clinical Psychology}, volume={13}, pages={23--47}, year={2017},
  doi={10.1146/annurev-clinpsy-032816-044949}
}
@article{Dzedzickis2020human,
  author={Dzedzickis, Andrius and Kaklauskas, Art{\={u}}ras and Bucinskas, Vytautas},
  title={Human Emotion Recognition: Review of Sensors and Methods},
  journal={Sensors}, volume={20}, number={3}, pages={592}, year={2020},
  doi={10.3390/s20030592}
}
@article{Mukhopadhyay2015wearable,
  author={Mukhopadhyay, Subhas Chandra},
  title={Wearable Sensors for Human Activity Monitoring: A Review},
  journal={IEEE Sensors Journal}, volume={15}, number={3}, pages={1321--1330}, year={2015},
  doi={10.1109/JSEN.2014.2376272}
}
@article{Oguz2023emotion,
  author={O{\u{g}}uz, Faruk Enes and Alkan, Ahmet and Sch{\"o}ler, Thorsten},
  title={Emotion detection from ECG signals with different learning algorithms and automated feature engineering},
  journal={Signal, Image and Video Processing}, volume={17}, number={7}, pages={3783--3791}, year={2023},
  doi={10.1007/s11760-023-02606-y}
}
@inproceedings{Mathias2020survey,
  author={Mathias, Sandeep and Kanojia, Diptesh and Mishra, Abhijit and Bhattacharyya, Pushpak},
  title={A Survey on Using Gaze Behaviour for Natural Language Processing},
  booktitle={Proceedings of the 29th International Joint Conference on Artificial Intelligence (IJCAI-20)}, pages={4907--4913}, year={2020},
  doi={10.24963/ijcai.2020/683}
}
@article{Hollenstein2021decoding,
  author={Hollenstein, Nora and Renggli, Cedric and Glaus, Benjamin and Barrett, Maria and Troendle, Marius and Langer, Nicolas and Zhang, Ce},
  title={Decoding EEG Brain Activity for Multi-Modal Natural Language Processing},
  journal={Frontiers in Human Neuroscience}, volume={15}, pages={659410}, year={2021},
  doi={10.3389/fnhum.2021.659410}
}
@inproceedings{Yang2023biot,
  author={Yang, Chaoqi and Westover, M. Brandon and Sun, Jimeng},
  title={BIOT: Biosignal Transformer for Cross-data Learning in the Wild},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, volume={36}, year={2023},
  doi={10.52202/075280-3420},
  note={Proceedings of the 37th Conference on Neural Information Processing Systems}
}
@article{Kumar2024deep,
  author={P., Sriram Kumar and Govarthan, Praveen Kumar and Gadda, Abdul Aleem Shaik and Ganapathy, Nagarajan and Ronickom, Jac Fredo Agastinose},
  title={Deep Learning-Based Automated Emotion Recognition Using Multimodal Physiological Signals and Time-Frequency Methods},
  journal={IEEE Transactions on Instrumentation and Measurement}, volume={73}, pages={2526912}, year={2024},
  doi={10.1109/TIM.2024.3420349}
}
@article{Dar2020emotion,
  author={Dar, Muhammad Najam and Akram, Muhammad Usman and Khawaja, Sajid Gul and Pujari, Amit N.},
  title={CNN and LSTM-Based Emotion Charting Using Physiological Signals},
  journal={Sensors}, volume={20}, number={16}, pages={4551}, year={2020},
  doi={10.3390/s20164551}
}
@article{Abbey2017attention,
  author={Abbey, James D. and Meloy, Margaret G.},
  title={Attention by design: Using attention checks to detect inattentive respondents and improve data quality},
  journal={Journal of Operations Management}, volume={53-56}, pages={63--70}, year={2017},
  doi={10.1016/j.jom.2017.06.001}
}
@article{Ibrahim2025crowdsourced,
  author={Ibrahim, Shahana and Traganitis, Panagiotis A. and Fu, Xiao and Giannakis, Georgios B.},
  title={Learning From Crowdsourced Noisy Labels: A Signal Processing Perspective},
  journal={IEEE Signal Processing Magazine}, volume={42}, number={3}, pages={84--106}, year={2025},
  doi={10.1109/MSP.2025.3572636}
}
@article{Xue2023ceap360vr,
  author={Xue, Tong and El Ali, Abdallah and Zhang, Tianyi and Ding, Gangyi and Cesar, Pablo},
  title={CEAP-360VR: A Continuous Physiological and Behavioral Emotion Annotation Dataset for 360{\textdegree} VR Videos},
  journal={IEEE Transactions on Multimedia}, volume={25}, pages={243--255}, year={2023},
  doi={10.1109/TMM.2021.3124080}
}
@article{Aygun2026multimodal,
  author={Aygun, Ayca and Blaney, Giles and Haga, Zachary and McWilliams, Thomas and Mertens, Julia and de Ruiter, J P and Ward, Nathan and Scheutz, Matthias},
  title={A Multimodal Virtual Reality Data Acquisition Platform and Dataset to Assess Systemic Human Cognitive States},
  journal={Scientific Data}, volume={13}, number={1}, pages={76}, year={2026},
  doi={10.1038/s41597-025-06384-9}
}
@article{Bussolan2025multiphysio,
  author={Bussolan, Andrea and Baraldo, Stefano and Avram, Oliver and Urcola, Pablo and Montesano, Luis and Gambardella, Luca Maria and Valente, Anna},
  title={MultiPhysio-HRC: A Multimodal Physiological Signals Dataset for Industrial Human-Robot Collaboration},
  journal={Robotics}, volume={14}, number={12}, pages={184}, year={2025},
  doi={10.3390/robotics14120184}
}
@article{Luzzani2025ecgrespiration,
  author={Luzzani, Gabriele and Pogliano, Marco and Burairoli, Irene and Colavincenzo, Manuel and Martorana, Stefano and Guglieri, Giorgio and Demarchi, Danilo},
  title={ECG, Respiration, fNIRS, and Eye Tracking for Stress and Mental Workload Monitoring in Human-Machine Interaction},
  journal={IEEE Access}, volume={13}, pages={122726--122741}, year={2025},
  doi={10.1109/ACCESS.2025.3588384}
}
@article{Wan2026novel,
  author={Wan, Xin and Wang, Yongxiong and Wang, Zhe and Liu, Benke},
  title={A Novel Self-Supervised Learning Based on Masked Signal Modeling for Multimodal Physiological Emotion Recognition},
  journal={IEEE Sensors Journal}, volume={26}, number={7}, pages={10231--10240}, year={2026},
  doi={10.1109/JSEN.2026.3663292}
}
@article{Shikha2024optimization,
  author={Shikha, Shikha and Sethia, Divyashikha and Indu, S},
  title={Optimization of Wearable Biosensor Data for Stress Classification Using Machine Learning and Explainable AI},
  journal={IEEE Access}, volume={12}, pages={169310--169327}, year={2024},
  doi={10.1109/ACCESS.2024.3463742}
}

% Datasets
@article{Hollenstein2018zuco,
  author={Hollenstein, Nora and Rotsztejn, Jonathan and Troendle, Marius and Pedroni, Andreas and Zhang, Ce and Langer, Nicolas},
  title={ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading},
  journal={Scientific Data}, volume={5}, pages={180180}, year={2018},
  doi={10.1038/sdata.2018.180}
}
% GAZE4HATE — gitlab.ub.uni-bielefeld.de/clause/gaze4hate
% eyeStyliency — github.com/minnesotanlp/eyeStyliency
```
