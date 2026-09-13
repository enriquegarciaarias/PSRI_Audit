# Análisis de Literatura Extenso — PSRI

Complemento de `docs/analisis_literatura.md` (índice ejecutivo). Ficha por paper, con la estructura de los deep-dives de `docs/analisis_literatura_ANNOTATE.odt`:

1. **Premisa / contribución**
2. **Mapeo al PSRI** → S_estab (estabilidad intra-trial / instrumento), S_coher (coherencia multimodal / organismo), S_cond (consistencia conductual / atención)
3. **Validación y resultados**
4. **Limitaciones que el PSRI supera**
5. **Cómo integrar / takeaway**

Cada ficha cierra con su entrada BibTeX (el bloque completo consolidado está en `docs/analisis_literatura.md` §8).

---

## A. Núcleo del SOTA — calidad de señal (11 papers)

Son los que más alinean con la premisa central del PSRI: **la señal fisiológica no es ground truth sin control de calidad**. La conclusión transversal: todos proponen métodos sensor-específicos y/o supervisados, y ninguno trata la calidad como variable de diseño reutilizable — exactamente el gap que el PSRI llena.

### A.1 Nahmias & Kontson (2021) — Quantifying signal quality from unimodal and multimodal sources (ML)

**Premisa.** La calidad de señal es el problema de fondo de los datos fisiológicos no controlados:
> "quality is defined by how much of the acquired signal is from the source of interest and not noise from external or internal sources"

Critica los SQI clásicos: *"previous works have generally used a limited number of quantitative features, have not used noise sources directly, and/or have characterized signal quality into no more than three discrete categories"*. Proponen puntuación continua Q∈[0,1].

**Mapeo al PSRI.**
- **S_estab**: usan 30 features cuantitativas (varianza, desviación estándar, energía, complejidad de Lempel-Ziv) con KDE para modelar las distribuciones de datos limpios vs. ruidosos, y el valor crítico de decisión bayesiana νf* para decidir si un valor pertenece a señal limpia o ruidosa. **Conexión directa**: la campana gaussiana sobre z-scores del PSRI es conceptualmente equivalente a la probabilidad bayesiana de pertenencia a la distribución "limpia". La desviación estándar y la mobility son las features más discriminativas.
- **S_coher**: en el caso multimodal (EEG + IMU) reconocen que "it may not be possible to directly compare distributions of quantitative features of both the signal of interest and noise" y usan una DCNN que clasifica limpio/ruidoso. Mismo principio que el PSRI (contrastar modalidades para inferir fiabilidad), distinto método (DL vs. decaimiento exponencial en V).
- **S_cond**: reconocen que artefactos conductuales (parpadeo/EOG, EMG, movimiento/IMU) corrompen la señal EEG — es la base conceptual de S_cond — pero no lo tratan como dimensión propia.
- **Agregación**: promedian sub-scores de varias modalidades — idéntico a la media de modalidades del PSRI.

**Validación.** Datos de validación reservados (10%) + dataset independiente de Shin et al. (2017) con tareas etiquetadas: distinguen correctamente datos con artefactos (blinking: 0.23±0.16) de tareas cognitivas (motor imagery: 0.40±0.14). Usan además su métrica para comparar algoritmos de eliminación de artefactos (MARA > AAR) → un índice de calidad sirve para evaluar preprocesamiento, no solo filtrar.

**Limitaciones que el PSRI supera.**
- Requiere ground truth etiquetado (limpio/ruidoso) para entrenar → PSRI no supervisado.
- Sensor-específico (features EEG + EOG; requiere que el ruido se mida con el mismo instrumento) → PSRI agnóstico al sensor.
- No aborda comportamiento/atención → PSRI añade S_cond.
- Requiere datos de calibración de múltiples sujetos → el PSRI usa referencia poblacional de cualquier dataset disponible.

**Takeaway.** Citar como fundamento de la premisa y de la metodología estadística (variabilidad vs. features morfológicas); señalar su dependencia de sensor como motivación del PSRI agnóstico.

```bibtex
@article{nahmias2021quantifying,
  author={Nahmias, David O and Kontson, Kimberly L},
  title={Quantifying Signal Quality From Unimodal and Multimodal Sources: Application to EEG With Ocular and Motion Artifacts},
  journal={Frontiers in Neuroscience}, volume={15}, pages={566004}, year={2021},
  doi={10.3389/fnins.2021.566004}
}
```

### A.2 Boulanger (2026) — Multi-modal DL robust arrhythmia screening (Sensors)

**Premisa.** Los artefactos de movimiento degradan señales fisiológicas y generan falsas alarmas:
> "Studies of intensive care unit (ICU) alarms reveal that 85–99% of arrhythmia alerts are false positives, primarily caused by patient movement."

Critica los SQI tradicionales: *"aggressive quality thresholds reduce false positives but risk missing genuine arrhythmia that occurs during movement"* → justifica un índice **continuo** como el PSRI, no umbrales binarios.

**Mapeo al PSRI.**
- **S_estab**: la desviación estándar del acelerómetro σ|a| correlaciona linealmente con la tasa de falsos positivos (r=0.982) → **evidencia directa de que la variabilidad de la señal es proxy fiable de su calidad** (respalda σ intra-trial como base de S_estab).
- **S_coher**: núcleo del paper. Fusión ECG (señal de interés) + acelerómetro tri-axial (movimiento) con attention gates aprendidos g∈[0,1]; el ratio g_ECG/g_ACC decrece de 1.87 a 1.40 al degradar SNR (24→-6 dB). La coherencia ECG-acelerómetro reduce falsos positivos un **67%**. **Paralelismo directo con K-EmoCon**: ECG/HR (cardiovascular) + acelerómetro/EDA (otro sistema) → si no hay coherencia, la señal no es fiable.
- **S_cond**: clasifican intensidad de movimiento (Rest/Light/Moderate/Vigorous) — conducta **física**, pero **no atención/compromiso cognitivo** (parpadeo, tiempos de reacción, fijación ocular) → dimensión que el PSRI añade.

**Validación.** MIT-BIH Arrhythmia + NST Noise en 6 niveles de SNR (24/18/12/6/0/-6 dB), con generalización a niveles no vistos; ScientISST MOVE con 20 sujetos sanos en actividades cotidianas (todo ritmo sinusal normal → cualquier arritmia es falso positivo). Resultados: precisión 99.5% en limpio degradando a 88.2% a -6 dB; reducción de falsos positivos del 67% (14.0%→4.7%); AUC 0.926 a -6 dB vs. 0.600 ECG-only. Los gates se ajustan con la calidad (pendientes significativas p=0.001/0.002) → **la fiabilidad es continua y dependiente del contexto**, no binaria.

**Limitaciones que el PSRI supera.** Requiere entrenamiento supervisado con datos etiquetados (MIT-BIH + NST); sensor-específico (ECG+ACC); salida de clasificación binaria (normal/arritmia) sin índice de fiabilidad explícito reutilizable; S_cond reducido a movimiento bruto.

**Takeaway.** La fusión multimodal basada en fiabilidad funciona, pero su enfoque supervisado y sensor-específico limita su generalización → abre espacio para un índice cerrado, agnóstico y no supervisado. Cita clave para justificar índice continuo y coherencia inter-modal.

```bibtex
@article{Boulanger2026,
  author={Boulanger, Pierre},
  title={From Motion Artifacts to Clinical Insight: Multi-Modal Deep Learning for Robust Arrhythmia Screening in Ambulatory ECG Monitoring},
  journal={Sensors}, volume={26}, number={4}, pages={1135}, year={2026},
  doi={10.3390/s26041135}
}
```

### A.3 Hyun et al. (2023) — Unsupervised seq2seq signal quality assessment (CVS)

**Premisa.** Etiquetar datos de calidad es caro y propenso a error humano:
> "Labeling a large amount of CVS data is extremely costly... the perfect annotation is almost impossible using only CVS data."

**Mapeo al PSRI.**
- **S_estab**: núcleo del paper. Auto-encoder punto-a-punto y ciclo-a-ciclo sobre cardiac volume signal; calidad = residual de reconstrucción; umbral τ con la **regla de los 2-sigmas** determinada sin etiquetas. El contexto temporal (secuencias de CVS) es la clave para distinguir señal normal de artefacto → respalda las ventanas deslizantes de σ en S_estab. Ambos asumen que la señal normal es aproximadamente gaussiana y que los atípicos (>2σ) indican artefacto.
- **S_coher**: alineación indirecta. Usan ECG solo para segmentar ciclos cardíacos ("the use of ECG is practically reasonable"), no para fusionar información → oportunidad de extender con PSRI comparando coherencia CVS↔ECG o CVS↔acelerómetro.
- **S_cond**: no abordada.

**Validación.** El modelo ciclo-a-ciclo alcanza ACC 0.9566, TPR 0.9743, AUC 0.9484, **compitiendo con el enfoque supervisado** (ACC 0.9672, AUC 0.9503) → **tesis directa del PSRI**: un índice no supervisado puede ser tan efectivo como uno supervisado. La J-statistic (que usa etiquetas) da algo mejor, pero la regla 2-sigma funciona sin ellas.

**Limitaciones que el PSRI supera.** Requiere entrenar un LSTM-VAE (costoso, dependiente de datos); sensor-específico (CVS de impedancia); una sola modalidad; no cubre coherencia multimodal ni conducta.

**Takeaway.** Evidencia de que los enfoques no supervisados son viables para evaluación de calidad y de que el contexto temporal es fundamental. Sugieren además que su método "can synergize with supervised learning as an aide" → el PSRI puede usarse como filtro previo o pseudo-etiquetado en sistemas supervisados.

```bibtex
@article{Hyun2023unsupervised,
  author={Hyun, Chang Min and Kim, Tae-Geun and Lee, Kyounghun},
  title={Unsupervised sequence-to-sequence learning for automatic signal quality assessment in multi-channel electrical impedance-based hemodynamic monitoring},
  journal={Biomedical Signal Processing and Control}, volume={86}, pages={105273}, year={2023},
  doi={10.1016/j.bspc.2023.105273}
}
```

### A.4 Gupta, Khomami Abadi et al. (2016) — Quality-adaptive multimodal affect recognition (ICMR)

**Premisa.** La contaminación por ruido perjudica el reconocimiento afectivo multimodal:
> "The signals from the above mentioned modalities are often contaminated with various sources of noise, that significantly hinders the task of affect recognition."

Un sistema multimodal no puede asumir ground truth válido; sin control de calidad, la fusión puede **empeorar** el rendimiento. Objetivo *cross-user* (funcionar para usuarios no vistos) → alineado con la exigencia de agnosticidad al sujeto del PSRI.

**Mapeo al PSRI.**
- **S_estab**: Signal Quality Estimators (SQEs) por modalidad (EEG, ECG, GSR, head-pose) con features **mayoritariamente estadísticas** (mean, median, skewness, kurtosis, power spectral) — prácticamente las mismas que la base de S_estab. Diferencia clave: sus SQEs son **supervisados** (entrenados con anotación humana, acuerdo inter-observador κ≥0.73); el PSRI es un índice cerrado continuo.
- **S_coher**: alineación parcial. No calculan coherencia entre modalidades, pero su fusión adaptativa pondera cada modalidad por su calidad (q_i∈{0,1} binario, t_i = F1 del clasificador, α_i = peso): la modalidad de baja calidad se pondera menos → mismo espíritu que el PSRI como peso continuo, pero más pobre (binario, sin coherencia).
- **S_cond**: alineación indirecta. Head-pose como conducta observable que correlaciona con la experiencia afectiva ("following the rhythm of music") → oportunidad de incluir head-pose entre las métricas conductuales de S_cond.

**Validación.** Cross-subject cross-validation. Los sistemas adaptativos a la calidad producen más resultados significativos y reducen la tasa de rechazo de muestras (de 22.58% en uni-modal a 0% en multimodal) manteniendo o mejorando el rendimiento → **la adaptación a la calidad mejora el rendimiento**.

**Limitaciones que el PSRI supera.** Requiere anotadores expertos (κ≥0.73); calidad binaria (buena/mala) pierde granularidad; no calcula coherencia entre modalidades; no aborda atención/compromiso; SQEs sensor-específicos.

**Takeaway.** El PSRI podría implementar la fusión aún más fina: S_estab/S_coher/S_cond como pesos **continuos**, sin entrenar clasificadores supervisados. Nota para EXIST: anotadores de calidad y sujetos coinciden (LOSO), mientras que en EXIST sujetos y anotadores son disjuntos.

```bibtex
@inproceedings{Abadi2016quality,
  author={Gupta, Rishabh and Khomami Abadi, Mojtaba and Cárdenes Cabré, Jesús Alejandro and Morreale, Fabio and Falk, Tiago H. and Sebe, Nicu},
  title={A Quality Adaptive Multimodal Affect Recognition System for User-Centric Multimedia Indexing},
  booktitle={Proceedings of the 2016 ACM on International Conference on Multimedia Retrieval (ICMR)}, pages={317--320}, year={2016},
  doi={10.1145/2911996.2912059}
}
```

### A.5 Liu et al. (2024) — Taxonomy & real-time classification of ECG artifacts

**Premisa.** Los artefactos durante la adquisición destruyen horas de registro:
> "Some researchers have experienced the pain of discovering signal problems in some channels during post-processing that can undo hours of data acquisition effort."

**Mapeo al PSRI.**
- **S_estab**: alineación muy fuerte en métricas. Usan 11 features estadísticas — "Mean, median, standard deviation, mean absolute deviation, variance, root-mean-square, max, min, interquartile range, kurtosis, and skewness" — prácticamente idénticas a la base de S_estab → **valida el uso de features estadísticas puras** frente a morfológicas. Diferencia de finalidad: ellos clasifican artefactos específicos (taxonomía TA-D-P, TA-I, etc.), el PSRI genera puntuación continua S∈[0,1].
- **S_coher**: aporta la justificación de la necesidad multimodal: los artefactos relacionados con movimiento (TA-M, TA-S) son **difíciles de clasificar con una sola modalidad (ECG)** → refuerza la necesidad de S_coher para validar la coherencia entre sensores en vez de detectar el artefacto.
- **S_cond**: la variabilidad intra-sujeto puede ser tan alta que un índice no supervisado como S_estab solo no basta → respalda añadir S_coher y S_cond como dimensiones.

**Validación.** Dataset propio de artefactos ECG con taxonomía y clasificación en tiempo real (starter study). Relevante para el SOTA como documentación sistemática de la complejidad del problema de artefactos.

**Limitaciones que el PSRI supera.** Clasificación supervisada de artefactos predefinidos (sensor-específico); no produce un índice continuo de fiabilidad; no multimodal.

**Takeaway.** Citar como evidencia de que las features estadísticas puras son las correctas y de que la clasificación de artefactos con una sola modalidad es insuficiente → motiva el enfoque multimodal del PSRI.

```bibtex
@article{Liu2024taxonomy,
  author={Liu, Hui and Zhang, Shiyao and Gamboa, Hugo and Xue, Tingting and Zhou, Congcong and Schultz, Tanja},
  title={Taxonomy and Real-Time Classification of Artifacts During Biosignal Acquisition: A Starter Study and Dataset of ECG},
  journal={IEEE Sensors Journal}, volume={24}, number={6}, pages={9162--9171}, year={2024},
  doi={10.1109/JSEN.2024.3356651}
}
```

### A.6 Ronca et al. (2026) — Real-world benchmarking of wearable EEG (Brain Informatics)

**Premisa.** Los dispositivos consumer-grade no son fiables por defecto:
> "The reliability of wearable EEG in real-world and task-oriented scenarios—particularly concerning its ability to reproduce meaningful neurometric indicators and their convergence with subjective experience—remains largely unexplored."

Comparan Emotiv EPOC X, Muse S y Mindtooth Touch en tareas ecológicas (conducción simulada, multitarea, vídeos).

**Mapeo al PSRI.**
- **S_estab**: alineación muy fuerte. Dos métricas de calidad: **porcentaje de artefactos** y **estabilidad espectral (PSD)** (correlación entre la primera y segunda mitad de la señal) — ambas análogas a la variabilidad intra-trial de S_estab.
- **S_coher**: no medida explícitamente.
- **S_cond**: alineación parcial. Usan neurométricas proxy de estado cognitivo — vigilance, attention (definida como −FrontalAlpha_GFP), mental workload (ratio theta frontal / alpha parietal) — pero **no miden conducta** (si el sujeto estaba atento a la tarea).

**Validación.** Hallazgo clave: los dispositivos de **mayor calidad de señal correlacionan mejor con la experiencia subjetiva** del usuario → la calidad de señal afecta a la validez de los resultados. Muse S (candidato para estudios tipo EXIST) tiene limitaciones severas de calidad y fiabilidad neurométrica.

**Limitaciones que el PSRI supera.** Es un benchmarking empírico entre dispositivos, no un índice de corrección/filtrado; no ofrece una métrica reutilizable para filtrar segmentos.

**Takeaway.** Evidencia empírica de que la calidad de señal no es un problema menor: afecta a la capacidad de detectar efectos y a la correlación con medidas subjetivas → justifica un filtro de calidad (PSRI) previo a cualquier análisis, aplicable directamente a datos de dispositivos consumer-grade.

```bibtex
@article{Ronca2026beyond,
  author={Ronca, Vincenzo and Cecchetti, Marianna and Capotorto, Rossella and Di Flumeri, Gianluca and Giorgi, Andrea and Germano, Daniele and Borghini, Gianluca and Babiloni, Fabio and Arico, Pietro},
  title={Beyond the lab: real-world benchmarking of wearable EEGs for passive brain-computer interfaces},
  journal={Brain Informatics}, volume={13}, pages={3}, year={2026},
  doi={10.1186/s40708-025-00290-x}
}
```

### A.7 Gao et al. (2021) — Reliability of self-report data (UbiComp/ISWC)

**Premisa.** Los autoinformes no son ground truth fiable por defecto; están sujetos a sesgo, subjetividad y falta de atención.

**Mapeo al PSRI.**
- **S_estab**: alineación parcial. Usan señales de Empatica E4 (EDA, PPG, ACC, ST) para inferir engagement, asumiéndolas como medidas objetivas sin evaluar su calidad → el PSRI añadiría una capa de validación. Hallazgo clave: *"participants with similar physiological patterns may report very different perceived engagement and participants with similar self-report annotations may also have very different physiological patterns"* → la señal fisiológica tampoco es ground truth perfecto.
- **S_coher**: no medida.
- **S_cond**: **alineación muy fuerte** (núcleo del paper). Dos métricas conductuales: nivel de confianza auto-reportado (1-5) y **tiempo de finalización del cuestionario**. Hallazgos: la mayoría tiene confianza moderada, pero hay perfiles sistemáticos (unos siempre >4, otros siempre bajos); el tiempo de finalización correlaciona positivamente con la confianza → el tiempo es un indicador de calidad de respuesta. El PSRI va un paso más allá usando **métricas objetivas** (tiempos de reacción, parpadeo) en vez de autoinformes de confianza (también subjetivos).

**Validación.** Estudio en la naturaleza con sensores del mundo real; correlaciones entre confianza, tiempo y consistencia fisiológica.

**Limitaciones que el PSRI supera.** No es un índice; solo identifica indicadores conductuales de fiabilidad; no propone pesos ni filtros accionables; no evalúa la calidad de las señales fisiológicas que usa.

**Takeaway.** Evidencia de que la fiabilidad del ground truth subjetivo es un problema **general** (no solo de EXIST) → cita clave: "it will be interesting to use survey completion time as an indicator of survey reliability and assign appropriate weights to self-report responses" — exactamente lo que el PSRI hace, pero sobre señales fisiológicas.

```bibtex
@inproceedings{Gao2021reliability,
  author={Gao, Nan and Rahaman, Mohammad Saiedur and Shao, Wei and Salim, Flora D.},
  title={Investigating the Reliability of Self-report Data in the Wild: The Quest for Ground Truth},
  booktitle={Adjunct Proceedings of the 2021 ACM International Joint Conference on Pervasive and Ubiquitous Computing (UbiComp-ISWC '21 Adjunct)}, pages={1--6}, year={2021},
  doi={10.1145/3460418.3479338}
}
```

### A.8 Chatzaki & Tsiknakis (2025) — Stress analysis: systematic review of open datasets

**Premisa.** Revisión sistemática de datasets abiertos de estrés con señales fisiológicas. Es el único paper del conjunto que propone **prácticas concretas de fiabilidad** en todo el pipeline.

**Mapeo al PSRI.** No propone un índice, pero su catálogo de prácticas es la "lista de verificación" que el PSRI formaliza como métrica:
- **Preprocesamiento**: filtros paso-bajo/paso-alto, normalización min-max, ventanas deslizantes (60 s, 50% overlap); ICA + DSWT para artefactos EEG; segmentación con ventanas cortas/largas (descubren que las largas son mejores para EEG, las cortas para señales periféricas).
- **Sincronización multimodal**: timestamps desde el PC de estímulos; dispositivo maestro (Empatica E4 como reloj de referencia); picos de señal extremos (doble toque en WESAD); software ad-hoc (CLAS, EMAP). La falta de sincronización degrada gravemente la fiabilidad.
- **Protocolo**: entornos controlados; bloques de línea base/relajación para calibrar frente a estado neutro.
- **Etiquetado (ground truth)**: enfoques híbridos (SAM/PANAS/STAI + anotación externa + validación biométrica como cortisol salival en MMSD) para reducir sesgo subjetivo.
- **Desbalanceo**: SMOTE, ADASYN, oversampling, weighted loss.
- **Evaluación**: LOSO y Leave-One-Trial-Out para comprobar generalización a nuevos sujetos/ensayos.
- Cita clave: "por muy avanzados que sean los modelos computacionales, la fiabilidad de los resultados sigue dependiendo en gran medida de la calidad de las señales de entrada, la validez del protocolo de inducción y la precisión del etiquetado".

**Limitaciones que el PSRI supera.** Todo son recomendaciones de práctica, no una métrica medible; la fiabilidad se gestiona por diseño del estudio, no por evaluación de la señal adquirida.

**Takeaway.** Citar para situar el PSRI como la formalización cuantitativa de estas buenas prácticas, y para respaldar LOSO/LOTO y la sincronización multimodal como requisitos de validación.

```bibtex
@article{Chatzaki2025,
  author={Chatzaki, C. and Tsiknakis, M.},
  title={An Overview of Stress Analysis Based on Physiological Signals: Systematic Review of Open Datasets and Current Trends},
  journal={Sensors}, volume={25}, number={23}, pages={7108}, year={2025},
  doi={10.3390/s25237108}
}
```

### A.9 Del Pup & Atzori (2023) — Self-supervised learning for biomedical signals (survey)

**Premisa.** Las señales fisiológicas están sujetas a alta variabilidad y artefactos (interferencias oculares y musculares en EEG, deriva de electrodos, ruido ambiental, heterogeneidad de protocolos). El SSL puede aprender representaciones robustas con aumentación de datos, **pero** su eficacia depende críticamente de distinguir patrones fisiológicos genuinos de degradaciones espurias.

**Mapeo al PSRI.** Identifica el gap de forma explícita:
> "If these data are to be effectively applied and correctly interpreted, it is important to understand the quality of data being recorded."

La necesidad de distinguir señal genuina de degradación es exactamente el papel de una métrica de calidad previa (PSRI).

**Limitaciones que el PSRI supera.** La SSL resuelve el problema de forma implícita (representaciones robustas), sin una métrica explícita, interpretable ni portable; el PSRI la proporciona de forma cerrada y sin entrenamiento.

**Takeaway.** Citar para justificar que incluso el aprendizaje de representaciones necesita un control de calidad previo, y como puente con el pre-entrenamiento autosupervisado (sección B.3).

```bibtex
@article{delpup2023applications,
  author={Del Pup, Federico and Atzori, Manfredo},
  title={Applications of Self-Supervised Learning to Biomedical Signals: a Survey},
  journal={IEEE Access}, volume={11}, pages={144680--144704}, year={2023},
  doi={10.1109/ACCESS.2023.3344531}
}
```

### A.10 Adams (2026) — High-reliability signal quality validation via sensor fusion + software indices (Sensors)

**Premisa.** Los métodos de calidad de señal existentes son o bien *software-only* (evaluación post-hoc que malgasta cómputo en intervalos corruptos y suele dar salida binaria acepta/rechaza) o bien *hardware-only* (indicadores IMU/impedancia no capturan todas las fuentes de corrupción, p.ej. interferencia EMG). Propone un **marco híbrido de dos etapas**: (I) gating por integridad de sensor (IMU sincronizado + impedancia/lead-off del electrodo) que rechaza tempranamente intervalos corruptos; (II) SQIs software sobre los latidos restantes. Salida: **puntuación ordinal de calidad 1–6 (1=inutilizable, 6=excelente)** — no binaria — válida para filtrado en tiempo real y curado offline de datasets para IA.

**Mapeo al PSRI.**
- **S_estab**: alineación muy fuerte. Sus cuatro índices software son estadísticos/de consistencia: plausibilidad RR (desviación frente a mediana móvil), **consistencia morfológica DTW contra plantillas adaptativas**, SNR en frecuencia y wander de línea base. "Under stable physiological conditions, successive RR intervals evolve smoothly, whereas abrupt deviations are often caused by motion artifacts" → **respaldo directo de la variabilidad como proxy de estabilidad**. El DTW morfológico es el discriminador individual más fuerte (90.4%) → sugiere añadir al PSRI una medida de consistencia morfológica/plantilla más allá de la σ pura.
- **S_coher**: el núcleo es la **fusión sensor+software**: usa una co-modalidad (IMU, impedancia) para decidir si la biosignal es fiable — mismo principio que S_coher (contrastar modalidades para inferir fiabilidad). Además, la Fig. 3 muestra HR estimada simultáneamente desde ECG/ICG/PPG con tendencias consistentes → **precedente explícito de coherencia inter-modal de HR** (aunque no se formaliza como índice).
- **S_cond**: el gating por movimiento (varianza IMU por latido, σ²_k) es un marcador **conductual/físico** usado como compuerta de calidad — análogo a S_cond (movimiento corrompe señal), pero sin atención/compromiso cognitivo (dimensión que el PSRI añade).

**Validación.** 8644 latidos ECG anotados por expertos, 20 participantes sanos (6153 Pass [71%] / 2491 Fail [29%]). Marco global: **98.13% acc, 98.81% sens, 96.70% spec, F1≈98%**, +7.89 pp sobre el mejor índice individual (DTW 90.4%); 5/20 datasets con 100%. Supera a 6 benchmarks publicados (Orphanidou, Behar, Liu, Zhou, Fu, Fotsing Kuetche) con el mejor equilibrio sensibilidad-especificidad. Calibración en 900 latidos de 3 participantes independientes, sin tuning post-hoc. **El rendimiento no viene de un solo índice sino de la fusión complementaria** → valida el compuesto S_estab/S_coher/S_cond del PSRI.

**Limitaciones que el PSRI supera.** Requiere hardware sincronizado IMU+impedancia (prerrequisito no disponible en dispositivos legacy/consumer) → PSRI agnóstico al sensor, opera solo con la señal; requiere anotación experta para calibrar umbrales y pesos (calibración supervisada) → PSRI cerrado/no supervisado; validado solo en ECG (extensión a otras señales es arquitectónica, no experimental) → PSRI multimodal por diseño; el gating por movimiento es conducta física, no atención; clasificación binaria Pass/Fail pese a la escala ordinal.

**Takeaway.** 
- Evidencia de que una **puntuación ordinal granular (1–6)** permite a la vez filtrar y curar datasets → respalda el diseño continuo del PSRI frente a umbrales binarios.
- La fusión sensor+software (IMU co-modal como compuerta de fiabilidad del ECG) y la consistencia inter-modal de HR (ECG/ICG/PPG) son los precedentes operativos más cercanos a S_coher.
- La consistencia morfológica DTW como discriminador más fuerte sugiere incorporar una medida de consistencia de forma/plantilla a S_estab.
- Su dependencia de hardware auxiliar motiva un índice software-only y agnóstico al sensor: el PSRI.

```bibtex
@article{Adams2026signalquality,
  author={Adams, Basel},
  title={High-Reliability Signal Quality Validation for Biosignals Using Sensor Fusion and Software Indices},
  journal={Sensors}, volume={26}, number={11}, pages={3478}, year={2026},
  doi={10.3390/s26113478}
}
```

### A.11 Diachenko et al. (2022) — CNN-guided annotation of EEG artifacts (eNeuro)

**Premisa.** La anotación manual de artefactos en EEG sigue siendo el gold standard en investigación y clínica, pero es costosa, lenta y propensa al error humano:
> "The quality and reliability of data analysis ultimately depend on the definition of artifacts, subjective decisions, concentration of the professional who preprocesses the data, and subsequently the resulting quality of the preprocessed signals."

Proponen un **modelo de aprendizaje iterativo con CNN** que, entrenado sobre datos anotados por expertos, selecciona los segmentos donde el modelo y la anotación original discrepan para que **dos expertos independientes los re-ecen** y, con ese gold standard revisado, se reentrena el modelo. Es decir: no usa el CNN solo para detectar artefactos, sino para **mejorar la fiabilidad de la propia anotación** (gold standard).

**Mapeo al PSRI.**
- **S_estab**: alineación muy fuerte. La premisa central coincide con la del PSRI: la calidad de señal depende de decisiones **subjetivas** y del estado del anotador. Su CNN opera sobre ventanas de 1 s con 50% de solape y un **umbral de probabilidad ajustable** que controla qué proporción de datos se etiqueta automáticamente y cuál requiere juicio experto → análogo a un índice continuo tipo PSRI con umbral seleccionable, frente a la clasificación binaria de artefactos.
- **S_coher**: no abordada (solo EEG).
- **S_cond**: no abordada, pero su evidencia de que los expertos **pasan por alto artefactos** (el modelo detectó intervalos no marcados, p.ej. por un misclick del anotador) respalda la necesidad de una métrica objetiva e independiente del anotador como el PSRI.
- **Fiabilidad de la anotación (clave)**: al re-evaluar el 23% de los datos discrepantes, los dos expertos cambiaron la anotación en el **25% de los casos** y alcanzaron solo un **Cohen's κ = 0.54** de acuerdo inter-observador → **la propia gold standard de calidad de señal es subjetiva y poco reproducible**, exactamente el gap que el PSRI aborda desde la señal.

**Validación.** CNN sobre EEG de reposo (ojos abiertos/cerrados) de 30 niños con desarrollo típico y 141 con trastornos del neurodesarrollo. Rendimiento original: sensitivity 71.0%, specificity 78.1%, precision 59.7%, balanced accuracy 74.6%. Tras reentrenar con el gold standard revisado: balanced accuracy sube a ~80% y precision a ~76% (76.7% vs 68.8%, specificity 87.0% vs 83.4%). El modelo con umbral 0.5 hizo el 83% de predicciones correctas, identificando el 76% de artefactos y el 86% de no-artefactos; detectó además intervalos de artefacto que el anotador omitió.

**Limitaciones que el PSRI supera.** Requiere anotación experta para entrenar (supervisado); sensor-específico (EEG); salida binaria artefacto/no-artefacto (aunque con umbral ajustable), no un índice continuo reutilizable; no multimodal; depende de la calidad de la gold standard inicial.

**Takeaway.** Es la evidencia más directa de que **la anotación de calidad de señal es subjetiva** (κ=0.54, 25% de cambios tras re-evaluación) → respalda la premisa del PSRI y su necesidad de métricas objetivas; además valida el uso de un índice continuo con umbral controlable frente a clasificación binaria.

```bibtex
@article{Diachenko2022improved,
  author={Diachenko, Marina and Houtman, Simon J and Juarez-Martinez, Erika L and Ramautar, Jennifer R and Weiler, Robin and Mansvelder, Huibert D and Bruining, Hilgo and Bloem, Peter and Linkenkaer-Hansen, Klaus},
  title={Improved Manual Annotation of EEG Signals through Convolutional Neural Network Guidance},
  journal={eNeuro}, volume={9}, number={5}, pages={ENEURO.0160--22.2022}, year={2022},
  doi={10.1523/ENEURO.0160-22.2022}
}
```

---

## B. Señales fisiológicas en NLP/IR — fichas de contexto y lecciones metodológicas

### B.1 Datasets con anotación subjetiva (validan la hipótesis)

#### B.1.1 GAZE4HATE — Subjective hate annotation and detection with gaze

**Premisa.** Detección de discurso de odio con eye-tracking durante la anotación. El sujeto que genera la señal (ET) **es el mismo que anota** → permite probar directamente la hipótesis original (señal fisiológica → subjetividad de anotación) que EXIST no permitía por el desacople poblacional.

**Datos.** 43 participantes, 90 frases sobre mujeres (3870 instancias únicas). Anotaciones: hatefulness 1-7, confianza y racionales (palabras seleccionadas). Alemán. Abierto CC-BY-NC (gitlab.ub.uni-bielefeld.de/clause/gaze4hate). Modelo MEANION integra gaze features con LMs.

**Mapeo al PSRI.**
- **S_estab**: métricas de fijación (fixation count, dwell time) análogas a la estabilidad de la señal de pupila; son predictores robustos de la subjetividad → candidatas a incorporar en S_estab/S_cond.
- **S_coher**: no abordado (solo ET, sin HR/EDA).
- **S_cond**: las métricas oculares predicen la subjetividad de la anotación, no mediante tiempos de reacción/parpadeo sino con fijaciones → refuerza la pupila como indicador de estado cognitivo.

**Takeaway.** Evidencia de que el ET captura la subjetividad del anotador cuando sujeto=anotador; dataset de validación adicional tras K-EmoCon (estudio de caso con ET puro), no sustituye la hoja de ruta principal.

```bibtex
% GAZE4HATE — gitlab.ub.uni-bielefeld.de/clause/gaze4hate (sin paper formal; ver fuente)
```

#### B.1.2 eyeStyliency — Textual saliency of styles from eye tracking (de Langis & Kang)

**Premisa.** Compara la saliencia textual derivada de eye-tracking, anotaciones humanas y modelos de lenguaje (BERT, GPT-2) en estilos (cortesía, descortesía, sentimiento ±). Diseño experimental con condiciones congruentes/incongruentes para aislar el efecto del estilo.

**Datos.** 20 participantes, 90 items, EyeLink 1000 Plus (1000 Hz). Anotaciones: saliencia de estilo (palabras que contribuyen al estilo). Abierto (github.com/minnesotanlp/eyeStyliency).

**Mapeo al PSRI.**
- **S_estab**: usan las mismas métricas que el PSRI (dwell time, first fixation duration, reread time, pupil size). Hallazgo: **dwell time es la métrica más robusta** → refuerza la elección de métricas basadas en fijaciones.
- **S_coher**: no entre señales, pero comparan ET con anotaciones humanas y LMs → análogo a comparar señales fisiológicas con anotación subjetiva; muestran intersección y divergencia (el ET captura información complementaria).
- **S_cond**: pupil size como métrica de arousal/carga cognitiva, similar a S_cond.

**Lecciones.** Normalización **z-score a nivel de participante** mejora la señal (alternativa/complemento a la MAD del PSRI); el ET destaca palabras (verbos, adverbios) que los humanos no seleccionan (adjetivos) → **el ET complementa, no sustituye, la anotación**; el ET es útil en escenarios few-shot.

**Limitaciones.** Pequeño (20×90); sin desacuerdo entre anotadores; dominio de estilo, no contenido sensible.

```bibtex
% eyeStyliency — github.com/minnesotanlp/eyeStyliency (sin paper formal; ver fuente)
```

#### B.1.3 Hollenstein et al. (2021) — Decoding EEG for multi-modal NLP (ZuCo)

**Premisa.** El EEG contiene información lingüística que mejora tareas de NLP, especialmente dividido en **bandas de frecuencia** (theta, alpha, beta, gamma). Corpus ZuCo: EEG + ET simultáneos en lectura naturalista.

**Datos (ZuCo).** 12-18 participantes, ~400-350 oraciones por tarea; anotación de sentiment (SST) y relation detection. Abierto (osf.io/q3zws, osf.io/2urht).

**Mapeo al PSRI.**
- **S_estab**: las bandas de frecuencia son más informativas que la señal completa → para EEG crudo futuro, trabajar en bandas en vez de señal cruda (mejor relación señal-ruido).
- **S_coher**: alineación fuerte — arquitectura multimodal texto+EEG; la **fusión late** supera a la early → valida el enfoque de representaciones separadas por modalidad antes de fusionar.
- **S_cond**: EEG durante lectura naturalista análogo al uso de ET/HR con estímulos.

**Resultados.** El EEG mejora más tareas simples (sentiment) que complejas (relation detection); el beneficio es mayor con **pocos datos** (ablation study) → relevante para datasets pequeños como EXIST.

**Limitaciones que el PSRI supera.** No aborda la calidad del EEG (artefactos, electrodos) como variable de diseño; sin anotación de desacuerdo (no sirve para la hipótesis central).

**Takeaway.** ZuCo como dataset complementario para validar S_estab (EEG) y S_coher (EEG+ET) en dominio lingüístico; no como sustituto de K-EmoCon.

```bibtex
@article{Hollenstein2021decoding,
  author={Hollenstein, Nora and Renggli, Cedric and Glaus, Benjamin and Barrett, Maria and Troendle, Marius and Langer, Nicolas and Zhang, Ce},
  title={Decoding EEG Brain Activity for Multi-Modal Natural Language Processing},
  journal={Frontiers in Human Neuroscience}, volume={15}, pages={659410}, year={2021},
  doi={10.3389/fnhum.2021.659410}
}
```

#### B.1.4 Xue et al. (2023) — CEAP-360VR: continuous physiological & behavioral emotion annotation in VR

**Premisa.** Las técnicas de anotación emocional existentes en VR son retrospectivas (SAM post-estímulo) o interrumpen la inmersión. Proponen **anotación continua de valencia-arousal (V-A) dentro del HMD** (joystick Joy-Con mapeado al espacio V-A de Russell) mientras se ven vídeos 360°, recogiendo simultáneamente señales conductuales (head/eye movements, pupila) y fisiológicas (HR, SKT, EDA, BVP, IBI). Publican el primer dataset multimodal 360° con anotación continua. **Relevancia**: es un caso casi idéntico al protocolo de anotación continua de K-EmoCon, con la diferencia de que aquí sujeto=anotador (self-report), lo que permite estudiar la fiabilidad de la anotación continua frente a la discreta.

**Datos.** 32 participantes (18–33, M=25), 8 vídeos 360° de 60 s validados por 95 sujetos (Li et al.), HTC Vive Pro Eye (ET Tobii 120 Hz), Empatica E4 (BVP, EDA, SKT, HR, IBI) + acelerómetro. Anotación continua V-A muestreada a 10 Hz + SAM within-VR post-estímulo + SSQ/IPQ/NASA-TLX. Abierto (github.com/cwi-dis/CEAP-360VR-Dataset, CC-BY-NC 4.0).

**Mapeo al PSRI.**
- **S_estab**: **la pupila se normaliza con z-score a nivel de participante** tras modelar y sustraer el efecto de luminancia (regresión lineal PD≈luz) → respalda la normalización robusta del PSRI y advierte de confusores de luminancia en la pupila. El ICC valida la consistencia inter-sujeto de las anotaciones (SAM V ICC=0.984, A ICC=0.951; acuerdo entre anotación continua y SAM: V ICC=0.882, A ICC=0.714).
- **S_coher**: **ablación explícita**: usar solo conducta o solo fisiológica da rendimiento razonable, pero **la fusión de ambas mejora la clasificación** → evidencia empírica del principio multimodal del PSRI. La consistencia de los mapas de saliencia HM/EM entre sujetos (CC>0.8) sugiere coherencia conductual inter-sujeto como señal de calidad de datos.
- **S_cond**: HM/EM como conducta observable que correlaciona con la experiencia emocional (p.ej. σ de yaw de HM correlaciona negativamente con valencia; σ de EM pitch con arousal) → refuerza S_cond como dimensión conductual. PD como marcador de arousal (HVHA>HVLA).

**Resultados.** Clasificación baseline (RF con segmentos de 2 s): SI 66.80% (V) / 64.26% (A) binario; 3-clases 49.92%/52.20%. SD supera a SI (esperable). El desbalanceo (mucha clase neutral) lastra clases finas. Los autores señalan que **sin anotación continua los algoritmos weak-supervised no pueden validarse** → analogía con la necesidad de ground truth fiable del PSRI.

**Limitaciones que el PSRI supera.** Sujeto=anotador sin desacuerdo entre anotadores externos → no sirve para la hipótesis central de PSRI (desacuerdo); anotación continua con división de atención (joystick + vídeo) que puede degradar la señal fisiológica → el PSRI añadiría un control de calidad sobre esa señal; sin EEG (lo descartan por dificultad de calidad en HMD); sin métrica de calidad de señal reutilizable (solo preprocesado manual: filtrado + normalización + z-score).

**Takeaway.** Dataset complementario para validar S_estab (pupila con z-score y corrección de luminancia), S_coher (ablación conducta+fisiológica) y S_cond (HM/EM) en VR; el protocolo de anotación continua lo convierte en un banco de pruebas natural para la fusión multimodal del PSRI, aunque sin desacuerdo entre anotadores.

```bibtex
@article{Xue2023ceap360vr,
  author={Xue, Tong and El Ali, Abdallah and Zhang, Tianyi and Ding, Gangyi and Cesar, Pablo},
  title={CEAP-360VR: A Continuous Physiological and Behavioral Emotion Annotation Dataset for 360{\textdegree} VR Videos},
  journal={IEEE Transactions on Multimedia}, volume={25}, pages={243--255}, year={2023},
  doi={10.1109/TMM.2021.3124080}
}
```

#### B.1.5 Aygun et al. (2026) — Multimodal VR data acquisition platform for systemic cognitive states (Scientific Data)

**Premisa.** Los agentes artificiales en equipos humano-máquina necesitan detectar **estados cognitivos sistémicos** (workload, sentido de urgencia, mind wandering, interferencia, distracción) a partir de múltiples modalidades. Publican una **plataforma de adquisición multimodal sincronizada** y un dataset abierto de 80 sujetos en tarea de conducción simulada con tareas secundarias (frenadas, diálogo, estimulación táctil DRT), con fNIRS, EEG, pupilometría/eye-tracking, respiración, EDA, pletismografía, audio y vídeo.

**Datos.** 80 participantes, dos sesiones de conducción. Modalidades: fNIRS (NIRScout, 7.81 Hz), EEG 8 canales (Enobio, 500 Hz) con control de calidad vía software del fabricante, Pupil Core (200 Hz, pupila + blink rate), respiración, EDA, pletismografía beat-to-beat, presión arterial, audio (sistema y micrófono de solapa), vídeo (no liberado por ética). Sincronización **LSL con timestamps universales**. Estados objetivo: workload, urgencia, mind wandering, interferencia, distracción; con marcadores de eventos (braking, dialogue, DRT). Cuestionarios pre/post (incluye SSQ).

**Mapeo al PSRI.**
- **S_estab**: alineación parcial. Verifican la calidad del EEG con el software del fabricante, pero es una comprobación puntual, no una métrica continua → el PSRI aportaría una evaluación de calidad por ventana reutilizable. La pupila (diámetro, blink rate) como señal sensible a artefactos de movimiento → respalda la variabilidad como proxy de estabilidad.
- **S_coher**: **alineación muy fuerte**. Plataforma diseñada para análisis conjunto de múltiples señales fisiológicas + conductuales sincronizadas; los autores ya usaron EEG+fNIRS+gaze para estimar workload → precedente directo de coherencia multimodal para inferir estados cognitivos. fNIRS explícitamente pensado como modalidad complementaria "incluso en presencia de contaminación" → concepto de redundancia inter-modal del PSRI.
- **S_cond**: **alineación muy fuerte**. Incluye estados cognitivos **conductuales/atencionales** (mind wandering, distracción, interferencia) con marcadores de eventos de tarea (braking, DRT) → análogo directo a S_cond (atención/compromiso). La distracción y el mind wandering son fenómenos atencionales medibles por conducta + fisiológica.

**Validación.** Data descriptor (Scientific Data); dataset abierto con descripción de protocolo, hardware, sincronización y formatos. No reportan métricas de clasificación (es un recurso, no un método).

**Limitaciones que el PSRI supera.** Es un dataset/plataforma, no un índice de calidad reutilizable; sin anotación subjetiva de contenido (no sirve para la hipótesis central de desacuerdo); sin EEG de alta densidad; no evalúa la calidad de señal como variable de diseño.

**Takeaway.** Dataset complementario para validar S_coher (fusión EEG+fNIRS+gaze+pupila) y S_cond (mind wandering/distracción con eventos de tarea) en entorno de conducción; el pipeline LSL de sincronización multimodal es una referencia de infraestructura para pipelines tipo EXIST/K-EmoCon.

```bibtex
@article{Aygun2026multimodal,
  author={Aygun, Ayca and Blaney, Giles and Haga, Zachary and McWilliams, Thomas and Mertens, Julia and de Ruiter, J P and Ward, Nathan and Scheutz, Matthias},
  title={A Multimodal Virtual Reality Data Acquisition Platform and Dataset to Assess Systemic Human Cognitive States},
  journal={Scientific Data}, volume={13}, number={1}, pages={76}, year={2026},
  doi={10.1038/s41597-025-06384-9}
}
```

#### B.1.6 Bussolan et al. (2025) — MultiPhysio-HRC: multimodal physiological dataset for industrial human-robot collaboration (Robotics)

**Premisa.** La robótica colaborativa industrial (Industry 5.0) necesita percibir el estado psico-físico del operador (estrés, carga cognitiva) para adaptarse. Presentan **MultiPhysio-HRC**, un dataset multimodal con EEG, ECG, EDA, respiración, EMG, voz y facial action units recogido en escenarios realistas (tareas cognitivas controladas, VR inmersiva y desensamblaje industrial manual y con asistencia robótica), con **ground truth mediante cuestionarios psicológicos validados** (STAI-Y1, NASA-TLX, SAM, NARS) y baselines de regresión/clasificación.

**Datos.** Publicado en *Robotics* 2025. Señales: EEG, ECG, EDA, RESP, EMG, audio/voz, facial AUs. Tareas heterogéneas: tests cognitivos, experiencias VR y actividades industriales de desensamblaje. Anotaciones: STAI-Y1 (estrés), NASA-TLX (carga), SAM (afecto), NARS (aceptación). Baselines: RandomForest, AdaBoost, XGBoost para regresión (STAI/NASA-TLX) y clasificación (3 clases estrés, 3 carga).

**Mapeo al PSRI.**
- **S_estab**: alineación parcial. Las señales fisiológicas periféricas (ECG, EDA, EMG, RESP) son las más informativas; el EEG "es más susceptible al ruido", rindiendo algo peor → confirma que el artefacto en EEG degrada su utilidad → el PSRI como filtro previo mejoraría la contribución del EEG.
- **S_coher**: **alineación muy fuerte**. Resultado clave: las señales fisiológicas superan a EEG y a voz en clasificación de estrés y carga cognitiva → la coherencia entre señales fisiológicas centrales+periféricas es más fiable que una sola modalidad → respalda el principio de S_coher. La fusión de features fisiológicas da los mejores F1.
- **S_cond**: alineación parcial. Voz y facial AUs son conducta observable → análogos a S_cond, aunque el paper los trata como features de clasificación, no como marcadores de atención/compromiso.

**Validación.** Baselines: regresión de STAI-Y1 y NASA-TLX y clasificación de 3 niveles de estrés y carga con RF/AdaBoost/XGB; las features fisiológicas alcanzan los mejores F1, sobre todo en carga cognitiva.

**Limitaciones que el PSRI supera.** Es un dataset (sin índice de calidad reutilizable); ground truth por autoinforme (STAI/NASA-TLX/SAM) sin desacuerdo entre anotadores → no sirve para la hipótesis central; el EEG se penaliza por ruido sin cuantificar ese ruido.

**Takeaway.** Dataset complementario para validar S_coher (fusión de señales fisiológicas supera a unimodales) y S_estab (evidencia de que el ruido EEG degrada su información → filtro de calidad previo); catálogo de anotación con cuestionarios validados (STAI/NASA-TLX/SAM/NARS) útil como referencia de protocolo de ground truth subjetivo.

```bibtex
@article{Bussolan2025multiphysio,
  author={Bussolan, Andrea and Baraldo, Stefano and Avram, Oliver and Urcola, Pablo and Montesano, Luis and Gambardella, Luca Maria and Valente, Anna},
  title={MultiPhysio-HRC: A Multimodal Physiological Signals Dataset for Industrial Human-Robot Collaboration},
  journal={Robotics}, volume={14}, number={12}, pages={184}, year={2025},
  doi={10.3390/robotics14120184}
}
```

### B.2 Metodológicos (enseñan cómo integrar señales en modelos)

#### B.2.1 Sood et al. (2020) — Gaze-guided neural attention (TSM, NeurIPS)

**Premisa.** Combina el modelo cognitivo de lectura E-Z Reader con un modelo data-driven para predecir fijaciones humanas y guiar la atención de redes neuronales en NLP. Joint training: el TSM se entrena junto a la tarea objetivo adaptando las predicciones de gaze a la tarea, sin necesidad de gaze en test.

**Mapeo al PSRI.** No aborda calidad de señal ni desacuerdo; relevante por lecciones de integración:
- Integrar la señal cognitiva en la **atención** del modelo (no solo como feature) → el PSRI podría modular la atención de un modelo de NLP.
- **Joint training** adapta la señal a la tarea; modelos híbridos (cognitivo + data-driven) superan a los puramente data-driven; usar **predicciones** de la señal (no solo mediciones) funciona → PSRI como regularización, no solo filtro.
- Síntesis con E-Z Reader para superar escasez de datos → análogo a generar datos de "fiabilidad" con modelos teóricos.

**Resultados.** Correlación ρ>0.88 con gaze ground truth (Provo, Geco, Dundee, MQA-RC); +10% BLEU-4 en paraphrase generation; sentence compression F1=85.0 (state of the art).

**Limitaciones.** No aporta datasets nuevos (usa corpora conocidos, sin desacuerdo entre anotadores); no mide fiabilidad de señal.

```bibtex
@inproceedings{Sood2020improving,
  author={Sood, Ekta and Tannert, Simon and M{\"u}ller, Philipp and Bulling, Andreas},
  title={Improving Natural Language Processing Tasks with Human Gaze-Guided Neural Attention},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, volume={33}, pages={1--15}, year={2020}
}
```

#### B.2.2 Vazquez-Rodriguez et al. (2022) — Transformer SSL for emotion recognition (ICPR)

**Premisa.** Transformer con aprendizaje autosupervisado (masked prediction, tipo BERT) sobre señales ECG; pre-entrenado con datos no etiquetados (ASCERTAIN, DREAMER, PsPM) y fine-tuning con AMIGOS.

**Mapeo al PSRI.** Metodológico indirecto:
- **Pre-entrenamiento autosupervisado** supera la escasez de datos y reduce overfitting → pre-entrenar modelos de fiabilidad con datos fisiológicos sin etiquetar.
- Los Transformers superan a CNN/LSTM en señales fisiológicas; **CLS token** para representación global → puntuación global de fiabilidad.
- Fusión de múltiples datasets para pre-entrenamiento.

**Limitaciones.** No aborda calidad de señal ni desacuerdo; datasets sin anotación múltiple (no relevantes para la hipótesis central).

```bibtex
@inproceedings{VazquezRodriguez2022transformer,
  author={Vazquez-Rodriguez, Juan and Lefebvre, Gr{\'e}goire and Cumin, Julien and Crowley, James L.},
  title={Transformer-Based Self-Supervised Learning for Emotion Recognition},
  booktitle={Proceedings of the 26th International Conference on Pattern Recognition (ICPR)}, year={2022},
  note={Also available as arXiv:2204.05103}
}
```

#### B.2.3 Zhang et al. (2024) — LLM + EEG + ET for word-level neural state classification (ZuCo)

**Premisa.** Clasifica palabras de alta/baja relevancia durante la lectura usando LLMs (GPT-3.5, GPT-4) para generar el ground truth de relevancia y EEG+ET para clasificar el estado neural (>60% accuracy).

**Mapeo al PSRI.**
- **S_estab**: features de estabilidad EEG (band power, entropía condicional, conectividad/PLV) durante la fijación de palabras.
- **S_coher**: integración EEG+ET con alineación **fixation-locked** → valida la alineación temporal de señales con estímulos (análoga a la alineación con memes en EXIST).
- **S_cond**: fixation counts, gaze duration, total reading time como métricas de ET → análogas a S_cond.

**Lección clave.** Los LLMs pueden generar ground truth cuando faltan anotaciones humanas → **plan B** para etiquetas de subjetividad/dificultad de memes en EXIST. La combinación EEG+ET supera a EEG solo → valida el enfoque multimodal.

**Limitaciones.** Usa ZuCo (sin desacuerdo entre anotadores); preprocesamiento MARA/EOG estándar sin métrica de calidad.

```bibtex
@misc{Zhang2024readingembedding,
  author={Zhang, Yuhong and Yang, Shilai and Cauwenberghs, Gert and Jung, Tzyy-Ping},
  title={From Word Embedding to Reading Embedding Using Large Language Model, EEG and Eye-tracking},
  year={2024}, eprint={2401.15681}, archivePrefix={arXiv}, primaryClass={cs.HC},
  note={Versión publicada (IEEE TNSRE 32:3465--3475, 2024, doi 10.1109/TNSRE.2024.3435460): "Integrating Large Language Model, EEG, and Eye-Tracking for Word-Level Neural State Classification in Reading Comprehension"}
}
```

#### B.2.4 Das et al. (2017) — VQA-HAT: human attention in VQA

**Premisa.** Estudia la atención humana en Visual Question Answering. La "atención humana" se recoge mediante una interfaz de desenfoque (deblurring) en Mechanical Turk, **no con ET real**.

**Mapeo al PSRI.** Prácticamente nulo:
- No usa señales fisiológicas (ET/EEG/HR); son mapas de atención subjetivos.
- **S_estab/S_coher/S_cond**: no aplican.
- VQA-HAT: 58,475 pares imagen-pregunta (train) + 1,374 (val), sin desacuerdo entre anotadores.

**Lección indirecta (débil).** La atención humana es **dependiente de la tarea** (diferentes preguntas → diferentes regiones) → refuerza que las señales fisiológicas reflejan procesamiento cognitivo específico de la tarea; los modelos supervisados con atención humana mejoran ligeramente → análogo a supervisión con señales fisiológicas. No aporta conceptos ni datasets al PSRI.

**Recomendación.** Prioridad nula; no usar VQA-HAT como validación.

```bibtex
@inproceedings{Das2016human,
  author={Das, Abhishek and Agrawal, Harsh and Zitnick, C. Lawrence and Parikh, Devi and Batra, Dhruv},
  title={Human Attention in Visual Question Answering: Do Humans and Deep Networks look at the same regions?},
  booktitle={Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP)}, pages={932--937}, year={2016}, address={Austin, Texas},
  doi={10.18653/v1/D16-1092}
}
```

#### B.2.5 Barrett et al. (2016) — Weakly supervised PoS tagging with eye-tracking

**Premisa.** Usa eye-tracking para mejorar el etiquetado gramatical débilmente supervisado.

**Mapeo al PSRI.** Indirecto y metodológico:
- **S_estab**: métricas de fijación (fixation duration, count, regressions) análogas a la estabilidad de la pupila; el paper no analiza la estabilidad en sí, sino la utilidad predictiva.
- Lección clave: la **agregación a nivel de tipo** (promediar sobre todas las ocurrencias de una palabra) funciona mejor que a nivel de token → **reduce ruido y produce señales más fiables** → valida el promediado entre sujetos de EXIST.

**Datos (Dundee Treebank).** 51,502 tokens / 9,776 tipos / 2,368 oraciones, 10 lectores nativos, ET 1000 Hz, anotación PoS (Universal Dependencies). Acceso restringido (requiere solicitud a los autores). Sin anotación subjetiva → no relevante para la hipótesis central.

```bibtex
@inproceedings{Barrett2016weakly,
  author={Barrett, Maria and Bingel, Joachim and Keller, Frank and S{\o}gaard, Anders},
  title={Weakly Supervised Part-of-speech Tagging Using Eye-tracking Data},
  booktitle={Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL)}, pages={579--584}, year={2016}, address={Berlin, Germany},
  doi={10.18653/v1/P16-2094}
}
```

#### B.2.6 Yang et al. (2023) — BIOT: Biosignal Transformer for cross-data learning (NeurIPS)

**Premisa.** Transformer de biosignales que tokeniza cada canal por separado en segmentos y los aplana en una "sentencia" unificada (estilo ViT/AST), permitiendo **aprender sobre datasets con canales no coincidentes, longitudes variables y valores faltantes**. Modelo flexible para EEG, ECG y señales de actividad.

**Mapeo al PSRI.** Metodológico indirecto:
- **Preprocesamiento robusto**: la normalización por **percentil 95 de la amplitud por canal** (en vez de z-score/mean-std) es robusta a outliers y variaciones de amplitud entre canales/datasets → respalda el uso de normalizaciones robustas tipo MAD en el PSRI.
- **Manejo de canales/segmentos faltantes sin imputación**: en vez de imputar con ceros, **descarta tokens corruptos** → lección para pipelines con canales defectuosos: eliminar señal de mala calidad (filosofía del PSRI como filtro) en vez de imputarla.
- **Pre-entrenamiento autosupervisado** (predictor sobre embedding de señal perturbada + pérdida contrastiva) sobre grandes corpus EEG/ECG → mejora downstream (+4pp balanced acc en CHB-MIT). → pre-entrenar modelos de fiabilidad sobre datos fisiológicos sin etiquetar (como Vazquez-Rodriguez).

**Resultados.** Supera a SPaRCNet/ContraWR/CNN-Transformer/FFCL/ST-Transformer en CHB-MIT (0.664 vs 0.639), IIIC (0.576), HAR (0.946) y PTB-XL (0.831); robusto a máscaras de canales/segmentos (la pérdida de canales afecta más que la de segmentos); el pre-entrenado en 6 datasets EEG da el mejor resultado (0.7068 CHB-MIT).

**Limitaciones.** Sin relación con la subjetividad de anotación ni el desacuerdo entre anotadores; datasets clínicos sin anotación múltiple; no mide calidad de señal explícita (los tokens corruptos se asumen por estructura, no se detectan por calidad).

**Takeaway.** Soporte para la normalización robusta (percentil 95) y la filosofía de descartar señal corrupta; referencia de arquitectura si el PSRI se integra como representación/token de calidad en un modelo multimodal.

```bibtex
@inproceedings{Yang2023biot,
  author={Yang, Chaoqi and Westover, M. Brandon and Sun, Jimeng},
  title={BIOT: Biosignal Transformer for Cross-data Learning in the Wild},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, volume={36}, year={2023},
  doi={10.52202/075280-3420},
  note={Proceedings of the 37th Conference on Neural Information Processing Systems}
}
```

#### B.2.7 Kumar et al. (2024) — DL-based AER with multimodal physiological signals and time-frequency methods (IEEE TIM)

**Premisa.** Clasificación de emociones con **EDA + ECG** usando tres representaciones tiempo-frecuencia (STFT, CWT, MFC) y tres arquitecturas DL (AlexNet, VGG16 pretrained, cCNN propia) en dos datasets públicos de afecto: **CASE** (anotación continua joystick, 4 emociones) y **WESAD** (3 estados). Resultado principal: **multimodal > unimodal**, y ECG > EDA.

**Mapeo al PSRI.**
- **S_coher**: evidencia empírica directa de que la **fusión de modalidades (EDA+ECG) supera a cada unimodal** (CASE: 86.66% vs 84.58% ECG vs 81.04% EDA; WESAD: 83.96% vs 80.74% vs 79.15%) → refuerza el principio de coherencia multimodal del PSRI (cuando las señales son consistentes entre sí, el conjunto es más fiable).
- **S_estab**: la descomposición **cvxEDA** separa la componente tónica (nivel basal lento) de la **fásica** (respuesta rápida al estímulo) y usa solo la fásica → analogía con separar componente estable de respuesta en la construcción de métricas de estabilidad.
- **CWT supera a STFT y MFC** (mejor resolución tiempo-frecuencia) → lección de representación si el PSRI se calcula en dominio tiempo-frecuencia.

**Datos (relevantes para el catálogo).** CASE: 30 participantes, 8 estímulos de vídeo, ECG/PPG/EDA/RSP/SKT/EMG, **anotación continua joystick** (valencia/arousal). WESAD: 15 participantes, RespiBAN+E4, 3 estados. **CASE es un dataset de anotación continua más para el catálogo** (sin desacuerdo entre anotadores → no sirve para la hipótesis central, sí para S_estab/S_coher).

**Resultados.** Pipeline EDA+ECG+CWT+VGG16: 86.66% (CASE, 4 clases) y 83.96% (WESAD, 3 clases); scary bien clasificada (92.9%); VGG16 > cCNN > AlexNet.

**Limitaciones.** No aborda calidad de señal (solo filtros Savitzky-Golay y downsample); anotación única (sin desacuerdo); resultados en ventanas cortas sin métrica de fiabilidad por segmento; sin valores de dispersión por participante.

**Takeaway.** Datasets CASE (anotación continua) y WESAD para complementar K-EmoCon en S_estab/S_coher; refuerza que la fusión multimodal es práctica recomendada (coherente con Dzedzickis, CEAP-360VR, Oguz).

```bibtex
@article{Kumar2024deep,
  author={P., Sriram Kumar and Govarthan, Praveen Kumar and Gadda, Abdul Aleem Shaik and Ganapathy, Nagarajan and Ronickom, Jac Fredo Agastinose},
  title={Deep Learning-Based Automated Emotion Recognition Using Multimodal Physiological Signals and Time-Frequency Methods},
  journal={IEEE Transactions on Instrumentation and Measurement}, volume={73}, pages={2526912}, year={2024},
  doi={10.1109/TIM.2024.3420349}
}
```

#### B.2.8 Dar et al. (2020) — CNN and LSTM-based emotion charting using physiological signals (Sensors)

**Premisa.** Clasificación de 4 emociones (HVHA/HVLA/LVHA/LVLA) con **2D-CNN para EEG (14 canales → imagen 81×128)** y **1D-CNN+LSTM para ECG y GSR**, con **fusión a nivel de decisión (majority voting)**. Sujeto-independiente, en AMIGOS y DREAMER (sensores low-cost Emotiv Epoc + Shimmer).

**Mapeo al PSRI.**
- **Evidencia directa del gap**: los autores atribuyen el bajo rendimiento de las muestras mal clasificadas a "strong input noise, which is unable to be removed through pre-processing steps" → **confirman que el preprocesamiento estándar (filtrado, baseline removal, z-score) no basta** y que el ruido residual degrada la clasificación → necesidad de una métrica de calidad de señal como el PSRI.
- **S_coher**: fusión multimodal (majority voting) eleva el rendimiento (AMIGOS: ECG(L)+ECG(R)+EEG → 98.8%; +GSR la degrada a 97.25%) → respalda S_coher, pero **con matiz**: añadir una modalidad débil (GSR, 63.67%) puede **perjudicar** la fusión → lección de ponderación por calidad (el PSRI podría ponderar GSR a la baja cuando su calidad sea pobre).
- **S_estab**: la comparación de espectros de señales "best" vs "worst" muestra **mayor variabilidad espectral inter-clase en las buenas** y señales baseline-shifted/noisy en las malas → la calidad del espectro correlaciona con el rendimiento → apoyo a S_estab como proxy de calidad.

**Datos (catálogo).** DREAMER (23 sujetos, EEG+ECG, SAM 1-5, 18 trials) y AMIGOS (40 sujetos, EEG+ECG+GSR, SAM 1-9, 16 trials) — **auto-reporte sin desacuerdo** → útiles para S_estab/S_coher, no para la hipótesis central. Relevante: los datasets **excluyen sujetos con datos inválidos** (AMIGOS 7 de 40; DREAMER 2 de 25) → práctica común de filtrado de calidad previo.

**Resultados.** ECG(R) 98.73% > ECG(L) 96.96% > EEG 74.65% > GSR 63.67% (AMIGOS); fusión ECG+EEG 98.8% (AMIGOS) y 90.55% (DREAMER); supera a DGCNN/GCB-Net/LSTM-RNN/Bayesian DNN en 4 clases.

**Limitaciones.** Sin calidad de señal explícita (solo filtros estándar); sin desacuerdo entre anotadores; mayor voting no ponderado (no usa fiabilidad de cada modalidad).

**Takeaway.** Refuerza S_coher (fusión multimodal) pero advierte que **añadir modalidades débiles sin ponderar por calidad degrada** → argumento para un índice de calidad que pondere/descuente modalidades (S_estab/S_coher/S_cond).

```bibtex
@article{Dar2020emotion,
  author={Dar, Muhammad Najam and Akram, Muhammad Usman and Khawaja, Sajid Gul and Pujari, Amit N.},
  title={CNN and LSTM-Based Emotion Charting Using Physiological Signals},
  journal={Sensors}, volume={20}, number={16}, pages={4551}, year={2020},
  doi={10.3390/s20164551}
}
```

#### B.2.9 Luzzani et al. (2025) — ECG, Respiration, fNIRS and Eye Tracking for stress & mental workload monitoring (IEEE Access)

**Premisa.** La monitorización en tiempo real de estrés y carga mental (MWL) en interacción humano-máquina requiere señales fisiológicas. Estudian la relación entre estrés/MWL y **cuatro señales** (ECG, respiración, fNIRS y eye tracking) combinadas con un **Self-Assessment Questionnaire (SAQ)** de 4 niveles, en 20 participantes con tareas Stroop y N-Back (visual, auditivo, dual). Extraen 83 features y evalúan con tests no paramétricos (Kruskal-Wallis, Mann-Whitney U con corrección Benjamini-Hochberg) su capacidad de discriminar niveles de estrés/MWL.

**Mapeo al PSRI.**
- **S_estab**: alineación parcial. Aplican **baseline correction** (sustraer la media de la fase de reposo) y **min-max normalization por participante** antes del análisis → respalda la normalización robusta del PSRI. Hallazgo: las features discriminan mejor estados binarios (relajado vs. alterado) que niveles finos entre estados alterados → sugiere que la fiabilidad es más fiable para distinguir "señal válida" de "señal degradada" que para graduar matices.
- **S_coher**: alineación fuerte. La combinación de cuatro modalidades da "mayor granularidad para diferenciar múltiples estados alterados"; respiración, fNIRS y movimientos oculares aportan discriminación complementaria al ECG → evidencia de que la **fusión multimodal** mejora la resolución de estados → respalda S_coher.
- **S_cond**: **alineación muy fuerte**. Los **movimientos oculares** (con análisis novedoso en frecuencia: ratio LF/HF de la mirada, tras filtrar drift y tremor) son una señal conductual de estado cognitivo → paralelismo directo con las métricas conductuales de S_cond (atención/compromiso). El SAQ captura la percepción subjetiva del sujeto → valida la dimensión subjetiva que el PSRI complementa con señal objetiva.

**Validación.** Análisis estadístico con K-W y M-W (p<0.05, FDR 0.05). >50% de los 83 features diferencian significativamente entre estados, sobre todo en la comparación binaria (baseline vs. alterado); menos de 20% discriminan entre más de 5 comparaciones por pares → más fácil detectar "alteración" que graduarla. Cada señal contribuye de forma distinta (ECG efectivo hasta 4 comparaciones por pares; respiración/fNIRS/ET dan más granularidad).

**Limitaciones que el PSRI supera.** Sin índice de calidad de señal reutilizable (solo filtrado + baseline + normalización); sin anotación de desacuerdo entre anotadores; no propone filtrado/ponderación accionable; el SAQ es autoinforme (subjetivo).

**Takeaway.** Evidencia de que las señales fisiológicas multimodales discriminan estados subjetivos auto-reportados, y de que **la mirada (eye tracking) es una señal conductual útil para S_cond**; respalda la normalización por participante y la comparación binaria (señal válida vs. no válida) del PSRI. Datos no públicos (estudio, no dataset).

```bibtex
@article{Luzzani2025ecgrespiration,
  author={Luzzani, Gabriele and Pogliano, Marco and Burairoli, Irene and Colavincenzo, Manuel and Martorana, Stefano and Guglieri, Giorgio and Demarchi, Danilo},
  title={ECG, Respiration, fNIRS, and Eye Tracking for Stress and Mental Workload Monitoring in Human-Machine Interaction},
  journal={IEEE Access}, volume={13}, pages={122726--122741}, year={2025},
  doi={10.1109/ACCESS.2025.3588384}
}
```

#### B.2.10 Wan et al. (2026) — Masked-signal SSL for multimodal physiological emotion recognition (IEEE Sensors Journal)

**Premisa.** Etiquetar señales fisiológicas es caro y difícil, sobre todo EEG. Proponen **MMSL** (Masked-signal Modeling-based Self-Supervised Learning): pre-entrenan un encoder por modalidad (EEG, ECG) **reconstruyendo señal enmascarada** (masked-signal modeling, MSM), luego transfieren el encoder a una fase SSL con **representación alignment (RA)** y **cross-modal transformer (CMT)** sin etiquetas, y finalmente fusionan con **multimodal tensor fusion (MTF)**. Evalúan en AMIGOS para reconocimiento de emociones (arousal/valence binarios y 4 clases).

**Mapeo al PSRI.**
- **S_estab**: alineación parcial. El MSM reconstruye señal enmascarada y asume que los artefactos/ruido degradan la reconstrucción → el residual de reconstrucción es un proxy de calidad (como Hyun et al.), pero no se usa como índice explícito.
- **S_coher**: **alineación muy fuerte**. El CMT alinea representaciones de EEG y ECG y MTF fusiona modalidades → principio de coherencia inter-modal del PSRI aplicado con transformers. La **fusión multimodal mejora** la clasificación frente a unimodal → respalda S_coher empíricamente.
- **S_cond**: no abordada.

**Validación.** AMIGOS, 10-fold CV: accuracy 89.91% (arousal) y 86.60% (valence), F1 92.72%/89.10%; **LOSO**: 81.47% (arousal) y 79.29% (valence) → generalización a sujetos no vistos; clasificación 4-clase 83.44%. Resultados superiores a métodos SSL previos (Sarkar & Etemad, Wang et al.).

**Limitaciones que el PSRI supera.** SSL supervisado implícito (necesita datos de pre-entrenamiento, aunque no etiquetados); no produce un índice de fiabilidad por segmento; no trata artefactos explícitamente (los asume en la reconstrucción); sin desacuerdo entre anotadores (AMIGOS es autoinforme).

**Takeaway.** Refuerza dos lecciones del PSRI: (1) el SSL/masked modeling reduce la dependencia de etiquetas costosas (como Del Pup, Vazquez-Rodriguez, BIOT); (2) la **fusión multimodal (MTF) y la alineación inter-modal (CMT) son el mecanismo correcto** → apoyo directo a S_coher y a la integración del PSRI como representación de calidad en modelos multimodales.

```bibtex
@article{Wan2026novel,
  author={Wan, Xin and Wang, Yongxiong and Wang, Zhe and Liu, Benke},
  title={A Novel Self-Supervised Learning Based on Masked Signal Modeling for Multimodal Physiological Emotion Recognition},
  journal={IEEE Sensors Journal}, volume={26}, number={7}, pages={10231--10240}, year={2026},
  doi={10.1109/JSEN.2026.3663292}
}
```

#### B.2.11 Shikha, Sethia & Indu (2024) — Wearable biosensor data optimization for stress classification + XAI (IEEE Access)

**Premisa.** Detección de estrés en tiempo real con wearables durante el Montreal Imaging Stress Task (MIST) en estudiantes, evaluando además si escuchar audio de meditación reduce el estrés. Usan **IBI-derived HRV, BVP y EDA** de dispositivos wearable, con un pipeline de **feature selection (Algoritmo Genético + Mutual Information)**, **optimización Bayesiana** de hiperparámetros y **SHAP (XAI)** para explicar las predicciones.

**Mapeo al PSRI.**
- **S_estab**: alineación parcial. El preprocesado elimina latidos ectópicos de IBI (HRV) porque "afectan a la precisión del análisis HRV" → reconocen implícitamente que artefactos de señal degradan las features → el PSRI haría explícita esa calidad. Las features estadísticas de HRV y EDA son las más discriminativas según SHAP → respaldan la base estadística de S_estab.
- **S_coher**: **alineación fuerte**. La combinación EDA+BVP+HRV alcanza la mayor precisión (98.28% en 2 niveles, 97.02% en 3 niveles) frente a EDA+HRV solas (97.07%/95.23%) → **multimodal > unimodal**, evidencia directa del principio de S_coher.
- **S_cond**: no abordada (solo señal fisiológica, sin conducta/atención).

**Validación.** 2 y 3 niveles de estrés; Gradient Boosting (GB) el mejor clasificador; EDA+HRV casi comparable a añadir BVP; SHAP confirma que **HRV y EDA son las features más significativas**; hallazgo de que la meditación reduce el estrés medido. Sin dataset liberado (estudio propio).

**Limitaciones que el PSRI supera.** No usa un índice de calidad de señal (solo filtrado + eliminación de ectópicos); sin desacuerdo entre anotadores (el ground truth de estrés es el protocolo MIST, no anotación subjetiva múltiple); no aborda conducta/atención; sin datos públicos.

**Takeaway.** Respaldo empírico a dos pilares: la **fusión multimodal de señales fisiológicas mejora el rendimiento** (S_coher) y las **features estadísticas de HRV/EDA son las más informativas** (S_estab). El reconocimiento explícito de que los ectópicos/ruido degradan el HRV justifica un índice de calidad previo como el PSRI.

```bibtex
@article{Shikha2024optimization,
  author={Shikha, Shikha and Sethia, Divyashikha and Indu, S},
  title={Optimization of Wearable Biosensor Data for Stress Classification Using Machine Learning and Explainable AI},
  journal={IEEE Access}, volume={12}, pages={169310--169327}, year={2024},
  doi={10.1109/ACCESS.2024.3463742}
}
```

### B.3 Surveys / contextuales

#### B.3.1 Barrett & Hollenstein (2020) — Sequence labelling and classification with gaze (survey)

**Premisa.** Revisión del uso de gaze en tareas de etiquetado y clasificación de secuencias en NLP (PoS, sintaxis, sarcasmo, NER, etc.).

**Mapeo al PSRI.** Estado del arte + catálogo de corpora (Dundee, GECO, Provo, ZuCo, CFILT/CITL). Valida que el ET es útil en tareas **subjetivas** (sarcasmo, sentimiento, hate speech). Estrategias para usar ET sin tenerlo en test: agregación type-level y multi-task learning → refuerzan la idea de las señales fisiológicas como señal "fortuita" (Plank, 2016) y del PSRI como señal de regularización.

**Datos relevantes.** CITL/Scansam (sentimiento, 72 lectores, reseñas) como el más cercano a la hipótesis de subjetividad; ZuCo (EEG+ET) para S_coher. Dundee/GECO/Provo sin anotación subjetiva → no relevantes.

```bibtex
@article{Barrett2020sequence,
  author={Barrett, Maria and Hollenstein, Nora},
  title={Sequence labelling and sequence classification with gaze: Novel uses of eye-tracking data for Natural Language Processing},
  journal={Language and Linguistics Compass}, volume={14}, number={11}, pages={1--16}, year={2020},
  doi={10.1111/lnc3.12396}
}
```

#### B.3.2 Mathias et al. (2020) — Survey on using gaze behaviour for NLP (IJCAI)

**Premisa.** Survey de gaze behaviour en NLP, con énfasis en **aprender** la mirada en tiempo de ejecución (sin tenerla en test) mediante multi-task learning, CRF, HMM, etc.

**Mapeo al PSRI.**
- **S_estab**: fijaciones, duración de fijaciones y regresiones como métricas estándar → validan las métricas basadas en fijaciones.
- **S_cond**: scanpath complexity y regresiones reflejan procesamiento tardío/dificultad → análogo a S_cond.
- El gaze es una señal **distribuida** (ninguna métrica basta) → justifica el compuesto S_estab+S_coher+S_cond.
- El gaze es especialmente útil en tareas subjetivas (sarcasmo, sentimiento, hate speech) → refuerza la hipótesis central.

**Datos.** Catálogo multilingüe: Dundee, GECO, Provo (no relevantes); ZuCo (EEG+ET, para S_coher); CFILT-Sarcasm/Sentiment/Quality (anotación subjetiva, pequeños pero útiles).

```bibtex
@inproceedings{Mathias2020survey,
  author={Mathias, Sandeep and Kanojia, Diptesh and Mishra, Abhijit and Bhattacharyya, Pushpak},
  title={A Survey on Using Gaze Behaviour for Natural Language Processing},
  booktitle={Proceedings of the 29th International Joint Conference on Artificial Intelligence (IJCAI-20)}, pages={4907--4913}, year={2020},
  doi={10.24963/ijcai.2020/683}
}
```

#### B.3.3 Dzedzickis et al. (2020) — Human emotion recognition: sensors and methods (survey)

**Premisa.** Revisión exhaustiva de sensores y métodos para reconocimiento de emociones (EEG, ECG, GSR, HRV, respiración, temperatura, EMG, EOG, facial, postura) y métodos de análisis (SVM, redes neuronales, lógica difusa).

**Mapeo al PSRI.**
- **S_estab**: revisa HRV, GSR y pupila como indicadores de arousal → refuerza la desviación estándar de pupila y HR como medidas de estabilidad.
- **S_coher**: la **fusión multimodal** (data fusion) es práctica recomendada → valida el enfoque multimodal del PSRI; ningún sensor basta por sí solo.
- **S_cond**: EOG (parpadeo), respiración y gestos/postura análogos a S_cond.
- **Resultados (Tabla 14)**: ECG, EEG y GSR son las técnicas más precisas; SVM y redes neuronales los clasificadores más efectivos.

**Limitaciones.** No aborda la subjetividad de anotación ni desacuerdo; no propone índice de fiabilidad (solo menciona ruido/artefactos como desafío y filtros Butterworth/notch). Datasets DEAP/DREAMER/DECAF/EMDB sin desacuerdo → no relevantes.

```bibtex
@article{Dzedzickis2020human,
  author={Dzedzickis, Andrius and Kaklauskas, Art{\={u}}ras and Bucinskas, Vytautas},
  title={Human Emotion Recognition: Review of Sensors and Methods},
  journal={Sensors}, volume={20}, number={3}, pages={592}, year={2020},
  doi={10.3390/s20030592}
}
```

#### B.3.4 Mohr et al. (2017) — Personal sensing for mental health (review)

**Premisa.** Revisión de sensado ubícuo (smartphones, wearables, redes sociales) con ML para salud mental.

**Mapeo al PSRI.** Marco conceptual y contextual:
- Jerarquía *raw sensor data → features → behavioral markers → clinical targets* análoga a la construcción del PSRI desde métricas de señal.
- "The Curse of Variability": variabilidad entre dispositivos, personas y entornos como desafío central → **refuerza la normalización MAD robusta** del PSRI.
- Señala artefactos, sesgo de selección, privacidad e incertidumbre como desafíos del campo, **sin proponer métrica** → es el más cercano al gap sin cubrirlo.

**Datos.** StudentLife, MONARCA, CrossCheck (smartphone sensing) sin señales fisiológicas de calidad ni desacuerdo → no relevantes.

```bibtex
@article{Mohr2017personal,
  author={Mohr, David C. and Zhang, Mi and Schueller, Stephen M.},
  title={Personal Sensing: Understanding Mental Health Using Ubiquitous Sensors and Machine Learning},
  journal={Annual Review of Clinical Psychology}, volume={13}, pages={23--47}, year={2017},
  doi={10.1146/annurev-clinpsy-032816-044949}
}
```

#### B.3.5 Mukhopadhyay (2015) — Wearable sensors for human activity monitoring (review)

**Premisa.** Survey tecnológico de sensores wearables (acelerómetros, ECG, temperatura) y sistemas de monitorización de actividad (salud, detección de caídas, deporte).

**Mapeo al PSRI.** Valor puramente contextual:
- Menciona artefactos de movimiento como fuente de falsas alarmas ("the generation of large false-alarm rates, and an inability to cope with sensor artefact in a principled manner") y la necesidad de algoritmos robustos con datos incompletos/ruidosos.
- No propone métodos de fiabilidad ni datasets; no analiza subjetividad de anotación.

**Recomendación.** Prioridad nula salvo contextualización general en la introducción.

```bibtex
@article{Mukhopadhyay2015wearable,
  author={Mukhopadhyay, Subhas Chandra},
  title={Wearable Sensors for Human Activity Monitoring: A Review},
  journal={IEEE Sensors Journal}, volume={15}, number={3}, pages={1321--1330}, year={2015},
  doi={10.1109/JSEN.2014.2376272}
}
```

#### B.3.6 Oguz et al. (2023) — Emotion detection from ECG with automated feature engineering

**Premisa.** Clasificación de emociones con ECG mediante features morfológicas (P-QRS-T), HRV y **feature engineering automatizada** (diferencias, ratios, z-score, logaritmos), con BiLSTM.

**Mapeo al PSRI.** Tangencial pero con lecciones:
- **S_estab**: las features morfológicas + HRV clasifican emociones con buena precisión → valida HR como señal relevante.
- La **feature engineering automatizada** mejora significativamente la precisión (Tabla 2) → podríamos explorar técnicas similares para optimizar la combinación/ponderación de S_estab, S_coher y S_cond.
- BiLSTM (dependencias temporales bidireccionales) → modelar series temporales de HR/ET/EEG en futuros trabajos.
- Tres tipos de ruido en ECG (baseline drift, power line interference, high-frequency noise) tratados con filtros estándar, sin índice de calidad.

**Datos (MAHNOB-HCI).** 27 participantes, 20 vídeos, EEG+ECG+GSR+temp+resp+ET+cámara+micrófono; anotaciones valence/arousal/dominance (auto-reporte). Sin desacuerdo entre anotadores → útil solo para validar S_estab/S_coher, no la hipótesis central.

```bibtex
@article{Oguz2023emotion,
  author={O{\u{g}}uz, Faruk Enes and Alkan, Ahmet and Sch{\"o}ler, Thorsten},
  title={Emotion detection from ECG signals with different learning algorithms and automated feature engineering},
  journal={Signal, Image and Video Processing}, volume={17}, number={7}, pages={3783--3791}, year={2023},
  doi={10.1007/s11760-023-02606-y}
}
```

#### B.3.7 Lin & Yang (2023) — EEG + large-scale NLP for virtual counseling (conceptual)

**Premisa.** Propuesta conceptual de combinar detección de emociones por EEG con LLMs (ChatGPT) para psicoterapia virtual. Sin experimentos, sin nuevos métodos, sin datasets.

**Mapeo al PSRI.** Solo contextual:
- Refuerza el modelo valencia-arousal como base teórica de las señales fisiológicas.
- El EEG requiere filtrado y extracción de características (tiempo, frecuencia, tiempo-frecuencia) → refuerza la importancia del preprocesamiento (normalización MAD del PSRI).
- Menciona DEAP, DREAMER, SEED (conocidos, sin desacuerdo → no relevantes).

**Recomendación.** Mención introductoria como ejemplo de aplicación EEG+NLP en salud mental; prioridad baja.

```bibtex
% Lin & Yang (2023) — propuesta conceptual EEG+LLM; sin paper formal en la fuente
```

#### B.3.8 Abbey & Meloy (2017) — Attention checks para detectar anotadores inatentos (JOM)

**Premisa.** Estudio metodológico sobre **attention checks** (preguntas trampa: directed queries, logical statements, manipulation checks, response pattern/time, honesty checks, reverse scaling, infrequency, memory recalls, outlier detection) para **detectar y eliminar encuestados inatentos** en recogida de datos primaria (encuestas, MTurk). Relevancia directa para el PSRI: documenta empíricamente qué pasa al **filtrar datos por calidad de respuesta**.

**Método.** 30 datasets donados por más de 30 estudios (~75% estudiantes en laboratorio, resto Qualtrics/MTurk; 2010–2016). Compara pérdida de muestra, ajuste de constructos/escalas (χ² con ML y rotación promax) y significación de las manipulaciones experimentales (F), con vs sin aplicar cada tipo de attention check.

**Resultados clave.**
- **Pérdida de muestra** media por check individual: 13.96%–19.56%, máx. 58.40%; **agregado (varios checks): media 35.79%, máx. 68.97%**.
- **Constructos y escalas: mejora consistente y significativa** (85–91% de las estadísticas de ajuste mejoran, p<0.01) → el filtrado por atención mejora la *validez de medida*.
- **Manipulaciones experimentales: NO mejora de forma consistente** (44.71%, 41.30%, 40.79% n.s.; solo manipulation checks 61.73%, p<0.05) → el filtrado **no garantiza mejores resultados predictivos/experimentales**.
- Trade-off explícito: eliminar inatentos reduce power (riesgo Type II) vs. conservarlos puede producir resultados espurios (Type I); aviso contra "cherry picking" de checks.

**Mapeo al PSRI.**
- **Analogía central**: los attention checks son el equivalente de encuestas al **PSRI como filtro de anotaciones de baja calidad**; la conclusión empírica de Abbey predice el resultado nulo del experimento3 (el filtrado mejora la fiabilidad de medida pero no la CCC/RMSE/MAE de una regresión continua).
- **S_estab**: response pattern/time (straight-line responses, respuestas repetidas) → análogos post-hoc a métricas de estabilidad de la señal.
- **S_coher**: manipulation checks (coherencia con el contenido de la tarea) → análogos a S_coher como verificación de consistencia con el estímulo.
- **Coste del filtrado**: la pérdida media de ~36% de muestra al filtrar por calidad **debe reportarse** en el paper como límite del PSRI cuando actúa como filtro excluyente.

**Limitaciones.** Dominio encuestas (sin señales fisiológicas); checks binarios (pasar/fallar), no una métrica continua como el PSRI; no propone un índice, solo una taxonomía de mecanismos.

**Takeaway.** Contextualiza el coste del filtrado por calidad: mejora la validez de medida pero puede no mejorar (ni dañar) la predicción; el PSRI debe posicionarse como índice de *fiabilidad de medida* más que como garantía de mejor rendimiento del modelo.

```bibtex
@article{Abbey2017attention,
  author={Abbey, James D. and Meloy, Margaret G.},
  title={Attention by design: Using attention checks to detect inattentive respondents and improve data quality},
  journal={Journal of Operations Management}, volume={53-56}, pages={63--70}, year={2017},
  doi={10.1016/j.jom.2017.06.001}
}
```

#### B.3.9 Ibrahim et al. (2025) — Learning from crowdsourced noisy labels: a signal processing perspective (survey)

**Premisa.** Tutorial (IEEE SPM) sobre **cómo aprender de etiquetas ruidosas producidas por múltiples anotadores en crowdsourcing**, desde una perspectiva de procesado de señal. Es el marco matemático del *desacuerdo entre anotadores*: la confusión de cada anotador se modela como una **matriz de confusión** y el objetivo es integrar etiquetas para estimar la etiqueta de ground truth latente. Relevancia máxima para la hipótesis central del PSRI (desacuerdo), aunque no usa señales fisiológicas.

**Modelos y métodos revisados.**
- **Label integration (dos etapas)**: *majority voting* (asume anotadores equiprobables) → *weighted majority voting* → **modelo de Dawid-Skene (DS)**: etiqueta latente + matrices de confusión A_m por anotador, aprendidas por EM. Variantes: one-coin (un parámetro por anotador), spammer-hammer, confusion vector, GLAD (añade dificultad del ítem), Bayesian (priors Dirichlet/Beta).
- **E2E (end-to-end)**: joint learning de la función f_θ (clasificador) y las confusiones A_m vía EM (crowdlayer de Rodrigues & Pereira); supera a la integración en dos etapas porque evita la propagación acumulativa de errores.
- **Extensiones**: dependencias temporales (DS-HMM), anotadores dependientes (agrupación espectral), identificabilidad vía tensor/NMF, spammer detection (rango de A_m), adversarios coludidos, RLHF/DPO para alinear LLMs, active learning y fairness.

**Resultados/garantías teóricas.**
- El error de la regla MAP **decrece exponencialmente con M** (número de anotadores): P_e ≤ α·exp(−βM) → "wisdom of the crowd".
- DS es identificable (CPD rank-K) si se dispone de la PMF conjunta; E2E suele superar a la integración en dos etapas (Fig. 7: GeoCrowdNet/TraceReg/CrowdLayer > CNMF/MV/DS-EM en LabelMe, Music, CIFAR-10, FashionMNIST).

**Mapeo al PSRI.**
- **S_coher como prior de fiabilidad**: las matrices de confusión estiman la fiabilidad *a posteriori* del anotador; el PSRI aportaría una estimación **en tiempo real, a priori y basada en señal** (sin esperar al consenso), combinable con DS/EM como prior o peso (w_m en WMV) → integración natural: `p(etiqueta|señal) = PSRI × DS`.
- **Punto ciego que el PSRI llena**: los métodos de crowdsourcing son *agnósticos al dato físico* (solo ven etiquetas); no pueden detectar anotadores distraídos que aun así responden consistentemente, ni degeneraciones de la señal (movimiento, artefactos). El PSRI añade la dimensión fisiológica que estos modelos ignoran → **complementariedad directa, no competencia**.
- **Lección metodológica**: modelos de ruido de etiquetas como marco teórico riguroso para las métricas de desacuerdo del paper (Krippendorff, ICC) → el PSRI puede situarse como un modelo generativo de ruido *con entradas fisiológicas*.

**Limitaciones.** Sin señales fisiológicas; asume que el ruido es solo del anotador (no del sensor/sujeto); requiere múltiples anotaciones por ítem (cara); las garantías teóricas asumen parámetros conocidos.

**Takeaway.** Es el **estado del arte formal de la hipótesis de desacuerdo**; el PSRI se posiciona como el puente entre este marco (etiquetas ruidosas) y la calidad de señal fisiológica que ningún trabajo de crowdsourcing considera.

```bibtex
@article{Ibrahim2025crowdsourced,
  author={Ibrahim, Shahana and Traganitis, Panagiotis A. and Fu, Xiao and Giannakis, Georgios B.},
  title={Learning From Crowdsourced Noisy Labels: A Signal Processing Perspective},
  journal={IEEE Signal Processing Magazine}, volume={42}, number={3}, pages={84--106}, year={2025},
  doi={10.1109/MSP.2025.3572636}
}
```

---

## C. Síntesis transversal

### C.1 El gap central

> **Ninguno de los ~37 papers analizados trata la calidad de la señal como variable de diseño** (medible, cuantificable y utilizable para filtrar/ponderar anotaciones). Solo 6 lo mencionan como problema a resolver con preprocesamiento (filtros, ICA, MARA): Mohr (variabilidad/artefactos como desafío), Dzedzickis (filtrado Butterworth/notch), Mukhopadhyay (falsas alarmas), Oguz (tres tipos de ruido en ECG), Hollenstein 2021 (pipeline MARA), Zhang 2024 (MARA/EOG). Abbey (2017) aborda la *calidad de respuesta* en encuestas con attention checks binarios, y Ibrahim et al. (2025) la *fiabilidad del anotador* vía matrices de confusión, ambos sin señales fisiológicas → siguen sin cubrir el gap. Diachenko et al. (2022) demuestra que **la propia anotación experta de artefactos es subjetiva** (Cohen's κ=0.54, 25% de cambios al re-anotar) → refuerza que ni siquiera la gold standard de calidad de señal es fiable. **Ninguno propone un índice** similar al PSRI → el PSRI llena un vacío real.

### C.2 Los más cercanos al gap (para diferenciar la contribución)

| Paper | Solución propuesta | Por qué no cubre el gap | Lo que el PSRI añade |
|---|---|---|---|
| Nahmias & Kontson | SQI continuo Q∈[0,1] con ML/KDE | Supervisado, sensor-específico (EEG) | Agnosticidad al sensor, no supervisión, S_cond |
| Boulanger | Fusión ECG+ACC con attention gates | DL supervisado, sensor-específico, salida binaria | Índice cerrado, continuo, sin reentrenar |
| Hyun et al. | Residual de auto-encoder + regla 2σ | Entrena LSTM-VAE, una modalidad | Índice cerrado sin entrenamiento, multimodal |
| Abadi et al. | SQEs supervisados + fusión adaptativa | Anotación experta (κ≥0.73), calidad binaria | Continuo, no supervisado, S_coher+S_cond |
| Adams | Fusión sensor (IMU+impedancia) + SQIs software, puntuación 1–6 | Requiere hardware sincronizado, calibración supervisada, solo ECG | Software-only, agnóstico al sensor, multimodal, sin calibrar |
| Ronca et al. | Benchmarking de dispositivos | Sin métrica de filtrado | PSRI aplicable a datos de cualquier dispositivo |
| Diachenko et al. | CNN que revisa y mejora la anotación experta de artefactos EEG | Supervisado (anotación experta), sensor-específico (EEG), salida binaria | Índice continuo, objetivo, independiente del anotador y de la gold standard |
| Abbey et al. | Attention checks binarios (encuestas) | Sin señales fisiológicas; filtrado binario con pérdida de muestra ~36% | Índice continuo basado en señal, sin intervención del encuestado |
| Ibrahim et al. | Matrices de confusión por anotador (DS/EM) | Solo etiquetas, agnóstico al dato físico; requiere multi-anotación | Estimación de fiabilidad a priori basada en señal (prior fisiológico para DS/EM) |

### C.3 Evidencia que respalda cada componente del PSRI

- **S_estab (variabilidad intra-trial como proxy de fiabilidad)**: Nahmias (σ/mobility discriminativas), Boulanger (σ|a|↔FP rate r=0.982), Liu (features estadísticas puras), Hyun (regla 2σ), Ronca (% artefactos y estabilidad espectral), eyeStyliency (dwell time robusta), Adams (DTW morfológico como discriminador más fuerte), CEAP-360VR (pupila con z-score y corrección de luminancia), Kumar (cvxEDA: separar componente fásica), BIOT (normalización por percentil 95), Luzzani (baseline correction + min-max por participante; binario mejor que graduar), Shikha (HRV/EDA como features más discriminativas vía SHAP), Diachenko (anotación experta de artefactos con κ=0.54 → necesidad de métrica objetiva).
- **S_coher (coherencia inter-modal como señal de fiabilidad)**: Boulanger (coherencia ECG-ACC reduce FP 67%), Abadi (fusión adaptativa a calidad), Hollenstein (fusión late), Hyun (contexto inter-modal como extensión), Del Pup (distinguir señal genuina de degradación), Adams (fusión IMU+impedancia como gating; HR consistente ECG/ICG/PPG), Kumar (EDA+ECG > unimodal en CASE/WESAD), Dar (majority voting ECG+EEG 98.8% — pero modalidad débil GSR degrada), CEAP-360VR (ablación conducta+fisiológica), MultiPhysio-HRC (señales fisiológicas > EEG > voz; fusión da mejor F1), Shikha (EDA+BVP+HRV > unimodal), Wan (CMT + MTF mejoran clasificación multimodal), Aygun (plataforma EEG+fNIRS+gaze+pupila sincronizada para estados cognitivos).
- **S_cond (conducta/atención como dimensión de fiabilidad)**: Gao (tiempo de finalización y confianza como indicadores), Boulanger (movimiento físico), Ronca (neurométricas), Mohr (comunicación/movimiento), Abadi (head-pose), CEAP-360VR (HM/EM correlacionados con valencia/arousal), Adams (IMU = movimiento físico), Luzzani (movimientos oculares con ratio LF/HF como señal conductual), Aygun (mind wandering/distracción con eventos de tarea y pupila).
- **Enfoque no supervisado**: Hyun (compite con supervisado), Nahmias (requiere ground truth — el PSRI no), Abadi (requiere anotadores expertos), Ibrahim (garantías de identificabilidad pero parámetros desconocidos en la práctica).
- **El preprocesamiento estándar no basta (evidencia del gap)**: Dar (señales mal clasificadas con ruido residual que los filtros no eliminaron), Adams (necesidad de gating adicional al filtrado), Gao (auto-reportes no fiables por defecto).

### C.4 Hoja de ruta de validación derivada de la revisión

1. **PhysioNet/CinC 2011** → validar S_estab (calidad técnica). ✅ Hecho (AUC 0.887).
2. **K-EmoCon** → hipótesis central (PSRI → desacuerdo entre anotadores; anotación tripartita con Krippendorff's alpha). 🔴 Principal objetivo.
3. **ZuCo** → S_coher (EEG+ET) en dominio lingüístico.
4. **GAZE4HATE / eyeStyliency** → S_estab/S_cond en tareas subjetivas donde sujeto=anotador.
5. **LLMs como generadores de ground truth** (Zhang et al.) → plan B si faltan anotaciones humanas.
6. **Joint training / PSRI como regularización** (Sood et al.) → extensión futura.
