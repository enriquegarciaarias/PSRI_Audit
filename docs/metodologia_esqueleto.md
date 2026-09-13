# Esqueleto de la Sección de Metodología

> Nota: esqueleto — cada apartado lista los puntos a desarrollar. Se irá expandiendo uno a uno.
> Numeración acordada: la validación con PhysioNet **no es un experimento**, es la validación externa de la transformación en U de la dimensión instrumental (S_estab). Los experimentos son: **Experimento 1 = EXIST 2026**, **Experimento 2 = K-EmoCon**. La forma en U es una herramienta anclada a su dominio validado; las demás dimensiones eligen su forma por la semántica de su observable (§2.4).

---

## 1. Visión general del enfoque

- PSRI como **instrumento de fiabilidad previo a la integración** de señales fisiológicas en modelos multimodales (motivación heredada del análisis de EXIST Working Notes).
- Tres dimensiones independientes (cadena de tres eslabones): `S_estab` (instrumento), `S_coher` (organismo), `S_cond` (conducta).
- Estrategia en dos fases:
  1. **Construcción + validación externa de la fórmula en U** contra ground truth de calidad conocido (PhysioNet/CinC Challenge 2011).
  2. **Experimentos de aplicación** a dos datasets sin ground truth de calidad: **Experimento 1** (EXIST 2026) y **Experimento 2** (K-EmoCon).
- Principio de diseño: no entrenar modelos — validar un principio matemático simple, sensor-agnóstico.

## 2. Formulación matemática del PSRI

### 2.1 El modelo de tres eslabones
- Motivación: un trial "poco fiable" es ambiguo hasta especificar dónde falló la cadena de medición.
- Instrumento → `S_estab`; Organismo → `S_coher`; Conducta → `S_cond`.
- Predicción: dimensiones con correlaciones bajas entre sí (independencia empírica).
- Correlaciones cruzadas observadas (ConstruccionPSRI §1): EXIST — S_estab–S_coher r≈0.110, S_estab–S_cond r≈0.075, S_coher–S_cond r≈0.001; K-EmoCon — S_estab–S_coher 0.075, S_estab–S_cond 0.193 (pero ρ_between=0.373, ρ_within=0.029, r²<0.001), S_coher–S_cond 0.013.

### 2.2 `S_estab` — Estabilidad intra-trial
- Pregunta: "¿el sensor capturó algo físicamente plausible?" — conceptualmente un SQI.
- Estadístico de entrada: desviación estándar intra-trial por canal (σ).
- **Etapa 1 — normalización robusta en espacio logarítmico**: `z = (log σ − mediana) / (1.4826·MAD)` sobre la población de referencia (inmune a sensores degenerados).
- **Etapa 2 — función en U** (campana gaussiana) que penaliza ambos extremos (señal plana = electrodo caído; señal errática = ruido): `R = exp(−z²/2)`.
- Corte duro (`τ_min`, `eps_hard`): σ por debajo del umbral físico (p.ej. percentil 1 o ε=1e-6) → `R = 0` directamente.
- Agregación: media aritmética entre modalidades por trial; media entre sujetos por estímulo.
- *(Nota EXIST)*: la normalización híbrida sujeto+población es una **limitación documentada, no probada** — requiere series temporales crudas para modelar artefactos intra-trial, no disponibles en el formato agregado de EXIST (ConstruccionPSRI §7.6 / EstudioEXIST §7).

### 2.3 `S_coher` — Coherencia multimodal
- Pregunta: "¿la respuesta es genuina y coordinada entre sistemas fisiológicos distintos, o es ruido de un canal aislado?" — principio psicofisiológico de arousal sistémico (Bradley & Lang).
- Z-scores por sujeto: `z = (M_ij − baseline_prev) / σ_sujeto`, con σ_sujeto = desviación estándar de las medias de ese sujeto entre sus trials (no la variabilidad intra-trial).
- Transformación: decaimiento exponencial en forma de V de la discrepancia entre z-scores: `exp(−|z_a − z_b|/2)` (a diferencia de la U de S_estab, busca discrepancia nula, no punto dulce).
- Variantes de baseline: `prev1` (trial inmediato anterior) y `prev5` (media de 5 anteriores) — ventana rodante, no media de sesión.
- Pares según estudio: EXIST = HR–pupila; K-EmoCon = **HR(E4)–EDA** (dos sistemas fisiológicos; se descartó HR(E4)–HR(Polar) por ser redundancia instrumental — misma magnitud, dos instrumentos).

### 2.4 `S_cond` — Consistencia conductual
- Pregunta: "¿estaba la persona genuinamente comprometida con la tarea?" — análogo a *insufficient effort responding* (Meade & Craig).
- **Caja de herramientas, no forma fija:** la transformación se elige por la semántica del observable (conceptual_psri.md §1.3). Observable con interpretación unívoca (RT "menos es mejor", atención "más es mejor") → transformación **monotónica**; observable con ambos extremos patológicos → función **en U** (log-MAD + campana gaussiana). La regla se decide a priori por la semántica, no por el resultado estadístico; la U queda reservada a su dominio validado (S_estab).
- EXIST: `reaction_time` (monotónica "menos es mejor" → `S_cond_mono`) + `blinks_count` (en U), promediadas.
- K-EmoCon: 7 variantes en-U probadas, **ninguna adoptada a nivel within-subject** — `S_cond_att_acc`, `S_cond_att_med`, `S_cond_att`, `S_cond_att_std`, `S_cond_att_med_std` (señales débiles/inconsistentes; `S_cond_att_acc` gana señal pooled tras la corrección del alineamiento, Sección 3.7 de EstudioKemocon.md), `S_cond_self` (correlación enorme ρ −0.45…−0.61 pero **inválida**: 73.8%/68.3% de valores exactamente 0, variable casi binaria + ambigüedad conceptual), `S_cond_combined` (hereda el problema de self). Las 4 variantes monotónicas contrastan la regla de selección (Sección 3.9): recuperan señal within-de-sujeto que la U enmascara pero pierden la validez de tarea. En K-EmoCon S_cond se documenta como **limitación de disponibilidad de sensores** (sin observables conductuales directos), no como componente operativo del compuesto; queda como módulo de task-validity (Sección 3.8) y su caja de herramientas lista para datasets con observables directos.

### 2.5 PSRI compuesto
- Pesos iguales `w1 = w2 = w3 = 1/3` (elección de diseño explícita, justificada por independencia empírica entre dimensiones).
- Imputación de valores faltantes por mediana de la dimensión antes de combinar.
- EXIST: EEG queda excluido del compuesto (sin métrica de estabilidad intra-trial y sin solapamiento de sujetos con HR/ET).
- Otras opciones descartadas por diseño: consenso entre sujetos (error de categoría), magnitud absoluta de respuesta (confunde arousal con fiabilidad), índice único de SNR (pierde capacidad diagnóstica de *por qué* falla un trial), calidad espectral fina (inviable con datos agregados).

## 3. Datasets

### 3.1 PhysioNet/CinC Challenge 2011 (validación externa)
- 998 grabaciones ECG de 12 derivaciones, 10 s, 500 Hz; etiquetadas aceptable (n=773) / inaceptable (n=225) por 3–18 expertos.
- Formato WFDB, estructura `set-a/` con `RECORDS`, `RECORDS-acceptable`, `RECORDS-unacceptable`.
- Uso: ground truth de calidad de señal para validar la fórmula en U (`S_estab`), **no** un experimento de aplicación.

### 3.2 EXIST 2026 (Experimento 1)
- Memes con señales fisiológicas de anotadores: HR (Garmin), EEG, eye-tracking (ET) + JSON con etiquetas hard/soft (3 subtareas de sexismo).
- Cobertura: 8 usuarios HR, 8 EEG, 8 ET; HR y ET comparten los mismos 8 sujetos; EEG es un grupo disjunto (0% solapamiento de pares) → `S_coher` solo HR–ET.
- 3982 memes con los tres sensores y etiqueta (99.9% del total).
- Variables: HR (`garmin_hr_mean/std`, baseline_prev); ET (diámetro pupilar medio/std + baseline_prev, `reaction_time`, `blinks_count`); EEG sin métrica intra-trial equivalente.

### 3.3 K-EmoCon (Experimento 2)
- Debates en pareja (10–30 min): el participante lleva sensores y es a la vez el objeto del juicio (self, partner, 5 anotadores externos R1–R5) — alineación poblacional que resuelve la limitación estructural de EXIST.
- Sensores: pulsera E4 (BVP, EDA, HR, IBI, TEMP, ACC), banda Polar (HR), diadema NeuroSky (Attention, Meditation, BrainWave).
- Anotaciones continuas de valence/arousal (SAM); métricas de desacuerdo: varianza/rango externo, discrepancias self-partner / self-external.
- Preprocesado: tablas de calidad E4 (completeness/zeros/outliers/durations), sincronización de timestamps ms→s, ventaneo de 5 s.
- Criterios de exclusión de sujetos: completeness <90%, ceros >20%, outliers >10%, ratio de duración entre modalidades <50% → de 27 elegibles a **23 sujetos** (excluidos pid 4/28 por caída IBI, pid 17/20 por EDA degenerada) → **5528 ventanas** de 5 s.

## 4. Validación externa de la fórmula (PhysioNet)

### 4.1 Diseño
- Aplicar la transformación en U sobre la variabilidad de la señal cruda (`std_signal`: desviación estándar promediada entre las 12 derivaciones) y, como análisis de sensibilidad, sobre una magnitud derivada (`rr_std`: std de intervalos RR vía `xqrs_detect`).
- Comparación contra SQIs de referencia: kSQI (curtosis) y Correlación inter-derivación (matriz 12×12).
- Casos degenerados (varianza <1e-6, n=135; 129/135 inaceptables): **sentinel** de calidad mínima en vez de exclusión, para no inflar el AUC de los SQIs de referencia.

### 4.2 Métricas
- AUC (curva ROC), G-mean (umbral 0.5), PR-AUC (desbalance 77%/23%).

### 4.3 Resultados principales
- PSRI (std_signal): **AUC 0.887 / G-mean 0.814**; kSQI: 0.842 / 0.707; Correlación: 0.794 / 0.678.
- PSRI supera a los dos índices diseñados específicamente para ECG usando una sola característica (σ) y sin conocimiento de dominio.
- Sensibilidad al nivel de señal: rr_std AUC 0.748 (todos) → 0.671 (extracción válida, 163/998 fallos 16.3%); el fallo de detección ya actúa como clasificador parcial de calidad.

### 4.4 Inferencia estadística
- DeLong para AUCs correlacionadas: vs. Correlación p=0.0003; vs. kSQI p=0.066 (consistente en dirección, no significativo).
- Bootstrap pareado (2000 réplicas): IC95 [0.043, 0.143] vs. Correlación; [−0.004, 0.093] vs. kSQI.
- PR-AUC (0.951 / 0.911 / 0.875) descarta que la ventaja dependa del desbalance.
- Robustez fuera de muestra: 200 particiones 50/50 calibración/evaluación → AUC medio 0.886 (IC95 [0.855, 0.920]) ≈ in-sample 0.887 → la calibración interna no explica la ventaja.
- Análisis del error residual (~11%): los falsos positivos (n=44) caen al 100% en la zona de solapamiento de std_signal [0.093, 0.400] (sobrerrepresentación ~4.4×); los falsos negativos (n=137) el 43.1% (infrarrepresentación ~2.1×) → limitación inherente a un único escalar (σ).

## 5. Análisis estadístico de los experimentos

### 5.1 Métricas de desacuerdo entre anotadores
- EXIST: entropía de Shannon de las soft labels por tarea (2.1, 2.2, 2.3).
- K-EmoCon: `external_valence_var` (varianza R1–R5, target principal), `external_valence_range`, `external_arousal_var`, `self_partner_diff`, `self_external_mean_diff`.

### 5.2 Corrección por comparaciones múltiples
- EXIST: 20 tests (2 Kruskal-Wallis + 18 Pearson: 6 métricas × 3 tareas) en un único bloque; umbral Bonferroni 0.05/20 = 0.0025.
- K-EmoCon: 10 tests del compuesto (2 variantes de baseline × 5 métricas de desacuerdo) en un único bloque, mismo estándar de rigor.

### 5.3 Experimento 1 (EXIST): nulo robusto
- Ninguna métrica (PSRI compuesto, S_estab, S_coher, S_cond, PSRI_hr, PSRI_et) supera Bonferroni ni FDR; tamaño de efecto trivial (máx. r² = 0.00118, <0.12% de varianza explicada).
- Kruskal-Wallis YES vs NO: p=0.0215 crudo → no significativo tras ajuste; DIRECT vs JUDGEMENTAL: p=0.8393.
- Con N≈3982 hay potencia suficiente para detectar r≈0.05 → el nulo no se debe a falta de potencia.
- Explicación estructural: sujetos fisiológicos (8) y población de anotadores LeWiDi son grupos **disjuntos** → hipótesis de causa común indirecta de dos eslabones, demasiado débil para detectarse (~4000 memes).

### 5.4 Experimento 2 (K-EmoCon): descomposición between/within
- `rho_between_subject` (medias por sujeto) vs `rho_within_subject` (person-mean-centering) — separa rasgos estables entre personas de relación trial-a-trial genuina.
- Diagnóstico de leverage: leave-one-subject-out (LOSO).
- **Hallazgo principal**: `external_valence_var` correlaciona **positiva** y significativamente *dentro de sujeto* con **S_coher HR-EDA** (ρ_within ≈ +0.046…+0.066, p 0.001–0.014 en el barrido) y **no** con S_estab (ρ_within ≈ +0.01, NS).
- Robustez: el signo positivo de S_coher sobrevive corrección por comparaciones múltiples, centrado por sujeto, LOSO, barrido de ventana de baseline (n_prev 1–10) y análisis de lag ±2.
- `external_valence_range`: significativo pooled; dentro-de-sujeto positivo en S_coher solo en ventanas de baseline cortas (p 0.005 prev1, 0.068 prev5) → réplica de apoyo.
- `external_arousal_var` **es significativo** pooled en el compuesto (ρ≈+0.10, p≈3.9e-8 prev1) pero sigue nulo dentro-de-sujeto; `self_partner_diff`/`self_external_mean_diff` **no** son significativos pooled; `self_partner_diff` gana significación within en S_estab (ρ_within≈+0.05, p≈0.004).
- `S_cond`: ninguna de las 7 variantes aporta evidencia within limpia (`S_cond_att_acc` gana señal pooled) → limitación parcial.

### 5.5 Tamaño del efecto y potencia
- Informar tamaños de efecto junto a p-valores.
- K-EmoCon: efecto modesto en magnitud (ρ_within positivo ≈ +0.05…+0.07 en S_coher; nulo en S_estab) pero genuino frente al nulo de EXIST; N=23 sujetos / 5528 ventanas → cautela en efectos que dependen de variación entre sujetos.

## 6. Robustez y análisis exploratorios

- Independencia cruzada entre componentes (Pearson, réplica Tabla 5 EXIST) → justifica pesos iguales.
- Barrido de ventana de baseline de `S_coher` (n_prev 1–10): efecto se mantiene con atenuación suave.
- Comparación de variantes de `S_cond` en paralelo.
- Cobertura real vs. imputada (S_estab/S_cond 100%, S_coher 98.8%).
- *(Historial del marco)*: anexo con correcciones metodológicas (CV→U, S_coher global→trial, S_coher HR-HR→HR-EDA, sentinel en PhysioNet).

### 6.1. Marco de tres niveles de validez (docs/dialogo.txt §"Qué análisis adicional")

La calidad se evalúa en **tres niveles separados**, de modo que una sesión con buena señal no se interprete automáticamente como una sesión de alta atención, ni un compromiso de ventana se confunda con calidad fisiológica:

| Nivel | Qué evalúa | Operacionalización en K-EmoCon | Evita |
|---|---|---|---|
| **Q_session** (session_validity) | Completitud, duración y cobertura de modalidad de la sesión/sujeto | Criterios de exclusión de `sanity_check` (completeness ≥90%, ceros ≤20%, outliers ≤10%, ratio de duración ≥50%) + `flag_low_quality_subjects` | Mezclar sesiones defectuosas con ventanas válidas |
| **Q_window** (window_task_validity) | Evidencia de que la ventana pertenece a una interacción activa (reposo vs. debate) | Asignación de `period` (pre/debate) por `startTime`/`endTime` en `task_validity.assign_period`; contraste reposo-vs-debate (modelo mixto + Wilcoxon por sujeto + LOSO + drop-first-windows) | Interpretar ventanas de reposo como tarea atencional |
| **Q_physio** (physiological_quality) | Estabilidad intra-canal y coherencia cross-sistema en la ventana | S_estab + S_coher (núcleo validado del compuesto confirmatorio) | Confundir calidad fisiológica con compromiso conductual |

La separación corrige el riesgo señalado en el diálogo: los criterios de completitud/ceros/outliers/duración son de **sujeto/sesión**, no deben mezclarse con un componente de ventana de 5 s. El módulo `task_validity.py` materializa Q_window; `sanity_check.py` materializa Q_session; y el compuesto confirmatorio (Sección 6.4 de EstudioKemocon.md) se restringe a Q_physio (compuesto de 2 patas) + S_cond como módulo de task-validity separado.
