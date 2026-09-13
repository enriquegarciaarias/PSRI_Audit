# PSRI — Physiological Signal Reliability Index

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![GitHub Pages](https://img.shields.io/badge/Docs-GitHub%20Pages-4FC08D?style=for-the-badge&logo=github&logoColor=white)](https://enriquegarciaarias.github.io/PSRI)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg?style=for-the-badge&logo=gnu&logoColor=white)](https://www.gnu.org/licenses/gpl-3.0)

Índice de fiabilidad de señales fisiológicas para tareas de Recuperación de Información Social y aprendizaje con desacuerdo (**LeWiDi**).

El **PSRI** evalúa la calidad de los datos fisiológicos en tres dimensiones independientes:

- 🛠️ **$S_{estab}$** — estabilidad instrumental (¿el sensor capturó algo físicamente plausible?)
- 🧬 **$S_{coher}$** — coherencia inter-sistémica (¿la respuesta es genuina y coordinada entre sistemas?)
- 🎯 **$S_{cond}$** — consistencia conductual (¿la persona estaba comprometida con la tarea?)

Agnóstico al dispositivo, robusto (Mediana/MAD) y libre de entrenamiento.

---

## 📚 Documentación

| Recurso | Enlace |
|---|---|
| 🌐 Sitio web del proyecto | <https://enriquegarciaarias.github.io/PSRI> |
| 🐙 Repositorio | <https://github.com/enriquegarciaarias/PSRI> |
| 📄 Paper (ECIR 2027) | `docs/ECIR2027/main.tex` |

---

## 🚀 Operativa

### 1. Entorno

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuración

Copia `config.json` en la raíz del repositorio si no existe (el fichero está ignorado por git). Contiene rutas, tokens y ajustes de GPU/LLM.

### 3. Ejecución

| Pipeline | Comando | Descripción |
|---|---|---|
| PhysioNet | `python main.py --proc PHYSIO` | Validación externa contra PhysioNet Challenge 2011 |
| EXIST 2026 | `python main.py --proc EXIST` | Experimento 1 (HR + EEG + eye-tracking) |
| K-EmoCon | `python main.py --proc KEMOCON` | Experimento 2 (wearables + anotaciones) |

Los resultados se escriben en `results/output/{PROC}` y los logs en `ProcessLog.txt` / `Process.txt`.

### 4. Datos de entrada

- 📥 `results/input/PHYSIO/set-a/` — `RECORDS`, `RECORDS-acceptable`, `RECORDS-unacceptable` (PhysioNet Challenge 2011)
- 📥 `results/input/EXIST/` — `HR_.xlsx`, `EEG_.xlsx`, `ET_.xlsx` + JSON de etiquetas (EXIST 2026)
- 📥 `results/input/KEMOCON/k-emocon/` — estructura del corpus K-EmoCon

---

## 🧪 Análisis exploratorios (no incluidos en el paper)

| Módulo | Descripción |
|---|---|
| `sources/kemocon/experiment3.py` | LOSO-CV del PSRI como filtro/peso en regresión continua (sin mejora significativa) |
| `sources/kemocon/experiment_estab_highvar.py` | Diagnóstico de la región de alta variabilidad de $S_{estab}$ |
| `sources/kemocon/task_validity.py` | Control de validez de tarea (pre-debate vs. debate) por componente |

---

## 🏗️ Estructura

```
sources/
├── psri/calculator.py      # Núcleo matemático del PSRI (NumPy puro)
├── physionet/              # Validación contra PhysioNet Challenge 2011
├── exist/                  # Pipeline del Experimento 1 (EXIST 2026)
└── kemocon/                # Pipeline del Experimento 2 (K-EmoCon)
docs/                       # Documentación y fuentes del paper
```

---

## 📖 Citar

```
@misc{garciaarias2026psri,
  title  = {PSRI: Physiological Signal Reliability Index},
  author = {Garcia-Arias, Enrique and Plaza, Laura and Carrillo-de-Albornoz, Jorge},
  year   = {2026},
  note   = {Submitted to ECIR 2027}
}
```
