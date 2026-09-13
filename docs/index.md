# PSRI - Physiological Signal Reliability Index

El **PSRI (Physiological Signal Reliability Index)** es un índice que cuantifica
la fiabilidad de las señales fisiológicas en estudios de emociones multimodales.
Agrega tres dimensiones de fiabilidad:

| Dimensión | Descripción |
|---|---|
| **S_estab** | Estabilidad temporal de la señal (variabilidad intra-trial) |
| **S_coher** | Coherencia entre dos sistemas fisiológicos distintos |
| **S_cond** | Consistencia conductual respecto a la tarea |

## Arquitectura del proyecto

- **`main.py`** — punto de entrada. Selecciona el pipeline mediante `--proc`
  (`PHYSIO`, `EXIST`, `KEMOCON`).
- **`sources/psri/`** — núcleo matemático del PSRI (NumPy puro).
- **`sources/physionet/`** — validación de calidad ECG contra el reto
  PhysioNet Challenge 2011.
- **`sources/exist/`** — pipeline del dataset EXIST 2026 (HR + EEG +
  eye-tracking).
- **`sources/kemocon/`** — pipeline completo K-EmoCon (wearable E4 +
  anotaciones).
- **`sources/common/`** — utilidades compartidas (config, logging, métricas,
  GPU).

## Documentación

- **Referencia de la API**: documentación autogenerada de todos los módulos
  con docstrings estilo Google.
- **Documentación del proyecto**: análisis de literatura, estudios EXIST /
  K-EmoCon y la validación PhysioNet.

## Uso rápido

```sh
python main.py --proc PHYSIO   # Validación PhysioNet Challenge 2011
python main.py --proc EXIST    # Pipeline EXIST 2026
python main.py --proc KEMOCON  # Pipeline K-EmoCon
```

## Desarrollo de la documentación

```sh
mkdocs serve    # Servidor local con recarga en vivo
mkdocs build    # Genera el sitio estático en site/
```

El despliegue en GitHub Pages es automático mediante GitHub Actions (ver
`.github/workflows/`).