"""
Punto de entrada principal del proyecto PSRI (Physiological Signal Reliability Index).

Este módulo inicializa el entorno, carga la configuración global y lanza el
proceso correspondiente al argumento `--proc` (`PHYSIO`, `EXIST`, `KEMOCON`).
Cada pipeline se importa y ejecuta de forma diferida dentro de su propia
función para evitar dependencias circulares y cargar únicamente lo necesario.

Autor: Enrique
"""
from sources.common.common import processControl, logger, writeLog
from sources.common.paramsManager import getConfigs

def process_physio():
    """Ejecuta el pipeline de validación del instrumento contra PhysioNet.

    Importa y llama a `sources.physionet.validator.process_physio`, que
    orquesta las cinco validaciones del Estudio 3 (comparativa de AUC,
    significación, sensibilidad de calibración, solapamiento y ablación).

    Returns:
        None
    """
    from sources.physionet.validator import process_physio
    process_physio()

    return

def process_kemocon():
    """Ejecuta el pipeline del Experimento 2 sobre el dataset K-EmoCon.

    Importa y llama a `sources.kemocon.run_pipeline.process_kemocon`, que
    carga, agrega por ventanas y valida el PSRI contra las métricas de
    desacuerdo entre anotadores.

    Returns:
        None
    """
    from sources.kemocon.run_pipeline import process_kemocon
    process_kemocon()

    return

def process_exist():
    """Ejecuta el pipeline del Experimento 1 sobre el dataset EXIST 2026.

    Construye el DataFrame de características (PSRI + coherencia) y ejecuta
    el sanity check con corrección por comparaciones múltiples.

    Returns:
        None
    """
    from sources.exist.build_features import process_dataframe
    process_dataframe()
    from sources.exist.sanity_check import process_sanity_check
    process_sanity_check()


def customProcess():
    """Proceso por defecto cuando `--proc` no es un valor conocido.

    Actualmente es un punto de extensión vacío: no realiza ninguna acción.

    Returns:
        None
    """
    return

if __name__ == '__main__':
    writeLog("info", logger, "********** STARTING PSRI (Physiological Signal Reliability Index) **********")
    getConfigs()

    if processControl.args.proc == "PHYSIO":
        process_physio()
    elif processControl.args.proc == "EXIST":
        process_exist()
    elif processControl.args.proc == "KEMOCON":
        process_kemocon()
    else:
        customProcess()

    writeLog("info", logger, "********** PROCESS COMPLETED **********")
