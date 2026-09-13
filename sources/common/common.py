"""
Utilidades globales de logging y control de proceso.

Este módulo configura el sistema de logging del proyecto (consola con color y
archivo rotatorio), define el objeto global de control de proceso
(`processControl`, un singleton) y expone las funciones auxiliares para
escribir mensajes de log. Es el punto común del que dependen todos los demás
módulos del proyecto.

Autor: Enrique
"""
from sources.common import global_vars

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from colorama import just_fix_windows_console

# ----------------------------------------------------------------------
# Enable ANSI support (Windows/Linux/macOS)
# ----------------------------------------------------------------------

just_fix_windows_console()

USE_COLORS = (
    sys.stderr.isatty()
    and os.getenv("TERM") not in (None, "dumb")
)

# ----------------------------------------------------------------------
# Global process control
# ----------------------------------------------------------------------

class controlProcess:
    """Contenedor global de la configuración y estado del proceso.

    Instancia única accesible vía `processControl` (véase el final del
    módulo). Almacena los argumentos de línea de comandos, los valores por
    defecto, las variables del dataset y los parámetros del entorno.

    Attributes:
        datasetVars (dict): Variables específicas del dataset procesado.
        args (argparse.Namespace): Argumentos parseados de línea de comandos.
        defaults (dict): Valores por defecto cargados de la configuración.
        parms (dict): Parámetros adicionales del proceso.
    """

    def __init__(self, datasetVars=None, args=None, defaults=None, parms=None):
        """Inicializa el control de proceso con los parámetros dados.

        Args:
            datasetVars (dict, optional): Variables del dataset. Por defecto
                es None.
            args (argparse.Namespace, optional): Argumentos de línea de
                comandos. Por defecto es None.
            defaults (dict, optional): Valores por defecto de configuración.
                Por defecto es None.
            parms (dict, optional): Parámetros adicionales. Por defecto es
                None.
        """

        self.datasetVars = datasetVars or {}
        self.args = args or {}
        self.defaults = defaults or {}
        self.parms = parms or {}

    def to_dict(self):
        """Serializa el estado del control de proceso en un diccionario.

        Returns:
            dict: Diccionario con las cuatro secciones del estado
                (datasetVars, args, defaults y parms).
        """

        return {
            "datasetVars": self.datasetVars,
            "args": self.args,
            "defaults": self.defaults,
            "parms": self.parms,
        }

global_vars.procCtrl = controlProcess()
processControl = global_vars.procCtrl

# ----------------------------------------------------------------------
# Colors
# ----------------------------------------------------------------------

COLORS = {
    "DEBUG": "\033[36m",       # Cyan
    "INFO": "\033[92m",        # Green
    "WARNING": "\033[93m",     # Yellow
    "ERROR": "\033[91m",       # Red
    "CRITICAL": "\033[95m",    # Magenta
}

RESET = "\033[0m"

# ----------------------------------------------------------------------
# Formatter
# ----------------------------------------------------------------------

class ColoredFormatter(logging.Formatter):
    """Formateador de logs que colorea el nivel del mensaje en consola.

    Aplica el código ANSI del color correspondiente al nivel (DEBUG, INFO,
    WARNING, ERROR, CRITICAL) solo al nombre del nivel, y lo restaura tras
    formatear para que los handlers de archivo no reciban secuencias ANSI.

    Attributes:
        USE_COLORS (bool): Si la salida es un terminal compatible.
    """

    def format(self, record):
        """Formatea el registro aplicando color al nombre del nivel.

        Args:
            record (logging.LogRecord): Registro a formatear.

        Returns:
            str: Mensaje formateado con el nivel coloreado si procede.
        """
        # 1. Obtener el nivel original
        level = record.levelname
        # 2. Aplicar el color y el reset solo al nombre del nivel
        color = COLORS.get(level, "")
        if USE_COLORS and color:
            record.levelname = f"{color}{level}{RESET}"

        # 3. Formatear el mensaje completo usando el super().format
        result = super().format(record)

        # 4. Restaurar el nivel para otros handlers (como el archivo)
        record.levelname = level
        return result

# ----------------------------------------------------------------------
# Logger configuration
# ----------------------------------------------------------------------

def configureLogger(log_type="log", logger_name="PSRI"):
    """Configura y devuelve un logger con handler de archivo y consola.

    Args:
        log_type (str, optional): Tipo de log. "log" escribe también en
            consola (ProcessLog.txt); cualquier otro valor solo escribe en
            archivo (Process.txt). Por defecto es "log".
        logger_name (str, optional): Nombre del logger. Por defecto "PSRI".

    Returns:
        logging.Logger: Logger configurado. Si ya existía con handlers, se
            devuelve tal cual (idempotente).
    """
    log_file = "./ProcessLog.txt" if log_type == "log" else "./Process.txt"
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # --------------------------------------------------
    # File Handler
    # --------------------------------------------------

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )

    file_formatter = logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s")
    file_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)

    # --------------------------------------------------
    # Console Handler
    # --------------------------------------------------

    if log_type == "log":
        console_handler = logging.StreamHandler()
        console_formatter = ColoredFormatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    return logger

# ----------------------------------------------------------------------
# Logging helper
# ----------------------------------------------------------------------

def writeLog(level: str, logger: logging.Logger, message: str):
    """Escribe un mensaje en el logger en el nivel indicado.

    Args:
        level (str): Nivel de log ("debug", "info", "warning", "error",
            "critical").
        logger (logging.Logger): Logger sobre el que escribir.
        message (str): Mensaje a registrar.

    Returns:
        None

    Raises:
        None: si el nivel no existe se registra un error y se retorna.
    """
    level = level.lower()
    if not hasattr(logger, level):
        logger.error(f"Invalid log level '{level}'")
        return
    getattr(logger, level)(message)

# ----------------------------------------------------------------------
# Global loggers
# ----------------------------------------------------------------------

logger = configureLogger("log", "enriqueLog")
logProc = configureLogger("proc", "enriqueProc")