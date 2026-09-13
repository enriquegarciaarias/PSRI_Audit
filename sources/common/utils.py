"""
Utilidades genéricas compartidas por todos los pipelines del proyecto.

Este módulo contiene funciones auxiliares de propósito general: login en
Hugging Face, hashing y normalización de texto, gestión de directorios,
marcas de tiempo, carga/serialización de JSON, la clase `configLoader` para
acceder a la configuración y la utilidad `inicioModulo` que construye las
rutas de entrada/salida de cada proceso.

Autor: Enrique
"""
from sources.common.common import logger, processControl, writeLog
import json

import time
import os
from os.path import isdir

import unicodedata
import re
from huggingface_hub import login
import hashlib
from pathlib import Path
from json import JSONDecodeError


def huggingface_login():
    """Autentica contra Hugging Face usando el token de configuración.

    Args:
        None

    Returns:
        None

    Raises:
        Exception: Si la autenticación falla, se registra el error y se
            relanza la excepción.
    """
    try:
        # Add your Hugging Face token here, or retrieve it from environment variables
        token = processControl.defaults['huggingFaceToken'] if 'huggingFaceToken' in processControl.defaults else ['', '']
        login(token)
        writeLog("info", logger, "Successfully logged in to Hugging Face.")
    except Exception as e:
        writeLog("error", logger, f"Error logging into Hugging Face {str(e)}")
        raise

def sha1(text: str) -> str:
    """Genera un hash SHA1 a partir de un texto.

    Args:
        text (str): Cadena a hashear.

    Returns:
        str: Hash SHA1 hexadecimal de la cadena.
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def mkdir(dir_path):
    """Crea un directorio si no existe.

    Args:
        dir_path (str o Path): Ruta del directorio a crear.

    Returns:
        None
    """
    if not isdir(dir_path):
        os.makedirs(dir_path)

def safe_filename(name: str) -> str:
    """Convierte una cadena en un nombre de archivo seguro.

    Elimina tildes y acentos, sustituye espacios por guiones bajos, elimina
    cualquier carácter no alfanumérico (salvo guión y subrayado) y pasa el
    resultado a minúsculas.

    Args:
        name (str): Nombre original.

    Returns:
        str: Nombre normalizado y seguro para usar como archivo.
    """
    # Normaliza caracteres (elimina tildes y acentos)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    # Reemplaza espacios por guiones bajos
    name = name.replace(" ", "_")
    # Elimina cualquier carácter no alfanumérico, guión o subrayado
    name = re.sub(r"[^A-Za-z0-9_\-]", "", name)
    # Convierte a minúsculas
    return name.lower()


def dbTimestamp():
    """Genera una marca de tiempo con formato YYYYMMDDHHMMSS.

    Returns:
        str: Marca de tiempo UTC formateada.
    """
    timestamp = int(time.time())
    formatted_timestamp = str(time.strftime("%Y%m%d%H%M%S", time.gmtime(timestamp)))
    return formatted_timestamp

class configLoader:
    """Carga y proporciona acceso a la configuración JSON del proyecto.

    Attributes:
        base_path (str): Ruta absoluta del directorio de trabajo.
        config (dict): Contenido del fichero JSON de configuración.
    """

    def __init__(self, config_path='config.json'):
        """Inicializa el loader leyendo el fichero de configuración.

        Args:
            config_path (str, optional): Ruta (relativa o absoluta) del
                fichero JSON. Por defecto es 'config.json'.
        """
        self.base_path = os.path.realpath(os.getcwd())
        realConfigPath = os.path.join(self.base_path, config_path)
        self.config = self.load_config(realConfigPath)

    def load_config(self, realConfigPath):
        """Lee y parsea el fichero JSON de configuración.

        Args:
            realConfigPath (str): Ruta absoluta del fichero JSON.

        Returns:
            dict: Contenido parseado del fichero.
        """
        with open(realConfigPath, 'r') as config_file:
            return json.load(config_file)

    def get_environment(self):
        """Devuelve la sección `environment` de la configuración.

        Añade la clave `realPath` con la ruta base del proyecto.

        Returns:
            dict: Variables de entorno, con `realPath` añadido.
        """
        environment =  self.config.get("environment", None)
        environment["realPath"] = self.base_path
        return environment

    def get_defaults(self):
        """Devuelve la sección `defaults` de la configuración.

        Returns:
            dict: Valores por defecto.
        """
        return self.config.get("defaults", {})

    def get_search(self):
        """Devuelve la sección `search` de la configuración.

        Returns:
            dict: Parámetros de búsqueda.
        """
        return self.config.get("search", {})

    def get_datasetVars(self):
        """Devuelve la sección `datasetVars` de la configuración.

        Returns:
            dict: Variables del dataset.
        """
        return self.config.get("datasetVars", {})
    def get_params(self):
        """Devuelve la sección `params` de la configuración.

        Returns:
            dict: Parámetros del proceso.
        """
        return self.config.get("params", {})

def image_parser(args):
    """Divide un argumento de imagen por el separador configurado.

    Args:
        args (argparse.Namespace): Argumentos con `image_file` y `sep`.

    Returns:
        list: Partes del `image_file` divididas por el separador.
    """
    out = args.image_file.split(args.sep)
    return out

def safe_int(value):
    """Convierte un valor a entero de forma segura.

    Args:
        value: Valor a convertir.

    Returns:
        int o None: Entero convertido, o None si no es posible.
    """

    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    try:
        return int(value)
    except (ValueError, TypeError):
        return None

def normalized_title(text: str) -> str:
    """Normaliza un título a minúsculas y sin signos diacríticos.

    Elimina acentos, convierte a minúsculas, conserva solo caracteres
    alfanuméricos y espacios, y colapsa espacios repetidos.

    Args:
        text (str): Título original.

    Returns:
        str: Título normalizado.
    """
    if not text:
        return ""

    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def normalize_text(text):
    """Normaliza un texto a una única línea sin espacios redundantes.

    Sustituye saltos de línea por espacios y colapsa espacios múltiples.

    Args:
        text (str): Texto original.

    Returns:
        str: Texto normalizado en una sola línea.
    """
    if not text:
        return ""

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_doi(doi):
    """Normaliza un DOI a minúsculas y sin espacios.

    Args:
        doi (str): DOI original.

    Returns:
        str: DOI normalizado (vacío si no se aporta).
    """
    if not doi:
        return ""

    return doi.strip().lower()

def inicioModulo(modulo):
    """Construye las rutas de entrada y salida de un proceso.

    Las rutas se derivan de `processControl.env` (`input`/`output`) y del
    argumento `processControl.args.proc`, según el patrón
    `results/{input|output}/{PROC}`.

    Args:
        modulo (str): Nombre del módulo/proceso, usado solo en el log.

    Returns:
        tuple: (Path de entrada, Path de salida).
    """
    writeLog("info", logger, "-" * 60)
    writeLog("info", logger, f"🚀 [START] Processing {modulo}")
    base_input_dir = Path(processControl.env.get("input", "")) / processControl.args.proc
    base_output_dir = Path(processControl.env.get("output", "")) / processControl.args.proc

    return base_input_dir, base_output_dir

def read_json(filepath: str | Path):
    """
    Reads and parses a JSON file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Parsed JSON object (dict, list, etc.).

    Raises:
        FileNotFoundError: If the file does not exist.
        IsADirectoryError: If the path points to a directory.
        PermissionError: If the file cannot be accessed.
        ValueError: If the JSON is malformed.
        OSError: For other I/O related errors.
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    if not path.is_file():
        raise IsADirectoryError(f"Expected a file, got: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            writeLog("info", logger, f"📄 JSON loaded from {filepath}")
            return json.load(f)

    except JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in '{path}': {e}") from e

def write_json(
    filepath: str | Path,
    data,
    *,
    indent: int = 4,
    ensure_ascii: bool = False,
):
    """
    Writes data to a JSON file.

    Args:
        filepath: Destination JSON file.
        data: Serializable Python object.
        indent: JSON indentation.
        ensure_ascii: Whether to escape non-ASCII characters.

    Raises:
        TypeError: If data is not JSON serializable.
        OSError: If the file cannot be written.
    """
    path = Path(filepath)

    # Create parent directories if they do not exist
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                indent=indent,
                ensure_ascii=ensure_ascii,
            )
            f.write("\n")  # POSIX-friendly final newline

        writeLog("info", logger, f"💾 JSON written to {path}")

    except TypeError as e:
        raise TypeError(f"Object is not JSON serializable: {path}") from e

    except OSError as e:
        raise OSError(f"Could not write JSON file: {path}") from e