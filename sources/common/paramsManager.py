"""
Gestión de parámetros y configuración global del proyecto.

Este módulo centraliza el arranque de la configuración: parsea los
argumentos de línea de comandos, carga la configuración JSON, detecta la GPU
disponible, configura el entorno (variables de entorno, caché de Python) y
rellena el singleton `processControl` con todos los valores necesarios para
cualquier pipeline.

Autor: Enrique
"""
from sources.common.common import processControl, logger, writeLog
from sources.common.utils import configLoader, dbTimestamp

import argparse
import os
import sys
import socket

import torch
from huggingface_hub import login

# Constants for parameter files
JSON_PARMS = "config.json"

def manageArgs():
    """Parse los argumentos de línea de comandos del proceso principal.

    Returns:
        argparse.Namespace: Argumentos parseados con los atributos
            `subject` y `proc`.
    """
    parser = argparse.ArgumentParser(description="Main process for Scientific Literature Intelligence Pipeline (SLIP) handling.")
    parser.add_argument('--subject', type=str, help="Subject of investigation", default="sensores")
    parser.add_argument('--proc', type=str, help="Process type: proc PHYSIO, EXIST, KEMOCON", default="KEMOCON")

    args = parser.parse_args()
    return args


def check_gpu(min_memory_gb=8.0):
    """Detecta GPUs CUDA disponibles con memoria suficiente.

    Filtra las GPUs con memoria total >= `min_memory_gb`, fija la variable de
    entorno `CUDA_VISIBLE_DEVICES` con las adecuadas y actualiza
    `processControl.defaults['device']` a 'cuda' o 'cpu' según el resultado.

    Args:
        min_memory_gb (float, optional): Memoria mínima exigida a cada GPU.
            Por defecto es 8.0 GB.

    Returns:
        None
    """
    suitable_gpus = []
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        writeLog("info", logger, f'{num_gpus} CUDA devices available')

        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024 ** 3)
            if memory_gb >= min_memory_gb:
                suitable_gpus.append(i)
                writeLog("info", logger, f"GPU {i} suitable: {props.name} ({memory_gb:.1f} GB)")
            else:
                writeLog("info", logger, f"GPU {i} skipped: {props.name} ({memory_gb:.1f} GB < {min_memory_gb} GB)")
    else:
        writeLog("info", logger, "No CUDA devices available")

    if suitable_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, suitable_gpus))
        processControl.defaults['device'] = 'cuda'
        writeLog("info", logger, f"Selected GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        processControl.defaults['device'] = 'cpu'
        writeLog("info", logger, "No suitable GPUs found; falling back to CPU")


def huggingface_login():
    """Autentica contra Hugging Face usando el token de configuración.

    Uses el token almacenado en `processControl.defaults['huggingFaceToken']`.
    Se emplea para descargar modelos privados del hub.

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


def setEnvironment():
    """Configura variables de entorno y caché de Python para el proceso.

    Fija `TOKENIZERS_PARALLELISM=false`, reubica la caché de bytecode de
    Python (`PYTHONPYCACHEPREFIX`) y, para la máquina PULSAR-PRO, prepara
    el entorno de entrenamiento distribuido (RANK/WORLD_SIZE, memoria de
    PyTorch).

    Returns:
        None
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    cache = os.environ.get('.pycache', os.path.expanduser('~/.cache'))
    os.environ['PYTHONPYCACHEPREFIX'] = cache
    os.makedirs(cache, exist_ok=True)
    sys.pycache_prefix = cache

    min_memory = getattr(processControl.defaults, 'min_gpu_memory_gb', 6.0)  # Default 8GB; override in config
    check_gpu(min_memory_gb=min_memory)


    if processControl.env['systemName'] == "PULSAR-PRO":
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.cuda.set_per_process_memory_fraction(0.98, device=0)
        torch.backends.cuda.max_split_size_mb = 64


def manageEnv():
    """Construye el diccionario de rutas y variables del entorno.

    Lee la sección `environment` de la configuración, resuelve las rutas
    relativas contra `realPath`, crea el directorio de caché y añade el
    nombre del sistema (`systemName`) vía `socket.getfqdn()`.

    Returns:
        dict: Diccionario con las rutas de entorno y el nombre del sistema.
    """
    config = configLoader()
    environment = config.get_environment()

    env_data = {}
    for key, value in environment.items():
        if "realPath" in key:
            env_data[key] = value
        else:
            env_data[key] = os.path.join(environment["realPath"], value)

    os.makedirs(env_data['cache'], exist_ok=True)
    os.environ['PYTHONPYCACHEPREFIX'] = env_data['cache']
    sys.pycache_prefix = env_data['cache']
    env_data['systemName'] = socket.getfqdn()
    return env_data


def manageDefaults():
    """Carga la sección de valores por defecto de la configuración.

    Returns:
        dict: Valores por defecto definidos en `config.json`.
    """
    config = configLoader()
    environment = config.get_defaults()
    return environment

def manageDatasetVars():
    """Carga las variables del dataset y añade un timestamp.

    Returns:
        dict: Variables del dataset con el campo `timestamp` añadido.
    """
    config = configLoader()
    datasetVars = config.get_datasetVars()
    datasetVars['timestamp'] = dbTimestamp()
    return datasetVars


def getConfigs():
    """Carga la configuración completa en el singleton `processControl`.

    Puebla `processControl.env`, `processControl.args`,
    `processControl.defaults` y `processControl.datasetVars`, y aplica la
    configuración del entorno (GPU, variables de entorno).

    Returns:
        None
    """
    processControl.env = manageEnv()
    processControl.args = manageArgs()

    processControl.defaults = manageDefaults()
    processControl.datasetVars = manageDatasetVars()

    setEnvironment()
    writeLog("info", logger, "Configuration loaded.")
