# src/physionet/loader.py
"""
Carga del ground truth y de las listas de registros de PhysioNet Challenge 2011.

Lee los ficheros `RECORDS`, `RECORDS-acceptable` y `RECORDS-unacceptable` del
directorio de datos, que definen los IDs de los registros y su etiqueta de
calidad oficial.

Autor: Enrique
"""
import os

def load_ground_truth(data_dir):
    """Carga las etiquetas oficiales de calidad de los registros.

    Args:
        data_dir (str o Path): Directorio que contiene los ficheros
            `RECORDS-acceptable` y `RECORDS-unacceptable`.

    Returns:
        dict: Mapa {record_id: 1 si aceptable, 0 si inaceptable}.
    """
    gt = {}
    with open(os.path.join(data_dir, 'RECORDS-acceptable'), 'r') as f:
        for line in f:
            gt[line.strip()] = 1
    with open(os.path.join(data_dir, 'RECORDS-unacceptable'), 'r') as f:
        for line in f:
            gt[line.strip()] = 0
    return gt

def load_record_ids(data_dir):
    """Lee la lista de IDs de registros del fichero `RECORDS`.

    Args:
        data_dir (str o Path): Directorio con el fichero `RECORDS`.

    Returns:
        list: IDs de registros (strings), en orden de procesamiento.
    """
    with open(os.path.join(data_dir, 'RECORDS'), 'r') as f:
        return [line.strip() for line in f if line.strip()]