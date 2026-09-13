"""
Variables globales de configuración y rutas del proyecto.

Este módulo declara los contenedores globales (argumentos, rutas de
directorios y estado del control de proceso) que comparten todos los
módulos. `procCtrl` se instancia en `sources.common.common` y se referencia
aquí como almacén compartido.

Autor: Enrique
"""
args = None
dirs = None
dirs = {
    'BASE_ROOT' : '../',
    'OUT_DIR' : "results",
    'TEXT_DIR' : "txt",
    'HTML_DIR' : "html",
    'CATEGORY_ATP_DIR' : "atpdocs",
    'APPS_MANAGER_DIR' : "corpuscrawl-manager",
    "CATEGORY_ATP_DIR" : "category-atp"
}
proc_ctrl = {}


