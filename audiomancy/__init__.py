"""
Audiomancy (nombre en proceso) es el nombre del algoritmo
de aprendizaje profundo dedicado a la separación de seis fuentes
musicales (batería, bajo, voz, de cuerda, de viento y acústicos).

La etimología es la unión entre las palabras 'Audio' y 'mancia'
(adivinación), que corresponde a la capacidad del modelo para
adivinar y segmentar las pistas musicales.

Se ha construido en sistema de paquete en Python para facilitar
la reproducibilidad, la organización de las funciones y archivos,
y la ejecución de operaciones embebidas.

https://github.com/Sofoclesias/music-source-separation
"""

import warnings
import os

from .constants import (
    LABELS, STEMS_PATH
)

from .common import (
    download_datasets, download_stems, read_from_jams
)

stream_handler = os.popen('sox -h')
if not len(stream_handler.readlines()):    
    warnings.warn("""No se ha encontrado el programa SoX.
                  
    La confección de audios aleatorios mediante .JAMS requiere
    de este programa.

    Para descargarlo, proceder acá:
     - - - http://sox.sourceforge.net/ - - -

    O descárguelo por chocolatey.
    """)
stream_handler.close()

if not os.path.exists(STEMS_PATH):
    warnings.warn("""No se ha encontrado el directorio de stems.
                  
    La confección de audios aleatorios mediante .JAMS requiere
    de esta carpeta en audiomancy/common.
    
    Para descargarlo, correr las siguientes líneas de código en 
    su directorio raíz:
    
    >import audiomancy
    >audiomancy.common.download_stems() 
    
    O descárguelo directamente desde https://mega.nz/file/eUwHCLCJ#7g4qRZnCxxgnQyY8WzWaWAjg14k_D59FiJRjLyz1MJo
    """)

__version__ = "v0.0.1"

