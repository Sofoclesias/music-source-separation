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
"""

from .constants import (
    LABELS
)

from .common import (
    download_datasets, download_stems, read_from_jams
)

__version__ = "v0.0.1"

