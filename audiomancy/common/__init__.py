"""Funciones para manejar los archivos comunes del dataset o también algunos
quality of life.
"""

import os

def download_stems():
    print(True)
    
def download_datasets():
    print(True)

def read_from_jams(jams_path: int | str = 200):
    """Función de atajo para recuperar al toque los tensores X e Y. Se asume que ya se tienen 
    jams files creados cuando se coloca un integer.

    Args:
        jams_path (int | str, optional): Ruta al archivo .jams completo si es un string; número
        de muestras si es un integer. Por defecto, 200.

    Returns:
        X: Array del mix con todas las frecuencias unidas. Dimensiones:
            (n_files, marco 1D (1), frecuencias)
        Y: Array del mix con todas las frecuencias separadas. Dimensiones:
            (n_files, marco 1D (1), frecuencias, pistas separadas (6))
    """
    
    from ..audioprocessing import cacophony
    from ..constants import JAMS_FILE_1000, JAMS_FILE_200,ABSOLUTE_PATH
    
    if type(jams_path)==int:
        if jams_path==200:
            jams_path = JAMS_FILE_200
        elif jams_path==1000:
            jams_path = JAMS_FILE_1000
        else:
            jams_path = os.path.join(ABSOLUTE_PATH,'common',f'{jams_path}_soundscapes.jams')
    elif type(jams_path)==str:
        pass
    else:
        raise AssertionError('Tipo de argumento no válido.')
    
    mixer = cacophony()
    X, Y = mixer.read_from_jams(jams_path)
    
    return X, Y