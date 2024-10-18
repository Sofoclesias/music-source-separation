"""
Funciones para manejar los archivos comunes del dataset o también algunos
quality of life.
"""

import os
import shutil

def download_stems():
    """
    Descarga las pistas de audio seleccionadas para el trabajo.
    Pesan cerca de 18 GB.
    
    Estos archivos sí son obligatorios para recuperar las muestras aleatorias
    y entrenar a los modelos.
    """
    stems_url = 'https://mega.nz/file/eUwHCLCJ#7g4qRZnCxxgnQyY8WzWaWAjg14k_D59FiJRjLyz1MJo'
    stems_size = 18 # 18 GB
    
    _, _, free = shutil.disk_usage("/")
    
    if free/(2**10 * 2**10 * 2**10) < stems_size:
        raise MemoryError(f"No hay suficiente espacio en disco para descargar los stems.\n El peso conjunto es {stems_size} GB, pero usted tiene disponibles {free} GB.")
    else:
        op = input(f"""Se instalarán los stems: {stems_size} GB ({stems_url})

Actualmente tiene {free/(2**10 * 2**10 * 2**10)} GB en el disco.

¿Desea continuar? [y/n]:""")
        
        if op=='y':
            from .utils import download_mega, unzip_file
            from ..constants import STEMS_PATH
            
            print('\nDescargando stems.zip')
            download_mega(stems_url,os.path.join(STEMS_PATH,'stems.zip'))
            print(f'\nDescomprimiendo stems.zip')
            unzip_file(os.path.join(STEMS_PATH,'stems.zip'),os.path.join(STEMS_PATH,'stems'))
                             
            print(f'Descarga y descompresión completada. Puede encontrar los archivos en {STEMS_PATH}.')
        else:
            pass
    
def download_datasets():
    """
    Descarga las pistas de audio sin alterar de los datasets originales.
    Pesan cerca de 72 GB.
    
    Solo descárguelo si quiere evaluar los tracks desde sus fuentes originales.
    """
    synthSOD_url = 'https://zenodo.org/record/13759492/files/SynthSOD.zip?download=1'
    musdb18hq_url = 'https://zenodo.org/record/3338373/files/musdb18hq.zip?download=1'
    musicnet_url = 'https://zenodo.org/record/5120004/files/musicnet.tar.gz?download=1'
    
    synthSOD_size = 45  # 45 GB para SynthSOD
    musdb18hq_size = 26  # 26 GB para MUSDB18HQ (desde Zenodo)
    musicnet_size = 11.1  # 11.1 GB para MusicNet
    
    _, _, free = shutil.disk_usage("/")
    
    if free/(2**10 * 2**10 * 2**10) < synthSOD_size + musdb18hq_size + musicnet_size:
        raise MemoryError(f"No hay suficiente espacio en disco para descargar los tres datasets.\n El peso conjunto es {synthSOD_size + musdb18hq_size + musicnet_size} GB, pero usted tiene disponibles {free} GB.")
    else:
        op = input(f"""Se instalarán los siguientes datasets:
              
SynthSOD: {synthSOD_size} GB ({synthSOD_url})
MusicNet: {musicnet_size} GB ({musicnet_url})
Musdb18-HQ: {musdb18hq_size} GB ({musdb18hq_url})

Actualmente tiene {free/(2**10 * 2**10 * 2**10)} GB en el disco.

¿Desea continuar? [y/n]:""")
        
        if op=='y':
            from .utils import download_file, untar_file, unzip_file
            import re
            from ..constants import DATASETS_PATH
            
            os.makedirs(DATASETS_PATH,exist_ok=True)
            filename = re.compile(r'(?<=/)[\w.]+(?=\?)')
            
            for dataset in [synthSOD_url, musicnet_url, musdb18hq_url]:
                f = filename.findall(dataset)[0]
                path = os.path.join(DATASETS_PATH,f)
                
                print(f'\nDescargando {f}')
                download_file(dataset,f)
                print(f'\nDescomprimiendo {f}')
                if 'zip' in f:
                    unzip_file(path,os.path.join(DATASETS_PATH,f.split('.')[0].lower()))
                elif 'tar.gz' in f:
                    untar_file(path,os.path.join(DATASETS_PATH,f.split('.')[0].lower()))
                    
                print(f'Descargas y descompresiones completadas. Puede encontrar los archivos en {DATASETS_PATH}.')
        else:
            pass

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