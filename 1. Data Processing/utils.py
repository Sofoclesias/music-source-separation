"""Este módulo conjunta todas las funciones utilizadas para
organizar las pistas de audio, explorarlas y procesarlas.

El resultado final de la ejecución de estas funciones, aplicadas 
en el archivo "1. Data organization.ipynb", es un archivo .jams 
que describe todas las mezclas aleatorias generadas.

Este contiene metadatos que permiten la reconstrucción de cada 
mezcla de manera eficiente. Cargando cada instancia, las muestras 
de audio se pueden reproducir y reconstruir fácilmente sin necesidad 
de crear archivos de audio pesados. De esta manera, se optimiza el 
uso de la memoria volátil y se reduce el consumo de espacio de 
almacenamiento, lo que hace un proceso de preparación más ligero.

Las funciones de data augmentation y confección de paisajes sonoros
se realizan mediante la librería scaper, del paper "Scaper: A Library 
for Soundscape Synthesis and Augmentation"

@inproceedings{Salamon:Scaper:WASPAA:17,
  author       = {Salamon, J. and MacConnell, D. and Cartwright, M. and Li, P. and Bello, J.~P.},
  title        = {Scaper: A Library for Soundscape Synthesis and Augmentation},
  booktitle.   = {IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
  month        = {Oct.},
  year         = {2017},
  pages        = {344--348}
}
"""
import numpy as np
np.float_ = np.float64
np.Inf = np.inf
import scaper
import librosa
import os
from tqdm import tqdm
import shutil
import jams

LABELS = {
    'accoustic':0,
    'bass':1,
    'drums':2,
    'piano':3,
    'strings':4,
    'vocals':5
}

def remove_silence(audio_path):
    """Elimina los espacios silenciosos en una pista de audio.
    Si se dan un enlace de directorio, carga el archivo.
    """
    audio, _ = librosa.load(audio_path,sr=44100)
    
    non_silent = librosa.effects.split(audio,top_db=60)
    audio_rel = np.empty((1,))
    
    for start, end in non_silent:
        audio_rel = np.hstack((audio_rel,audio[start:end+1]))
        
    return audio_rel

def normalize_track(arr,length = 220500): 
    """Si se encuentran diferencias con la longitud dada, agrega
    ceros como elementos de array o recorta su longitud. El objetivo
    es que todos los archivos de audio tengan la misma extensión de 
    segundos de muestreo frecuencial.    
    """
    if arr.shape[0] < length:
        return np.hstack((arr,np.zeros((length - arr.shape[0],1),dtype='float64')))
    elif arr.shape[0] > length:
        return arr[:length]
    else:
        return arr

class cacophony:
    """Clase que facilita la creación de muestras aleatorias y la carga
    de metadatos ya pre-confeccionados.
    """
    def __init__(self, duration: float = 5.0, sampling: int = 44100, n_channels: int = 1, ref_db: int = -20, fg_path: str = 'stems', seed: int = 42):
        """Configura los hiperparámetros del creador de muestras
        aleatorias.

        Args:
            duration (float, optional): Determina la duración de cada 
            audio generado (segundos). Por defecto, 5.0 segundos.
            sampling (int, optional): Número de muestras por unidad de
            tiempo que se toman a las señales continuas de audio (hercios). 
            Por defecto, 44100 hercios, el formato de descompresión de
            nuestros archivos.
            n_channels (int, optional): Si es estéreo o mono. Por defecto,
            1, mono.
            ref_db (int, optional): Volumen de referencia para normalizar
            el rango de volúmenes de los audios (decibelios, dB). Por
            defecto, -20 dB.
            seed (int, optional): Semilla de aleatoriedad. 42.
            fg_path (str, optional): Directorio donde se ubican los audios.
            Por defecto, 'stems'. Los archivos dentro de esta carpeta
            deben estructurarse de la siguiente manera:
            stems/
              +----- accoustic/
              |         +---- acc1.wav
              |         +---- acc2.wav
              |         +---- acc3.wav
              |        ...
              +----- bass/
              |         +---- bss1.wav
              |         +---- bss2.wav
              |         +---- bss3.wav
              |        ...
              +----- drums/
              |         +---- drm1.wav
              |         +---- drm2.wav
              |         +---- drm3.wav
              |        ...
              +----- piano/
              |         +---- pia1.wav
              |         +---- pia2.wav
              |         +---- pia3.wav
              |        ...
              +----- string/
              |         +---- str1.wav
              |         +---- str2.wav
              |         +---- str3.wav
              |        ...
              +----- vocals/
                        +---- voc1.wav
                        +---- voc2.wav
                        +---- voc3.wav
                       ...

            (El nombre de los archivos .wav da igual, pero las ramas de
            directorios internos deben ser exactos.)
        """
        self.seed = seed
        self.duration = duration
        self.sr = sampling
        self.n_channels = n_channels
        self.ref_db = ref_db
        self.fg_path = fg_path
        
    def generate_random(self, n: int = 1000, jams_path: str = 'soundscapes.jams', snr: tuple = (-5, 5), pitch_shift: tuple = (-2, 2), time_stretch: tuple = (0.8, 1.2)):
        """Genera iterativamente metadatos de mixes. Por defecto, no
        exporta ninguna pista de audio para evitar sobrecargas de memoria.
        
        La función crea una carpeta temporal ("temp") en el directorio
        nativo de donde se llame al módulo. En esta se exportan los metadatos
        individuales de Scaper, dado que el módulo trabajo solo con directorios
        y por mixes individuales.
        
        Luego de la confección del archivo .jams completo, se borra "temp"
        y todos los archivos transitorios. Queda en la carpeta nativa el
        archivo .jams bajo el directorio indicado en 'jams_path'.

        Args:
            n (int, optional): Cantidad de iteraciones. Por defecto, 1000.
            jams_path (str, optional): Directorio para exportar el archivo
            .jams final. Por defecto, 'soundscapes.jams'.
            snr (tuple, optional): Signal-to-Noise Ratio en decibelios (dB).
            Altera el volumen de la pista agregada. Por defecto, entre (-5, 5).
            pitch_shift (tuple, optional): Variación de tono en la pista agregada. 
            Por defecto, entre (-2, 2).
            time_stretch (tuple, optional): Extensión de tiempo en la pista
            agregada. Por defecto, entre (0.8, 1.2).
        
        Las configuraciones de preprocesamiento especificadas se han considerado
        gracias a la investigación de "Why does music source separation benefit 
        from cacophony?" (inspiración del nombre de la clase también :p).
        
        @misc{jeon2024doesmusicsourceseparation,
            title={Why does music source separation benefit from cacophony?}, 
            author={Chang-Bin Jeon and Gordon Wichern and François G. Germain and Jonathan Le Roux},
            year={2024},
            eprint={2402.18407},
            archivePrefix={arXiv},
            primaryClass={eess.AS},
            url={https://arxiv.org/abs/2402.18407}, 
        }
        """
        
        if not os.path.exists('temp'): # Creación de la carpeta "temp/"
            print('Creada carpeta temporal para resguardar memoria.\n')
            os.makedirs('temp',exist_ok=True)
        
        
        sc = scaper.Scaper(             
                duration=self.duration,  
                fg_path=self.fg_path,
                bg_path=self.fg_path,   
                random_state=self.seed
            )
        '''
        El objeto de scaper que conjunta las herramientas de preprocesamiento.
        
        Se repite la variable de fg_path en bg_path porque no es un argumento
        relevante: no estamos agregando sonidos de fondo a nuestros mixes. No
        obstante, si no se declara, produce un error. Por ello se ha declarado
        con el comodín.
        '''
        
        sc.sr = self.sr                 # Frecuencia de muestreo. 
        sc.n_channels = self.n_channels # Mono o estéreo.
        sc.ref_db = self.ref_db         # Volumen de referencia.
        
        print('Confeccionando mixes aleatorios.')
        for i in tqdm(range(n)):
            sc.reset_fg_event_spec()    # Reiniciar eventos del confeccionador para que no se acumulen. 
            
            for label in LABELS.keys():
                sc.add_event(
                    label=('const', label),         # Selecciona un track del instrumento seleccionado.
                    source_file=('choose', []),     # Argumento para que seleccione aleatoriamente lo que haya adentro.
                    source_time=('uniform', 0, 7),  
                    event_time=('const', 0),
                    event_duration=('const', sc.duration),
                    snr=('uniform', snr[0], snr[1]),
                    pitch_shift=('uniform', pitch_shift[0], pitch_shift[1]),
                    time_stretch=('uniform', time_stretch[0], time_stretch[1])
                )
                '''
                Se establecen configuraciones estocásticas con "uniform". En el
                rango de valores dados, se escogerá una pista musical y se realizarán
                todas las manipulaciones mencionadas.
                '''
                
                sc.generate(jams_path=os.path.join('temp',f'soundscape_{i+1}.jams'),fix_clipping=True)
        
        print('\nFusión de archivos .jams')
        all_jams = jams.JAMS()
        for jams_file in tqdm([os.path.join('temp',f) for f in os.listdir('temp')]):
            jam = jams.load(jams_file,strict=False)
            
            for annot in jam.annotations:
                all_jams.annotations.append(annot) # Combina todo en un mismo archivo.
                
        all_jams.save(jams_path,strict=False)
        shutil.rmtree('temp')
        print('Creación de metadata terminada. Eliminada carpeta temporal.\n')
        
    def read_from_jams(self,jams_path: str = 'soundscapes.jams'):
        """Sea para cargar los datos que recién crees o para el archivo
        .jams compartido, con este método reconstruyes los audios en
        tensores de numpy para el procesamiento posterior.
        
        Args:
            jams_path (str, optional): Ubicación del archivo .jams.
        
        Returns
            X: Array del mix con todas las frecuencias unidas. Dimensiones:
            (n_files, marco 1D (1), frecuencias)
            Y: Array del mix con todas las frecuencias separadas. Dimensiones:
            (n_files, marco 1D (1), frecuencias, pistas separadas (6))
        """
        
        if not os.path.exists('temp'):
            print('Creada carpeta temporal para resguardar memoria.\n')
            os.makedirs('temp',exist_ok=True)
        
        mixtures = []
        stems = []
        
        print('Reconstrucción de audios.')
        for ann in tqdm(jams.load(jams_path,strict=False).annotations):
            temp = jams.JAMS()
            temp.annotations.append(ann)
            temp.file_metadata.duration = 5.0           # Sin esto, suelta error.
            temp.save(r'temp\temp.jams',strict=False)
            """De nuevo, como Scaper trabaja con directorios y no con
            arrays, es necesario exportar cada elemento del archivo .jams
            en uno individual.
            
            Por ello, se vuelve a utilizar la carpeta temporal "temp". 
            """        
            
            mix_audio, _, _, stem_list = scaper.generate_from_jams(
                jams_infile = r'temp\temp.jams',
                fg_path = 'stems',
                bg_path = 'stems'
            )
            
            mixtures.append(normalize_track(mix_audio).T)

            stem = []
            for stem_audio in stem_list:
                stem.append(normalize_track(stem_audio))
                
            stems.append(np.array(stem).T)
            os.remove(r'temp\temp.jams')    # Quita el archivo .jams

        X = np.array(mixtures)
        Y = np.array(stems)
        shutil.rmtree('temp')
        
        return X, Y
    
'''
En adelante, para cargar X e Y, se puede acudir al siguiente shortcut de copia y pega.

    mixer = cacophony()
    X, Y = mixer.read_from_jams()
'''