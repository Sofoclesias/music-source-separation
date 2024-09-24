import numpy as np
import pandas as pd
import nussl
from tqdm import tqdm
import scaper
from pydub import AudioSegment

def load_stems():
    '''
    Asumiendo que se tenga stems/ en el directorio, carga todos los stems.
    PC de Bisetti: al stem n° 273 ocurre error de memoria.
    '''
    
    metadata = pd.read_json('stems/metadata.json')
    stem_files = []
    for diris in tqdm(metadata['newdir']):
        stem_files.append(nussl.AudioSignal(diris))

def wav_to_flac(old_dir,new_dir):
    '''
    Dado un directorio de inicio y uno de destino, carga el archivo .flac y lo exporta en .wav.
    Normaliza frecuencias al pasarla a .wav descomprimido.
    '''
    
    flac = AudioSegment.from_file(old_dir,format="flac")
    flac.export(new_dir, format='wav')

def random_mix(N):
    '''
    Una sola función de data augmentation y random mixing.
    '''
    
    sc = scaper.Scaper(duration=5.0,fg_path='stem'random_state=42)

    sc.sr = 44100       # Frecuencia del audio final
    sc.n_channels = 1   # Canales mono
    sc.ref_db = -20     # Volumen de referencia para el mix
    
    for label in ['vocal','drum','bass','piano','string','accoustic']:
        sc.add_event(
            label = ('const',label),
            source_file = ('choose',[]),
            source_time = ('uniform', 0, 7),
            event_time = ('const', sc.duration),
            snr = ('uniform', -5, 5),           # Volumen de stems en +-10
            pitch_shift = ('uniform',-2,2),     # Cambios de pitch
            time_stretch = ('uniform',0.8,1.2)  # Cambios de velocidad
        )

    for i in range(N):
        mix_audio, _, _, stem_audios = sc.generate(fix_clipping=True)

        ## Falta exportar (problema de memoria)

if __name__=='__main__':
    pass