import numpy as np
np.float_ = np.float64
np.Inf = np.inf
import scaper
import librosa
import os
from tqdm import tqdm
import shutil
import time
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
    audio, _ = librosa.load(audio_path,sr=44100)
    
    non_silent = librosa.effects.split(audio,top_db=60)
    audio_rel = np.empty((1,))
    
    for start, end in non_silent:
        audio_rel = np.hstack((audio_rel,audio[start:end+1]))
        
    return audio_rel

def normalize_track(arr,length = 220500): 
    if arr.shape[0] < length:
        return np.hstack((arr,np.zeros((length - arr.shape[0],1),dtype='float64')))
    elif arr.shape[0] > length:
        return arr[:length]
    else:
        return arr

class cacophony:
    def __init__(self, duration: float = 5.0, sampling: int = 44100, n_channels: int = 1, ref_db: int = -20, fg_path: str = 'stems', seed: int = 42):
        self.seed = seed
        self.duration = duration
        self.sr = sampling
        self.n_channels = n_channels
        self.ref_db = ref_db
        self.fg_path = fg_path
        
    def generate_random(self, n: int = 100, jams_path: str = 'soundscapes.jams', snr: tuple = (-5, 5), pitch_shift: tuple = (-2, 2), time_stretch: tuple = (0.8, 1.2)):
        if not os.path.exists('temp'):
            print('Creada carpeta temporal para resguardar memoria.\n')
            os.makedirs('temp',exist_ok=True)
        
        sc = scaper.Scaper(
                duration=self.duration,  # Duration of each soundscape
                fg_path=self.fg_path,
                bg_path=self.fg_path,
                random_state=self.seed
            )
        
        sc.sr = self.sr
        sc.n_channels = self.n_channels
        sc.ref_db = self.ref_db
        
        print('Confeccionando mixes aleatorios.')
        for i in tqdm(range(n)):
            sc.reset_fg_event_spec()
            
            for label in LABELS.keys():
                sc.add_event(
                    label=('const', label),
                    source_file=('choose', []),  # Scaper will choose randomly
                    source_time=('uniform', 0, 7),
                    event_time=('const', 0),
                    event_duration=('const', sc.duration),
                    snr=('uniform', snr[0], snr[1]),
                    pitch_shift=('uniform', pitch_shift[0], pitch_shift[1]),
                    time_stretch=('uniform', time_stretch[0], time_stretch[1])
                )
                
                sc.generate(jams_path=os.path.join('temp',f'soundscape_{i+1}.jams'),fix_clipping=True)
        
        print('\nFusión de archivos .jams')
        all_jams = jams.JAMS()
        for jams_file in tqdm([os.path.join('temp',f) for f in os.listdir('temp')]):
            jam = jams.load(jams_file,strict=False)
            
            for annot in jam.annotations:
                all_jams.annotations.append(annot)
                
        all_jams.save(jams_path,strict=False)
        shutil.rmtree('temp')
        print('Creación de metadata terminada. Eliminada carpeta temporal.\n')
        
    def read_from_jams(self,jams_path: str = 'soundscapes.jams'):
        if not os.path.exists('temp'):
            print('Creada carpeta temporal para resguardar memoria.\n')
            os.makedirs('temp',exist_ok=True)
        
        mixtures = []
        stems = []
        
        print('Reconstrucción de audios.')
        for ann in tqdm(jams.load(jams_path,strict=False).annotations):
            temp = jams.JAMS()
            temp.annotations.append(ann)
            temp.file_metadata.duration = 5.0
            temp.save(r'temp\temp.jams',strict=False)

            mix_audio, _, _, stem_list = scaper.generate_from_jams(
                jams_infile = r'temp\temp.jams',
                fg_path = 'stems',
                bg_path = 'stems'
            )
            
            mixtures.append(normalize_track(mix_audio))

            stem = []
            for stem_audio in stem_list:
                stem.append(normalize_track(stem_audio))
                
            stems.append(np.array(stem).T)
            
            os.remove(r'temp\temp.jams')

        X = np.array(mixtures)
        Y = np.array(stems)
        shutil.rmtree('temp')
        
        return X, Y