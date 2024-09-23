import numpy as np
import pandas as pd
from multiprocessing import Pool
import nussl
import soundfile as sf
from tqdm import tqdm

def load_stems():
    '''
    Asumiendo que se tenga stems/ en el directorio, carga todos los stems.
    '''
    
    metadata = pd.read_json('stems/metadata.json')
    stem_files = []
    for diris in tqdm(metadata['newdir']):
        stem_files.append(nussl.AudioSignal(diris))

if __name__=='__main__':
    pass