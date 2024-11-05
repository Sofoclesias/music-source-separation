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
import librosa
import os
from tqdm import tqdm
import shutil
import jams
import torch
import typing as tp
import subprocess as sp
import torchaudio as ta
from pathlib import Path
import lameenc
from einops import rearrange
import julius
from .architecture.utils import temp_filenames

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
    def __init__(self, duration: float = 5.0, sampling: int = 44100, n_channels: int = 1, ref_db: int = -20, fg_path: str | None = None, seed: int = 42):
        import scaper
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
        
        if fg_path is None:
            from .constants import STEMS_PATH
            self.fg_path = STEMS_PATH
        else:
            pass
        
    def generate_random(self, n: int = 1000, jams_path: str | None = None, snr: tuple = (-5, 5), pitch_shift: tuple = (-2, 2), time_stretch: tuple = (0.8, 1.2)):
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
        
        from .constants import ABSOLUTE_PATH, TEMP_PATH, LABELS
        
        if jams_path is None:
            jams_path = os.path.join(ABSOLUTE_PATH,'common',f'{n}_soundscapes.jams')
        else:
            pass
        
        if not os.path.exists(TEMP_PATH): # Creación de la carpeta "temp/"
            print('Creada carpeta temporal para resguardar memoria.\n')
            os.makedirs(TEMP_PATH,exist_ok=True)
        import scaper
        
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
                
                sc.generate(jams_path=os.path.join(TEMP_PATH,f'soundscape_{i+1}.jams'),fix_clipping=True)
        
        print('\nFusión de archivos .jams')
        all_jams = jams.JAMS()
        for jams_file in tqdm([os.path.join(TEMP_PATH,f) for f in os.listdir(TEMP_PATH)]):
            jam = jams.load(jams_file,strict=False)
            
            for annot in jam.annotations:
                all_jams.annotations.append(annot) # Combina todo en un mismo archivo.
                
        all_jams.save(jams_path,strict=False)
        shutil.rmtree(TEMP_PATH)
        print('Creación de metadata terminada. Eliminada carpeta temporal.\n')
        
    def read_from_jams(self,jams_path: str | None = None):
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
        
        from .constants import STEMS_PATH, TEMP_PATH, JAMS_FILE_200
        
        if jams_path is None:
            jams_path = JAMS_FILE_200
        else:
            pass
        
        if not os.path.exists(TEMP_PATH): # Creación de la carpeta "temp/"
            print('Creada carpeta temporal para resguardar memoria.\n')
            os.makedirs(TEMP_PATH,exist_ok=True)
        
        mixtures = []
        stems = []
        import scaper
        print('Reconstrucción de audios.')
        for ann in tqdm(jams.load(jams_path,strict=False).annotations):
            temp = jams.JAMS()
            temp.annotations.append(ann)
            temp.file_metadata.duration = 5.0           # Sin esto, suelta error.
            temp.save(os.path.join(TEMP_PATH,'temp.jams'),strict=False)
            """De nuevo, como Scaper trabaja con directorios y no con
            arrays, es necesario exportar cada elemento del archivo .jams
            en uno individual.
            
            Por ello, se vuelve a utilizar la carpeta temporal "temp". 
            """        
            
            mix_audio, _, _, stem_list = scaper.generate_from_jams(
                jams_infile = os.path.join(TEMP_PATH,'temp.jams'),
                fg_path = STEMS_PATH,
                bg_path = STEMS_PATH
            )
            
            mixtures.append(normalize_track(mix_audio).T)

            stem = []
            for stem_audio in stem_list:
                stem.append(normalize_track(stem_audio))
                
            stems.append(np.array(stem).T)
            os.remove(os.path.join(TEMP_PATH,'temp.jams'))    # Quita el archivo .jams

        X = np.array(mixtures)
        Y = rearrange(np.array(stems), 'b c l s -> b s c l')
        shutil.rmtree(TEMP_PATH)
        
        return X, Y
    
class AudioFile:
    """
    Allows to read audio from any format supported by ffmpeg, as well as resampling or
    converting to mono on the fly. See :method:`read` for more details.
    """
    def __init__(self, path: Path):
        self.path = Path(path)
        self._info = None

    def __repr__(self):
        features = [("path", self.path)]
        features.append(("samplerate", self.samplerate()))
        features.append(("channels", self.channels()))
        features.append(("streams", len(self)))
        features_str = ", ".join(f"{name}={value}" for name, value in features)
        return f"AudioFile({features_str})"

    @property
    def duration(self):
        return float(self.info['format']['duration'])

    @property
    def _audio_streams(self):
        return [
            index for index, stream in enumerate(self.info["streams"])
            if stream["codec_type"] == "audio"
        ]

    def __len__(self):
        return len(self._audio_streams)

    def channels(self, stream=0):
        return int(self.info['streams'][self._audio_streams[stream]]['channels'])

    def samplerate(self, stream=0):
        return int(self.info['streams'][self._audio_streams[stream]]['sample_rate'])

    def read(self,
             seek_time=None,
             duration=None,
             streams=slice(None),
             samplerate=None,
             channels=None):
        """
        Slightly more efficient implementation than stempeg,
        in particular, this will extract all stems at once
        rather than having to loop over one file multiple times
        for each stream.

        Args:
            seek_time (float):  seek time in seconds or None if no seeking is needed.
            duration (float): duration in seconds to extract or None to extract until the end.
            streams (slice, int or list): streams to extract, can be a single int, a list or
                a slice. If it is a slice or list, the output will be of size [S, C, T]
                with S the number of streams, C the number of channels and T the number of samples.
                If it is an int, the output will be [C, T].
            samplerate (int): if provided, will resample on the fly. If None, no resampling will
                be done. Original sampling rate can be obtained with :method:`samplerate`.
            channels (int): if 1, will convert to mono. We do not rely on ffmpeg for that
                as ffmpeg automatically scale by +3dB to conserve volume when playing on speakers.
                See https://sound.stackexchange.com/a/42710.
                Our definition of mono is simply the average of the two channels. Any other
                value will be ignored.
        """
        streams = np.array(range(len(self)))[streams]
        single = not isinstance(streams, np.ndarray)
        if single:
            streams = [streams]

        if duration is None:
            target_size = None
            query_duration = None
        else:
            target_size = int((samplerate or self.samplerate()) * duration)
            query_duration = float((target_size + 1) / (samplerate or self.samplerate()))

        with temp_filenames(len(streams)) as filenames:
            command = ['ffmpeg', '-y']
            command += ['-loglevel', 'panic']
            if seek_time:
                command += ['-ss', str(seek_time)]
            command += ['-i', str(self.path)]
            for stream, filename in zip(streams, filenames):
                command += ['-map', f'0:{self._audio_streams[stream]}']
                if query_duration is not None:
                    command += ['-t', str(query_duration)]
                command += ['-threads', '1']
                command += ['-f', 'f32le']
                if samplerate is not None:
                    command += ['-ar', str(samplerate)]
                command += [filename]

            sp.run(command, check=True)
            wavs = []
            for filename in filenames:
                wav = np.fromfile(filename, dtype=np.float32)
                wav = torch.from_numpy(wav)
                wav = wav.view(-1, self.channels()).t()
                if channels is not None:
                    wav = convert_audio_channels(wav, channels)
                if target_size is not None:
                    wav = wav[..., :target_size]
                wavs.append(wav)
        wav = torch.stack(wavs, dim=0)
        if single:
            wav = wav[0]
        return wav


def spectro(x, n_fft=512, hop_length=None, pad=0):
    *other, length = x.shape
    x = x.reshape(-1, length)
    is_mps = x.device.type == 'mps'
    if is_mps:
        x = x.cpu()
    z = torch.stft(x,
                n_fft * (1 + pad),
                hop_length or n_fft // 4,
                window=torch.hann_window(n_fft).to(x),
                win_length=n_fft,
                normalized=True,
                center=True,
                return_complex=True,
                pad_mode='reflect')
    _, freqs, frame = z.shape
    return z.view(*other, freqs, frame)

def ispectro(z, hop_length=None, length=None, pad=0):
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    win_length = n_fft // (1 + pad)
    is_mps = z.device.type == 'mps'
    if is_mps:
        z = z.cpu()
    x = torch.istft(z,
                 n_fft,
                 hop_length,
                 window=torch.hann_window(win_length).to(z.real),
                 win_length=win_length,
                 normalized=True,
                 length=length,
                 center=True)
    _, length = x.shape
    return x.view(*other, length)

def prevent_clip(wav, mode='rescale'):
    """
    different strategies for avoiding raw clipping.
    """
    if mode is None or mode == 'none':
        return wav
    assert wav.dtype.is_floating_point, "too late for clipping"
    if mode == 'rescale':
        wav = wav / max(1.01 * wav.abs().max(), 1)
    elif mode == 'clamp':
        wav = wav.clamp(-0.99, 0.99)
    elif mode == 'tanh':
        wav = torch.tanh(wav)
    else:
        raise ValueError(f"Invalid mode {mode}")
    return wav

def convert_audio_channels(wav, channels=2):
    """Convert audio to the given number of channels."""
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1:
        # The caller asked 1-channel audio, but the stream have multiple
        # channels, downmix all channels.
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file have
        # one single channel, replicate the audio over all channels.
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file have
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav

def i16_pcm(wav):
    """Convert audio to 16 bits integer PCM format."""
    if wav.dtype.is_floating_point:
        return (wav.clamp_(-1, 1) * (2**15 - 1)).short()
    else:
        return wav

def encode_mp3(wav, path, samplerate=44100, bitrate=320, quality=2, verbose=False):
    """Save given audio as mp3. This should work on all OSes."""
    C, T = wav.shape
    wav = i16_pcm(wav)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(samplerate)
    encoder.set_channels(C)
    encoder.set_quality(quality)  # 2-highest, 7-fastest
    if not verbose:
        encoder.silence()
    wav = wav.data.cpu()
    wav = wav.transpose(0, 1).numpy()
    mp3_data = encoder.encode(wav.tobytes())
    mp3_data += encoder.flush()
    with open(path, "wb") as f:
        f.write(mp3_data)

def save_audio(wav: torch.Tensor,
               path: tp.Union[str, Path],
               samplerate: int,
               bitrate: int = 320,
               clip: tp.Literal["rescale", "clamp", "tanh", "none"] = 'rescale',
               bits_per_sample: tp.Literal[16, 24, 32] = 16,
               as_float: bool = False,
               preset: tp.Literal[2, 3, 4, 5, 6, 7] = 2):
    """Save audio file, automatically preventing clipping if necessary
    based on the given `clip` strategy. If the path ends in `.mp3`, this
    will save as mp3 with the given `bitrate`. Use `preset` to set mp3 quality:
    2 for highest quality, 7 for fastest speed
    """
    wav = prevent_clip(wav, mode=clip)
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".mp3":
        encode_mp3(wav, path, samplerate, bitrate, preset, verbose=True)
    elif suffix == ".wav":
        if as_float:
            bits_per_sample = 32
            encoding = 'PCM_F'
        else:
            encoding = 'PCM_S'
        ta.save(str(path), wav, sample_rate=samplerate,
                encoding=encoding, bits_per_sample=bits_per_sample)
    elif suffix == ".flac":
        ta.save(str(path), wav, sample_rate=samplerate, bits_per_sample=bits_per_sample)
    else:
        raise ValueError(f"Invalid suffix for path: {suffix}")
    
def convert_audio(wav, from_samplerate, to_samplerate, channels) -> torch.Tensor:
    """Convert audio from a given samplerate to a target one and target number of channels."""
    wav = convert_audio_channels(wav, channels)
    return julius.resample_frac(wav, from_samplerate, to_samplerate)

def compute_ideal_binary_mask(source_magnitudes):
    ibm = (
            source_magnitudes == np.max(source_magnitudes, axis=-1, keepdims=True)
    ).astype(float)

    ibm = ibm / np.sum(ibm, axis=-1, keepdims=True)
    ibm[ibm <= .5] = 0
    return ibm

from collections import OrderedDict
class PhaseSensitiveSpectrumApproximation(object):
    """
    Takes a dictionary and looks for two special keys, defined by the
    arguments ``mix_key`` and ``source_key``. These default to `mix` and `sources`.
    These values of these keys are used to calculate the phase sensitive spectrum 
    approximation [1]. The input dictionary is modified to have additional
    keys:

    - mix_magnitude: The magnitude spectrogram of the mixture audio signal.
    - source_magnitudes: The magnitude spectrograms of each source spectrogram.
    - assignments: The ideal binary assignments for each time-frequency bin.

    ``data[self.source_key]`` points to a dictionary containing the source names in
    the keys and the corresponding AudioSignal in the values. The keys are sorted
    in alphabetical order and then appended to the mask. ``data[self.source_key]``
    then points to an OrderedDict instead, where the keys are in the same order
    as in ``data['source_magnitudes']`` and ``data['assignments']``.

    This transform uses the STFTParams that are attached to the AudioSignal objects
    contained in ``data[mix_key]`` and ``data[source_key]``.

    [1] Erdogan, Hakan, John R. Hershey, Shinji Watanabe, and Jonathan Le Roux. 
        "Phase-sensitive and recognition-boosted speech separation using 
        deep recurrent neural networks." In 2015 IEEE International Conference 
        on Acoustics, Speech and Signal Processing (ICASSP), pp. 708-712. IEEE, 
        2015.
    
    Args:
        mix_key (str, optional): The key to look for in data for the mixture AudioSignal. 
          Defaults to 'mix'.
        source_key (str, optional): The key to look for in the data containing the list of
          source AudioSignals. Defaults to 'sources'.
        range_min (float, optional): The lower end to use when truncating the source 
          magnitudes in the phase sensitive spectrum approximation. Defaults to 0.0 (construct
          non-negative masks). Use -np.inf for untruncated source magnitudes.
        range_max (float, optional): The higher end of the truncated spectrum. This gets
          multiplied by the magnitude of the mixture. Use 1.0 to truncate the source 
          magnitudes to `max(source_magnitudes, mix_magnitude)`. Use np.inf for untruncated
          source magnitudes (best performance for an oracle mask but may be beyond what a
          neural network is capable of masking). Defaults to 1.0.
          
    Raises:
            TransformException: if the expected keys are not in the dictionary, an
              Exception is raised.
        
    Returns:
        data: Modified version of the input dictionary.
    """

    def __init__(self, mix_key='mix', source_key='sources',
                 range_min=0.0, range_max=1.0,nfft=4096):
        self.mix_key = mix_key
        self.source_key = source_key
        self.range_min = range_min
        self.range_max = range_max
        self.nfft = nfft

    def __call__(self, data):
        mixture = data[self.mix_key]

        mix_stft = torch.from_numpy(mixture).stft(self.nfft,self.nfft//4,2048,return_complex=True)
        mix_magnitude = np.abs(mix_stft)
        mix_angle = np.angle(mix_stft)
        data['mix_magnitude'] = mix_magnitude

        if self.source_key not in data:
            return data

        _sources = data[self.source_key]
        source_names = sorted(list(_sources.keys()))

        sources = OrderedDict()
        for key in source_names:
            sources[key] = _sources[key]
        data[self.source_key] = sources

        source_angles = []
        source_magnitudes = []
        for key in source_names:
            s = sources[key]
            _stft = torch.from_numpy(s).stft(self.nfft,self.nfft//4,2048,return_complex=True)
            source_magnitudes.append(np.abs(_stft))
            source_angles.append(np.angle(_stft))

        source_magnitudes = np.stack(source_magnitudes, axis=-1)
        source_angles = np.stack(source_angles, axis=-1)
        range_min = self.range_min
        range_max = self.range_max * mix_magnitude[..., None]

        # Section 3.1: https://arxiv.org/pdf/1909.08494.pdf
        source_magnitudes = np.minimum(
            np.maximum(
                source_magnitudes * np.cos(source_angles - mix_angle[..., None]),
                range_min
            ),
            range_max
        )

        data['ideal_binary_mask'] = compute_ideal_binary_mask(source_magnitudes.numpy())
        data['source_magnitudes'] = source_magnitudes

        return data

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"mix_key = {self.mix_key}, "
            f"source_key = {self.source_key}, "
            f"range_min = {self.range_min}, "
            f"range_max = {self.range_max}"
            f")"
        )