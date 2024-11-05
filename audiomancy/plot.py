"""
Módulo para gráficos y tal.
"""
import matplotlib.pyplot as plt
import matplotlib
import librosa.display
import torch
import numpy as np
from copy import deepcopy
import importlib_resources as pkg_resources
import random, string
from tempfile import NamedTemporaryFile
import os
from contextlib import contextmanager
from . import common

multitrack_template = pkg_resources.read_text(common, 'multitrack.html')

def _check_imports():
    try:
        import ffmpy
    except:
        ffmpy = False

    try:
        import IPython
    except:
        raise ImportError('IPython must be installed in order to use this function!')
    return ffmpy, IPython

@contextmanager
def _close_temp_files(tmpfiles):
    """
    Utility function for creating a context and closing all temporary files
    once the context is exited. For correct functionality, all temporary file
    handles created inside the context must be appended to the ```tmpfiles```
    list.

    This function is taken wholesale from Scaper.

    Args:
        tmpfiles (list): List of temporary file handles
    """
    def _close():
        for t in tmpfiles:
            try:
                t.close()
                os.unlink(t.name)
            except:
                pass
    try:
        yield
    except:
        _close()
        raise
    _close()

def visualize_waveform(audio_signal, ch=0, do_mono=False, x_axis='time', **kwargs):
    """
    Wrapper around `librosa.display.waveplot` for usage with AudioSignals.
    
    Args:
        audio_signal (AudioSignal): AudioSignal to plot
        ch (int, optional): Which channel to plot. Defaults to 0.
        do_mono (bool, optional): Make the AudioSignal mono. Defaults to False.
        x_axis (str, optional): x_axis argument to librosa.display.waveplot. Defaults to 'time'.
        kwargs: Additional keyword arguments to librosa.display.waveplot.
    """

    if do_mono:
        audio_signal = librosa.to_mono(audio_signal)
        
    data = np.asfortranarray(audio_signal)
    
    x_coords = librosa.display.__mesh_coords(x_axis, None, data.shape[0],
    sr=44100, hop_length=4096//4)
    extent = [x_coords.min(), x_coords.max()//1000]

    librosa.display.waveshow(data, sr=44100, axis=x_axis, **kwargs)
    plt.ylabel('Amplitude')
    plt.xlim(extent)

def visualize_sources_as_waveform(audio_signals, ch=0, do_mono=False, x_axis='time', 
                                  colors=None, alphas=None, show_legend=True, **kwargs):
    """
    Visualizes a dictionary or list of sources with overlapping waveforms with transparency.
    
    The labels of each source are either the key, if a dictionary, or the 
    path to the input audio file, if a list.
    
    Args:
        audio_signals: (sr,ch,wv)
        ch (int, optional): Which channel to plot. Defaults to 0.
        do_mono (bool, optional): Make each AudioSignal mono. Defaults to False.
        x_axis (str, optional): x_axis argument to librosa.display.waveplot. Defaults to 'time'.
        colors (list, optional): Sequence of colors to use for each signal. 
          Defaults to None, which uses the default matplotlib color cycle.
        alphas (list, optional): Sequence of alpha transparency to use for each signal. 
          Defaults to None.
        kwargs: Additional keyword arguments to librosa.display.waveplot.
    """
    sorted_keys = sorted(
        audio_signals.keys(),
        key=lambda k: librosa.feature.rms(y=audio_signals[k]).mean(),
        reverse=True
    )

    alphas = (
        np.linspace(0.25, .75, len(audio_signals)) 
        if alphas is None else alphas
    )
    colors = (
        plt.rcParams['axes.prop_cycle'].by_key()['color'] 
        if colors is None else colors
    )

    for i, key in enumerate(sorted_keys):
        val = audio_signals[key]
        color = colors[i % len(audio_signals)]
        visualize_waveform(val, ch=ch, do_mono=do_mono, x_axis=x_axis, 
                           alpha=alphas[i % len(audio_signals)],
                           label=key, color=color)

    if show_legend:
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2)

def visualize_sources_as_masks(audio_signals, ch=0, do_mono=False, x_axis='time', 
                               y_axis='linear', db_cutoff=-60, colors=None, alphas=None, 
                               alpha_amount=1.0, nfft=4096, **kwargs):
    """
    Visualizes a dictionary or list of sources with overlapping waveforms with transparency.
    
    The labels of each source are either the key, if a dictionary, or the 
    path to the input audio file, if a list.
    
    Args:
        audio_signals (list or dict): List or dictionary of audio signal objects to be
          plotted.
        ch (int, optional): Which channel to plot. Defaults to 0.
        do_mono (bool, optional): Make each AudioSignal mono. Defaults to False.
        x_axis (str, optional): x_axis argument to librosa.display.waveplot. Defaults to 'time'.
        colors (list, optional): Sequence of colors to use for each signal. 
          Defaults to None, which uses the default matplotlib color cycle.
        alphas (list, optional): Sequence of alpha transparency to use for each signal. 
          Defaults to None.
        kwargs: Additional keyword arguments to librosa.display.specshow.
    """
    import matplotlib.pyplot as plt
    
    if do_mono:
        for key in audio_signals:
            audio_signals[key] = librosa.to_mono(audio_signals[key])

    sorted_keys = sorted(
        audio_signals.keys(),
        key=lambda k: librosa.feature.rms(y=audio_signals[k]).mean(),
        reverse=True
    )

    source_names = sorted(list(audio_signals.keys()))
    mix = np.vstack([src for src in audio_signals.values()]).sum(axis=0)  # sum(audio_signals.values())
    data = {
        'mix': mix,
        'sources': audio_signals
    }
    
    from .audioprocessing import PhaseSensitiveSpectrumApproximation
    data = PhaseSensitiveSpectrumApproximation()(data)
    
    alphas = (
        np.linspace(0.25, .75, len(audio_signals)) 
        if alphas is None else alphas
    )
    colors = (
        plt.rcParams['axes.prop_cycle'].by_key()['color'] 
        if colors is None else colors
    )
    

    # construct each image with alpha values
    masks = data['source_magnitudes'] / (np.maximum(
            data['mix_magnitude'][..., None], data['source_magnitudes']) 
            + 1e-16
        )
    legend_elements = []

    silence_mask = librosa.amplitude_to_db(np.abs(torch.from_numpy(mix).stft(nfft,nfft//4,2048,return_complex=True)),ref=np.max) > db_cutoff
    masks *= silence_mask[..., None]

    y_coords = librosa.display.__mesh_coords(y_axis, None, masks.shape[0], 
        sr=44100, hop_length=nfft//4)
    x_coords = librosa.display.__mesh_coords(x_axis, None, masks.shape[1],
        sr=44100, hop_length=nfft//4)

    extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]

    for j, key in enumerate(sorted_keys):
        i = source_names.index(key)
        
        mask = masks[..., i]
        color = colors[j % len(colors)]

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            'custom', ['white', color])
        image = cmap(mask)
        image[:, :,-1] = mask ** alpha_amount
        plt.imshow(image, origin='lower', aspect='auto', 
            interpolation='none', extent=extent, alpha=alphas[i % len(audio_signals)])

        legend_elements.append(
            matplotlib.patches.Patch(facecolor=color, label=key))

    axes = librosa.display.__check_axes(None)

    axes.set_xlim(x_coords.min(), x_coords.max())
    axes.set_ylim(y_coords.min(), y_coords.max())

    # Set up axis scaling
    librosa.display.__scale_axes(axes, x_axis, 'x',None,None)
    librosa.display.__scale_axes(axes, y_axis, 'y',None,None)

    # Construct tickers and locators
    librosa.display.__decorate_axis(axes.xaxis, x_axis)
    librosa.display.__decorate_axis(axes.yaxis, y_axis)

def multitrack(audio_signals, names=None, ext='.mp3', display=False):
    """
    Takes a bunch of audio sources, converts them to mp3 to make them smaller, and
    creates a multitrack audio player in the notebook that lets you
    toggle between the sources and the mixture. Heavily adapted
    from https://github.com/binarymind/multitrackHTMLPlayer,
    designed by Bastien Liutkus.

    Args:
        audio_signals (list): List of AudioSignal objects that add up to the mixture.
        names (list): List of names to give to each object (e.g. foreground, background).
        ext (str): What extension to use when embedding. '.mp3' is more lightweight
          leading to smaller notebook sizes.
        display (bool): Whether or not to display the object immediately, or to return
          the html object for display later by the end user.
    """
    ffmpy, IPython = _check_imports()
    div_id = ''.join(random.choice(string.ascii_uppercase) for _ in range(20))
    _names = None

    if isinstance(audio_signals, dict):
        _names = list(audio_signals.keys())
        audio_signals = [audio_signals[k] for k in _names]

    if names is not None:
        if len(names) != len(audio_signals):
            raise ValueError("len(names) must be equal to len(audio_signals)!")
    else:
        if _names is not None:
            names = _names

    template = (
        f"<div id={div_id} class=audio-container "
        f"preload=auto name={div_id}>")

    for name, signal in zip(names, audio_signals):
        encoded_audio = embed_audio(signal, ext=ext, display=False).src_attr()
        audio_element = (
            f"<audio name='{name}' url={encoded_audio}></audio>")
        template += audio_element

    template += "</div>"
    template += multitrack_template
    template = template.replace('NAME', div_id)

    return template

def embed_audio(audio_signal, ext='.mp3', display=False):
    """
    Write a numpy array to a temporary mp3 file using ffmpy, then embeds the mp3
    into the notebook.

    Args:
        audio_signal (AudioSignal): AudioSignal object containing the data.
        ext (str): What extension to use when embedding. '.mp3' is more lightweight 
          leading to smaller notebook sizes. Defaults to '.mp3'.
        display (bool): Whether or not to display the object immediately, or to return
          the html object for display later by the end user. Defaults to True.

    Example:
        >>> import nussl
        >>> audio_file = nussl.efz_utils.download_audio_file('schoolboy_fascination_excerpt.wav')
        >>> audio_signal = nussl.AudioSignal(audio_file)
        >>> audio_signal.embed_audio()

    This will show a little audio player where you can play the audio inline in 
    the notebook.      
    """
    ext = f'.{ext}' if not ext.startswith('.') else ext
    audio_signal = deepcopy(audio_signal)
    ffmpy, IPython = _check_imports()
    sr = 44100
    tmpfiles = []

    with _close_temp_files(tmpfiles):
        tmp_wav = NamedTemporaryFile(
            mode='w+', suffix='.wav', delete=False)
        tmpfiles.append(tmp_wav)
        
        from .audioprocessing import save_audio
        save_audio(torch.from_numpy(audio_signal[None]),tmp_wav.name,44100)
        if ext != '.wav' and ffmpy:
            tmp_converted = NamedTemporaryFile(
                mode='w+', suffix=ext, delete=False)
            tmpfiles.append(tmp_wav)
            ff = ffmpy.FFmpeg(
                inputs={tmp_wav.name: None},
                outputs={tmp_converted.name: '-write_xing 0 -codec:a libmp3lame -b:a 128k -y'})
            ff.run()
        else:
            tmp_converted = tmp_wav

        audio_element = IPython.display.Audio(data=tmp_converted.name, rate=sr)
    return audio_element

def show_sources(tensor,sourlist):
    sources = {s: tensor[i][0] for i, s in enumerate(sourlist)}
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(211)
    visualize_sources_as_waveform(sources)
    plt.subplot(212)
    visualize_sources_as_masks(sources, db_cutoff=-80)
    plt.tight_layout()
    
    _sources = {k: v * 1 / len(sources) for k, v in sources.items()}
    return fig ,multitrack(_sources, ext='.wav')