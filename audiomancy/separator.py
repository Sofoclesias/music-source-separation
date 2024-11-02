import subprocess

import argparse
import torch as th
import torchaudio as ta

from pathlib import Path
from typing import Optional, Callable, Dict, Tuple, Union

from .architecture.apply import apply_model, _replace_dict
from .audioprocessing import AudioFile, convert_audio, save_audio

def get_parser():
    parser = argparse.ArgumentParser("demucs.separate",
                                     description="Separate the sources for the given tracks")
    parser.add_argument("tracks", nargs='*', type=Path, default=[], help='Path to tracks')
    parser.add_argument("--list-models", action="store_true", help="List available models "
                        "from current repo and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-o",
                        "--out",
                        type=Path,
                        default=Path("separated"),
                        help="Folder where to put extracted tracks. A subfolder "
                        "with the model name will be created.")
    parser.add_argument("--filename",
                        default="{track}/{stem}.{ext}",
                        help="Set the name of output file. \n"
                        'Use "{track}", "{trackext}", "{stem}", "{ext}" to use '
                        "variables of track name without extension, track extension, "
                        "stem name and default output file extension. \n"
                        'Default is "{track}/{stem}.{ext}".')
    parser.add_argument("-d",
                        "--device",
                        default="cuda" if th.cuda.is_available() else "cpu",
                        help="Device to use, default is cuda if available else cpu")
    parser.add_argument("--shifts",
                        default=1,
                        type=int,
                        help="Number of random shifts for equivariant stabilization."
                        "Increase separation time but improves quality for Demucs. 10 was used "
                        "in the original paper.")
    parser.add_argument("--overlap",
                        default=0.25,
                        type=float,
                        help="Overlap between the splits.")
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument("--no-split",
                             action="store_false",
                             dest="split",
                             default=True,
                             help="Doesn't split audio in chunks. "
                             "This can use large amounts of memory.")
    split_group.add_argument("--segment", type=int,
                             help="Set split size of each chunk. "
                             "This can help save memory of graphic card. ")
    parser.add_argument("--two-stems",
                        dest="stem", metavar="STEM",
                        help="Only separate audio into {STEM} and no_{STEM}. ")
    parser.add_argument("--other-method", dest="other_method", choices=["none", "add", "minus"],
                        default="add", help='Decide how to get "no_{STEM}". "none" will not save '
                        '"no_{STEM}". "add" will add all the other stems. "minus" will use the '
                        "original track minus the selected stem.")
    depth_group = parser.add_mutually_exclusive_group()
    depth_group.add_argument("--int24", action="store_true",
                             help="Save wav output as 24 bits wav.")
    depth_group.add_argument("--float32", action="store_true",
                             help="Save wav output as float32 (2x bigger).")
    parser.add_argument("--clip-mode", default="rescale", choices=["rescale", "clamp", "none"],
                        help="Strategy for avoiding clipping: rescaling entire signal "
                             "if necessary  (rescale) or hard clipping (clamp).")
    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument("--flac", action="store_true",
                              help="Convert the output wavs to flac.")
    format_group.add_argument("--mp3", action="store_true",
                              help="Convert the output wavs to mp3.")
    parser.add_argument("--mp3-bitrate",
                        default=320,
                        type=int,
                        help="Bitrate of converted mp3.")
    parser.add_argument("--mp3-preset", choices=range(2, 8), type=int, default=2,
                        help="Encoder preset of MP3, 2 for highest quality, 7 for "
                        "fastest speed. Default is 2")
    parser.add_argument("-j", "--jobs",
                        default=0,
                        type=int,
                        help="Number of jobs. This can increase memory usage but will "
                             "be much faster when multiple cores are available.")
    return parser

class Separator:
    def __init__(
        self,
        model: str = "htdemucs",
        repo: Optional[Path] = None,
        device: str = "cuda" if th.cuda.is_available() else "cpu",
        shifts: int = 1,
        overlap: float = 0.25,
        split: bool = True,
        segment: Optional[int] = None,
        jobs: int = 0,
        progress: bool = False,
        callback: Optional[Callable[[dict], None]] = None,
        callback_arg: Optional[dict] = None,
    ):
        """
        `class Separator`
        =================

        Parameters
        ----------
        model: Pretrained model name or signature. Default is htdemucs.
        repo: Folder containing all pre-trained models for use.
        segment: Length (in seconds) of each segment (only available if `split` is `True`). If \
            not specified, will use the command line option.
        shifts: If > 0, will shift in time `wav` by a random amount between 0 and 0.5 sec and \
            apply the oppositve shift to the output. This is repeated `shifts` time and all \
            predictions are averaged. This effectively makes the model time equivariant and \
            improves SDR by up to 0.2 points. If not specified, will use the command line option.
        split: If True, the input will be broken down into small chunks (length set by `segment`) \
            and predictions will be performed individually on each and concatenated. Useful for \
            model with large memory footprint like Tasnet. If not specified, will use the command \
            line option.
        overlap: The overlap between the splits. If not specified, will use the command line \
            option.
        device (torch.device, str, or None): If provided, device on which to execute the \
            computation, otherwise `wav.device` is assumed. When `device` is different from \
            `wav.device`, only local computations will be on `device`, while the entire tracks \
            will be stored on `wav.device`. If not specified, will use the command line option.
        jobs: Number of jobs. This can increase memory usage but will be much faster when \
            multiple cores are available. If not specified, will use the command line option.
        callback: A function will be called when the separation of a chunk starts or finished. \
            The argument passed to the function will be a dict. For more information, please see \
            the Callback section.
        callback_arg: A dict containing private parameters to be passed to callback function. For \
            more information, please see the Callback section.
        progress: If true, show a progress bar.

        Callback
        --------
        The function will be called with only one positional parameter whose type is `dict`. The
        `callback_arg` will be combined with information of current separation progress. The
        progress information will override the values in `callback_arg` if same key has been used.
        To abort the separation, raise `KeyboardInterrupt`.

        Progress information contains several keys (These keys will always exist):
        - `model_idx_in_bag`: The index of the submodel in `BagOfModels`. Starts from 0.
        - `shift_idx`: The index of shifts. Starts from 0.
        - `segment_offset`: The offset of current segment. If the number is 441000, it doesn't
            mean that it is at the 441000 second of the audio, but the "frame" of the tensor.
        - `state`: Could be `"start"` or `"end"`.
        - `audio_length`: Length of the audio (in "frame" of the tensor).
        - `models`: Count of submodels in the model.
        """
        self._name = model
        self._repo = repo
        self._load_model()
        self.update_parameter(device=device, shifts=shifts, overlap=overlap, split=split,
                              segment=segment, jobs=jobs, progress=progress, callback=callback,
                              callback_arg=callback_arg)

    def update_parameter(
        self,
        device,
        shifts,
        overlap,
        split,
        segment,
        jobs,
        progress,
        callback,
        callback_arg,
    ):
        """
        Update the parameters of separation.

        Parameters
        ----------
        segment: Length (in seconds) of each segment (only available if `split` is `True`). If \
            not specified, will use the command line option.
        shifts: If > 0, will shift in time `wav` by a random amount between 0 and 0.5 sec and \
            apply the oppositve shift to the output. This is repeated `shifts` time and all \
            predictions are averaged. This effectively makes the model time equivariant and \
            improves SDR by up to 0.2 points. If not specified, will use the command line option.
        split: If True, the input will be broken down into small chunks (length set by `segment`) \
            and predictions will be performed individually on each and concatenated. Useful for \
            model with large memory footprint like Tasnet. If not specified, will use the command \
            line option.
        overlap: The overlap between the splits. If not specified, will use the command line \
            option.
        device (torch.device, str, or None): If provided, device on which to execute the \
            computation, otherwise `wav.device` is assumed. When `device` is different from \
            `wav.device`, only local computations will be on `device`, while the entire tracks \
            will be stored on `wav.device`. If not specified, will use the command line option.
        jobs: Number of jobs. This can increase memory usage but will be much faster when \
            multiple cores are available. If not specified, will use the command line option.
        callback: A function will be called when the separation of a chunk starts or finished. \
            The argument passed to the function will be a dict. For more information, please see \
            the Callback section.
        callback_arg: A dict containing private parameters to be passed to callback function. For \
            more information, please see the Callback section.
        progress: If true, show a progress bar.

        Callback
        --------
        The function will be called with only one positional parameter whose type is `dict`. The
        `callback_arg` will be combined with information of current separation progress. The
        progress information will override the values in `callback_arg` if same key has been used.
        To abort the separation, raise `KeyboardInterrupt`.

        Progress information contains several keys (These keys will always exist):
        - `model_idx_in_bag`: The index of the submodel in `BagOfModels`. Starts from 0.
        - `shift_idx`: The index of shifts. Starts from 0.
        - `segment_offset`: The offset of current segment. If the number is 441000, it doesn't
            mean that it is at the 441000 second of the audio, but the "frame" of the tensor.
        - `state`: Could be `"start"` or `"end"`.
        - `audio_length`: Length of the audio (in "frame" of the tensor).
        - `models`: Count of submodels in the model.
        """

        self._device = device
        self._shifts = shifts
        self._overlap = overlap
        self._split = split
        self._segment = segment
        self._jobs = jobs
        self._progress = progress
        self._callback = callback
        self._callback_arg = callback_arg

    def _load_model(self):
        self._model = th.load(self._name)
        self._audio_channels = self._model.audio_channels
        self._samplerate = self._model.samplerate

    def _load_audio(self, track: Path):
        errors = {}
        wav = None

        try:
            wav = AudioFile(track).read(streams=0, samplerate=self._samplerate,
                                        channels=self._audio_channels)
        except FileNotFoundError:
            errors["ffmpeg"] = "FFmpeg is not installed."
        except subprocess.CalledProcessError:
            errors["ffmpeg"] = "FFmpeg could not read the file."

        if wav is None:
            try:
                wav, sr = ta.load(str(track))
            except RuntimeError as err:
                errors["torchaudio"] = err.args[0]
            else:
                wav = convert_audio(wav, sr, self._samplerate, self._audio_channels)

        return wav

    def separate_tensor(
        self, wav: th.Tensor, sr: Optional[int] = None
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        """
        Separate a loaded tensor.

        Parameters
        ----------
        wav: Waveform of the audio. Should have 2 dimensions, the first is each audio channel, \
            while the second is the waveform of each channel. Type should be float32. \
            e.g. `tuple(wav.shape) == (2, 884000)` means the audio has 2 channels.
        sr: Sample rate of the original audio, the wave will be resampled if it doesn't match the \
            model.

        Returns
        -------
        A tuple, whose first element is the original wave and second element is a dict, whose keys
        are the name of stems and values are separated waves. The original wave will have already
        been resampled.

        Notes
        -----
        Use this function with cautiousness. This function does not provide data verifying.
        """
        if sr is not None and sr != self.samplerate:
            wav = convert_audio(wav, sr, self._samplerate, self._audio_channels)
        ref = wav.mean(0)
        wav -= ref.mean()
        wav /= ref.std() + 1e-8
        out = apply_model(
                self._model,
                wav[None],
                segment=self._segment,
                shifts=self._shifts,
                split=self._split,
                overlap=self._overlap,
                device=self._device,
                num_workers=self._jobs,
                callback=self._callback,
                callback_arg=_replace_dict(
                    self._callback_arg, ("audio_length", wav.shape[1])
                ),
                progress=self._progress,
            )
        if out is None:
            raise KeyboardInterrupt
        out *= ref.std() + 1e-8
        out += ref.mean()
        wav *= ref.std() + 1e-8
        wav += ref.mean()
        return (wav, dict(zip(self._model.sources, out[0])))

    def separate_audio_file(self, file: Path):
        """
        Separate an audio file. The method will automatically read the file.

        Parameters
        ----------
        wav: Path of the file to be separated.

        Returns
        -------
        A tuple, whose first element is the original wave and second element is a dict, whose keys
        are the name of stems and values are separated waves. The original wave will have already
        been resampled.
        """
        return self.separate_tensor(self._load_audio(file), self.samplerate)

    @property
    def samplerate(self):
        return self._samplerate

    @property
    def audio_channels(self):
        return self._audio_channels

    @property
    def model(self):
        return self._model


if __name__ == "__main__":
    # Test API functions
    # two-stem not supported
    args = get_parser().parse_args()
    separator = Separator(
        model=args.name,
        device=args.device,
        shifts=args.shifts,
        overlap=args.overlap,
        split=args.split,
        segment=args.segment,
        jobs=args.jobs,
        progress=True
    )
    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    for file in args.tracks:
        separated = separator.separate_audio_file(file)[1]
        if args.mp3:
            ext = "mp3"
        elif args.flac:
            ext = "flac"
        else:
            ext = "wav"
        kwargs = {
            "samplerate": separator.samplerate,
            "bitrate": args.mp3_bitrate,
            "clip": args.clip_mode,
            "as_float": args.float32,
            "bits_per_sample": 24 if args.int24 else 16,
        }
        for stem, source in separated.items():
            stem = out / args.filename.format(
                track=Path(file).name.rsplit(".", 1)[0],
                trackext=Path(file).name.rsplit(".", 1)[-1],
                stem=stem,
                ext=ext,
            )
            stem.parent.mkdir(parents=True, exist_ok=True)
            save_audio(source, str(stem), **kwargs)