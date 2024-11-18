import torch
from . import distrib
from dora.log import LogProgress
import logging
from concurrent import futures
import numpy as np
import museval
from ..audioprocessing import convert_audio
from .apply import apply_model

logger = logging.getLogger(__name__)

def new_sdr(references, estimates):
    """
    Compute the SDR according to the MDX challenge definition.
    Adapted from AIcrowd/music-demixing-challenge-starter-kit (MIT license)
    """
    assert references.dim() == 4
    assert estimates.dim() == 4
    delta = 1e-7  # avoid numerical errors
    num = torch.sum(torch.square(references), dim=(2, 3))
    den = torch.sum(torch.square(references - estimates), dim=(2, 3))
    num += delta
    den += delta
    scores = 10 * torch.log10(num / den)
    return scores

def eval_track(references, estimates, win, hop, compute_sdr=True):
    references = references.double()
    estimates = estimates.double()

    new_scores = new_sdr(references.cpu(), estimates.cpu())[0]

    if not compute_sdr:
        return None, new_scores
    else:
        references = references.numpy()
        estimates = estimates.numpy()
        scores = museval.metrics.bss_eval(
            references, estimates,
            compute_permutation=False,
            window=win,
            hop=hop,
            framewise_filters=False,
            bsseval_sources_version=False)[:-1]
        return scores, new_scores

def evaluate(solver,test_set, compute_sdr=False):
    """
    Evaluate model using museval.
    compute_sdr=False means using only the MDX definition of the SDR, which
    is much faster to evaluate.
    """
    args = solver.args

    # we load tracks from the original musdb set
    src_rate = 44100
    eval_device = 'cpu'

    model = solver.model
    win = int(1. * model.samplerate)
    hop = int(1. * model.samplerate)

    logprog = LogProgress(logger, test_set, updates=args.misc.num_prints,
                          name='Eval')
    pendings = []

    pool = futures.ProcessPoolExecutor
    with pool(args.test.workers) as pool:
        for idx, (mix, sources) in enumerate(logprog):
            mix = mix.to(solver.device)
            ref = mix.mean(dim=1)  # mono mixture
            mix = (mix - ref.mean()) / ref.std()
            mix = convert_audio(mix, src_rate, model.samplerate, model.audio_channels)
            estimates = apply_model(model, mix,
                                    shifts=args.test.shifts, split=args.test.split,
                                    overlap=args.test.overlap)[0]
            estimates = estimates[None] * ref.std() + ref.mean()
            estimates = estimates.to(eval_device)

            sources = sources.to(eval_device)
            sources = convert_audio(sources, src_rate,
                                       model.samplerate, model.audio_channels)
            
            pendings.append((idx, pool.submit(
                eval_track, sources, estimates, win=win, hop=hop, compute_sdr=compute_sdr)))

        pendings = LogProgress(logger, pendings, updates=args.misc.num_prints,
                               name='Eval (BSS)')
        tracks = {}
        for idx, pending in pendings:
            pending = pending.result()
            scores, nsdrs = pending
            tracks[idx] = {}
            
            for idy, target in enumerate(model.sources):
                tracks[idx][target] = {'nsdr': [float(nsdrs[idy])]}
            if scores is not None:
                (sdr, isr, sir, sar) = scores
                for idz, target in enumerate(model.sources):
                    values = {
                        "SDR": sdr[idz].tolist(),
                        "SIR": sir[idz].tolist(),
                        "ISR": isr[idz].tolist(),
                        "SAR": sar[idz].tolist()
                    }
                    tracks[idx][target].update(values)

        all_tracks = {}
        for src in range(distrib.world_size):
            all_tracks.update(distrib.share(tracks, src))

        result = {}
        metric_names = next(iter(all_tracks.values()))[model.sources[0]]
        for metric_name in metric_names:
            avg = 0
            avg_of_medians = 0
            for source in model.sources:
                medians = [
                    np.nanmedian(all_tracks[track][source][metric_name])
                    for track in all_tracks.keys()]
                mean = np.mean(medians)
                median = np.median(medians)
                result[metric_name.lower() + "_" + source] = mean
                result[metric_name.lower() + "_med" + "_" + source] = median
                avg += mean / len(model.sources)
                avg_of_medians += median / len(model.sources)
            result[metric_name.lower()] = avg
            result[metric_name.lower() + "_med"] = avg_of_medians
        return result