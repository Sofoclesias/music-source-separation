
import torch
import random
from . import distrib
from dora.log import LogProgress
import logging
from concurrent import futures
import numpy as np
import os
import museval
from ..audioprocessing import save_audio, convert_audio
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


def power_iteration(m, niters=1, bs=1):
    """This is the power method. batch size is used to try multiple starting point in parallel."""
    assert m.dim() == 2
    assert m.shape[0] == m.shape[1]
    dim = m.shape[0]
    b = torch.randn(dim, bs, device=m.device, dtype=m.dtype)

    for _ in range(niters):
        n = m.mm(b)
        norm = n.norm(dim=0, keepdim=True)
        b = n / (1e-10 + norm)

    return norm.mean()
    
penalty_rng = random.Random(1234)


def svd_penalty(model, min_size=0.1, dim=1, niters=2, powm=False, convtr=True,
                proba=1, conv_only=False, exact=False, bs=1):
    """
    Penalty on the largest singular value for a layer.
    Args:
        - model: model to penalize
        - min_size: minimum size in MB of a layer to penalize.
        - dim: projection dimension for the svd_lowrank. Higher is better but slower.
        - niters: number of iterations in the algorithm used by svd_lowrank.
        - powm: use power method instead of lowrank SVD, my own experience
            is that it is both slower and less stable.
        - convtr: when True, differentiate between Conv and Transposed Conv.
            this is kept for compatibility with older experiments.
        - proba: probability to apply the penalty.
        - conv_only: only apply to conv and conv transposed, not LSTM
            (might not be reliable for other models than Demucs).
        - exact: use exact SVD (slow but useful at validation).
        - bs: batch_size for power method.
    """
    total = 0
    if penalty_rng.random() > proba:
        return 0.

    for m in model.modules():
        for name, p in m.named_parameters(recurse=False):
            if p.numel() / 2**18 < min_size:
                continue
            if convtr:
                if isinstance(m, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d)):
                    if p.dim() in [3, 4]:
                        p = p.transpose(0, 1).contiguous()
            if p.dim() == 3:
                p = p.view(len(p), -1)
            elif p.dim() == 4:
                p = p.view(len(p), -1)
            elif p.dim() == 1:
                continue
            elif conv_only:
                continue
            assert p.dim() == 2, (name, p.shape)
            if exact:
                estimate = torch.svd(p, compute_uv=False)[1].pow(2).max()
            elif powm:
                a, b = p.shape
                if a < b:
                    n = p.mm(p.t())
                else:
                    n = p.t().mm(p)
                estimate = power_iteration(n, niters, bs)
            else:
                estimate = torch.svd_lowrank(p, dim, niters)[1][0].pow(2)
            total += estimate
    return total / proba