"""
Módulo y algoritmos para entrenar los modelos creados.
"""


import logging
import os
from pathlib import Path
import sys
import torch
from omegaconf import OmegaConf
from .architecture import distrib
from .architecture.solver import Solver
import hydra

logger = logging.getLogger(__name__)

def charge_model(obj,args):
    extra = {
        'sources': list(args.dset.sources),
        'audio_channels': args.dset.channels,
        'samplerate': args.dset.samplerate,
        'segment': args.model_segment or 4 * args.dset.segment,
    }
    kw = OmegaConf.to_container(getattr(args, args.model), resolve=True)
    model = obj(**extra, **kw)
    return model

def get_optimizer(model, args):
    seen_params = set()
    other_params = []
    groups = []
    for n, module in model.named_modules():
        if hasattr(module, "make_optim_group"):
            group = module.make_optim_group()
            params = set(group["params"])
            assert params.isdisjoint(seen_params)
            seen_params |= set(params)
            groups.append(group)
    for param in model.parameters():
        if param not in seen_params:
            other_params.append(param)
    groups.insert(0, {"params": other_params})
    parameters = groups
    if args.optim.optim == "adam":
        return torch.optim.Adam(
            parameters,
            lr=args.optim.lr,
            betas=(args.optim.momentum, args.optim.beta2),
            weight_decay=args.optim.weight_decay,
        )
    elif args.optim.optim == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=args.optim.lr,
            betas=(args.optim.momentum, args.optim.beta2),
            weight_decay=args.optim.weight_decay,
        )
    else:
        raise ValueError("Invalid optimizer %s", args.optim.optimizer)


def splitter(args):
    return True

def get_solver(args, model_only=False):
    distrib.init()

    torch.manual_seed(args.seed)
    model = charge_model(args)
    if args.misc.show:
        logger.info(model)
        mb = sum(p.numel() for p in model.parameters()) * 4 / 2**20
        logger.info('Size: %.1f MB', mb)
        if hasattr(model, 'valid_length'):
            field = model.valid_length(1)
            logger.info('Field: %.1f ms', field / args.dset.samplerate * 1000)
        sys.exit(0)

    # torch also initialize cuda seed if available
    if torch.cuda.is_available():
        model.cuda()

    # optimizer
    optimizer = get_optimizer(model, args)

    assert args.batch_size % distrib.world_size == 0
    args.batch_size //= distrib.world_size

    if model_only:
        return Solver(None, model, optimizer, args)

    train_set, valid_set = splitter(args)

    logger.info("train/valid set size: %d %d", len(train_set), len(valid_set))
    train_loader = distrib.loader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.misc.num_workers, drop_last=True)
    if args.dset.full_cv:
        valid_loader = distrib.loader(
            valid_set, batch_size=1, shuffle=False,
            num_workers=args.misc.num_workers)
    else:
        valid_loader = distrib.loader(
            valid_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.misc.num_workers, drop_last=True)
    loaders = {"train": train_loader, "valid": valid_loader}

    # Construct Solver
    return Solver(loaders, model, optimizer, args)

def main(args):
    global __file__
    for attr in ["musdb", "wav", "metadata"]:
        val = getattr(args.dset, attr)
        if val is not None:
            setattr(args.dset, attr, hydra.utils.to_absolute_path(val))

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    if args.misc.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)
    from dora import get_xp
    logger.debug(get_xp().cfg)

    solver = get_solver(args)
    solver.train()