"""
Módulo y algoritmos para entrenar los modelos creados.
"""


import logging
import os
import sys
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from .architecture import distrib
from .architecture.model import Audiomancer
from .architecture.solver import Solver
from .common import read_from_jams
from .constants import LABELS

logging.basicConfig(level=logging.DEBUG,force=True)
logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)


class stems(Dataset):
    def __init__(self,X,Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return self.X[idx], self.Y[idx]

def charge_model(obj,args):
    extra = {
        'sources': list(args.dset.sources),
        'audio_channels': args.dset.channels,
        'samplerate': args.dset.samplerate,
        'segment': args.dset.segment,
    }
    model = obj(**extra, **args.get('audiomancy'))
    return model

def get_optimizer(model, args):
    seen_params = set()
    other_params = []
    groups = []
    for _, module in model.named_modules():
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
            lr=float(args.optim.lr),
            betas=(args.optim.momentum, args.optim.beta2),
            weight_decay=args.optim.weight_decay,
        )
    elif args.optim.optim == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=float(args.optim.lr),
            betas=(args.optim.momentum, args.optim.beta2),
            weight_decay=args.optim.weight_decay,
        )
    else:
        raise ValueError("Invalid optimizer %s", args.optim.optimizer)


def splitter(args):
    ret = []
    for key, value in LABELS.items():
        if key in args.dset.sources:
           ret.append(value) 
    
    X, Y = read_from_jams(args.dset.jams)
    Y = Y[:,ret,:,:]
    prop = args.dset.training_split.split('/')
    
    X_T, X_tv, Y_T, Y_tv = train_test_split(X,Y,test_size=(int(prop[-1])+int(prop[-2]))/100, random_state=args.seed)
    X_t, X_v, Y_t, Y_v = train_test_split(X_tv,Y_tv,test_size=0.5,random_state=args.seed)

    return stems(X_T,Y_T), stems(X_v,Y_v), stems(X_t,Y_t)

def get_solver(args):
    distrib.init()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = charge_model(Audiomancer,args)
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

    train_set, valid_set, test_set = splitter(args)
    args.weights = args.weights[:len(args.dset.sources)]

    logger.info("train/valid set size: %d %d", len(train_set), len(valid_set))
    train_loader = distrib.loader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.misc.num_workers, drop_last=True)
    test_loader = distrib.loader(
        test_set, batch_size=1, shuffle=True,
        num_workers=args.misc.num_workers, drop_last=True)
    if args.dset.full_cv:
        valid_loader = distrib.loader(
            valid_set, batch_size=1, shuffle=False,
            num_workers=args.misc.num_workers)
    else:
        valid_loader = distrib.loader(
            valid_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.misc.num_workers, drop_last=True)
    loaders = {"train": train_loader, "valid": valid_loader,"test":test_loader}

    # Construct Solver
    return Solver(loaders, model, optimizer, device, args)

def start(args):
    global __file__
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    if args.misc.verbose:
        logger.setLevel(logging.DEBUG)
        stream_handler.setLevel(logging.DEBUG)
        logger.addHandler(stream_handler)

    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)

    solver = get_solver(args)
    solver.train()