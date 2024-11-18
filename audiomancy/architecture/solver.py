
import logging
import os
import gc
import torch
import torch.nn.functional as F
from dora.utils import write_and_rename
from dora.log import LogProgress, bold
from dora.link import Link

from . import distrib, states
from .utils import EMA, pull_metric
from .evaluate import evaluate, new_sdr
from ..constants import OUTPUT_PATH
from .apply import apply_model

logger = logging.getLogger(__name__)

def _summary(metrics):
    return " | ".join(f"{key.capitalize()}={val}" for key, val in metrics.items())

class Solver(object):
    def __init__(self, loaders, model, optimizer, device,args):
        self.args = args
        self.loaders = loaders

        self.model = model
        self.optimizer = optimizer
        self.dmodel = distrib.wrap(model)        
        self.device = device

        if args.exp.out_path is None:
            abs_path = OUTPUT_PATH
        else:
            abs_path = args.exp.out_path
        
        self.folder = os.path.join(abs_path,args.exp.name)
        os.makedirs(self.folder,exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(self.folder,'log.txt'),mode='a',encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        # Checkpoints
        self.checkpoint_file = os.path.join(self.folder,'checkpoint.th')
        self.best_file = os.path.join(self.folder,'best.th')
        logger.debug("Checkpoint will be saved to %s", self.checkpoint_file)
        self.best_state = None
        self.best_changed = False

        self.link = Link(os.path.join(self.folder,'history.json'))
        self.history = self.link.history
        self._reset()

    def _serialize(self):
        package = {}
        package['state'] = self.model.state_dict()
        package['optimizer'] = self.optimizer.state_dict()
        package['history'] = self.history
        package['best_state'] = self.best_state
        package['args'] = self.args

        with write_and_rename(self.checkpoint_file) as tmp:
            torch.save(package, tmp)

        if self.best_changed:
            # Saving only the latest best model.
            with write_and_rename(self.best_file) as tmp:
                package = states.serialize_model(self.model, self.args)
                package['state'] = self.best_state
                torch.save(package, tmp)
            self.best_changed = False

    def _reset(self):
        """Reset state of the solver, potentially using checkpoint."""
        if os.path.exists(self.checkpoint_file):
            logger.info(f'Loading checkpoint model: {self.checkpoint_file}')
            package = torch.load(self.checkpoint_file, 'cpu')
            self.model.load_state_dict(package['state'])
            self.optimizer.load_state_dict(package['optimizer'])
            self.history[:] = package['history']
            self.best_state = package['best_state']

    def _format_train(self, metrics: dict) -> dict:
        """Formatting for train/valid metrics."""
        losses = {
            'loss': format(metrics['loss'], ".4f"),
            'reco': format(metrics['reco'], ".4f"),
        }
        if 'nsdr' in metrics:
            losses['nsdr'] = format(metrics['nsdr'], ".3f")
        if 'grad' in metrics:
            losses['grad'] = format(metrics['grad'], ".4f")
        if 'best' in metrics:
            losses['best'] = format(metrics['best'], '.4f')
        if 'bname' in metrics:
            losses['bname'] = metrics['bname']
        if 'penalty' in metrics:
            losses['penalty'] = format(metrics['penalty'], ".4f")
        if 'hloss' in metrics:
            losses['hloss'] = format(metrics['hloss'], ".4f")
        return losses

    def _format_test(self, metrics: dict) -> dict:
        """Formatting for test metrics."""
        losses = {}
        if 'sdr' in metrics:
            losses['sdr'] = format(metrics['sdr'], '.3f')
        if 'nsdr' in metrics:
            losses['nsdr'] = format(metrics['nsdr'], '.3f')
        for source in self.model.sources:
            key = f'sdr_{source}'
            if key in metrics:
                losses[key] = format(metrics[key], '.3f')
            key = f'nsdr_{source}'
            if key in metrics:
                losses[key] = format(metrics[key], '.3f')
        return losses

    def train(self):
        # Optimizing the model
        epoch = 0
        for epoch in range(len(self.history), self.args.epochs):
            # Train one epoch
            self.model.train()  # Turn on BatchNorm & Dropout
            metrics = {}
            logger.info('-' * 70)
            logger.info("Training...")
            metrics['train'] = self._run_one_epoch(epoch)
            formatted = self._format_train(metrics['train'])
            logger.info(
                bold(f'Train Summary | Epoch {epoch + 1} | {_summary(formatted)}'))

            # Cross validation
            logger.info('-' * 70)
            logger.info('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            with torch.no_grad():
                valid = self._run_one_epoch(epoch, train=False)
                state = states.copy_state(self.model.state_dict())
                metrics['valid'] = valid
                key = self.args.test.metric

            valid_loss = metrics['valid'][key]
            mets = pull_metric(self.link.history, f'valid.{key}') + [valid_loss]
            if key.startswith('nsdr'):
                best_loss = max(mets)
            else:
                best_loss = min(mets)
            metrics['valid']['best'] = best_loss

            formatted = self._format_train(metrics['valid'])
            logger.info(
                bold(f'Valid Summary | Epoch {epoch + 1} | {_summary(formatted)}'))

            # Save the best model
            if valid_loss == best_loss or self.args.dset.train_valid:
                logger.info(bold('New best valid loss %.4f'), valid_loss)
                self.best_state = states.copy_state(state)
                self.best_changed = True

            # Eval model every `test.every` epoch or on last epoch
            should_eval = (epoch + 1) % self.args.test.every == 0
            is_last = epoch == self.args.epochs - 1

            if should_eval or is_last:
                # Evaluate on the testset
                logger.info('-' * 70)
                logger.info('Evaluating on the test set...')
                # We switch to the best known model for testing
                if self.args.test.best:
                    state = self.best_state
                else:
                    state = states.copy_state(self.model.state_dict())
                compute_sdr = self.args.test.sdr and is_last
                with states.swap_state(self.model, state):
                    with torch.no_grad():
                        metrics['test'] = evaluate(self, self.loaders['test'],compute_sdr=compute_sdr)
                formatted = self._format_test(metrics['test'])
                logger.info(bold(f"Test Summary | Epoch {epoch + 1} | {_summary(formatted)}"))
            self.link.push_metrics(metrics)

            if distrib.rank == 0:
                # Save model each epoch
                self._serialize()
                logger.debug("Checkpoint saved to %s", self.checkpoint_file)
            if is_last:
                break

    def _run_one_epoch(self, epoch, train=True):
        args = self.args
        data_loader = self.loaders['train'] if train else self.loaders['valid']
        if distrib.world_size > 1 and train:
            data_loader.sampler.set_epoch(epoch)

        label = ["Valid", "Train"][train]
        name = label + f" | Epoch {epoch + 1}"
        total = len(data_loader)
        if args.max_batches:
            total = min(total, args.max_batches)
        logprog = LogProgress(logger, data_loader, total=total,
                              updates=self.args.misc.num_prints, name=name)
        averager = EMA()

        for idx, (mix,sources) in enumerate(logprog):
            sources = sources.to(self.device)
            mix = mix.to(self.device)

            if not train and self.args.valid_apply:
                estimate = apply_model(self.model, mix, split=self.args.test.split, overlap=0.25)
            else:
                estimate = self.dmodel(mix)
            assert estimate.shape == sources.shape, (estimate.shape, sources.shape)
            dims = tuple(range(2, sources.dim()))

            if args.optim.loss == 'l1':
                loss = F.l1_loss(estimate, sources, reduction='none')
                loss = loss.mean(dims).mean(0)
                reco = loss
            elif args.optim.loss == 'mse':
                loss = F.mse_loss(estimate, sources, reduction='none')
                loss = loss.mean(dims)
                reco = loss**0.5
                reco = reco.mean(0)
            else:
                raise ValueError(f"Invalid loss {self.args.loss}")
            weights = torch.tensor(args.weights).to(sources)
            loss = (loss * weights).sum() / weights.sum()

            ms = 0

            losses = {}
            losses['reco'] = (reco * weights).sum() / weights.sum()
            losses['ms'] = ms

            if not train:
                nsdrs = new_sdr(sources, estimate.detach()).mean(0)
                total = 0
                for source, nsdr, w in zip(self.model.sources, nsdrs, weights):
                    losses[f'nsdr_{source}'] = nsdr
                    total += w * nsdr
                losses['nsdr'] = total / weights.sum()

            losses['loss'] = loss

            for k, source in enumerate(self.model.sources):
                losses[f'reco_{source}'] = reco[k]

            # optimize model in training mode
            if train:
                loss.backward()
                grad_norm = 0
                grads = []
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm()**2
                        grads.append(p.grad.data)
                losses['grad'] = grad_norm ** 0.5
                if args.optim.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        args.optim.clip_grad)

                self.optimizer.step()
                self.optimizer.zero_grad()
            losses = averager(losses)
            logs = self._format_train(losses)
            logprog.update(**logs)
            # Just in case, clear some memory
            del loss, estimate, reco, ms
            gc.collect()
            torch.cuda.empty_cache()
        return distrib.average(losses, idx + 1)