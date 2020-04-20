import os
import time
import math
import operator
import warnings
import numpy as np
from collections import deque

import torch
from typing import List, Tuple


class CallbackList(object):
    """Container abstracting a list of callbacks.
    # Arguments
        callbacks: List of `Callback` instances.
        queue_length: Queue length for keeping
            running statistics over callback execution time.
    """

    def __init__(self, callbacks=None, queue_length=10):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.queue_length = queue_length

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_runner(self, runner):
        for callback in self.callbacks:
            callback.set_runner(runner)

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.
        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
        self._delta_t_batch = 0.
        self._delta_ts_batch_begin = deque([], maxlen=self.queue_length)
        self._delta_ts_batch_end = deque([], maxlen=self.queue_length)

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.
        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        """Called right before processing a batch.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
        self._delta_ts_batch_begin.append(time.time() - t_before_callbacks)
        delta_t_median = np.median(self._delta_ts_batch_begin)
        if (self._delta_t_batch > 0. and
                delta_t_median > 0.95 * self._delta_t_batch and
                delta_t_median > 0.1):
            warnings.warn('Method on_batch_begin() is slow compared '
                          'to the batch update (%f). Check your callbacks.'
                          % delta_t_median)
        self._t_enter_batch = time.time()

    def on_batch_end(self, batch, logs=None):
        """Called at the end of a batch.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        if not hasattr(self, '_t_enter_batch'):
            self._t_enter_batch = time.time()
        self._delta_t_batch = time.time() - self._t_enter_batch
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)
        self._delta_ts_batch_end.append(time.time() - t_before_callbacks)
        delta_t_median = np.median(self._delta_ts_batch_end)
        if (self._delta_t_batch > 0. and
                (delta_t_median > 0.95 * self._delta_t_batch and delta_t_median > 0.1)):
            warnings.warn('Method on_batch_end() is slow compared '
                          'to the batch update (%f). Check your callbacks.'
                          % delta_t_median)

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.
        # Arguments
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        """Called at the end of training.
        # Arguments
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def __iter__(self):
        return iter(self.callbacks)


class Callback(object):
    """Abstract base class used to build new callbacks.
    # Properties
        params: dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: instance of `keras.models.Model`.
            Reference of the model being trained.
    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch.
    Currently, the `.fit()` method of the `Sequential` model class
    will include the following quantities in the `logs` that
    it passes to its callbacks:
        on_epoch_end: logs include `acc` and `loss`, and
            optionally include `val_loss`
            (if validation is enabled in `fit`), and `val_acc`
            (if validation and accuracy monitoring are enabled).
        on_batch_begin: logs include `size`,
            the number of samples in the current batch.
        on_batch_end: logs include `loss`, and optionally `acc`
            (if accuracy monitoring is enabled).
    """

    def __init__(self):
        self.validation_data = None
        self.runner = None

    def set_params(self, params):
        self.params = params

    @property
    def model(self):
        return self.runner.model

    def set_runner(self, runner):
        self.runner = runner

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class ModelCheckpoint(Callback):

    def __init__(
            self,
            directory,
            monitor='val_loss',
            save_best=True,
            save_last=False,
            save_top_k=0,
            mode='min',
            prefix=None,
            verbose=0
    ):

        self.directory = directory
        os.makedirs(directory, exist_ok=True)

        self.save_last = save_last
        self.save_best = save_best
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.verbose = verbose
        self.prefix = prefix
        self.mode = mode

        self.last_template = os.path.join(directory, 'last.pth')
        self.best_template = os.path.join(directory, 'best.pth')
        self.top_k_template = os.path.join(directory, 'k-ep[{epoch}]-{%s:.4f}.pth') % monitor

        self._top_k: List[Tuple[float, str]] = []
        self._compare_fn = operator.gt if mode == 'max' else operator.lt  # first current!
        self._reverse = True if mode == 'max' else False
        self._best_score = - math.inf if mode == 'max' else math.inf

        super(ModelCheckpoint, self).__init__()

    def reset(self):
        self._top_k: List[Tuple[float, str]] = []
        self._compare_fn = operator.gt if self.mode == 'max' else operator.lt  # first current!
        self._reverse = True if self.mode == 'max' else False
        self._best_score = - math.inf if self.mode == 'max' else math.inf

    def _add_prefix(self, path, prefix):
        if prefix:
            directory = os.path.dirname(path)
            name = os.path.basename(path)
            prefix_name = '{}_{}'.format(prefix, name)
            path = os.path.join(directory, prefix_name)
        return path

    def save_checkpoint(self, path, epoch, logs):
        state = {}
        state['state_dict'] = self.model.state_dict()
        state['logs'] = logs
        state['epoch'] = epoch

        path = self._add_prefix(path, self.prefix)
        torch.save(state, path)

        if self.verbose:
            print('Saved: {}'.format(path))

    def on_epoch_end(self, epoch, logs=None):

        score = logs[self.monitor]

        # save last checkpoint always
        if self.save_last:
            self.save_checkpoint(self.last_template, epoch, logs)

        # save best if score is better
        if self.save_best and self._compare_fn(score, self._best_score):
            self.save_checkpoint(self.best_template, epoch, logs)
            self._best_score = score

        # save top k
        if self.save_top_k:
            path = self.top_k_template.format(epoch=epoch, **logs)
            if len(self._top_k) < self.save_top_k:
                self._top_k.append((score, path))
                self.save_checkpoint(path, epoch, logs)
            else:
                self._top_k = list(sorted(self._top_k, key=lambda x: x[0], reverse=self._reverse))
                for score_, path_ in self._top_k:
                    if self._compare_fn(score, score_):
                        _, remove_checkpoint_path = self._top_k.pop(-1)
                        self._top_k.append((score, path))
                        self.save_checkpoint(path, epoch, logs)
                        os.remove(remove_checkpoint_path)
                        break


class TensorBoard(Callback):
    """TensorBoard basic visualizations.
    [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)
    is a visualization tool provided with TensorFlow.
    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.
    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
    ```sh
    tensorboard --logdir=/full_path_to_your_logs
    ```
    When using a backend other than TensorFlow, TensorBoard will still work
    (if you have TensorFlow installed), but the only feature available will
    be the display of the losses and metrics plots.
    # Arguments
        log_dir: the path of the directory where to save the log
            files to be parsed by TensorBoard.
    """

    def __init__(self, log_dir='./logs'):
        super(TensorBoard, self).__init__()
        self.log_dir = log_dir

        import tensorflow as tf
        self.writer = tf.summary.FileWriter(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self._write_logs(logs, epoch)

    def _write_logs(self, logs, index):
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            import tensorflow as tf
            summary = tf.Summary()
            summary_value = summary.value.add()
            if isinstance(value, np.ndarray):
                summary_value.simple_value = value.item()
            else:
                summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, index)
        self.writer.flush()

    def __del__(self):
        self.writer.close()


class Scheduler(Callback):

    def __init__(self, scheduler, freq='epoch', verbose=True):
        super(Scheduler, self).__init__()
        assert freq in ['epoch', 'batch']
        self.scheduler = scheduler
        self.freq = freq
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose:
            lrs = self.scheduler.get_lr()
            print(
                'Learning rates: {}'.format(
                    ' / '.join(['{:.6f}'.format(lr) for lr in lrs])
                )
            )

    def on_batch_end(self, batch, logs=None):
        if self.freq == 'batch':
            self.scheduler.step()
        if logs is not None:
            logs['lr'] = np.mean(self.scheduler.get_lr())

    def on_epoch_end(self, epoch, logs=None):
        if self.freq == 'epoch':
            self.scheduler.step()
        if logs is not None:
            lrs = self.scheduler.get_lr()
            logs['mean_lr'] = np.mean(lrs)


class CosineAnnealingWithCheckpoints(Callback):

    def __init__(self, optimizer, cycle_len_iter, log_dir, monitor, mode, verbose=True):
        super().__init__()
        self.optimizer = optimizer
        self.log_dir = log_dir

        self.checkpointer = ModelCheckpoint(
            log_dir,
            monitor=monitor,
            mode=mode,
            save_best=True,
            save_last=False,
            save_top_k=0,
            verbose=verbose,
            prefix='cycle-0'
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, cycle_len_iter)
        self.scheduler = Scheduler(scheduler=scheduler, freq='batch', verbose=verbose)

        self.cycle_len_iter = cycle_len_iter
        self.cycle = 0
        self.iter = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.scheduler.on_epoch_begin(epoch, logs)
        self.checkpointer.on_epoch_begin(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        self.scheduler.on_batch_begin(batch, logs)
        self.checkpointer.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        self.scheduler.on_batch_end(batch, logs)
        self.checkpointer.on_batch_end(batch, logs)
        self.iter += 1
        if self.iter > self.cycle_len_iter:
            self.iter = 0
            self.cycle += 1
            self.checkpointer.reset()
            self.checkpointer.prefix = 'cycle-{}'.format(self.cycle)

    def on_epoch_end(self, epoch, logs=None):
        self.scheduler.on_epoch_end(epoch, logs)
        self.checkpointer.on_epoch_end(epoch, logs)


class BatchNormFreezer(Callback):

    def __init__(self, start_epoch, freeze_affine=False, verbose=True):
        super().__init__()
        self.start_epoch = start_epoch
        self.freeze_affine = freeze_affine
        self.verbose = verbose

    def _set_bn(self, is_trainable=False):
        n_freezed = 0
        for m in self.model.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                m.train(mode=is_trainable)
                if self.freeze_affine:
                    m.weight.requires_grad = is_trainable
                    m.bias.requires_grad = is_trainable
                n_freezed += 1
        return n_freezed

    def on_epoch_begin(self, epoch, logs=None):
        if epoch >= self.start_epoch:
            n_freezed = self._set_bn(is_trainable=False)
            if self.verbose:
                print('Freezed {} BatchNorm layers.'.format(n_freezed))

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_epoch:
            self._set_bn(is_trainable=True)


class MetricCallback(Callback):

    def __init__(self, valid_loader, valid_labels, metric_name, metric_fn):
        super().__init__()
        self.valid_labels = valid_labels
        self.valid_loader = valid_loader
        self.metric_name = metric_name
        self.metric_fn = metric_fn

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        output = self.runner.predict_generator(self.valid_loader)
        score = self.metric_fn(self.valid_labels, output)
        print("{}: {:.4f}".format(self.metric_name, score))
        logs[self.metric_name] = score
        return logs


class FreezeEncoderCallback(Callback):

    def __init__(self, start_epoch=-1, end_epoch=-1, unfreeze_bn=True, verbose=True):
        super().__init__()
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.unfreeze_bn = unfreeze_bn
        self.verbose = verbose

    def _set_module(self, module, trainable=True):
        module.train(mode=trainable)
        for p in module.parameters():
            p.requires_grad = trainable
        if self.verbose:
            print("Encoder set trainable={}".format(trainable))

    def _set_bn(self, module, trainable=False):
        for m in module.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                m.train(mode=trainable)
                m.weight.requires_grad = trainable
                m.bias.requires_grad = trainable

    def on_epoch_begin(self, epoch, logs=None):
        if epoch >= self.start_epoch and epoch < self.end_epoch:
            self._set_module(self.model.encoder, trainable=False)
        elif epoch == self.end_epoch:
            self._set_module(self.model.encoder, trainable=True)

        # stay batch norm layers freeze
        if epoch >= self.end_epoch and not self.unfreeze_bn:
            self._set_bn(self.model.encoder, trainable=False)
