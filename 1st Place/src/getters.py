import torch.nn as nn
import segmentation_models_pytorch as smp
from .training import losses, metrics, optimizers, callbacks
from . import datasets


def get_model(architecture, init_params):
    model_class = smp.__dict__[architecture]
    return model_class(**init_params)


def get_dataset(name, init_params, **kwargs):
    dataset_class = datasets.__dict__[name]
    dataset = dataset_class(**init_params, **kwargs)
    return dataset


def get_loss(name, init_params):
    init_params = init_params or {}
    loss_class = losses.__dict__[name]
    return loss_class(**init_params)


def get_metric(name, init_params):
    init_params = init_params or {}
    metric_class = metrics.__dict__[name]
    return metric_class(**init_params)


def get_optimizer(name, model_params, init_params):
    assert init_params is not None
    optim_class = optimizers.__dict__[name]
    # TODO: make parsing of model parameters for different LR
    return optim_class(model_params, **init_params)


def get_scheduler(name, optimizer, init_params):
    init_params = init_params or {}
    scheduler_class = optimizers.__dict__[name]
    scheduler = scheduler_class(optimizer, **init_params)
    return scheduler


def get_callback(name, init_pararams):
    init_pararams = init_pararams or {}
    callback_class = callbacks.__dict__[name]
    callback = callback_class(**init_pararams)
    return callback
