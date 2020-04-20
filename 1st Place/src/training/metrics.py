import torch
from . import base
from . import functional as F
from . import _modules as modules


class IoU(base.Metric):
    __name__ = "iou"

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None,
                 per_image=False, class_weights=None, drop_empty=False, take_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = modules.Activation(activation, dim=1)
        self.ignore_channels = ignore_channels
        self.per_image = per_image
        self.class_weights = class_weights
        self.drop_empty = drop_empty
        self.take_channels = take_channels

    @torch.no_grad()
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.iou(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
            per_image=self.per_image,
            class_weights=self.class_weights,
            drop_empty=self.drop_empty,
            take_channels=self.take_channels,
        )


class MicroIoU(base.Metric):
    __name__ = "micro_iou"

    def __init__(self, threshold=0.5):
        super().__init__()
        self.eps = 1e-5
        self.intersection = 0.
        self.union = 0.
        self.threshold = threshold

    def reset(self):
        self.intersection = 0.
        self.union = 0.

    @torch.no_grad()
    def __call__(self, prediction, target):
        prediction = (prediction > self.threshold).float()

        intersection = (prediction * target).sum()
        union = (prediction + target).sum() - intersection

        self.intersection += intersection.detach()
        self.union += union.detach()

        score = (self.intersection + self.eps) / (self.union + self.eps)
        return score
