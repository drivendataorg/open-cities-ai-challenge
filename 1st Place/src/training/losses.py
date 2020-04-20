import torch
import torch.nn as nn

from . import base
from . import functional as F
from . import _modules as modules


class JaccardLoss(base.Loss):

    def __init__(self, eps=1e-7, activation=None, ignore_channels=None,
                 per_image=False, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = modules.Activation(activation, dim=1)
        self.per_image = per_image
        self.ignore_channels = ignore_channels
        self.class_weights = class_weights

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(
            y_pr, y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
            per_image=self.per_image,
            class_weights=self.class_weights,
        )


class DiceLoss(base.Loss):

    def __init__(self, eps=1e-7, beta=1., activation=None, ignore_channels=None,
                 per_image=False, class_weights=None, drop_empty=False, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = modules.Activation(activation, dim=1)
        self.ignore_channels = ignore_channels
        self.per_image = per_image
        self.class_weights = class_weights
        self.drop_empty = drop_empty

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
            per_image=self.per_image,
            class_weights=self.class_weights,
            drop_empty=self.drop_empty,
        )


class L1Loss(nn.L1Loss, base.Loss):
    pass


class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    pass


class BCELoss(base.Loss):

    def __init__(self, pos_weight=1., neg_weight=1., reduction='mean', label_smoothing=None):
        super().__init__()
        assert reduction in ['mean', None, False]
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, pr, gt):
        loss = F.binary_crossentropy(
            pr, gt,
            pos_weight=self.pos_weight,
            neg_weight=self.neg_weight,
            label_smoothing=self.label_smoothing,
        )

        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


class BinaryFocalLoss(base.Loss):
    def __init__(self, alpha=1, gamma=2, class_weights=None, logits=False, reduction='mean', label_smoothing=None):
        super().__init__()
        assert reduction in ['mean', None]
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction
        self.class_weights = class_weights if class_weights is not None else 1.
        self.label_smoothing = label_smoothing

    def forward(self, pr, gt):
        if self.logits:
            bce_loss = nn.functional.binary_cross_entropy_with_logits(pr, gt, reduction='none')
        else:
            bce_loss = F.binary_crossentropy(pr, gt, label_smoothing=self.label_smoothing)

        pt = torch.exp(- bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        focal_loss = focal_loss * torch.tensor(self.class_weights).to(focal_loss.device)

        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()

        return focal_loss


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass


class FocalDiceLoss(base.Loss):

    def __init__(self):
        super().__init__()
        self.focal = BinaryFocalLoss()
        self.dice = DiceLoss(eps=10.)

    def __call__(self, y_pred, y_true):
        return 2 * self.focal(y_pred, y_true) + self.dice(y_pred, y_true)


class BCEDiceLoss(base.Loss):

    def __init__(self):
        super().__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss(eps=10.)

    def __call__(self, y_pred, y_true):
        return 2 * self.bce(y_pred, y_true) + self.dice(y_pred, y_true)
