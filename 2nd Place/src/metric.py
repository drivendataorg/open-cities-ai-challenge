def calc_iu(pred, target, thresh=0.5, min_area=float('-inf')):
    pred = pred > thresh
    if pred.sum() < min_area:
        pred[:] = False

    pred = pred.flatten()
    target = target.flatten()

    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    return inter, union


def calc_dice(pred, target, thresh=0.5, eps=1e-8, min_area=float('-inf')):
    inter, union = calc_iu(pred, target, thresh=thresh, min_area=min_area)

    return (inter + eps) / (union - inter + eps)


class Dice:
    def __init__(self, n_classes=1, thresh=None, w3m=False):
        self.thresh = thresh
        self.n_classes = n_classes - 1 if thresh is None else n_classes
        self.w3m = w3m
        if self.w3m:
            self.n_classes = 1

        self.clean()

    def clean(self):
        self.dice = {i: 0.0 for i in range(self.n_classes)}
        self.n = 0

    def update(self, preds, targets):
        assert len(preds) == len(targets)

        for p, t in zip(preds, targets):
            if self.thresh is None:
                t = t.squeeze()
                p = p.argmax(0)
                
                if self.w3m:
                    p = (p == 1) | (p == 2)
                    
                for c in range(self.n_classes):
                    self.dice[c] += calc_dice(p == c + 1, t == c + 1)
            else:
                for c in range(self.n_classes):
                    self.dice[c] += calc_dice(p[c], t[c], self.thresh)
        self.n += len(preds)

    def evaluate(self, reduce=True):
        if not reduce:
            return [self.dice[c]/self.n for c in self.dice]

        if self.n > 0:
            return sum(self.dice.values())/self.n_classes/self.n

        return 0.0
    
    
class JaccardMicro:
    def __init__(self, n_classes=1, thresh=None, eps=1e-6, w3m=False):
        self.thresh = thresh
        self.eps = eps
        self.n_classes = n_classes - 1 if thresh is None else n_classes
        self.w3m = w3m
        if self.w3m:
            self.n_classes = 1

        self.clean()

    def clean(self):
        self.inter = {i: 0.0 for i in range(self.n_classes)}
        self.union = {i: 0.0 for i in range(self.n_classes)}

    def update(self, preds, targets):
        assert len(preds) == len(targets)

        for p, t in zip(preds, targets):
            if self.thresh is None:
                t = t.squeeze()
                p = p.argmax(0)
                
                if self.w3m:
                    p = (p == 1) | (p == 2)
                    
                for c in range(self.n_classes):
                    inter, union = calc_iu(p == c + 1, t == c + 1)
                    self.inter[c] += inter
                    self.union[c] += union
            else:
                for c in range(self.n_classes):
                    inter, union = calc_iu(p[c], t[c], self.thresh)
                    self.inter[c] += inter
                    self.union[c] += union

    def evaluate(self):
        return sum((self.inter[c] + self.eps)/(self.union[c] - self.inter[c] + self.eps) for c in range(self.n_classes))/self.n_classes
