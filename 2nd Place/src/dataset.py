import math

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.distributed as dist
import numpy as np

from utils import read_img, read_img_cv


p = 0.5
albu_train = A.Compose([
    A.RandomCrop(512, 512),

    A.HorizontalFlip(p=p),
    A.VerticalFlip(p=p),

    A.OneOf([
        A.RandomBrightnessContrast(p=1),
        A.RandomGamma(p=1),
    ], p=p),
    
    A.OneOf([
        A.GaussianBlur(p=1),
        A.GaussNoise(p=1),
        A.IAAAdditiveGaussianNoise(p=1),
    ], p=p),

#     A.OneOf([
#         A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1),
#         A.GridDistortion(p=1),
#         A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1),
#     ], p=p),
    
#     A.ShiftScaleRotate(border_mode=0),

    A.Normalize(),
    ToTensorV2(),
])

aresize = A.Resize(512, 512)

albu_dev = A.Compose([
    A.Normalize(),
    ToTensorV2(),
])


def train_transform(img, mask):
    data = albu_train(image=img, mask=mask)
    img, mask = data['image'], data['mask']

    return img, mask.permute(2, 0, 1)


def dev_transform(img, mask):
    data = albu_dev(image=img, mask=mask)
    img, mask = data['image'], data['mask']
    
    return img, mask.permute(2, 0, 1)


def to_cat(mask):
    mask[..., 0] -= mask[..., 1]
    mask = np.concatenate([np.zeros_like(mask[..., [0]]), mask], -1).argmax(-1)[..., None]
    
    return mask

    
class CloudsDS(torch.utils.data.Dataset):
    def __init__(self, items, root, transform, w3m=False):
        self.items = items
        self.root = root
        self.transform = transform
        self.w3m = w3m

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items.iloc[index]

#         root = self.root.parent /  self.root.name.replace('_1_', f'_{item.tier}_')
#         path_to_img = root / f'z{item.zoom}' / 'images' / item.fname
#         path_to_mask = root / f'z{item.zoom}' / 'masks' / item.fname
        path_to_img = item.fname
        path_to_mask = item.mask_path
        if self.w3m:
            img = read_img_cv(path_to_img)
            mask = to_cat(read_img_cv(path_to_mask))
        else:
            img = read_img_cv(path_to_img)      
#             mask = (read_img_cv(path_to_mask, is_grey=True)).astype('float32')[..., None]
            if not item.is_test:
                mask = (read_img_cv(path_to_mask)[..., [0]]).astype('float32')
                mask /= 255
            else:
                mask = (read_img_cv(path_to_mask, is_grey=True)).astype('float32')[..., None]
#                 mask = aresize(image=mask)['image']
        
        return self.transform(img, mask)


def collate_fn(x):
    x, y = list(zip(*x))

    return torch.stack(x), torch.stack(y)


class DistributedWeightedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, weights, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.weights = weights
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
#         if self.shuffle:
#             indices = torch.randperm(len(self.dataset), generator=g).tolist()
#         else:
#             indices = list(range(len(self.dataset)))
        indices = torch.multinomial(self.weights, len(self.dataset), True).tolist()


        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
