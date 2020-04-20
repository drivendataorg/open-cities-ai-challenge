import os
import glob
import rasterio
import numpy as np
import pandas as pd
from typing import Optional
import torch
from torch.utils.data import Dataset
from . import transforms

import warnings

warnings.simplefilter("ignore")


class SegmentationDataset(Dataset):

    def __init__(
            self,
            images_dir: str,
            masks_dir: str,
            ids: Optional[list] = None,
            transform_name: Optional[str] = None,
    ):
        super().__init__()
        self.ids = ids if ids is not None else os.listdir(images_dir)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transforms.__dict__[transform_name] if transform_name else None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id = self.ids[i]
        image_path = os.path.join(self.images_dir, id)
        mask_path = os.path.join(self.masks_dir, id)

        # read data sample
        sample = dict(
            id=id,
            image=self.read_image(image_path),
            mask=self.read_mask(mask_path),
        )

        # apply augmentations
        if self.transform is not None:
            sample = self.transform(**sample)

        sample["mask"] = sample["mask"][None]  # expand first dim for mask

        return sample

    def read_image(self, path):
        with rasterio.open(path) as f:
            image = f.read()
        image = image.transpose(1, 2, 0)
        return image

    def read_mask(self, path):
        return self.read_image(path).squeeze()

    def read_image_profile(self, id):
        path = os.path.join(self.images_dir, id)
        with rasterio.open(path) as f:
            return f.profile


class TestSegmentationDataset(Dataset):

    def __init__(self, images_dir, transform_name=None):
        super().__init__()
        self.ids = os.listdir(images_dir)
        self.images_dir = images_dir
        self.transform = transforms.__dict__[transform_name] if transform_name else None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id = self.ids[i]
        path = os.path.join(self.images_dir, id)

        sample = dict(
            id=id,
            image=self.read_image(path),
        )

        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    def read_image(self, path):
        with rasterio.open(path) as f:
            image = f.read()[:3]
        image = image.transpose(1, 2, 0)
        return image

    def read_image_profile(self, id):
        path = os.path.join(self.images_dir, id)
        with rasterio.open(path) as f:
            return f.profile