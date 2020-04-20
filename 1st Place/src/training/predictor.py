import torch
import rasterio
import numpy as np
from rasterio.windows import Window
from tqdm import tqdm
from multiprocessing.pool import ThreadPool


class Predictor:

    def __init__(self, sample_size=1024, cut_edge=128, batch_size=1, **save_kwargs):
        self.sample_size = sample_size
        self.cut_edge = cut_edge
        self.step = sample_size - (cut_edge * 2)
        self.save_kwargs = save_kwargs
        self.batch_size = batch_size

    def read(self, src, x, y):
        return src.read(window=Window(
            x, y, self.sample_size, self.sample_size),
            boundless=True,
        )

    def write(self, dst, data, x, y):
        
        if np.all(data == 0):
            return

        h, w = data.shape[1:3]
        write_x = x + self.cut_edge
        write_y = y + self.cut_edge
        write_h = min(h - (2 * self.cut_edge), dst.height - write_y)
        write_w = min(w - (2 * self.cut_edge), dst.width - write_x)
        if write_h < 0 or write_w < 0:
            return
        data = data[:, self.cut_edge: self.cut_edge + write_h, self.cut_edge:self.cut_edge + write_w]
        dst.write(
            data,
            window=Window(
                write_x,
                write_y,
                write_w,
                write_h,
            ),
        )

    def read_batch(self, src, blocks):

        data = []
        new_blocks = []

        for x, y in blocks:
            sample = self.read(src, x, y)
            if not np.all(sample == 0):
                data.append(sample)
                new_blocks.append((x, y))

        data = np.stack(data, axis=0) if data else None

        return data, new_blocks

    def write_batch(self, dst, data, blocks):
        for (x, y), sample in zip(blocks, data):
            if not np.all(sample == 0):
                self.write(dst, sample, x, y)

    def _compute_blocks(self, h, w):
        return [
            (x, y)
            for x in range(-self.cut_edge, w + self.cut_edge, self.step)
            for y in range(-self.cut_edge, h + self.cut_edge, self.step)
        ]

    def __call__(self, src_path, dst_path):
        with rasterio.open(src_path) as src:
            profile = src.profile
            profile.update(self.save_kwargs)

            with rasterio.open(dst_path, "w", **profile) as dst:
                blocks = self._compute_blocks(h=profile["height"], w=profile["width"])
                n_batches = (len(blocks) - 1) // self.batch_size + 1
                for i in tqdm(range(n_batches)):
                    batch_blocks = blocks[i * self.batch_size: (i + 1) * self.batch_size]
                    batch, batch_blocks = self.read_batch(src, batch_blocks)
                    if batch is not None:  # check that batch is not empty
                        prediction = self.predict(batch)
                        self.write_batch(dst, prediction, batch_blocks)

    def predict(self, raster):
        raise NotImplementedError


class TorchTifPredictor(Predictor):

    def __init__(self, runner, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.runner = runner

    def predict(self, raster):
        x = torch.from_numpy(raster).float()
        y = self.runner.predict_on_batch(dict(image=x))["mask"]
        y = y.round().int().detach().cpu().numpy().astype("uint8")
        return y
