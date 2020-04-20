import os
import fire
import ttach
import torch
import addict
import argparse
import rasterio
import pandas as pd

from tqdm import tqdm
from multiprocessing import Pool

from rasterio.windows import Window

from .training.runner import GPUNormRunner
from .training.config import parse_config

from . import getters
from .training.predictor import TorchTifPredictor
from .datasets import TestSegmentationDataset


class EnsembleModel(torch.nn.Module):
    """Ensemble of torch models, pass tensor through all models and average results"""

    def __init__(self, models: list):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        result = None
        for model in self.models:
            y = model(x)
            if result is None:
                result = y
            else:
                result += y
        result /= torch.tensor(len(self.models)).to(result.device)
        return result


def model_from_config(path: str):
    """Create model from configuration specified in config file and load checkpoint weights"""
    cfg = addict.Dict(parse_config(config=path))  # read and parse config file
    init_params = cfg.model.init_params  # extract model initialization parameters
    init_params["encoder_weights"] = None  # because we will load pretrained weights for whole model
    model = getters.get_model(architecture=cfg.model.architecture, init_params=init_params)
    checkpoint_path = os.path.join(cfg.logdir, "checkpoints", "best.pth")
    state_dict = torch.load(checkpoint_path)["state_dict"]
    model.load_state_dict(state_dict)
    return model


def main(args):
    # set GPUS
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # --------------------------------------------------
    # define model
    # --------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Available devices:", device)

    # loading trained models
    models = [model_from_config(config_path) for config_path in args.configs]

    # create ensemble
    model = EnsembleModel(models)

    # add test time augmentations (flipping and rotating input image)
    if args.tta:
        model = ttach.SegmentationTTAWrapper(model, ttach.aliases.d4_transform(), merge_mode='mean')

    # create Multi GPU model if number of GPUs is more than one
    n_gpus = len(args.gpu.split(","))
    if n_gpus > 1:
        gpus = list(range(n_gpus))
        model = torch.nn.DataParallel(model, gpus)

    print("Done loading...")
    model.to(device)

    # --------------------------------------------------
    # start evaluation
    # --------------------------------------------------
    runner = GPUNormRunner(model, model_device=device)
    model.eval()

    # predict big tif files
    predictor = TorchTifPredictor(
        runner=runner, sample_size=1024, cut_edge=256,
        batch_size=args.batch_size,
        count=1, NBITS=1, compress=None, driver="GTiff",
        blockxsize=256, blockysize=256, tiled=True, BIGTIFF='IF_NEEDED',
    )

    
    print(f"Inference for {args.src_path}")
    predictor(args.src_path, args.dst_path)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--configs', nargs="+", required=True, type=str,
        help="Path(s) to models configuration file."
    )
    parser.add_argument(
        '--src_path', type=str, required=True,
        help="path to RGB tif file, it is recommended to reproject file" + \
        " to UTM zone and resample to 0.1 m/pixel resolution"
    )
    parser.add_argument(
        '--dst_path', type=str, required=True,
        help="Path where preiction will be saved",
    )
    parser.add_argument('--batch_size', type=int, default=8, help="Number of images in batch")
    parser.add_argument('--gpu', type=str, default='0', help="GPUs to use, e.g. --gpu 0,1,2")
    parser.add_argument('--tta', action='store_true', help="Flag to enable test time augmentation")
    args = parser.parse_args()

    main(args)
    os._exit(0)