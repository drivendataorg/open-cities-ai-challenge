import os
import glob
import json
import rasterio
import argparse
import numpy as np
import aeronet.dataset as ds
from rasterio import Affine


def sample(raster, y, x, height, width):
    """
    Read sample of of band to memory with specified:
        x, y - pixel coordinates of left top corner
        width, height - spatial dimension of sample in pixels
    Return: raster, profile
    """

    coord_x = raster.transform.c + x * raster.transform.a
    coord_y = raster.transform.f + y * raster.transform.e

    dst_crs = raster.crs
    dst_name = os.path.basename(raster.name)
    dst_nodata = raster.nodata if raster.nodata is not None else 0
    dst_transform = Affine(raster.transform.a, raster.transform.b, coord_x,
                           raster.transform.d, raster.transform.e, coord_y)

    dst_raster = raster.read(window=((y, y + height), (x, x + width)),
                             boundless=True, fill_value=dst_nodata)

    return dst_raster, dict(transform=dst_transform, crs=dst_crs, nodata=dst_nodata)


def generate_samples(raster, width, height):
    """
    Yield `Sample`s with defined grid
    Args:
        width: dimension of sample in pixels and step along `X` axis
        height: dimension of sample in pixels and step along `Y` axis
    Returns:
        Generator object
    """
    for x in range(0, raster.width, width):
        for y in range(0, raster.height, height):
            yield sample(raster, y, x, height, width)


def save_raster(path, raster, **profile):
    """Save raster on disk"""
    c, h, w = raster.shape
    _profile = dict(
        driver="GTiff",
        height=h,
        width=w,
        count=c,
        dtype=raster.dtype,
    )
    _profile.update(profile)

    with rasterio.open(path, "w", **_profile) as dst:
        dst.write(raster)


def main(path_pattern, dst_dir, size=1024):

    size = (size, size)  # convert to tuple

    # collect imagery tif file paths
    tif_files = glob.glob(path_pattern, recursive=True)

    # collect corresponding masks tif file paths
    masks_files = []
    for tif_file_path in tif_files:
        base_dir = os.path.dirname(tif_file_path)
        masks_files.append(os.path.join(base_dir, "mask.tif"))

    print(f"Generating masks with size {size}...")
    dst_image_dir = os.path.join(dst_dir, "images")
    dst_mask_dir = os.path.join(dst_dir, "masks")
    os.makedirs(dst_image_dir, exist_ok=True)
    os.makedirs(dst_mask_dir, exist_ok=True)

    i = 0
    for tif_path, mask_path in zip(tif_files, masks_files):
        with rasterio.open(tif_path) as image_src, rasterio.open(mask_path) as mask_src:
            city = tif_path.split("/")[-3]
            name = tif_path.split("/")[-2]
            for image_data, mask_data in zip(generate_samples(image_src, *size), generate_samples(mask_src, *size)):
                image, image_profile = image_data
                mask, mask_profile = mask_data

                image = image[:3]  # take only 3 RGB channels
                if image.sum() > 100:  # prevent empty masks
                    i += 1
                    dst_name = "{}_{}_{}.tif".format(city, name, str(i).zfill(5))
                    dst_image_path = os.path.join(dst_image_dir, dst_name)
                    dst_mask_path = os.path.join(dst_mask_dir, dst_name)

                    print(f"Saving: {dst_image_path}")
                    save_raster(dst_image_path, image, **image_profile)
                    print(f"Saving: {dst_mask_path}")
                    save_raster(dst_mask_path, mask, **mask_profile)
                    print()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_pattern", type=str, required=True)
    parser.add_argument("--dst_dir", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=1024)
    args = parser.parse_args()
    
    main(path_pattern=args.path_pattern, dst_dir=args.dst_dir, size=args.sample_size)
