import os
import glob
import argparse
import rasterio
import numpy as np

from .cut_train import sample, generate_samples, save_raster


def main(images_dir, masks_dir, dst_dir, size=1024):
    
    size = (size, size)

    # collect images file paths
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.tif")))
    mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.tif")))

    # prepare output dirs
    dst_image_dir = os.path.join(dst_dir, "images")
    dst_mask_dir = os.path.join(dst_dir, "masks")
    os.makedirs(dst_image_dir, exist_ok=True)
    os.makedirs(dst_mask_dir, exist_ok=True)

    print(f"Generating masks with size {size}...")

    i = 0
    for image_path, mask_path in zip(image_paths, mask_paths):

        # check thak image and mask names the same
        assert os.path.basename(image_path) == os.path.basename(mask_path)

        # open rasters and generate samples according to specified size
        with rasterio.open(image_path) as image_src, \
             rasterio.open(mask_path) as mask_src: 

            # check dimentions is the same
            assert image_src.height == mask_src.height
            assert image_src.width == mask_src.width

            name = mask_path.split("/")[-1][:-4]  # remove extension
            for image_data, mask_data in zip(generate_samples(image_src, *size), generate_samples(mask_src, *size)):
                image, image_profile = image_data
                mask, mask_profile = mask_data

                image = image[:3]  # take only 3 RGB channels
                if image.sum() > 100:  # prevent empty (only nodata) images and masks
                    i += 1
                    dst_name = "{}_{}.tif".format(name, str(i).zfill(5))
                    dst_image_path = os.path.join(dst_image_dir, dst_name)
                    dst_mask_path = os.path.join(dst_mask_dir, dst_name)

                    print(f"Saving: {dst_image_path}")
                    save_raster(dst_image_path, image, **image_profile)
                    print(f"Saving: {dst_mask_path}")
                    save_raster(dst_mask_path, mask, NBITS=1, **mask_profile)
                    print()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--masks_dir", type=str, required=True)
    parser.add_argument("--dst_dir", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=1024)
    args = parser.parse_args()
    
    main(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        dst_dir=args.dst_dir,
        size=args.sample_size,
    )
