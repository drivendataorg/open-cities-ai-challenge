import os
import glob
import json
import argparse
import rasterio
import numpy as np
import aeronet.dataset as ds

from rasterio.features import geometry_mask


def gj_correct_crs(path: str):
    "Make GeoJSON file crs correct (fix error reading file)"
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    gj["crs"] = "EPSG:4326"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(gj, f)


def read_profile(path: str):
    "Read tif file profile (see rasterio documentation about profile)"
    with rasterio.open(path) as src:
        return src.profile


def rasterize(geometries, transform, shape):
    """Convert geometries to raster mask"""
    if len(geometries) > 0:
        mask = geometry_mask(geometries, out_shape=shape, transform=transform, invert=True).astype('uint8')
    else:
        mask = np.zeros(shape, dtype='uint8')
    return mask


def save_raster(path, raster, crs, transform):
    """Write raster to file"""
    profile = dict(
        driver="GTiff",
        height=raster.shape[0],
        width=raster.shape[1],
        count=1,
        crs=crs,
        transform=transform,
        dtype=raster.dtype,
        NBITS=1,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(raster, 1)


## prepare train data
def main(path_pattern):

    # collect tif file paths
    tif_files = glob.glob(path_pattern, recursive=True)
    print(f"Collected {len(list(tif_files))} files.")

    # collect geojson file paths
    geojson_files = []
    for tif_file_path in tif_files:
        base_dir = os.path.dirname(tif_file_path)
        dir, name = os.path.split(base_dir)
        geojson_dir_name = name + "-labels"
        geojson_file_name = name + ".geojson"
        path = os.path.join(dir, geojson_dir_name, geojson_file_name)
        assert os.path.exists(path)
        geojson_files.append(path)

    # generate raster mask for each geojson
    print("Generating masks...")
    for tif_path, geojson_path in zip(tif_files, geojson_files):

        # make destination file path
        dst_dir = os.path.dirname(tif_path)
        dst_path = os.path.join(dst_dir, "mask.tif")
        #print(dst_path)

        # read Feature Collection with geometries and reproject ot same projection as tif file
        profile = read_profile(tif_path)
        gj_correct_crs(geojson_path)
        fc = ds.FeatureCollection.read(geojson_path)
        fc = fc.reproject(profile["crs"])

        # generate raster from geometries
        geometries = [f.geometry for f in fc]
        mask = rasterize(
            geometries,
            transform=profile["transform"],
            shape=(profile["height"], profile["width"]),
        )

        # save raster on disk
        save_raster(dst_path, mask, crs=profile["crs"], transform=profile["transform"])


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_pattern", type=str, required=True)
    args = parser.parse_args()
    
    main(path_pattern=args.path_pattern)