import os
import glob
import shutil
import rasterio
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from rasterio.windows import Window


def read_raster(path):
    """Read raster and profile"""
    with rasterio.open(path) as src:
        return src.read(), src.profile


def get_max(data):
    """Get max indices for x and y axes"""
    x = max([v[0] for v in data.values()])
    y = max([v[1] for v in data.values()])
    return x, y


def get_cluster_tiles(cluster_id, df):
    """Select cluster tiles"""
    df = df[df.cluster_id == cluster_id]
    data = {}
    for i, row in df.iterrows():
        data[row.id] = (row.x, row.y)
    return data


def get_cluster_tile_size(cluster_id, df):
    """Select tile size for specified cluster id"""
    df = df[df.cluster_id == cluster_id]
    return df.iloc[0]["tile_size"]


def create_raster(dst_path, data, id_to_path, step=1, tile_size=1024, dtype="uint8", count=1, resolution=0.1, **kwargs):
    """Create stitched tif file from tiles"""
    assert tile_size % step == 0
    
    max_x, max_y = get_max(data)
    h = (max_y + 1) * tile_size // step
    w = (max_x + 1) * tile_size // step
    profile = dict(
        driver='GTiff',
        nodata=0,
        count=count,
        height=h,
        width=w,
        dtype=dtype,
        crs="EPSG:3857",  # just to make it not empty provide fake crs
        transform=[step * resolution, 0, 0, 0, - step * resolution, 0],  # provide fake transform
    )
    profile.update(kwargs)

    # create big raster on disk and start writing each tile by window writing
    with rasterio.open(dst_path, "w", **profile) as dst:
        with tqdm(data.items()) as data_items:
            for id, position in data_items:
                raster, _ = read_raster(id_to_path[id])
                raster = raster[:3, ::step, ::step]
                x_start = position[0] * tile_size // step
                y_start = position[1] * tile_size // step
                dst.write(raster, window=Window(x_start, y_start, tile_size // step, tile_size // step))


def main(df_path, path_pattern, dst_dir):

    # read data about tile positions
    df = pd.read_csv(df_path)

    # collect test data file paths
    tile_paths = glob.glob(path_pattern)

    # create output dir
    stitched_dst_dir = os.path.join(dst_dir, "stitched")  # dir for stitched test tiles
    sliced_dst_dir = os.path.join(dst_dir, "sliced")  # dir for not stiched test tiles
    
    os.makedirs(stitched_dst_dir, exist_ok=True)
    os.makedirs(sliced_dst_dir, exist_ok=True)
    
    id_to_path = {os.path.basename(p):p for p in tile_paths}
    cluster_ids = df.cluster_id.unique()

    # create stitced files
    for cluster_id in cluster_ids:
        if cluster_id == -1:  # -1 is not clustered tiles
            continue
    
        dst_path = os.path.join(stitched_dst_dir, f"{cluster_id}.tif".zfill(7))
        data = get_cluster_tiles(cluster_id, df)
        tile_size = get_cluster_tile_size(cluster_id, df)

        print(f"Creating stitched cluster #{cluster_id}")
        create_raster(
            dst_path, data, id_to_path, step=1, count=3, resolution=0.1, 
            tile_size=tile_size, tiled=True, blockxsize=256, blockysize=256, BIGTIFF='IF_NEEDED',
        )

    # copy not stitched files
    ids = df[df.cluster_id == -1].id.values
    for image_id in ids:
        src_path = id_to_path[image_id]
        dst_path = os.path.join(sliced_dst_dir, image_id)
        shutil.copy(src_path, dst_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--df_path", type=str, required=True)
    parser.add_argument("--path_pattern", type=str, required=True)
    parser.add_argument("--dst_dir", type=str, required=True)
    args = parser.parse_args()
    
    main(df_path=args.df_path, path_pattern=args.path_pattern, dst_dir=args.dst_dir)