import os
import glob
import argparse


def resample(src_path: str, dst_path: str, dst_res: float):
    command = f"gdalwarp -co COMPRESS=JPEG -co TILED=YES -co BLOCKXSIZE=256 -co BLOCKYSIZE=256 -co NUM_THREADS=ALL_CPUS -r bilinear " + \
                   f"-tr {dst_res} -{dst_res} {src_path} {dst_path}"
    os.system(command)

def main(path_pattern: str, dst_res: float = 0.1):
    
    prefix = "res-"

    # collect all tif files
    tif_files = glob.glob(path_pattern, recursive=True)
    print(path_pattern)
    print(f"Num files: {len(list(tif_files))}")

    # resample all tif files
    for src_path in tif_files:
        file_name = os.path.basename(src_path)
        directory = os.path.dirname(src_path)
        dst_path = os.path.join(directory, prefix + file_name)

        print(f"Resampling {src_path} -> {dst_path}")
        resample(src_path, dst_path, dst_res)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_pattern", type=str, required=True)
    parser.add_argument("--dst_res", type=float, required=True)
    args = parser.parse_args()
    
    main(path_pattern=args.path_pattern, dst_res=args.dst_res)