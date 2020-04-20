import argparse
from pathlib import Path

import cv2
import numpy as np
from pystac import Catalog
import rasterio
import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--win-sz', type=int, default=1024)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    data_path = Path(args.data_path)
    
    to_save = data_path / f'test_tiles_{args.win_sz}'
    if not to_save.exists():
        to_save.mkdir(parents=True)

    test_cat = Catalog.from_file(str(data_path / 'test' / 'catalog.json'))
    with tqdm.tqdm(test_cat.get_items()) as pbar:
        for one_item in pbar:
            rst = rasterio.open(one_item.make_asset_hrefs_absolute().assets['image'].href)
            win_arr = rst.read()
            win_arr = np.transpose(win_arr, (1, 2, 0))[..., :3]
            win_arr = win_arr[..., ::-1]
            res = cv2.imwrite(str(to_save / f'{one_item.id}.jpg'), win_arr)
            assert res, f'Cannot write image {one_item.id}'
            rst.close()

            
if __name__ == '__main__':
    main()
