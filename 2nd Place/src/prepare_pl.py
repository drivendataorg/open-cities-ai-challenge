import argparse
from pathlib import Path
import multiprocessing

import pandas as pd
import tqdm

from utils import read_img_cv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='./data/test_tiles_1024')
    parser.add_argument('--preds-path', type=str, required=True)
    
    return parser.parse_args()


def process_img(row):
    _, item = row

    return item.fname, read_img_cv(item.mask_path).any()


def main():
    args = parse_args()
    print(args)
    
    data_path = Path(args.data_path)
    img_paths = list(data_path.glob('*.jpg'))
    
    test = pd.DataFrame(img_paths, columns=['fname'])
    preds_path = Path(args.preds_path)
    test['mask_path'] = test.fname.apply(lambda p: preds_path / f'{p.stem}.TIF')
    
    with multiprocessing.Pool(32) as p:
        with tqdm.tqdm(list(test.iterrows())) as pbar:
            res = dict(p.imap_unordered(func=process_img, iterable=pbar))
            
    res = pd.DataFrame.from_dict(res, orient='index', columns=['has_mask']).reset_index().rename({'index': 'fname'}, axis=1)
    test = pd.merge(test, res, how='left', on='fname')
    test['area'] = 'mon'
    test['scene_id'] = 'f15272'
    save_path = preds_path.parent / f'{preds_path.name}.csv'
    test.to_csv(save_path, index=False)

            
if __name__ == '__main__':
    main()
