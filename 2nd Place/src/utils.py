import pickle

import cv2
import jpeg4py
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def split_df(df, args):
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=1)
    
    df['fold'] = -1
    fold_idx = len(df.columns) - 1

    df['y'] = df.scene_id + '_' + df.area
    
    df_y_vc = df.y.value_counts()
    df = df[df.y.isin(df_y_vc[df_y_vc >= args.n_folds].index)].reset_index(drop=True)

    for i, (_, dev_index) in enumerate(skf.split(range(len(df)), df.y.values)):
        df.iloc[dev_index, fold_idx] = i
        
    return df[df.fold != args.fold].reset_index(drop=True), df[df.fold == args.fold].reset_index(drop=True)


def get_data_groups(path, args):
    if not path.exists():
        root = path.parent
        ims = [str(p) for p in (root / 'images').glob('*.png')]
        df = pd.DataFrame({
            'fname':ims,
            'mask_path':[im.replace('images', 'masks') for im in ims],
        })

        df['scene_id'] = df['fname'].apply(lambda x: x.split("_")[-5])
        df['area'] = df['fname'].apply(lambda x: x.split("_")[-6].split("/")[-1])
        df.to_csv(path, index=False)
        del df

    train = pd.read_csv(path)
    train['is_test'] = False
    
    train, dev = split_df(train.copy(), args)
    
    if args.ft:
        train = pd.concat([train, dev]).reset_index(drop=True)
    
    if args.pl is not None:
        test = pd.read_csv(args.pl)
        test = test[test.has_mask].copy()
        test['y'] = test.scene_id + '_' + test.area
        test['is_test'] = True
        train = pd.concat([train, test]).reset_index(drop=True)
        
    if args.csv2 is not None:
        tier2 = pd.read_csv(path.parent / args.csv2)
        tier2 = tier2[tier2.has_mask].copy()
        tier2['y'] = tier2.scene_id + '_' + tier2.area
        tier2['is_test'] = False
        train = pd.concat([train, tier2]).reset_index(drop=True)
    
    return train, dev


def read_img(path):
    return jpeg4py.JPEG(str(path)).decode()


def read_img_cv(path, is_grey=False):
    if is_grey:
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    return img


def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
