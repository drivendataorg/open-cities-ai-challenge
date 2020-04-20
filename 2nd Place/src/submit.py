import argparse
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import tqdm
import PIL
import PIL.Image as Image

from dataset import dev_transform
from utils import read_img


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/test_tiles_1024',
                        help='Path to data')
    parser.add_argument('--exp', type=str, required=True,
                        help='Path to models checkpoints in jit format')
    parser.add_argument('--to-save', type=str, required=True,
                        help='Folder path to save test masks')
    
    parser.add_argument('--n-parts', type=int, default=1)
    parser.add_argument('--part', type=int, default=0)
    
    parser.add_argument('--res', type=int, default=512,
                        help='Image resolution')
    parser.add_argument('--batch-size', type=int, default=32)

    return parser.parse_args()


class DS(torch.utils.data.Dataset):
    def __init__(self, imgs, root):
        self.imgs = imgs
        self.root = root

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_id = self.imgs[index]
        
        img = read_img(self.root / img_id)
        mask = np.zeros_like(img)[..., [0]]

        return dev_transform(img, mask)[0], img_id


def collate(x):
    x, y = list(zip(*x))

    return torch.stack(x), y


def main():
    args = parse_args()
    path_to_data = Path(args.data)
    print(args)

    test_anns = [p.name for p in (path_to_data).glob('*.jpg')]
    
    n = len(test_anns)
    k = n//args.n_parts
    start = args.part*k
    end = k*(args.part + 1) if args.part + 1 != args.n_parts else n
    test_anns = test_anns[start:end]
    print(f'test size: {len(test_anns)}')
    
    batch_size = args.batch_size
    batch_size = batch_size if args.res != 1024 else batch_size//4
    ds = DS(test_anns, path_to_data)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        collate_fn=collate,
        pin_memory=True,
    )

    models = [
        torch.jit.load(str(p)).cuda().eval()
        for p in Path(args.exp).rglob('*.pt')
    ]
    print(f'#models: {len(models)}')
    n_models = len(models)
    tta = 4
    n_augs = n_models * tta
    
    to_save = Path(args.to_save)
    print(f'save path: {to_save}')
    if not to_save.exists():
        to_save.mkdir(parents=True)

    def get_submit(thresh_dice=None):
        masks = torch.zeros((batch_size, 1 if thresh_dice is not None else 2, args.res, args.res), dtype=torch.float32, device='cuda')
        with torch.no_grad():
            with tqdm.tqdm(loader, mininterval=2) as pbar:
                for img, anns in pbar:
                    bs = img.size(0)
                    img = img.cuda()

                    masks.zero_()
                    for model in models:
                        mask = model(img)
                        if thresh_dice is not None:
                            masks[:bs] += torch.sigmoid(mask)
                        else:
                            masks[:bs] += torch.softmax(mask, dim=1)

                        # vertical flip
                        if tta > 1:
                            mask = model(torch.flip(img, dims=[-1]))
                            if thresh_dice is not None:
                                masks[:bs] += torch.flip(torch.sigmoid(mask), dims=[-1])
                            else:
                                masks[:bs] += torch.flip(torch.softmax(mask, dim=1), dims=[-1])

                        # horizontal flip
                        if tta > 2:
                            mask = model(torch.flip(img, dims=[-2]))
                            if thresh_dice is not None:
                                masks[:bs] += torch.flip(torch.sigmoid(mask), dims=[-2])
                            else:
                                masks[:bs] += torch.flip(torch.softmax(mask, dim=1), dims=[-2])

                        if tta > 3:
                            # vertical + horizontal flip
                            mask = model(torch.flip(img, dims=[-1, -2]))
                            if thresh_dice is not None:
                                masks[:bs] += torch.flip(torch.sigmoid(mask), dims=[-1, -2])
                            else:
                                masks[:bs] += torch.flip(torch.softmax(mask, dim=1), dims=[-1, -2])

                    masks /= n_augs
                    for mask, annotation in zip(masks, anns):
                        if thresh_dice is None:
                            mask = mask.argmax(0)[None]

                        for cls, m in enumerate(mask):
                            if thresh_dice is not None:
                                m = m > thresh_dice

                            m = m.cpu().numpy().astype('float32')
                            if args.res != 1024:
                                m = A.Resize(1024, 1024)(image=np.zeros((512, 512, 3), dtype='uint8'),
                                                         mask=m)['mask']

                            m = Image.fromarray(((m*255).astype('uint8')/255).astype('uint8'))
                            m.save(str(to_save / annotation.replace('.jpg', '.TIF')), compression='tiff_deflate')

    get_submit()


if __name__ == '__main__':
    main()
