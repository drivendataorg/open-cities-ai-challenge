import argparse
from pathlib import Path

import apex
import numpy as np
import segmentation_models_pytorch as smp
import torch
import tqdm
import pandas as pd

from dataset import CloudsDS, dev_transform, collate_fn
from metric import Dice, JaccardMicro
from utils import get_data_groups


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/train_tier_1_tiles',
                        help='Path to data')
    parser.add_argument('--load', type=str, required=True,
                        help='Load model')
    parser.add_argument('--save', type=str, default='',
                        help='Save predictions')
    parser.add_argument('--tta', type=int, default=0,
                        help='Test time augmentations')

    return parser.parse_args()
    

def epoch_step(loader, desc, model, metrics):
    model.eval()

    pbar = tqdm.tqdm(total=len(loader), desc=desc, leave=False, mininterval=2)
    loc_targets, loc_preds = [], []
    if args.cls:
        loc_preds_cls = []
    
    
    for x, y in loader:
        x, y = x.cuda(args.gpu), y.cuda(args.gpu).float()
        
        masks = []
        if args.cls:
            clsss = []

        logits = model(x)
        if args.cls:
            logits, cls = logits
            clsss.append(torch.sigmoid(cls).cpu().numpy())
        if args.n_classes == 1:
            masks.append(torch.sigmoid(logits).cpu().numpy())
        else:
            masks.append(torch.softmax(logits, dim=1).cpu().numpy())
        
        if args.tta > 0:
            logits = model(torch.flip(x, dims=[-1]))
            if args.cls:
                logits, cls = logits
                clsss.append(torch.sigmoid(cls).cpu().numpy())
            if args.n_classes == 1:
                masks.append(torch.flip(torch.sigmoid(logits), dims=[-1]).cpu().numpy())
            else:
                masks.append(torch.flip(torch.softmax(logits, dim=1), dims=[-1]).cpu().numpy())

        if args.tta > 1:
            logits = model(torch.flip(x, dims=[-2]))
            if args.cls:
                logits, cls = logits
                clsss.append(torch.sigmoid(cls).cpu().numpy())
            if args.n_classes == 1:
                masks.append(torch.flip(torch.sigmoid(logits), dims=[-2]).cpu().numpy())
            else:
                masks.append(torch.flip(torch.softmax(logits, dim=1), dims=[-2]).cpu().numpy())

        if args.tta > 2:
            logits = model(torch.flip(x, dims=[-1, -2]))
            if args.cls:
                logits, cls = logits
                clsss.append(torch.sigmoid(cls).cpu().numpy())
            if args.n_classes == 1:
                masks.append(torch.flip(torch.sigmoid(logits), dims=[-1, -2]).cpu().numpy())
            else:
                masks.append(torch.flip(torch.softmax(logits, dim=1), dims=[-1, -2]).cpu().numpy())

        trg = y.cpu().numpy()
        loc_targets.extend(trg)
        preds = np.mean(masks, 0)
        loc_preds.extend(preds)
    
        for metric in metrics.values():
            metric.update(preds, trg)
        
        if args.cls:
            loc_preds_cls.extend(np.mean(clsss, 0))

        torch.cuda.synchronize()

        if args.local_rank == 0:
            pbar.set_postfix(**{
                k: f'{metric.evaluate():.4}' for k, metric in metrics.items()
            })
            pbar.update()

    pbar.close()
    
    if args.cls:
        return loc_targets, loc_preds, loc_preds_cls
    
    return loc_targets, loc_preds

    
def main():
    global args
    
    args = parse_args()
    print(args)

    torch.backends.cudnn.benchmark = True

#     args.distributed = False
#     if 'WORLD_SIZE' in os.environ:
#         args.distributed = int(os.environ['WORLD_SIZE']) > 1
    
    args.gpu = 0
#     args.world_size = 1
#     if args.distributed:
#         args.gpu = args.local_rank
#         torch.cuda.set_device(args.gpu)
#         torch.distributed.init_process_group(backend='nccl',
#                                              init_method='env://')
#         args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, 'Amp requires cudnn backend to be enabled.'
    
    to_save = args.save
    path_to_load = Path(args.load)
    if path_to_load.is_file():
        print(f"=> Loading checkpoint '{path_to_load}'")
        checkpoint = torch.load(path_to_load, map_location=lambda storage, loc: storage.cuda(args.gpu))
        print(f"=> Loaded checkpoint '{path_to_load}'")
    else:
        raise

    tta = args.tta
    args = checkpoint['args']
    args.tta = tta
    print(args)

    n_classes = args.n_classes
    if args.cls:
        print('With classification')
    else:
        print('Without classification')

    model = smp.Unet(encoder_name=args.encoder,
                     encoder_weights='imagenet' if 'dpn92' not in args.encoder else 'imagenet+5k',
                     classes=n_classes,
                     decoder_attention_type=args.attention_type,
                     activation='sigmoid',)
    
#     if args.sync_bn:
#         print('using apex synced BN')
#         model = apex.parallel.convert_syncbn_model(model)
        
    model.cuda()
     
    # Initialize Amp. Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    assert args.fp16 == False, "torch script doesn't work with amp"
    if args.fp16:
        model = apex.amp.initialize(model,
                                    opt_level=args.opt_level,
                                    keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                    loss_scale=args.loss_scale
                                   )
    
    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
#     if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with 
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
#         model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
        
    work_dir = path_to_load.parent
    
    import copy
    state_dict = copy.deepcopy(checkpoint['state_dict'])
    for p in checkpoint['state_dict']:
        if p.startswith('module.'):
            state_dict[p[len('module.'):]] = state_dict.pop(p)
#             state_dict[p.replace('module.', '')] = state_dict.pop(p)
    
    model.load_state_dict(state_dict)
    
    x = torch.rand(2, 3, 512, 512).cuda()
    model = model.eval()
    if 'efficientnet' in args.encoder:
        model.encoder.set_swish(memory_efficient=False)
    with torch.no_grad():
        traced_model = torch.jit.trace(model, x)

    traced_model.save(str(work_dir / f'model_{path_to_load.stem}.pt'))
    del traced_model
    del model
    
    return
    
    path_to_data = Path(args.data)
    train_gps, dev_gps = get_data_groups(path_to_data / 'train_512_kds.csv', args)  #  / 'train.csv', args)

    dev_gps = pd.concat([train_gps, dev_gps]).reset_index(drop=True)
    dev_gps = dev_gps[dev_gps.is_test].copy()
    
    dev_ds = CloudsDS(dev_gps, root=path_to_data, transform=dev_transform)
#     dev_sampler = None
#     if args.distributed:
#         dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_ds)
        
    batch_size = args.batch_size
    dev_loader = torch.utils.data.DataLoader(dev_ds,
                                             batch_size=max(min(batch_size, 32), 32),
                                             shuffle=False,
                                             sampler=None,
                                             num_workers=4,
                                             collate_fn=collate_fn,
                                             pin_memory=True)

    if n_classes == 1:
        metrics = {
            'dice_4': Dice(n_classes=n_classes, thresh=0.4),
            'dice': Dice(n_classes=n_classes, thresh=0.5),
            'dice_6': Dice(n_classes=n_classes, thresh=0.6),
        }
    else:
        metrics = {
            'score': JaccardMicro(n_classes=n_classes, thresh=None),
            'jaccard': Dice(n_classes=n_classes, thresh=None),
        }
    
    model = torch.jit.load(str(work_dir / f'model_{path_to_load.stem}.pt')).cuda().eval()
        
    with torch.no_grad():
        for metric in metrics.values():
            metric.clean()

        trgs, preds, *preds_cls = epoch_step(dev_loader, f'[ Validating dev.. ]',
                                             model=model,
                                             metrics=metrics)
        for key, metric in metrics.items():
            print(f'{key} dev {metric.evaluate()}')

    if str(to_save) == '': #or n_classes > 1:
        return

    to_save = Path(to_save)
    to_save1 = to_save / 'pred_masks_tta'
    if not to_save1.exists():
        to_save1.mkdir(parents=True)

    if args.cls:
        to_save2 = to_save / 'pred_clss_tta'
        if not to_save2.exists():
            to_save2.mkdir(parents=True)

        with tqdm.tqdm(zip(dev_gps.iterrows(), preds, preds_cls[0]), total=len(preds)) as pbar:
            for (_, row), p1, p2 in pbar:
                np.save(to_save1 / row.fname, p1)
                np.save(to_save2 / row.fname, p2)
    else:
        with tqdm.tqdm(zip(dev_gps.iterrows(), preds), total=len(preds)) as pbar:
            for (_, row), p1 in pbar:
                np.save(to_save1 / row.fname.split('/')[-1], p1)


if __name__ == '__main__':
    main()
