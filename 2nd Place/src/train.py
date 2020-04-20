import argparse
from pathlib import Path
import random
import os

import apex
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm
import segmentation_models_pytorch as smp

from dataset import CloudsDS, train_transform, dev_transform, collate_fn, DistributedWeightedSampler
from lovasz_losses import symmetric_lovasz, lovasz_softmax
from metric import Dice, JaccardMicro
from utils import get_data_groups


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/train_tier_1_tiles',
                        help='Path to data')
    parser.add_argument('--csv', type=str, default='train_512_kds.csv')
    parser.add_argument('--csv2', type=str, default=None)
    
    parser.add_argument('--work-dir', default='', type=str,
                        help='Working directory')
    parser.add_argument('--load', default='', type=str,
                        help='Load model (default: none)')
    parser.add_argument('--resume', default='', type=str,
                        help='Path to latest checkpoint (default: none)')
    parser.add_argument('--encoder', type=str, default='efficientnet-b7')
    parser.add_argument('--cls', action='store_true')
    parser.add_argument('--n-classes', type=int, default=1)
    parser.add_argument('--attention-type', type=str, default=None)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--mixup', type=float, default=-1)
    parser.add_argument('--lovasz', type=int, default=25)
    
    parser.add_argument('--zoom', type=int, default=0)
    parser.add_argument('--w3m', action='store_true')
    parser.add_argument('--t2', action='store_true')
    parser.add_argument('--hm', action='store_true')
    
    parser.add_argument('--pl', type=str, default=None)  # './data/test_sub_eff5_512_sm_kds_f012.csv'
    parser.add_argument('--ft', action='store_true')
    
    parser.add_argument('--teachers', type=str, default=None)
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--temperature', type=float, default=20)

    parser.add_argument('--workers', '-j', type=int, default=4, required=False)

    parser.add_argument('--epochs', '-e', type=int, default=300)
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', '-b', type=int, default=8,
                        help='Batch size per process (default: 8)')
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--lr', '--learning-rate', type=float, default=1e-4,
                        metavar='LR',
                        help='Initial learning rate. Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256. A warmup schedule will also be applied over the first 5 epochs.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--scheduler', type=str, default='cos')
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--T-max', type=int, default=5)

    parser.add_argument('--seed', type=int, default=314159,
                        help='Random seed')
    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--sync-bn', action='store_true',
                        help='Enabling apex sync BN.')

    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--opt-level', type=str, default='O1')
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
def loss_fn_kd(outputs, labels, teacher_outputs):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = args.alpha
    T = args.temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)  + \
              nn.CrossEntropyLoss()(outputs, labels) * (1. - alpha)

    return KD_loss


def epoch_step(loader, desc, model, criterion, metrics, opt=None, batch_accum=1, teachers=None):
    is_train = opt is not None
    if is_train:
        model.train()
    else:
        model.eval()
    use_mixup = (args.mixup > 0) and is_train

    pbar = tqdm.tqdm(total=len(loader), desc=desc, leave=False, mininterval=2)
    loc_loss = n = 0
    loc_accum = 1
    if teachers is not None:
        loc_loss_t = 0

    for x, y in loader:
        x, y = x.cuda(args.gpu, non_blocking=True), y.cuda(args.gpu, non_blocking=True).float()

        if use_mixup:
            x, y_a, y_b, lam = mixup_data(x, y, args.mixup)
            logits = model(x)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)/batch_accum
        else:
            logits = model(x)
            loss = criterion(logits, y)/batch_accum

        if is_train:
            if teachers is not None and args.alpha != 1:
                alpha = args.alpha
                T = args.temperature
                logits_over_T = logits/T
                loss_t = 0
                for teacher in teachers:
                    with torch.no_grad():
                        logits_t = teacher(x)
                    loss_t += nn.KLDivLoss(reduction='batchmean')(torch.log_softmax(logits_over_T, dim=1), torch.softmax(logits_t/T, dim=1))
#                 loss_t = sum(nn.KLDivLoss(reduction='batchmean')(torch.log_softmax(logits_over_T, dim=1), torch.softmax(teacher(x)/T, dim=1))
#                              for teacher in teachers)
                if alpha == 0:
                    loss = loss_t/len(teachers)/batch_accum
                else:
                    loss = alpha*loss + (1 - alpha)*T*T*loss_t/len(teachers)/batch_accum
            
            if args.fp16:
                with apex.amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if loc_accum == batch_accum:
                opt.step()
                opt.zero_grad()
                loc_accum = 1
            else:
                loc_accum += 1

            if args.cls:
                logits, _ = logits

            logits = logits.detach()
        elif args.cls:  # inference
            logits, _ = logits

        bs = x.size(0)
        loc_loss += loss.item() * bs * batch_accum
        if teachers is not None:
            loc_loss_t += loss_t.item() * bs * batch_accum

        n += bs

        y_cpu_np = y.cpu().numpy()
        if args.n_classes > 1:
            logits_cpu_np = logits.cpu().numpy()
        else:
            logits_cpu_np = torch.sigmoid(logits).cpu().numpy()

        for metric in metrics.values():
            metric.update(logits_cpu_np, y_cpu_np)

        torch.cuda.synchronize()

        if args.local_rank == 0:
            postfix = {
                'loss': f'{loc_loss / n:.3f}',
            }
            if teachers is not None:
                postfix.update({'loss_t': f'{loc_loss_t/n:.3f}'})

            postfix.update({k: f'{metric.evaluate():.3f}' for k, metric in metrics.items()})
            if is_train:
                postfix.update({'lr': next(iter(opt.param_groups))['lr']})
            pbar.set_postfix(**postfix)
            pbar.update()

    if is_train and loc_accum != batch_accum:
        opt.step()
        opt.zero_grad()

    pbar.close()

    return loc_loss / n


def plot_hist(history, path):
    history_len = len(history)
    n_rows = history_len // 2 + 1
    n_cols = 2
    plt.figure(figsize=(12, 4 * n_rows))
    for i, (m, vs) in enumerate(history.items()):
        plt.subplot(n_rows, n_cols, i + 1)
        for k, v in vs.items():
            if 'loss' in m:
                ep = np.argmin(v)
            else:
                ep = np.argmax(v)
            plt.title(f'{v[ep]:.4} on {ep}')
            plt.plot(v, label=f'{k} {v[-1]:.4}')

        plt.xlabel('#epoch')
        plt.ylabel(f'{m}')
        plt.legend()
        plt.grid(ls='--')

    plt.tight_layout()
    plt.savefig(path / 'evolution.png')
    plt.close()


def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    bs = x.size(0)
    index = torch.randperm(bs)
    mixed_x = lam * x + (1 - lam) * x[index, :]

    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


def add_weight_decay(model, weight_decay=1e-4, skip_list=('bn',)):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}
    ]


def main():
    global args

    args = parse_args()
    print(args)

    torch.backends.cudnn.benchmark = True
    if args.deterministic:
        set_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_printoptions(precision=10)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, 'Amp requires cudnn backend to be enabled.'

    # create model
    n_classes = args.n_classes
    if args.cls:
        print('With classification')
    else:
        print('Without classification')

    model = smp.Unet(encoder_name=args.encoder,  # Unet
                     encoder_weights='imagenet' if 'dpn92' not in args.encoder else 'imagenet+5k',
                     classes=n_classes,
                     decoder_attention_type=args.attention_type,
                     activation='sigmoid', )

    if args.sync_bn:
        print('using apex synced BN')
        model = apex.parallel.convert_syncbn_model(model)

    model.cuda()

    # Scale learning rate based on global batch size
    print(f'lr={args.lr}, opt={args.opt}')
    if args.opt == 'adam':
        opt = apex.optimizers.FusedAdam(model.parameters(),  # add_weight_decay(model, args.weight_decay, ('bn', )),
                                        lr=args.lr,
                                        weight_decay=args.weight_decay,
                                        )
    elif args.opt == 'sgd':
        opt = torch.optim.SGD(add_weight_decay(model, args.weight_decay, ('bn',)),  # model.parameters(),
                              args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay
                              )
    else:
        raise

    # Initialize Amp. Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    if args.fp16:
        model, opt = apex.amp.initialize(model, opt,
                                         opt_level=args.opt_level,
                                         keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                         loss_scale=args.loss_scale
                                         )

    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with 
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)

    jac_loss = smp.utils.losses.JaccardLoss()
    dice_loss = smp.utils.losses.DiceLoss()
    bce_loss = nn.BCEWithLogitsLoss()
    if args.cls:
        def BCEBCE(logits, target):
            prediction_seg, prediction_cls = logits
            y_cls = (target.sum([2, 3]) > 0).float()

            return bce_loss(prediction_seg, target) + bce_loss(prediction_cls, y_cls)

        def symmetric_lovasz_fn(logits, target):
            prediction_seg, prediction_cls = logits
            y_cls = (target.sum([2, 3]) > 0).float()

            return symmetric_lovasz(prediction_seg, target) + bce_loss(prediction_cls, y_cls)
    else:
        if n_classes == 1:
            def BCEBCE(logits, target):
                #             return bce_loss(logits, target)
                #             return jac_loss(logits, target) + bce_loss(logits, target)

                return dice_loss(logits, target) + bce_loss(logits, target)

            symmetric_lovasz_fn = symmetric_lovasz
        else:
            def BCEBCE(logits, target):
                return nn.CrossEntropyLoss()(logits, target[:, 0].long())

            def symmetric_lovasz_fn(logits, target):
                return lovasz_softmax(torch.softmax(logits, dim=1), target[:, 0].long())

    criterion = BCEBCE

    history = {
        k: {k_: [] for k_ in ['train', 'dev']}
        for k in ['loss']
    }
    best_score = 0
    if n_classes == 1:
        metrics = {
            'j6': JaccardMicro(n_classes=n_classes, thresh=0.6, w3m=args.w3m),
            'score': JaccardMicro(n_classes=n_classes, thresh=0.55, w3m=args.w3m),
            'j5': JaccardMicro(n_classes=n_classes, thresh=0.5, w3m=args.w3m),
            'd55': Dice(n_classes=n_classes, thresh=0.55, w3m=args.w3m),
        }
    else:
        metrics = {
            'score': JaccardMicro(n_classes=2, thresh=None, w3m=args.w3m),
            'jaccard': Dice(n_classes=2, thresh=None, w3m=args.w3m),
        }

    history.update({k: {v: [] for v in ['train', 'dev']} for k in metrics})

    base_name = f'{args.encoder}_b{args.batch_size}_{args.opt}_lr{args.lr}_c{int(args.cls)}_fold{args.fold}_z{args.zoom}_w3m{int(args.w3m)}_t2{int(args.t2)}_hm{int(args.hm)}'
    work_dir = Path(args.work_dir) / base_name
    if args.local_rank == 0 and not work_dir.exists():
        work_dir.mkdir(parents=True)

    # Optionally load model from a checkpoint
    if args.load:
        def _load():
            path_to_load = Path(args.load)
            if path_to_load.is_file():
                print(f"=> loading model '{path_to_load}'")
                checkpoint = torch.load(path_to_load, map_location=lambda storage, loc: storage.cuda(args.gpu))
                model.load_state_dict(checkpoint['state_dict'])
                if args.fp16 and checkpoint['amp'] is not None:
                    apex.amp.load_state_dict(checkpoint['amp'])
                print(f"=> loaded model '{path_to_load}'")
            else:
                print(f"=> no model found at '{path_to_load}'")

        _load()

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def _resume():
            nonlocal history, best_score
            path_to_resume = Path(args.resume)
            if path_to_resume.is_file():
                print(f"=> loading resume checkpoint '{path_to_resume}'")
                checkpoint = torch.load(path_to_resume, map_location=lambda storage, loc: storage.cuda(args.gpu))
                args.start_epoch = checkpoint['epoch'] + 1
                history = checkpoint['history']
                best_score = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                opt.load_state_dict(checkpoint['opt_state_dict'])
                if args.fp16 and checkpoint['amp'] is not None:
                    apex.amp.load_state_dict(checkpoint['amp'])
                print(f"=> resume from checkpoint '{path_to_resume}' (epoch {checkpoint['epoch']})")
            else:
                print(f"=> no checkpoint found at '{args.resume}'")

        _resume()
    history.update({k: {v: [] for v in ['train', 'dev']} for k in metrics if k not in history})

    scheduler = None
    if args.scheduler == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                               T_max=args.T_max,
                                                               eta_min=max(args.lr * 1e-2, 1e-6),
                                                               last_epoch=args.start_epoch if args.resume else -1)

    path_to_data = Path(args.data)
    train_gps, dev_gps = get_data_groups(path_to_data / args.csv, args)  #  / 'train.csv', args)
    
    train_ds = CloudsDS(train_gps, root=path_to_data, transform=train_transform, w3m=args.w3m)
    dev_ds = CloudsDS(dev_gps, root=path_to_data, transform=dev_transform, w3m=args.w3m)

    proba = train_gps.y.value_counts().values
    proba = proba / proba.sum()
    to_weights = dict(zip(train_gps.y.value_counts().index, 1 / proba))
    train_gps['w'] = train_gps.y.apply(lambda x: to_weights[x])
    weights = torch.from_numpy(train_gps.w.values.astype('float32'))

    train_sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
    dev_sampler = None
    if args.distributed:
        train_sampler = DistributedWeightedSampler(train_ds, weights)
#         train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_ds)

    batch_size = args.batch_size
    num_workers = args.workers
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=batch_size,
                                               shuffle=train_sampler is None,
                                               sampler=train_sampler,
                                               num_workers=num_workers,
                                               collate_fn=collate_fn,
                                               pin_memory=True)

    dev_loader = torch.utils.data.DataLoader(dev_ds,
                                             batch_size=20,  # 27
                                             shuffle=False,
                                             sampler=dev_sampler,
                                             num_workers=num_workers,
                                             collate_fn=collate_fn,
                                             pin_memory=True)

    saver = lambda path: torch.save({
        'epoch': epoch,
        'best_score': best_score,
        'history': history,
        'state_dict': model.state_dict(),
        'opt_state_dict': opt.state_dict(),
        'amp': apex.amp.state_dict() if args.fp16 else None,
        'args': args,
    }, path)
    
    teachers = None
    if args.teachers is not None:
        teachers = [
            torch.jit.load(str(p)).cuda().eval()
            for p in Path(args.teachers).rglob('*.pt')
        ]

        if args.distributed:
            for i in range(len(teachers)):
                teachers[i] = apex.parallel.DistributedDataParallel(teachers[i], delay_allreduce=True)

        print(f'#teachers: {len(teachers)}')
    
    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if epoch >= args.lovasz:
            criterion = symmetric_lovasz_fn

        for metric in metrics.values():
            metric.clean()
        loss = epoch_step(train_loader, f'[ Training {epoch}/{args.epochs}.. ]',
                          model=model, criterion=criterion, metrics=metrics, opt=opt, batch_accum=1, teachers=teachers)
        history['loss']['train'].append(loss)
        for k, metric in metrics.items():
            history[k]['train'].append(metric.evaluate())

        if not args.ft:
            with torch.no_grad():
                for metric in metrics.values():
                    metric.clean()
                loss = epoch_step(dev_loader, f'[ Validating {epoch}/{args.epochs}.. ]',
                                  model=model, criterion=criterion, metrics=metrics, opt=None)
                history['loss']['dev'].append(loss)
                for k, metric in metrics.items():
                    history[k]['dev'].append(metric.evaluate())
        else:
            history['loss']['dev'].append(loss)
            for k, metric in metrics.items():
                history[k]['dev'].append(metric.evaluate())

        if scheduler is not None:
            scheduler.step()

        if args.local_rank == 0:
            saver(work_dir / 'last.pth')
            if history['score']['dev'][-1] > best_score:
                best_score = history['score']['dev'][-1]
                saver(work_dir / 'best.pth')

            plot_hist(history, work_dir)


if __name__ == '__main__':
    main()
