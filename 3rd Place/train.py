'''
Created on Nov 22, 2019

@author: Michal.Busta at gmail.com
'''

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

import numpy as np
from tqdm import tqdm

from dataset import provider_cuts

import time
import random
import neptune

import hydra

import sklearn
import torch.nn.functional as F
import cv2

from meter import Meter
from pathlib import Path
from dataset import CCDatasetCuts, get_training_augmentation
from torch.utils.data import DataLoader

from skimage import data, filters

def predict(X, threshold):
    '''X is sigmoid output of the model'''
    X_p = np.copy(X.cpu())
    preds = (X_p > threshold).astype('uint8')
    return preds

def compute_kaggle_metric(probability_label, truth_label):
    eps = 1e-15
    t = truth_label.reshape(-1,5)
    p = probability_label.reshape(-1,5)
    p = torch.clamp(  p, eps, 1-eps)
    # kaggle score ----------
    log_p = -torch.log(p)

    metric = t*log_p
    metric = metric.sum(1).mean()
    return metric

class FocalLoss(nn.Module):
  def __init__(self, gamma=2.0, alpha=0.25, size_average=True):
    super(FocalLoss, self).__init__()
    self.gamma = gamma
    self.alpha = alpha
    if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
    if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
    self.size_average = size_average

  def forward(self, input, target):
    if input.dim()>2:
      input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
      input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
      input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
    target = target.view(-1,1)

    logpt = F.log_softmax(input, 1)
    logpt = logpt.gather(1,target)
    logpt = logpt.view(-1)
    pt = logpt.data.exp()

    if self.alpha is not None:
      if self.alpha.type()!=input.data.type():
        self.alpha = self.alpha.type_as(input.data)
      at = self.alpha.gather(0,target.data.view(-1))
      logpt = logpt * at

    loss = -1 * (1-pt)**self.gamma * logpt
    if self.size_average: return loss.mean()
    else: return loss.sum()
    
class DiceLoss(nn.Module):
    
  def __init__(self):
    super(DiceLoss, self).__init__()
      
  def forward(self, input, target):
      
    smooth = 1.
    intersection = (input * target).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (input.sum() + target.sum() + smooth))
  
class SoftTargetCrossEntropy(nn.Module):

  def __init__(self, reduce='mean'):
    super(SoftTargetCrossEntropy, self).__init__()
    self.criterion = nn.KLDivLoss(reduction = reduce)
    self.reduce = reduce

  def forward(self, x, target, mask = None): 
    x = F.log_softmax(x, dim=1)   
    if mask is not None:
      loss = self.criterion(x[mask], target[mask])
    else:
      loss = self.criterion(x, target)
    return loss


class Trainer(object):
  '''This class takes care of training and validation of our model'''
  def __init__(self, model, batch_size = {"train": 8, "val": 2}, base_dir='',
               optimizer=None, train_width=512, num_epochs=20, num_workers=1, 
               use_tqdm = True, from_epoch = -1, debug=False, base_threshold = 0.5, post_process=False,
               max_lr=5e-5, fold=0, scale_factor=1.0, do_ohem=True, ref_scale = 0.07, soft_labels_dir = '', 
               use_neptune=False, convert_soft_to_hard=False):
      
    self.num_workers = num_workers
    self.batch_size = batch_size
    self.num_epochs = num_epochs
    self.use_tqdm = use_tqdm
    self.best_loss = np.inf
    self.phases = ["train", "val"]
    self.device = torch.device("cuda:0")
    self.do_ohem = do_ohem
    self.use_neptune = use_neptune

    self.net = model
    self.from_epoch = from_epoch
    self.train_width = train_width
    
    self.criterion = FocalLoss()
    self.criterion_ohem = nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='none')
    self.criterion_cls = FocalLoss()
    self.criterion_dice = DiceLoss()
    self.criterionL1 = nn.L1Loss()
    self.criterion_soft = SoftTargetCrossEntropy()
    self.convert_soft_to_hard = convert_soft_to_hard
    
    self.base_threshold = base_threshold
    self.optimizer = optimizer
    self.debug = debug
    self.post_process = post_process
    
    self.net = self.net.to(self.device)
    cudnn.benchmark = True
    
    self.base_dir = base_dir
    self.soft_labels_dir = soft_labels_dir
    self.batch_size = batch_size
    self.train_width = train_width
    self.fold = fold
    self.ref_scale = ref_scale
    self.dataloaders = {
      phase: provider_cuts(
        base_dir,
        phase,
        batch_size=batch_size[phase],
        num_workers=num_workers,
        train_width = train_width,
        train_height = train_width,
        fold = fold,
        debug = debug,
        scale_factor = scale_factor,
        ref_scale = self.ref_scale,
      )
      for phase in self.phases
    }
    
    if from_epoch > 0:
      self.scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=num_epochs, steps_per_epoch=len(self.dataloaders['train']), last_epoch=from_epoch * len(self.dataloaders['train']))
    else:
      self.scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=num_epochs, steps_per_epoch=len(self.dataloaders['train']), last_epoch=-1)
    
    self.dataloader2 = None  
    if soft_labels_dir is not None and len(soft_labels_dir) > 0:
      print('using weak labels')
      files2 = []
      for file_path in Path(soft_labels_dir).glob('**/*.tif'):
        files2.append(str(file_path))
      for file_path in Path(soft_labels_dir).glob('**/*.jpg'):
        files2.append(str(file_path))
      train_dataset2 = CCDatasetCuts(files2, get_training_augmentation(width=train_width, height=train_width), scale_factor=scale_factor, ref_scale = ref_scale, scales = {}, is_infinite = True)
      self.dataloader2 = DataLoader(
        train_dataset2,
        sampler= None,
        batch_size=batch_size['train'],
        num_workers=1,
        pin_memory=True,
        shuffle=False,
        drop_last=True     
      )
      
  def ohem(self, outputs, targets, outputs2):
      
    o = outputs.reshape(-1, 2)
    t = targets.reshape(-1)
    o2 = outputs2.reshape(-1)
    inst_losses = self.criterion_ohem(o, t.long()) 
    _, idxs = inst_losses.topk(inst_losses.shape[0] * 2 // 3)
    
    cutoff = len(idxs) // 10
    idxs = idxs[cutoff:]
    
                                           
    outputs_hn = o.index_select(0, idxs)                                             
    outputs2_hn = o2.index_select(0, idxs) 
    targets_hn = t.index_select(0, idxs) 
    
    return outputs_hn, targets_hn, outputs2_hn
      
  def forward(self, images, masks, scales, meter, idxs):
    #print(images.shape)
    images = images.to(self.device)
    masks = masks.to(self.device)    
    
    if self.phase == 'train':
      ri2 = random.randint(0, 3)
      if ri2 == 0:
        images = images.permute(0, 1, 3, 2)
        masks = masks.permute(0, 1, 3, 2)
          
    outputs, dice, cls, scl = self.net(images, masks)
    masks = F.interpolate(masks, size=(outputs.shape[2], outputs.shape[3]), mode='bilinear', align_corners=False)
      
    outputs2 = outputs.permute(0, 2, 3, 1)
    masks2 = masks.permute(0, 2, 3, 1)
    dice2 = dice.permute(0, 2, 3, 1)
  
    if self.do_ohem and self.phase == 'train':
      outputs2, masks2, dice2 = self.ohem(outputs2, masks, dice2)
    else:
      outputs2 = outputs2.reshape(-1, 2)
      masks2 = masks.reshape(-1)
      dice2 = dice2.reshape(-1)
      masks2[masks2 < 0.4] = 0
      dice_loss2 = self.criterion_dice(torch.sigmoid(dice2), masks2)
     
    loss = self.criterion(outputs2.reshape(-1, 2), masks2.long().reshape(-1))
    if self.phase == 'val':
      
      outputs_log = F.softmax(outputs, 1)
      outputs_log = outputs_log[:, 1]
      cls = F.softmax(cls, 1)
      dice = torch.sigmoid(dice)
      
      dice[dice >= self.base_threshold] = 1
      dice[dice < self.base_threshold] = 0
      
      olog = outputs_log.cpu().detach().numpy() 
      olog[olog >= self.base_threshold] = 1
      olog[olog < self.base_threshold] = 0
      j1 = sklearn.metrics.jaccard_score(masks2.cpu().detach().numpy().astype(np.int).reshape(-1), olog.astype(np.int).reshape(-1), average='micro')
      j2 = sklearn.metrics.jaccard_score(masks2.cpu().detach().numpy().astype(np.int).reshape(-1), dice.cpu().detach().numpy().astype(np.int).reshape(-1), average='micro')        
      meter.update( j1 = j1, j2 = j2, itr=-1)
    
 
    outputs_log = F.softmax(outputs2, 1)
    outputs_log = outputs_log[:, 1]  
    outputs_log[outputs_log >= self.base_threshold] = 1
    outputs_log[outputs_log < self.base_threshold] = 0
    
    dice_loss = self.criterion_dice(outputs_log.reshape(-1), masks2.reshape(-1))
    
    cls_target = masks.max(1)[0].max(1)[0].max(1)[0]
    cls_loss = self.criterion_cls(cls, cls_target.long())
    
    return loss, dice_loss, dice_loss2, cls_loss
  
  def forward2(self, images, masks, scales, meter, idxs):
    
    
    images = images.to(self.device)
    masks = masks.to(self.device)    
    
    if self.phase == 'train':
      ri2 = random.randint(0, 3)
      if ri2 == 0:
        images = images.permute(0, 1, 3, 2)
        masks = masks.permute(0, 1, 3, 2)
          
    outputs, dice, cls, scl = self.net(images, masks)
    masks = F.interpolate(masks, size=(outputs.shape[2], outputs.shape[3]), mode='bilinear', align_corners=False)
    
    if self.convert_soft_to_hard:
    
      mask2 = (masks < 0.6) * ( masks > 0.4)
      mask2 = ~mask2
      masks[masks > 0.6] = 1
      masks[masks < 0.4] = 0
    
      masks2s = torch.cat([1 - masks, masks], dim=1)
      mask2 = torch.cat([mask2, mask2], dim=1)
      loss = self.criterion_soft(outputs, masks2s, mask2)
    else:
      masks2s = torch.cat([1 - masks, masks], dim=1)
      loss = self.criterion_soft(outputs, masks2s)
    return loss
    
  
  def iterate(self, epoch, phase, max_samples=1000000, flip = 0, transpose=False, use_neptune = True):
    
    start = time.strftime("%H:%M:%S")
    print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
    self.phase = phase
    self.net.train(phase == "train")
    self.net.freeze_bn()
    dataloader = self.dataloaders[phase]
    total_batches = len(dataloader)
    meter = Meter(phase, epoch, use_neptune = use_neptune, total_batches=total_batches)
    if self.use_tqdm:
      tk0 = tqdm(dataloader)
    else:
      tk0 = dataloader
    self.optimizer.zero_grad()
    if self.dataloader2 is not None:
      dli = iter(self.dataloader2)
    for itr, batch in enumerate(tk0): # replace `dataloader` with `tk0` for tqdm
      images, masks, idxs, scales  = batch
      
      if flip > 10:
        images = images.flip([2, 3])
        masks = masks.flip([2, 3])
      elif flip != 0:
        images = images.flip(flip)
        masks = masks.flip(flip)
        
      if transpose:
        images = images.permute(0, 1, 3, 2)
        masks = masks.permute(0, 1, 3, 2)
      
      loss, dice_loss, dice_loss2, cls_loss = self.forward(images, masks, scales, meter, idxs)
      lo = loss + 0.1 *dice_loss2 + cls_loss
      
      if phase == "train":
        lo.backward()
        if self.scheduler is not None:
          try:
            self.scheduler.step()
          except:
            pass #TODO so uggly :) 
        
        if self.dataloader2 is not None and itr % 5 == 0:
          images, masks, idxs, scales = next(dli)
          lo = self.forward2(images, masks, scales, meter, idxs)
          lo.backward()
          meter.update(KLDiv = lo.item(), itr=-1)
          
        self.optimizer.step()
        self.optimizer.zero_grad()  
      
      if itr > 0 and itr % 100 == 0:
        if phase == 'train':
          for i, o in enumerate(self.optimizer.param_groups):
            if self.use_neptune:
              neptune.log_metric("lr/group_{}".format(i), itr + epoch * total_batches, o["lr"])
                        
        if itr % 5000 == 0:
          state = {
          "state_dict": self.net.state_dict(),
          "optimizer": self.optimizer.state_dict(),
          }
          torch.save(state, f'm{itr}.pth')
        
      if itr > max_samples:
        break    
      
      meter.update(loss = loss.item(), lossdice = dice_loss.item(), lossdice2 = dice_loss2.item(), itr=itr)
    #             tk0.set_postfix(loss=(running_loss / ((itr + 1))))
    meter.get_metrics()        
    torch.cuda.empty_cache()
    
    return loss
  
  def start(self):
    for epoch in range(self.from_epoch, self.num_epochs):
      
      self.iterate(epoch, "train", use_neptune= self.use_neptune)
      
      state = {
        "epoch": epoch,
        "best_loss": self.best_loss,
        "state_dict": self.net.state_dict(),
        "optimizer": self.optimizer.state_dict(),
      }
      torch.save(state, "m.pth")
      with torch.no_grad():
        val_loss = self.iterate(epoch, "val", use_neptune=self.use_neptune)
      if val_loss < self.best_loss:
        print("******** New optimal found, saving state ********")
        state["best_loss"] = self.best_loss = val_loss
        torch.save(state, "model.pth")
      else:
        torch.save(state, "last.pth")

from model import EfficientFPN, load_net
import os
from optim import Over9000

@hydra.main(config_path='config/config.yaml')
def main(cfg):
    
  print(cfg.pretty())
  
  if cfg.use_neptune:
    neptune.init(f'{cfg.neptune_user}/{cfg.neptune_project}')
    neptune.create_experiment(f'Train-{cfg.model_name}', upload_stdout=False, upload_stderr=False, send_hardware_metrics=False, run_monitoring_thread=False)
      
  model = EfficientFPN(encoder_name=cfg.model_name, use_attention=cfg.use_attention, use_context_block=cfg.use_context_block, use_mish=cfg.use_mish, drop_connect_rate=cfg.drop_connect_rate) #TODO - evaluate losses
  epoch  = 0
  device = torch.device("cuda:0")
  model = model.to(device)
  optimizer = None
  if os.path.exists(cfg.model):
    state = torch.load(cfg.model, map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])
    device = torch.device("cuda:0")
    model = model.to(device)
    print('state ok')
    if cfg.use_over9000_optimizer:
      optimizer = Over9000(model.parameters()) 
    else:
      optimizer = optim.AdamW(model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay)
    if cfg.load_state == 1:
      optimizer.load_state_dict(state["optimizer"])
      #epoch = state['epoch'] + 1
  else:
    if cfg.use_over9000_optimizer:
      optimizer = Over9000(model.parameters()) 
    else:
      optimizer = optim.AdamW(model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay) 
  
  print(f'using {cfg.model_name}')
  print(epoch)
  
  model_trainer = Trainer(
    model, batch_size = {"train": cfg.batch_size, "val": cfg.batch_size}, 
    base_dir=cfg.data_dir, optimizer=optimizer, train_width=cfg.width, num_epochs=cfg.num_epochs, num_workers=cfg.num_workers, 
    use_tqdm = True, from_epoch = epoch, debug=cfg.debug, base_threshold = 0.5,
    max_lr=cfg.max_lr, fold=cfg.fold, do_ohem=cfg.do_ohem, soft_labels_dir=cfg.soft_labels_dir, use_neptune=cfg.use_neptune)
  
  model_trainer.start()
  neptune.stop()

if __name__ == '__main__':
  main()
    
    
    
