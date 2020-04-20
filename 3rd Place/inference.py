'''
Created on Nov 22, 2019

@author: Michal.Busta at gmail.com
'''

import torch

import numpy as np
from tqdm import tqdm

from dataset import test_provider

import hydra

from model import EfficientFPN
import os

import torch.nn.functional as F
import cv2


@hydra.main(config_path='config/config_eval.yaml')
def main(cfg):
    
  print(cfg.pretty())    
  model = EfficientFPN(encoder_name=cfg.model_name, use_context_block=cfg.use_context_block, use_mish=cfg.use_mish, use_attention=cfg.use_attention)    
  state = torch.load(cfg.model, map_location=lambda storage, loc: storage)
    
  model.load_state_dict(state["state_dict"])
  
  device = torch.device("cuda:0")
  model = model.to(device)
  model = model.eval()
  print('state ok')
  print('using {0}'.format(cfg.model_name))
  
  os.mkdir('submission_format')
  os.mkdir('dice')
  os.mkdir('bce')
  
  dataloader = test_provider(
    cfg.data_dir,
    batch_size=1,
    num_workers=0,
    train_width = cfg.width,
    train_height = cfg.width,
    debug = cfg.debug,
    phase = 'test',
    scale_factor = cfg.scale_factor,           
  )
  
  
  if cfg.mine_empty and not os.path.exists(cfg.output_dir_empty):
    os.mkdir(cfg.output_dir_empty)
  

  tk0 = tqdm(dataloader)
  with torch.no_grad():
    for itr, batch in enumerate(tk0): # replace `dataloader` with `tk0` for tqdm
      images, masks, idx, sf  = batch
      
      base_name = os.path.basename(idx[0])
      
      if cfg.flip > 10:
        images = images.flip([2, 3])
      elif cfg.flip != 0:
        images = images.flip(cfg.flip)
          
      if cfg.transpose:
        images = images.permute(0, 1, 3, 2)

      images = images.to(device)
                  
      outputs, dice, cls, scale = model(images, masks)
      cls = F.softmax(cls, 1)
      
      if cfg.flip > 10:
        outputs = outputs.flip([2, 3])
        dice = dice.flip([2, 3])
      elif cfg.flip != 0:
        outputs = outputs.flip(cfg.flip)
        dice = dice.flip(cfg.flip)
        
      if cfg.transpose:
        outputs = outputs.permute(0, 1, 3, 2)
        dice = dice.permute(0, 1, 3, 2)
      
      
      outputs_log = F.softmax(outputs, 1)
      outputs_log2 = F.interpolate(outputs_log, size=(1024, 1024), mode='bilinear', align_corners=False)
      outputs_log_out = outputs_log2[:, 1].cpu().numpy() * 255
      outputs_log_out = outputs_log_out[0].astype(np.uint8)
      cv2.imwrite(f'bce/{base_name}', outputs_log_out)
      
      max_cls = outputs_log[:, 1].max().item()  
      dice = torch.sigmoid(dice)
      
      if cfg.mine_empty and cls[0, 1] < cfg.empty_threshold and max_cls < cfg.empty_threshold:
        dimg = cv2.imread(idx[0])
        print(f'empty image {idx[0]}')
        if not os.path.exists(f'{cfg.output_dir_empty}/{os.path.basename(base_name)}'):
          cv2.imwrite(f'{cfg.output_dir_empty}/{base_name}', dimg)
          
      
      dice_out = dice[0, 0].cpu().numpy() * 255
      dice_out = dice_out.astype(np.uint8)
      cv2.imwrite(f'dice/{base_name}', dice_out)
      
      dice = F.interpolate(dice, size=(1024, 1024), mode='bilinear', align_corners=False)
      outputs_log = F.interpolate(outputs_log, size=(1024, 1024), mode='bilinear', align_corners=False)
      
      outputs_log = outputs_log[:, 1]
      
      avg = outputs_log
      outputs_log[outputs_log >= cfg.threshold] = 1
      outputs_log[outputs_log < cfg.threshold] = 0
      
      dice[dice < 0.5] = 0
      dice[dice >= 0.5] = 1
      dice = dice[0, 0, :, :].cpu().numpy() * 255
      dice = dice.astype(np.uint8)
      
      avg[avg < cfg.threshold] = 0
      avg[avg >= cfg.threshold] = 1 
      
      avg = avg.cpu().numpy() * 255
      avg = avg.astype(np.uint8)
      avg = avg[0, :, :]
      outputs_log = outputs_log.cpu().numpy() * 255
      outputs_log = outputs_log.astype(np.uint8)
      
      outputs_log = outputs_log[0, :, :]
      if cfg.debug:
        dimg = cv2.imread(idx[0])
        dimgr = cv2.resize(dimg, (1024, 1024))
        dimgr[:, :, 2][avg > 0] = 255        
        cv2.imshow('resized', dimgr)
        cv2.imshow('di', dice_out)
        cv2.imshow('bce', outputs_log_out)
        cv2.waitKey(0)
      
      cv2.imwrite(f'submission_format/{base_name}', avg)
      #cv2.imwrite(f'{cfg.data_dir}/{base_name[:-4]}_mask.png', outputs_log_out)
                

if __name__ == '__main__':
    main()
    
    
    
