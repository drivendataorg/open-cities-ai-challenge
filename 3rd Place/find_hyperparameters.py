'''
Created on Nov 21, 2019

@author: michal.busta at gmail.com
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import optim

import optuna
optuna.logging.disable_default_handler()

from tqdm import tqdm as tqdm

from dataset import provider_cuts
from model import EfficientFPN
import hydra

EPOCH = 1

def get_optimizer(trial, model):
  weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
  adam_lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)
  optimizer = optim.AdamW(model.parameters(), lr=adam_lr, weight_decay=weight_decay)
  return optimizer

def train(model, device, train_loader, optimizer):
  model.train()
  criterion = nn.CrossEntropyLoss(weight=None, ignore_index=255)
  sum_loss = 0
  for batch_idx, (data, mask, idx, scales) in enumerate(tqdm(train_loader)):
    data = data.to(device)
    mask = mask.to(device)
    optimizer.zero_grad()
    output, dice, xc, xs = model(data, mask)
    mask = F.interpolate(mask, size=(output.shape[2], output.shape[3]), mode='bilinear', align_corners=False)
    loss = criterion(output.reshape(-1, 2), mask.long().reshape(-1))
    loss.backward()
    optimizer.step()
    sum_loss += loss.item()
        
def test(model, device, test_loader):
  model.eval()
  criterion = nn.CrossEntropyLoss(weight=None, ignore_index=255)
  loss = 0
  with torch.no_grad():
    for batch_idx, (data, mask, idx, scales) in enumerate(test_loader):
      data = data.to(device)
      mask = mask.to(device)
      output, dice, xc, xs = model(data, mask)
      mask = F.interpolate(mask, size=(output.shape[2], output.shape[3]), mode='bilinear', align_corners=False)
      loss += criterion(output.reshape(-1, 2), mask.long().reshape(-1)).item()

  return loss / len(test_loader.dataset)

def objective_wrapper(pbar):
    
  def objective(trial):
      
    device = "cuda"
    model = EfficientFPN(encoder_name='efficientnet-b0').to(device)
    optimizer = get_optimizer(trial, model)

    for step in range(EPOCH):
        train(model, device, train_loader, optimizer)
        error_rate = test(model, device, test_loader)
        trial.report(error_rate, step)
        if trial.should_prune(step):
            pbar.update()
            raise optuna.exceptions.TrialPruned()

    pbar.update()

    return error_rate
  
  return objective

@hydra.main(config_path='config/config.yaml') 
def main(cfg):
    
  BATCHSIZE = cfg.batch_size
  base_dir = cfg.data_dir
  
  size = cfg.width
  
  global train_loader, test_loader
  
  train_loader = provider_cuts(
    base_dir,
    'train',
    batch_size=BATCHSIZE,
    num_workers=1,
    train_width = size,
    train_height = size,
    fold = 0,
    debug = True
    )
  
  test_loader = provider_cuts(
    base_dir,
    'val',
    batch_size=BATCHSIZE,
    num_workers=1,
    train_width = size,
    train_height = size,
    fold = 0,
    debug = True
    )
  
  TRIAL_SIZE = 50
  with tqdm(total=TRIAL_SIZE) as pbar:
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
    study.optimize(objective_wrapper(pbar), n_trials=TRIAL_SIZE)
    
  print('Best params:')
  print(study.best_trial.params)

if __name__ == '__main__':
  main()
    
