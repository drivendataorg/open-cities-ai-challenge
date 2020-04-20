'''
Created on Jan 29, 2020

@author: Michal.Busta at gmail.com
'''

import numpy as np
import neptune

class Meter:
  
  '''A meter to keep track of losses scores throughout an epoch'''
  def __init__(self, phase, epoch, use_neptune=False, log_interval = 100, total_batches = 100):
    
    self.metrics = {}
    self.rmetrics = {}
    self.phase = phase  
    self.epoch = epoch  
    self.use_neptune = use_neptune 
    self.log_interval = log_interval 
    self.total_batches = total_batches
  
  def update(self, **kwargs):
    
    itr = 0
    for name, value in kwargs.items():  
      if name == 'itr':
        itr = value
        continue
      try:      
        self.metrics[name].append(value)
        self.rmetrics[name].append(value)
      except:
        self.metrics[name] = []
        self.metrics[name].append(value)
        self.rmetrics[name] = []
        self.rmetrics[name].append(value)
    
    if itr % self.log_interval == 0:
      if self.use_neptune: 
        for key in self.rmetrics.keys():
          mean = np.mean(self.rmetrics[key])
          self.rmetrics[key] = []
          neptune.log_metric(f'{key}_{self.phase}', itr + self.epoch * self.total_batches, mean)
      else:
        for key in self.rmetrics.keys():
          mean = np.mean(self.rmetrics[key])
          self.rmetrics[key] = []
          print(f'  - {key}: {mean}')
  
  def get_metrics(self):
    ret = {}
    log_str = ''
    for key in self.metrics.keys():
      mean = np.mean(self.metrics[key])
      ret[key] = mean
      log_str = '%s | %s: %0.4f '  % (log_str, key, mean)
      if self.use_neptune:
        neptune.log_metric(f'epoch_{key}_{self.phase}', self.epoch, mean)      
        
    print(log_str)    
    return ret
    
    