'''
Created on Mar 22, 2020

@author: Michal.Busta at gmail.com
'''

#get rid of the optimizer state ... 
import torch
MODEL_PATH = '/models/model-b2-2.pth'
state = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)

state_out = {
  "state_dict": state["state_dict"],
}

torch.save(state_out, 'model-b2-2.pth')