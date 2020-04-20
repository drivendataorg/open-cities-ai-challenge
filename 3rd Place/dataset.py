'''
Created on Nov 22, 2019

@author: michal.busta at gmail.com
'''

import albumentations as albu
from albumentations.pytorch import ToTensor

from torch.utils.data import DataLoader, Dataset, ConcatDataset

import pandas as pd
import geopandas as gpd

import rasterio
from rasterio.mask import mask, geometry_mask

import multiprocessing

import numpy as np
import cv2

from pathlib import Path
import hydra

mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)

from sklearn.model_selection import KFold
import pickle
import random

def get_training_augmentation(grayscale=False, height=320, width=640, crop_mode = 0):
    
  mea = mean
  st = std
  if grayscale:
      mea = (mean[0] + mean[1] + mean[2]) / 3
      st = (std[0] + std[1] + std[2]) / 3
  
  if crop_mode == 0:
    train_transform = [
      albu.PadIfNeeded(height * 3 // 2, width * 3 // 2),
      albu.RandomCrop(height * 3 // 2, width * 3 // 2),
      albu.HorizontalFlip(p=0.5),
      albu.VerticalFlip(p=0.5),
      albu.CoarseDropout(p=0.1),
      albu.ShiftScaleRotate(scale_limit=0.4, rotate_limit=45, shift_limit=0.1, p=0.5, border_mode=0),
      albu.GridDistortion(p=0.3),
      albu.OpticalDistortion(p=0.3, distort_limit=2, shift_limit=0.5),
      albu.RGBShift(p=0.3),
      albu.Blur(p=0.3),
      albu.MotionBlur(p=0.3),
      albu.PadIfNeeded(height, width),
      albu.RandomCrop(height, width)
    ]
  else:
    train_transform = [
      albu.HorizontalFlip(p=0.5),
      albu.VerticalFlip(p=0.5),
      albu.ShiftScaleRotate(scale_limit=0.4, rotate_limit=45, shift_limit=0.1, p=0.5, border_mode=0),
      albu.GridDistortion(p=0.5),
      albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
      albu.RGBShift(p=0.5),
      albu.ToGray(p=0.5),
      albu.Resize(height, width)
    ]
      
  
  train_transform.extend(
        [
            #Equalize(p=1.0, by_channels=False),
            albu.Normalize(mean=mea, std=st, p=1),
            ToTensor(),
        ]
    )
  return albu.Compose(train_transform)

def get_validation_augmentation(grayscale=False, height=320, width=640, crop_mode = 0):
  """Add paddings to make image shape divisible by 32"""   
  mea = mean
  st = std
  if grayscale:
      mea = (mean[0] + mean[1] + mean[2]) / 3
      st = (std[0] + std[1] + std[2]) / 3
  
  if crop_mode == 0:    
      test_transform = [
          albu.PadIfNeeded(height, width),
          albu.CenterCrop(height, width)
      ]
  else:
      test_transform = [
          albu.PadIfNeeded(height, width),
          albu.Resize(height, width)
      ]
  
  test_transform.extend(
          [
              #Equalize(p=1.0, by_channels=False),
              albu.Normalize(mean=mea, std=st, p=1),
              ToTensor(),
          ]
      )
  return albu.Compose(test_transform)

def get_test_augmentation(grayscale=False, height=320, width=640, crop_mode = 0):
  """Add paddings to make image shape divisible by 32"""   
  mea = mean
  st = std
  if grayscale:
    mea = (mean[0] + mean[1] + mean[2]) / 3
    st = (std[0] + std[1] + std[2]) / 3
    
  test_transform = [
      #albu.PadIfNeeded(height, width),
      albu.Resize(height, width)
  ]
  
  test_transform.extend(
    [
        #Equalize(p=1.0, by_channels=False),
        albu.Normalize(mean=mea, std=st, p=1),
        ToTensor(),
    ]
  )
  return albu.Compose(test_transform)

class CCDataset(Dataset):
  def __init__(self, tiff, json, ids, transforms = albu.Compose([albu.HorizontalFlip(), ToTensor()]), crop_mode=0, scale_factor=1.0, filled=False):
             
    self.img_ids = ids
    self.tiff_name = tiff
    self.process_tiffs = {} 
    self.df_roof_geometries = json
    tiff = rasterio.open(self.tiff_name) 
    tiff_crs = tiff.crs.data
    self.df_roof_geometries['projected_geometry'] = (
        self.df_roof_geometries['geometry'].to_crs(tiff_crs)
    )
    self.transforms = transforms
    self.crop_mode = crop_mode
    self.padding = 200
    self.scale_factor = scale_factor
    self.filled = filled
    
    print(f'scale factor: {self.scale_factor}')
      
      
  def __getitem__(self, idx):
      
    ident = multiprocessing.current_process().ident
    try:
      if ident not in self.process_tiffs:
          self.process_tiffs[ident] = rasterio.open(self.tiff_name) 
      tiff =   self.process_tiffs[ident]  
      geometry = self.df_roof_geometries.loc[idx]['projected_geometry']
      roof_image, transform = mask(tiff, [geometry], crop=True, pad=True, filled=self.filled, pad_width=self.padding )
       
      roof_image = roof_image[0:3, :, :]   
      roof_mask = geometry_mask(self.df_roof_geometries['projected_geometry'], transform=transform, invert=True, out_shape=(roof_image.shape[1], roof_image.shape[2]), all_touched=True)
      
      #roof_mask = roof_image[0, :, :]
      
      roof_mask = roof_mask.astype(np.uint8)
      roof_mask[roof_mask > 0] = 255
      
      roof_image = np.transpose(roof_image, (1, 2, 0))
      roof_image = roof_image[:, :, ::-1]
      
      if self.scale_factor != 1.0:
          #print('resizing ... ')
          roof_image = cv2.resize(roof_image, dsize=(int(roof_image.shape[1] * self.scale_factor), int(roof_image.shape[0] * self.scale_factor)))
          roof_mask = cv2.resize(roof_mask, dsize=(int(roof_mask.shape[1] * self.scale_factor), int(roof_mask.shape[0] * self.scale_factor)))
   
      augmented = self.transforms(image=roof_image, mask=roof_mask)
      img = augmented['image']
      roof_mask = augmented['mask']
      
      return img, roof_mask, idx
    except:
      self.process_tiffs[ident] = rasterio.open(self.tiff_name) 
      return self.__getitem__(idx)

  def __len__(self):
      
    return len(self.img_ids)
    

class CCDatasetCuts(Dataset):
  def __init__(self, files, transforms = albu.Compose([albu.HorizontalFlip(), ToTensor()]), crop_mode=0, scale_factor=1.0, train_width=512, ref_scale=0.07, scales = {}, phase='train', is_fastai=False, is_infinite=False):
             
      self.files = files
      self.transforms = transforms
      self.crop_mode = crop_mode
      self.scale_factor = scale_factor
      self.train_width = train_width
      self.ref_scale = ref_scale
      self.scales = scales 
      self.phase = phase
      self.is_fastai = is_fastai
      self.is_infinite = is_infinite
      
  def __getitem__(self, idx):
    
    if self.is_infinite:
      idx = random.randint(0, len(self.files) - 1)  
    try:
      file_name = self.files[idx]
      roof_image = cv2.imread(file_name)
      
      '''
      a1 = self.pre_t(image=roof_image)
      roof_image = a1['image']
      cv2.imwrite(file_name, roof_image)
      '''
      
      mask_name =  file_name.replace('.jpg', '_mask.png')  
      mask_name =  mask_name.replace('.tif', '_mask.png')  
      scale = 0
      if not os.path.exists(mask_name):
        roof_mask = np.copy(roof_image[:, :, 0])
        roof_mask[:] = 0  
      else:  
        roof_mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        
      p = Path(file_name)
      sc_name = os.path.basename(str(p.parent))
      if sc_name in self.scales:
        scale = self.scales[sc_name]
      
      if self.train_width == 512 and (roof_image.shape[0] > 1024 or roof_image.shape[1] > 1024) and random.randint(0, 10) > 1:
        roof_image = cv2.resize(roof_image, (roof_image.shape[1] // 2, roof_image.shape[0] // 2), interpolation=cv2.INTER_AREA)
        roof_mask = cv2.resize(roof_mask, (roof_mask.shape[1] // 2, roof_mask.shape[0] // 2), interpolation=cv2.INTER_AREA)
      csf = 1.0
      if scale > 0:
        csf = scale / self.ref_scale     
      
      if random.randint(0, 5) == 1 and self.phase == 'train':
        roof_image = cv2.resize(roof_image, (int(roof_image.shape[1] * csf), int(roof_image.shape[0] * csf)), interpolation=cv2.INTER_AREA)
        roof_mask = cv2.resize(roof_mask, (int(roof_mask.shape[1] * csf), int(roof_mask.shape[0] * csf)), interpolation=cv2.INTER_AREA)
        
        
      
      roof_image = roof_image[:, :, ::-1]
      #roof_mask[roof_image[:, :, 0] == 0] = 0 
      
      if self.scale_factor != 1.0:
        #print('resizing ... ')
        roof_image = cv2.resize(roof_image, dsize=(int(roof_image.shape[1] * self.scale_factor), int(roof_image.shape[0] * self.scale_factor)))
        roof_mask = cv2.resize(roof_mask, dsize=(int(roof_mask.shape[1] * self.scale_factor), int(roof_mask.shape[0] * self.scale_factor)))
        scale = 0
 
      augmented = self.transforms(image=roof_image, mask=roof_mask)
      img = augmented['image']
      roof_mask = augmented['mask']
      
      if self.is_fastai:
        return img, roof_mask
      
      return img, roof_mask, file_name, scale
    except:
      return self.__getitem__(random.randint(0, len(self.files) - 1))

  def __len__(self): 
    if self.is_infinite:
      return 100000000 
    return len(self.files)

def provider(
        base_dir,
        phase,
        batch_size=4,
        num_workers=2,
        train_width = 256,
        train_height = 256,
        fold = 0,
        debug = False,
        scale_factor=1.0,
        filled = False
        ):
  '''Returns dataloader for the model training'''
  
  folds = 5
  meta = pd.read_csv(f'{base_dir}/train_metadata.csv')
  datasets = []
  for index, row in meta.iterrows():
    image = row['img_uri']
    train_label = row['label_uri']
          
    print(f'processing {image}')
    
    df_roof_geometries = gpd.read_file(f'{base_dir}/{train_label}')
    
    ids = df_roof_geometries.index 
    kf = KFold(n_splits=folds, shuffle=True, random_state=777)
        
    cf = 0
    for train_idx, valid_idx in kf.split(ids):
        
      if debug:
        train_idx = train_idx[0:500]
        valid_idx = valid_idx[0:500]
    
      if fold !=-1 and cf != fold:
        cf += 1
        continue
    
      if  phase == 'train':    
        train_dataset = CCDataset(f'{base_dir}/{image}', df_roof_geometries, ids[train_idx], transforms = get_training_augmentation(width=train_width, height=train_height), scale_factor=scale_factor, filled=filled, phase=phase)
      else:
        train_dataset = CCDataset(f'{base_dir}/{image}', df_roof_geometries, ids[valid_idx], get_validation_augmentation(width=train_width, height=train_height), scale_factor=scale_factor, filled = filled, phase=phase)
          
      datasets.append(train_dataset)
      if fold != -1:
          break
    
    if debug:
      break  
  
  cdataset = ConcatDataset(datasets)    
  
  
  dataloader = DataLoader(
    cdataset,
    sampler= None,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    shuffle=True if phase=='train' else False,
    drop_last=True     
  )
  return dataloader

def provider_cuts(
        base_dir,
        phase,
        batch_size=4,
        num_workers=2,
        train_width = 256,
        train_height = 256,
        fold = 0,
        debug = False,
        scale_factor=1.0,
        ref_scale = 0.07,
        is_fastai = False
        ):
  '''Returns dataloader for the model training'''
  
  folds = 7
  files = []
  for file_path in Path(base_dir).glob('**/*.jpg'):
    files.append(str(file_path))
    if debug and len(files) > 5000:
      break
  for file_path in Path(base_dir).glob('**/*.tif'):
    files.append(str(file_path))
    if debug and len(files) > 5000:
      break
  
  files = np.asarray(files)
  
  scales = {}
  if os.path.exists(f'{base_dir}/resolutions.pkl'):
    f = open(f'{base_dir}/resolutions.pkl','rb')
    scales = pickle.load(f)
      
  kf = KFold(n_splits=folds, shuffle=True, random_state=777)        
  cf = 0
  for train_idx, valid_idx in kf.split(files):
          
    if debug:
      train_idx = train_idx[0:1000]
      valid_idx = valid_idx[0:1000]
    
    if fold !=-1 and cf != fold:
      cf += 1
      continue
    
    if  phase == 'train':    
      train_dataset = CCDatasetCuts(files[train_idx], transforms = get_training_augmentation(width=train_width, height=train_height), scale_factor=scale_factor, ref_scale = ref_scale, scales = scales, is_fastai=is_fastai)
    else:
      train_dataset = CCDatasetCuts(files[valid_idx], get_validation_augmentation(width=train_width, height=train_height), scale_factor=scale_factor, ref_scale = ref_scale, scales = scales, is_fastai=is_fastai)
        
    if fold != -1:
      break
  
  
  dataloader = DataLoader(
    train_dataset,
    sampler= None,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    shuffle=True if phase=='train' else False,
    drop_last=True     
  )
  
  return dataloader

def test_provider(
        base_dir,
        batch_size=4,
        num_workers=2,
        train_width = 1024,
        train_height = 1024,
        debug = False,
        phase='test',
        scale_factor=1.0,
        ):
  '''Returns dataloader for the model training'''
  
  files = []
  for file_path in Path(base_dir).glob('**/*.tif'):
      files.append(str(file_path))
  for file_path in Path(base_dir).glob('**/*.JPG'):
      files.append(str(file_path))
  for file_path in Path(base_dir).glob('**/*.jpg'):
      files.append(str(file_path))
  files = np.asarray(files)
  files = sorted(files)
  #files = files[::-1]
  
  train_dataset = CCDatasetCuts(files, get_test_augmentation(width=train_width, height=train_height), scale_factor=scale_factor, phase = phase, train_width = train_width)
  dataloader = DataLoader(
      train_dataset,
      sampler= None,
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=True,
      shuffle=False,
      drop_last=False     
  )
  return dataloader
    
import os, sys
from tqdm import tqdm
  
@hydra.main(config_path='config/dataset.yaml')  
def main(cfg):
        
  if not os.path.exists(cfg.out_dir):
      os.mkdir(cfg.out_dir)
  resolutions = {}
  measure = []

  train_df = pd.read_csv(f'{cfg.base_dir}/train_metadata.csv')
  for index, row in train_df.iterrows():
    image = row['img_uri']
    train_label = row['label_uri']
    base_name = os.path.basename(image)[:-4]
    
    if train_label.find('nia') == -1:
      continue
    
    prefix = image[0:12]
    out_dir = f'{cfg.out_dir}/{prefix}/{base_name}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    #else:
    #  continue
        
    print(f'processing {image}')
    df_roof_geometries = gpd.read_file(f'{cfg.base_dir}/{train_label}')
    
    tiff = rasterio.open(f'{cfg.base_dir}/{image}') 
    tiff_crs = tiff.crs.data
    df_roof_geometries['projected_geometry'] = (
      df_roof_geometries['geometry'].to_crs(tiff_crs)
    )
    
    pad_width = 1024 + 512
    proccesed = 0
    print(tiff.res[0])
    resolutions[base_name] = tiff.res[0]
    measure.append(tiff.res[0])
    
    if cfg.shuffle:
      df_roof_geometries = df_roof_geometries.sample(frac=1)
    
    for index, row in df_roof_geometries.iterrows():      
      mask_name = f'{out_dir}/{str(cfg.scale).replace(".", "_")}_{index}_mask.png'
      if mask_name.find(cfg.process_tier) == -1:
        continue
      if os.path.exists(mask_name):
        continue
      
      
      geometry = row['projected_geometry']
      roof_image, transform = mask(tiff, [geometry], crop=True, pad=True, filled=False, pad_width=pad_width)
      roof_image = roof_image[0:3, :, :]   
      roof_mask = geometry_mask(df_roof_geometries['projected_geometry'], transform=transform, invert=True, out_shape=(roof_image.shape[1], roof_image.shape[2]), all_touched=True)
      
      roof_mask = roof_mask.astype(np.uint8)
      roof_mask[roof_mask > 0] = 255
      
      roof_image = np.transpose(roof_image, (1, 2, 0))
      roof_image = roof_image[:, :, ::-1]
      
      if cfg.scale != 1:
        roof_image = cv2.resize(roof_image, (int(roof_image.shape[1] / cfg.scale), int(roof_image.shape[0] / cfg.scale)), interpolation=cv2.INTER_LINEAR)
        roof_mask = cv2.resize(roof_mask, (int(roof_mask.shape[1]/ cfg.scale), int(roof_mask.shape[0] / cfg.scale)), interpolation=cv2.INTER_LINEAR)
      
      if cfg.crop1024:
        roof_image = roof_image[0:1024, 0:1024, :]
        roof_mask = roof_mask[0:1024, 0:1024]
      
      cv2.imwrite(f'{out_dir}/{str(cfg.scale).replace(".", "_")}_{index}.jpg', roof_image)
      cv2.imwrite(f'{out_dir}/{str(cfg.scale).replace(".", "_")}_{index}_mask.png', roof_mask)  
      
      if index % 500 == 0:
        print(f'processed {index}/{len(df_roof_geometries)}')
      if proccesed > cfg.max_samples:
        break
      proccesed += 1
      
  f = open(f'{out_dir}/resolutions.pkl',"wb")
  pickle.dump(resolutions,f)
  f.close()

if __name__ == '__main__':
  main()
        
    

