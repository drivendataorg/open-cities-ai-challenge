'''
Created on Feb 7, 2020

@author: michal.busta at gmail.com
'''
import os

import cv2
from pathlib import Path
import numpy as np
import hydra

def geo_mean(iterable):
  a = np.array(iterable)
  return a.prod(axis=0)**(1.0/a.shape[0])

def arit_mean(iterable):
  a = np.array(iterable)
  return a.sum(axis=0) / a.shape[0]

def variance_of_laplacian(image):
  return cv2.Laplacian(image, cv2.CV_64F).var()

def get_list_of_first_level_directories(root_dir):
  entries = os.listdir(root_dir)
  all_dirs = []
  for entry in entries:
    path = os.path.join(root_dir, entry)
    if os.path.isdir(path):
      all_dirs.append(entry)
  return all_dirs

@hydra.main(config_path='config/combine.yaml') 
def main(cfg):
  
  threshold = cfg.threshold
  
  childs = get_list_of_first_level_directories(cfg.base_dir)
  ref_dir  = os.path.join(cfg.base_dir, childs[0])
  ref_dir  = os.path.join(ref_dir, 'bce')
  out_dir = 'submission_format'
  os.mkdir(out_dir)
  cnt = 0
  
  for file_path in sorted(Path(ref_dir).glob('*.tif')):
    
    file_path = str(file_path)
    bce1 = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).astype(np.float) / 255
    arr2 = [bce1]
    for i in range(1, len(childs)):
      file_path2 = file_path.replace(f'/{childs[0]}/', f'/{childs[i]}/')
      arr2.append(cv2.imread(file_path2, cv2.IMREAD_GRAYSCALE).astype(np.float) / 255)
      
    arr_out2 = []
    vars = []
    for img in arr2:
      var = img[img > 0].std()
      #var = img[img > 0].max() - img[img > 0].min()
      vars.append(var)
      
      im_out = cv2.resize(img, (512, 512))
      #im_out = im_out.astype(np.float) / 255
      arr_out2.append(im_out)
    
    ind = np.argsort(vars)
    avg2 = arit_mean(arr_out2)
    avg2 = cv2.resize(avg2, (1024, 1024))
    
    if arr2[ind[0]].max() > 0.5 and avg2.max() < 0.9:
      avg2 = arit_mean(np.array(arr_out2)[ind][1:])
      avg2 = cv2.resize(avg2, (1024, 1024))
      
    if True :
      cv2.imshow('var0', arr2[ind[0]])
      cv2.imshow('varl', arr2[ind[-1]])
      
      base_name = os.path.basename(file_path)
      src_img = f'/home/busta/git/DeblurGANv2/submit2/{base_name}'
      if os.path.exists(src_img):
        out_img = cv2.imread(src_img)
        cv2.imshow('orig', out_img)
        out_img = out_img // 2
        out_img[:, :, 2][avg2 > threshold] = 255
        cv2.imshow('out_img', out_img)
        cv2.waitKey(0)
      
    avg = avg2
    
    if cfg.add_mask_suffix:
      bce_out = avg * 255
      bce_out = bce_out.astype(np.uint8)
      cv2.imwrite(f'{out_dir}/{os.path.basename(file_path)[:-4]}_mask.png', bce_out)
      continue
    
    avg[avg < threshold] = 0
    avg[avg >= threshold] = 1
    
    avg = avg * 255
    avg = avg.astype(np.uint8)
    
    
    cv2.imwrite(f'{out_dir}/{os.path.basename(file_path)}', avg)
    cnt += 1
    if cnt % 500 == 0:
      print(cnt)
      
if __name__ == '__main__':
  main()
  
    