'''
Created on Nov 22, 2019

@author: michal.busta at gmail.com
'''

import geopandas as gpd

import rasterio
from rasterio.mask import mask, geometry_mask

import numpy as np
import cv2

from pathlib import Path
import traceback
import os, sys
    
if __name__ == '__main__':
  
  
  base_dir = '/data/ssd/Nadir/Atlanta_nadir16_catid_1030010002649200/Pan-Sharpen'  
  out_dir = '/ssd/Nadir/Atlanta_nadir16_catid_1030010002649200/RGB_1030010002649200'
  if not os.path.exists(out_dir):
    os.mkdir(out_dir)
  
  max_val = 0
  max_val1 = 0
  max_val2 = 0
  for file_path in Path(base_dir).glob('*.tif'):
    
    file = str(file_path)
    tiff = rasterio.open(file)
    img = tiff.read()
    
    try:
      img = img[0:3, :, :]   
      img = np.transpose(img, (1, 2, 0))
      if img.sum() == 0:
        continue
      
      percentiles = [2, 98]
      
      maxBand = np.percentile(img.flatten(), percentiles[1])
      minBand = np.percentile(img.flatten(), percentiles[0])
      scale_ratio = 255/(maxBand - minBand)
      
      img = img.astype(np.float)
      img[img>=maxBand] = maxBand
      img[img<minBand] = minBand
      img = img - minBand
      img=img*scale_ratio

      img = img.astype(np.uint8)
      
      out_img = f'{out_dir}/{os.path.basename(file)}'
      cv2.imwrite(out_img, img)
      out_img_mask = out_img.replace('.tif', '_mask.png')
      
      geo_name = file.replace('/Atlanta_nadir16_catid_1030010002649200/Pan-Sharpen/Pan-Sharpen_Atlanta_nadir16_catid_1030010002649200_', '/geojson/spacenet-buildings/spacenet-buildings_')
      geo_name = geo_name.replace('.tif', '.geojson')  
      
      df_roof_geometries = gpd.read_file(f'{geo_name}')
      tiff_crs = tiff.crs.data
      df_roof_geometries['projected_geometry'] = (
        df_roof_geometries['geometry'].to_crs(tiff_crs)
      )

      roof_image, transform = mask(tiff, df_roof_geometries['projected_geometry'], crop=False, pad=False, filled=False)
      
      roof_mask = geometry_mask(df_roof_geometries['projected_geometry'], transform=transform, invert=True, out_shape=(img.shape[0], img.shape[1]), all_touched=True)
      roof_mask = roof_mask.astype(np.uint8)
      roof_mask[roof_mask > 0] = 255
      
      print(out_img)
      cv2.imwrite(out_img_mask, roof_mask)
    except:
      traceback.print_exc(file=sys.stdout)
      cv2.imwrite(out_img, img)
      pass
        
    

