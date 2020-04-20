# Open Cities AI Challenge: Segmenting Buildings for Disaster Resilience

This is the code for winning solution for ["Open Cities AI Challenge: Segmenting Buildings for Disaster Resilience"](https://www.drivendata.org/competitions/60/building-segmentation-disaster-resilience). 

Semantic Segmentation track: Build models to identify building footprints from aerial imagery across diverse African cities.

My solution is based on Unet-like CNN models and below you will find description of full pipeline and instructions how to run training, inference on competitions data or inference on your own data.   

Archive with pretrained models could be loaded [here](https://drive.google.com/file/d/1Zn2BR2vgzGzS6FjAzT-ymAWKbSFeY0Qx/view?usp=sharing) (Google Drive).

Solution have been packed using Docker to simplify environment preparation.  

## Table of content

- [Requirements](#requirements)
    - [Software](#software)
    - [Hardware](#hardware)
- [Pipeline short summary](#pipeline-short-summary)
- [Pipeline in-depth view](#pipeline-in-depth-view)
    - [Stage 0.](#stage-0)
    - [Stage 1.](#stage-1)
        - [Step 1.](#step-1)
        - [Step 2.](#step-2)
        - [Step 3.](#step-3)
    - [Stage 2.](#stage-2)
        - [Step 1.](#step-1-1)
        - [Step 2.](#step-2-1)
        - [Step 3.](#step-3-1)
    - [Stage 3.](#stage-3)
        - [Step 1.](#step-1-2)
        - [Step 2.](#step-2-2)
        - [Step 3.](#step-3-2)
- [Your data prediction](#your-data-prediction)

## Requirements

#### Software

- Docker (19.03.6, build 369ce74a3c)
- Docker-Compose (1.24.0-rc1, build 0f3d4dda)
- Nvidia-Docker (Nvidia Driver Version: 396.44)

 All other packages and software specified in `Dockerfile` and `requirements.txt`

#### Hardware

Recommended minimal configuration:

  - Nvidia GPU with at least 16GB Memory *
  - Disk space 256 GB (free)
  - 64GB RAM

\* - during inference it is possible to reduce batch size to reduce memory consumption, however training configuration need at least 16GB.


## Pipeline short summary

**Step1. Starting service**

Build docker image, start docker-compose service in daemon mode and install requirements inside container.

```bash
$ make build && make start && make install
```

**Step 2. Starting pipelines inside container**


Data preparation 

 - Put competition data to 'data/raw/' (`train_tier_1.tgz` and `test.tgz` required)
 - Load models archive [here](https://drive.google.com/file/d/1Zn2BR2vgzGzS6FjAzT-ymAWKbSFeY0Qx/view?usp=sharing) and extract to `models/` directory

```
<project_dir>/
  data/
    ...
    raw/
      train_tier_1.tgz
      test.tgz
  models/
    stage1/
      ...
    stage2/
      ...
    stage3/
      ...
```

Start only inference (`models/` directory should be provided with pretrained models)
```bash
$ make inference
```

Start training and inference
```bash
$ make train-inference
```

After pipeline execution final prediction will appear in `data/predictions/stage3/sliced/` directory.

**Step 3. Stop service**

After everything is done stop docker container
```bash
$ make clean
```

## Pipeline in-depth view

Before start, please, make sure:  
1. You clone the repo and have specified [requirements](#requirements)
2. Put data (`train_tier_1.tgz` and `test.tgz`) in `data/raw/` folder
3. Optionally downoad and extract pretrained models (`models/`)
4. Start service (see above instructions) 

The whole pipeline consist of 4 stages stages 1 to 3 steps each.
The whole structure is as follows:

 - Stage 0
   - Step 0 - preparing test data
 - Stage 1
   - Step 1 - prepraring train data (`train_tier_1`)
   - Step 2 - training 10 Unet models on that data
   - Step 3 - make prediction with ensemble of 10 models
 - Stage 2
   - Step 1 - take prediction from `stage 1` and prepare as train data (pseudolabels)
   - Step 2 - train 10 Unet models with `train_tier_1` and `stage_2` data (pseudolabels) 
   - Step 3 - make predictions with new models
 - Stage 3
   - Step 1 - take prediction from `stage 2` and prepare as train data (pseudolabels round 2)
   - Step 2 - train 5 Unet models with `train_tier_1` and `stage_3` data (pseudolabels round 2)
   - Step 3 - make prediction with new models

After whole pipeline execution final prediction will appear in `data/predictions/stage3/sliced/` directory. In case you have pretrained models you can run just `stage 0 step 0` and `stage 3 step 3` blocks.

---
#### Stage 0.
---

Command to run:  
```bash
$ make stage0
```

Consist of one step - preparaing `test.tgz` data. 
We will extract data and create mosaic from test tiles (it is better to stitch separate tiles into big images (scenes), so prediction network will have more context). CSV file with data about mosaic is located in `data/processed/test_mosaic.csv` and created by jyputer notebook (`notebooks/mosaic.ipynb`). You dont need to generate it again, it is already exist.

---
#### Stage 1.
---

Train data preparation, models training and prediction with ensemble of models (consist of three steps).     


Command to run:  
```bash
$ make stage1
```

##### Step 1.

Prepraing data:

 - Extract `train_tier_1` data 
 - Resample data to 0.1 meters per pixel resolution
 - Generate raster masks from provided geojson files
 - Slice masks to small tiles with shape `1024 x 1024` (more convinient for training the models)

##### Step 2.

On this step 10 `Unet` models are going to be trained on data. 5 Unet models for `eficientnet-b1` encoder and 5 for `se_resnext_32x4d` encoder (all encoders are pretrained on Imagenet). We train 5 models for each encoder because of 5 folds validation scheme. Models trained with hard augmentations using `albumetations` library and random data sampling. Training lasts 50 epochs with continious learining rate decay from 0.0001 to 0.

##### Step 3.

Pretrained models aggregeated to `EnsembleModel` with test time augmentation (flip, rotate90) - all predicitons averaged by simple mean and thresholded by 0.5 value. First, prediction is made for stitched test images, than for others (which not present on mosaic).

---
#### Stage 2.
---

Train data preparation (add pseudolabels), models training and prediction with ensemble of models (consist of three steps).     


Command to run:  
```bash
$ make stage2
```

##### Step 1.

Take predictions from previous stage and prepare them to use as training data for current stage. This technique is called pseudolabeling (when we use models prediction for training). I used all test data because leadeboard score was high enough (~0.845 jaccard score), but usually you should take only confident predictions.

##### Step 2.

(same as stage 1 step 1, but with exta data labeled on previous stage)

On this step 10 `Unet` models are going to be trained on data. 5 Unet models for `eficientnet-b1` encoder and 5 for `se_resnext_32x4d` encoder (all encoders are pretrained on Imagenet). We train 5 models for each encoder because of 5 folds validation scheme. Models trained with hard augmentations using `albumetations` library and random data sampling. Trainings last 50 epochs with continious learining rate decay from 0.0001 to 0.

##### Step 3.

(same as stage 1 step 1)

Pretrained models aggregeated to `EnsembleModel` with test time augmentation (flip, rotate90) - all predicitons averaged by simple mean and thresholded by 0.5 value. First, prediction is made for stitched test images, than for others (which not present on mosaic).

---
#### Stage 3.
---

The stage is the same as previous one, just another round of pseudolabelng with better trained models.   
Train data preparation (pseudolabels), models training and prediction with ensemble of models (consist of three steps).     


Command to run:  
```bash
$ make stage3
```

##### Step 1.

Take predictions from previous stage and prepare them to use as training data for current stage (same as stage 2 step 1). 

##### Step 2.

(same as stage 2 step 1, but with exta data labeled on previous stage)

On this step 4 `Unet` models and 1 `FPN` are going to be trained on `tier 1` data and `pseudolabels round 2`. Models trained with hard augmentations using `albumetations` library and random data sampling. Training lasts 50 epochs with continious learining rate decay from 0.0001 to 0.

##### Step 3.

(same as stage 2 step 1)

Pretrained models aggregeated to `EnsembleModel` with test time augmentation (flip, rotate90) - all predicitons averaged by simple mean and thresholded by 0.5 value. First, prediction is made for stitched test images, than for others (which not present on mosaic).


## Your data prediction

1. Start service ```$ make build && make start && make install```
2. Put your `.tif` file somewhere in data folder (make sure you reproject it in UTM zone and resample to 0.1 m/pixel). You can use GDAL for example.
3. Run prediction

```bash
$ docker exec open-cities-dev \
    python -m src.predict_tif \
      --configs configs/stage3-srx50-2-f0.yaml \
      --src_path <path/to/your/tif/file.tif> \
      --dst_path <path/for/result.tif> \
      --batch_size 8 \
      --gpu 0 \
      --tta
```

4. For help:  
```bash 
$ docker exec open-cities-dev python -m src.predict_tif --help
```