#!/bin/bash


GPU=0,1,2,3,4,5
N_GPUS=6


# FIRST TRAINING

ENCODER=efficientnet-b4
WORK_DIR=${ENCODER}_1024

# train
CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node=$N_GPUS ./src/train.py --work-dir $WORK_DIR --batch-size 18 --lr 0.0001 --encoder $ENCODER --n-folds 5 --fold 0 --epochs 60 --attention-type scse --n-classes 2 --csv train_1024.csv --data ./data/train_tier_1_tiles_1024/z19
CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node=$N_GPUS ./src/train.py --work-dir $WORK_DIR --batch-size 18 --lr 0.0001 --encoder $ENCODER --n-folds 5 --fold 1 --epochs 60 --attention-type scse --n-classes 2 --csv train_1024.csv --data ./data/train_tier_1_tiles_1024/z19
CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node=$N_GPUS ./src/train.py --work-dir $WORK_DIR --batch-size 18 --lr 0.0001 --encoder $ENCODER --n-folds 5 --fold 2 --epochs 60 --attention-type scse --n-classes 2 --csv train_1024.csv --data ./data/train_tier_1_tiles_1024/z19

# save models using torch.jit
CUDA_VISIBLE_DEVICES=4 python ./src/predict.py --load efficientnet-b4_1024/efficientnet-b4_b18_adam_lr0.0001_c0_fold0_z0_w3m0_t20_hm0/best.pth
CUDA_VISIBLE_DEVICES=4 python ./src/predict.py --load efficientnet-b4_1024/efficientnet-b4_b18_adam_lr0.0001_c0_fold1_z0_w3m0_t20_hm0/best.pth
CUDA_VISIBLE_DEVICES=4 python ./src/predict.py --load efficientnet-b4_1024/efficientnet-b4_b18_adam_lr0.0001_c0_fold2_z0_w3m0_t20_hm0/best.pth

# submission
PREDS=./data/test_sub_eff4_1024_f012
CUDA_VISIBLE_DEVICES=0 python ./src/submit.py --exp efficientnet-b4_1024 --to-save ./data/test_sub_eff4_1024_f012 --n-parts 4 --part 0 --batch-size 128 --res 1024 --data ./data/test_tiles_1024
CUDA_VISIBLE_DEVICES=1 python ./src/submit.py --exp efficientnet-b4_1024 --to-save ./data/test_sub_eff4_1024_f012 --n-parts 4 --part 1 --batch-size 128 --res 1024 --data ./data/test_tiles_1024
CUDA_VISIBLE_DEVICES=2 python ./src/submit.py --exp efficientnet-b4_1024 --to-save ./data/test_sub_eff4_1024_f012 --n-parts 4 --part 2 --batch-size 128 --res 1024 --data ./data/test_tiles_1024
CUDA_VISIBLE_DEVICES=5 python ./src/submit.py --exp efficientnet-b4_1024 --to-save ./data/test_sub_eff4_1024_f012 --n-parts 4 --part 3 --batch-size 128 --res 1024 --data ./data/test_tiles_1024

# END FIRST TRAINING

# SECOND TRAINING

# pseudo-labeling
python ./src/prepare_pl.py --preds-path $PREDS --data-path ./data/test_tiles_1024

PL=${PREDS}.csv
WORK_DIR=dpn_dense_1024_pl

# train
CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node=$N_GPUS ./src/train.py --work-dir $WORK_DIR --batch-size 22 --lr 0.0001 --encoder dpn92 --n-folds 5 --fold 3 --epochs 60 --n-classes 2 --csv train_1024.csv --data ./data/train_tier_1_tiles_1024/z19 --pl ${PL}
CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node=$N_GPUS ./src/train.py --work-dir $WORK_DIR --batch-size 16 --lr 0.0001 --encoder densenet161 --n-folds 5 --fold 4 --epochs 60 --n-classes 2 --csv train_1024.csv --data ./data/train_tier_1_tiles_1024/z19 --pl ${PL}

# save models using torch.jit
CUDA_VISIBLE_DEVICES=1 python ./src/predict.py --load dpn_dense_1024_pl/dpn92_b22_adam_lr0.0001_c0_fold3_z0_w3m0_t20_hm0/best.pth
CUDA_VISIBLE_DEVICES=1 python ./src/predict.py --load dpn_dense_1024_pl/densenet161_b16_adam_lr0.0001_c0_fold4_z0_w3m0_t20_hm0/best.pth

# submission
PREDS=./data/test_sub_dpn92dense_1024_f34_pl
CUDA_VISIBLE_DEVICES=0 python ./src/submit.py --exp dpn_dense_1024_pl --to-save ./data/test_sub_dpn92dense_1024_f34_pl --n-parts 5 --part 0 --batch-size 128 --res 1024 --data ./data/test_tiles_1024
CUDA_VISIBLE_DEVICES=1 python ./src/submit.py --exp dpn_dense_1024_pl --to-save ./data/test_sub_dpn92dense_1024_f34_pl --n-parts 5 --part 1 --batch-size 128 --res 1024 --data ./data/test_tiles_1024
CUDA_VISIBLE_DEVICES=2 python ./src/submit.py --exp dpn_dense_1024_pl --to-save ./data/test_sub_dpn92dense_1024_f34_pl --n-parts 5 --part 2 --batch-size 128 --res 1024 --data ./data/test_tiles_1024
CUDA_VISIBLE_DEVICES=3 python ./src/submit.py --exp dpn_dense_1024_pl --to-save ./data/test_sub_dpn92dense_1024_f34_pl --n-parts 5 --part 3 --batch-size 128 --res 1024 --data ./data/test_tiles_1024
CUDA_VISIBLE_DEVICES=5 python ./src/submit.py --exp dpn_dense_1024_pl --to-save ./data/test_sub_dpn92dense_1024_f34_pl --n-parts 5 --part 4 --batch-size 128 --res 1024 --data ./data/test_tiles_1024

# END SECOND TRAINING

# THIRD TRAINING

# pseudo-labeling
python ./src/prepare_pl.py --preds-path ./data/test_sub_eff4_1024_f0 --data-path ./data/test_tiles_1024

PL=${PREDS}.csv
WORK_DIR=last_1024_pl2_ft

# train
CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node=$N_GPUS ./src/train.py --work-dir $WORK_DIR --batch-size 28 --lr 0.0001 --encoder inceptionv4 --n-folds 5 --fold 3 --epochs 27 --n-classes 2 --csv train_1024.csv --data ./data/train_tier_1_tiles_1024/z19 --pl ${PL} --ft
CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node=$N_GPUS ./src/train.py --work-dir $WORK_DIR --batch-size 12 --lr 0.0001 --encoder senet154 --n-folds 5 --fold 4 --epochs 27 --n-classes 2 --csv train_1024.csv --data ./data/train_tier_1_tiles_1024/z19 --pl ${PL} --ft
CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node=$N_GPUS ./src/train.py --work-dir $WORK_DIR --batch-size 26 --lr 0.0001 --encoder se_resnext50_32x4d --n-folds 5 --fold 3 --epochs 27 --n-classes 2 --csv train_1024.csv --data ./data/train_tier_1_tiles_1024/z19 --pl ${PL} --ft
CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node=$N_GPUS ./src/train.py --work-dir $WORK_DIR --batch-size 16 --lr 0.0001 --encoder densenet161 --n-folds 5 --fold 1 --epochs 27 --n-classes 2 --csv train_1024.csv --data ./data/train_tier_1_tiles_1024/z19 --pl ${PL} --ft
CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node=$N_GPUS ./src/train.py --work-dir $WORK_DIR --batch-size 18 --lr 0.0001 --encoder efficientnet-b4  --attention-type scse --n-folds 5 --fold 0 --epochs 27 --n-classes 2 --csv train_1024.csv --data ./data/train_tier_1_tiles_1024/z19 --pl ${PL} --ft

# save models using torch.jit
CUDA_VISIBLE_DEVICES=0 python ./src/predict.py --load last_1024_pl2_ft/senet154_b12_adam_lr0.0001_c0_fold4_z0_w3m0_t20_hm0/best.pth
CUDA_VISIBLE_DEVICES=1 python ./src/predict.py --load last_1024_pl2_ft/se_resnext50_32x4d_b26_adam_lr0.0001_c0_fold3_z0_w3m0_t20_hm0/best.pth
CUDA_VISIBLE_DEVICES=2 python ./src/predict.py --load last_1024_pl2_ft/densenet161_b16_adam_lr0.0001_c0_fold1_z0_w3m0_t20_hm0/best.pth
CUDA_VISIBLE_DEVICES=3 python ./src/predict.py --load last_1024_pl2_ft/inceptionv4_b28_adam_lr0.0001_c0_fold3_z0_w3m0_t20_hm0/best.pth
CUDA_VISIBLE_DEVICES=4 python ./src/predict.py --load last_1024_pl2_ft/efficientnet-b4_b18_adam_lr0.0001_c0_fold0_z0_w3m0_t20_hm0/best.pth

# submission
PREDS=./data/test_sub_inseefdensenet_1024_f34_pl2_ft
CUDA_VISIBLE_DEVICES=0 python ./src/submit.py --exp last_1024_pl2_ft --to-save ./data/test_sub_inseefdensenet_1024_f34_pl2_ft --n-parts 6 --part 0 --batch-size 100 --res 1024 --data ./data/test_tiles_1024
CUDA_VISIBLE_DEVICES=1 python ./src/submit.py --exp last_1024_pl2_ft --to-save ./data/test_sub_inseefdensenet_1024_f34_pl2_ft --n-parts 6 --part 1 --batch-size 100 --res 1024 --data ./data/test_tiles_1024
CUDA_VISIBLE_DEVICES=2 python ./src/submit.py --exp last_1024_pl2_ft --to-save ./data/test_sub_inseefdensenet_1024_f34_pl2_ft --n-parts 6 --part 2 --batch-size 100 --res 1024 --data ./data/test_tiles_1024
CUDA_VISIBLE_DEVICES=3 python ./src/submit.py --exp last_1024_pl2_ft --to-save ./data/test_sub_inseefdensenet_1024_f34_pl2_ft --n-parts 6 --part 3 --batch-size 100 --res 1024 --data ./data/test_tiles_1024
CUDA_VISIBLE_DEVICES=4 python ./src/submit.py --exp last_1024_pl2_ft --to-save ./data/test_sub_inseefdensenet_1024_f34_pl2_ft --n-parts 6 --part 4 --batch-size 100 --res 1024 --data ./data/test_tiles_1024
CUDA_VISIBLE_DEVICES=5 python ./src/submit.py --exp last_1024_pl2_ft --to-save ./data/test_sub_inseefdensenet_1024_f34_pl2_ft --n-parts 6 --part 5 --batch-size 100 --res 1024 --data ./data/test_tiles_1024

# END THIRD TRAINING
