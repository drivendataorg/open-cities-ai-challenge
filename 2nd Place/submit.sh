#!/bin/bash


python ./src/submit.py --exp last_1024_pl2_ft --to-save ./data/test_sub_inseefdensenet_1024_f34_pl2_ft --batch-size 100 --res 1024 --data ./data/test_tiles_1024

# parallize it
# CUDA_VISIBLE_DEVICES=0 python ./src/submit.py --exp last_1024_pl2_ft --to-save ./data/test_sub_inseefdensenet_1024_f34_pl2_ft --n-parts 6 --part 0 --batch-size 100 --res 1024 --data ./data/test_tiles_1024
# CUDA_VISIBLE_DEVICES=1 python ./src/submit.py --exp last_1024_pl2_ft --to-save ./data/test_sub_inseefdensenet_1024_f34_pl2_ft --n-parts 6 --part 1 --batch-size 100 --res 1024 --data ./data/test_tiles_1024
# CUDA_VISIBLE_DEVICES=2 python ./src/submit.py --exp last_1024_pl2_ft --to-save ./data/test_sub_inseefdensenet_1024_f34_pl2_ft --n-parts 6 --part 2 --batch-size 100 --res 1024 --data ./data/test_tiles_1024
# CUDA_VISIBLE_DEVICES=3 python ./src/submit.py --exp last_1024_pl2_ft --to-save ./data/test_sub_inseefdensenet_1024_f34_pl2_ft --n-parts 6 --part 3 --batch-size 100 --res 1024 --data ./data/test_tiles_1024
# CUDA_VISIBLE_DEVICES=4 python ./src/submit.py --exp last_1024_pl2_ft --to-save ./data/test_sub_inseefdensenet_1024_f34_pl2_ft --n-parts 6 --part 4 --batch-size 100 --res 1024 --data ./data/test_tiles_1024
# CUDA_VISIBLE_DEVICES=5 python ./src/submit.py --exp last_1024_pl2_ft --to-save ./data/test_sub_inseefdensenet_1024_f34_pl2_ft --n-parts 6 --part 5 --batch-size 100 --res 1024 --data ./data/test_tiles_1024
