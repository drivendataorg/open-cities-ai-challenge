# Open Cities AI Challenge: Segmenting Buildings for Disaster Resilience

[Open Cities AI Challenge: Segmenting Buildings for Disaster Resilience](https://www.drivendata.org/competitions/60/building-segmentation-disaster-resilience/page/150/).
The pipeline follows [severstal](https://github.com/kbrodt/severstal).

[2nd place](https://www.drivendata.org/competitions/60/building-segmentation-disaster-resilience/leaderboard/)
out of 1106 with 0.8575 jaccard index (top 1 -- 0.8598).

### Prerequisites

- GPU(s) with 32Gb RAM (e.g. Tesla V100)
- [NVIDIA apex](https://github.com/NVIDIA/apex)

```bash
pip install -r requirements.txt
```

### Usage

#### Download

First download the train and test data from the competition link into [data](./data) folder.

Then you must prepare train and test datasets. For train use:

```python
python ./src/data/prepare_train.py
```

Actually, it's hard to install all dependencies like [`GDAL`](https://gdal.org)
etc. So, it's better to use [code](https://colab.research.google.com/drive/1PvWONzToxReltd2quODDRhsQRd_T9oEA)
from competition forum in [Getting started](https://community.drivendata.org/t/getting-started/4022)
topic by `@johnowhitaker`. Or even better download extracted tiles from kaggle dataset

```bash
kaggle d download kbrodt/oc-t1-1024-z19-p1
kaggle d download kbrodt/oc-t1-1024-z19-p2
kaggle d download kbrodt/oc-t1-1024-z19-p3
```

For test simply use

```python
python ./src/data/prepare_test.py
```

#### Train

To train the model run

```bash
sh ./train.sh
```

On 6 GPUs Tesla V100 it will take around 50h. This will generates trained models and submission file.

#### Predict

If you want only predict the test set,
first you need to download model weights from [yandex disk](https://yadi.sk/d/RDSYG6PzSlc5dQ),
unzip and execute:

```bash
sh ./submit.sh
```

On 6 GPUs Tesla V100 it takes around 1h.

### Approach

Final solution turned out to be quite straightforward.

#### Summary
 
- Train tier 1, 1024x1024 tile's size with 19 zoom level
- 5 folds stratified by `area_scene`
- Balanced sampler by `area_scene`
- Unet-like with heavy encoders: `senet154`, `se_resnext50_32x4d`, `densenet161`, `inceptionv4`, `efficientnet-b4`
- Cross-entropy loss (not to search threshold for binarization) 
- 2 rounds of pseudo-labeling

#### Tried

- Although we have train tier 2 dataset with "dirty" labels,
we can pretrain on it and finetune on tier 1, but it doesn't work for me.
Another way to use tier 2 is to fix "dirty" labels. I tried to train on
tier 2 predictions of models trained on tier 1 with knowledge-distillation (KD),
but it works the same if we train only on tier 1. I only managed train a single model
`efficientnet-b3` with score 85.02.

- Different zoom levels (18 and 20) greatly increases data size,
hence increases training time, so I gave up it.

- Instead of 1-channel footprints I used 3-channel mask footprint/boundary/contact,
but I didn't managed better results.

- MixUp training

#### Possible improvements

- Do first rounds of pseudo-labeling with heavier encoders (instead using `effnets`)
- Use different zoom levels if have time
- Fix/filter/clean/remove "dirty" labels of tier 2 using trained models
on tier 1 by calculating jaccard index between "dirty" label and 
prediction. Add labels with high score, because it signifies that
labels are close enough to predictions, where latter are obtained
on clean tier 1 dataset. Remove labels with low score.
See some examples and ideas [here](https://docs.google.com/document/d/1M6dMO1wYB2n93qgbNxUwmmvcgi4G5gymI7sq5z63_hI/edit?usp=sharing).
