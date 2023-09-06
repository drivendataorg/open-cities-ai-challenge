[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

![Banner Image](https://s3.amazonaws.com/drivendata-public-assets/opendri_mon_labeled.jpg)

# Open Cities AI Challenge: Segmenting Buildings for Disaster Resilience 

## Goal of the Competition

As urban populations grow, more people are exposed to the benefits and hazards of city life. One challenge for cities is managing the risk of disasters in a constantly changing built environment. Buildings, roads, and critical infrastructure need to be mapped frequently, accurately, and in enough detail to represent assets important to every community. Knowing where and how assets are vulnerable to damage or disruption by natural hazards is key to disaster risk management (DRM).

In this challenge, participants segment building footprints from aerial imagery. The data consists of drone imagery from 10 different cities and regions across Africa. The goal is to classify the presence or absence of a building on a pixel-by-pixel basis.

## What's in this Repository

This repository contains code from winning competitors in the [Open Cities AI Challenge: Segmenting Buildings for Disaster Resilience](https://www.drivendata.org/competitions/60/building-segmentation-disaster-resilience/) DrivenData challenge. Code for all winning solutions are open source under the MIT License.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

Place |Team or User | Public Score | Private Score | Summary of Model
--- | --- | --- | --- | ---
1 | qubvel | 0.858401 | 0.859849 | The first stage was ensemble of Unet models trained on noisy labels with hard image augmentations. The next two stages were trained on noisy train data and automatically labeled by stage 1 model test data (pseudolabels).
2 | kbrodt | 0.857154 | 0.857532 | Unet-like models with heavy encoders. To account for rare tiles, assign some class to the tile and use the inverse probability of that class in the dataset to oversample them. Also change binary cross-entropy loss to multiclass cross-entropy (in our case 2 output channels) and take argmax instead of searching an optimal threshold. To overfit on the test set you can do pseudo-labeling. After obtaining a strong single model, train five more models and ensemble them by simple averaging.
3 | MichalBusta | 0.839302 | 0.840065 | First, trained a network using FPN with efficient-net(b1) backbone with Focal loss and Dice loss. In the second stage, self-training with a noisy student and negative mining strategies were used with the efficient-net(b2) backbone and KL divergence loss. Ensemble models from steps 1 and 2 with TTA augmentation (scale and flipping).

Additional solution details can be found in the `reports` folder inside the directory for each submission.

**Benchmark Blog Post: ["Open Cities AI Challenge benchmark model"](https://github.com/azavea/open-cities-ai-challenge-benchmark-model)**

**["Meet the Winners" Blog Post"](https://drivendata.co/blog/open-cities-disaster-winners/)**

## Responsible AI track

This competition also included a **[Responsible AI track](https://www.drivendata.org/competitions/60/building-segmentation-disaster-resilience/page/152/)** in which participants were asked to apply an ethical lens to the design and use of AI systems for DRM. The top three submissions can be viewed in OpenDRI's [Perspectives on Responsible AI for Disaster Risk Management](https://opendri.org/resource/perspectives-on-responsible-ai-for-drm/). The winning reports are also included in this repository.
