# MarginDistillation: distillation for margin-basedsoftmax

This repository contains an implementation of the distillation methods compared in this [paper](https://arxiv.org/pdf/2003.02586v1.pdf). Using the code from this repository, you can train a lightweight network to recognize faces for embedded devices.
The repository contains the code for the following methods:
* [Angular distillation](https://arxiv.org/pdf/1905.10620.pdf)
* [Triplet distillation L2](https://arxiv.org/pdf/1905.04457.pdf)
* [Triplet distillation Cos](https://arxiv.org/pdf/1905.04457.pdf)
* [Margin based with T](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11433/114330O/Margin-based-knowledge-distillation-for-mobile-face-recognition/10.1117/12.2557244.short?SSO=1)
* [MarginDistillation (our)](https://arxiv.org/pdf/2003.02586v1.pdf)

# Data preparation

  1) Download dataset https://github.com/deepinsight/insightface/wiki/Dataset-Zoo 
  2) Extract images using: data_prepare/bin_get_images.ipynb
  3) Save vectors from Resnet100 using: data_prepare/save_embedings.ipynb
  4) Prepare a list for conversion to .bin file using: data_prepare/save_lst.ipynb
  5) Convert to .bin file using: insightface/blob/master/src/data/dir2rec.py

### Training

* ##### Resnet100 (Teacher network)
Download from [google drive](https://drive.google.com/open?id=1OjQ15abzTga2ixqUeCiQ6OEMtMRi1XDX).
Train Resnet100 with Arcface:
```sh
$ CUDA_VISIBLE_DEVICES='0,1' python3 -u train.py --network r100 --loss arcface --dataset emore
```
Performance:
| lfw | cfp-fp | agedb-30 | megaface |
| ------ | ------ | ------ | ------ |
| 99.76% | 98.38% | 98.25% | 98.35% |
* ##### Arcface:
Download from [google drive](https://drive.google.com/open?id=1dMfDS0VF_moi8m6JsK8Sro8HX7OyAjS4).
Train MobileFaceNet with Arcface:
```sh
$ CUDA_VISIBLE_DEVICES='0,1' python3 -u train.py --network y1 --loss arcface --dataset emore
```
Performance:
| lfw | cfp-fp | agedb-30 | megaface |
| ------ | ------ | ------ | ------ |
| 99.51% | 92.68% | 96.13% | 90.62% |
* ##### Angular distillation:
Download from [google drive](https://drive.google.com/open?id=16Iiv8ks07A5DzUunBWfacvKiwN9_haci).
Train MobileFaceNet with Angular distillation:
```sh
$ CUDA_VISIBLE_DEVICES='0,1' python3 -u train.py --network y1 --loss angular_distillation --dataset emore_soft
```
Performance:
| lfw | cfp-fp | agedb-30 | megaface |
| ------ | ------ | ------ | ------ |
| 99.55% | 91.90% | 96.01% | 90.73% |
* ##### Triplet distillation L2:
Download from [google drive](https://drive.google.com/open?id=1R_PSMxyfZOcpMXxZ5z_SKypH6fevQ9lx).
Finetune MobileFaceNet with Triplet distillation L2:
```sh
$ CUDA_VISIBLE_DEVICES='0,1' python3 -u train.py --network y1 --loss triplet_distillation_L2 --dataset emore_soft --pretrained ./models/y1-arcface-emore/model
```
Performance:
| lfw | cfp-fp | agedb-30 | megaface |
| ------ | ------ | ------ | ------ |
| 99.56% | 93.30% | 96.23% | 89.10% |
* ##### Triplet distillation cos:
Download from [google drive](https://drive.google.com/open?id=1zeTxzRmxkzhmPzu_yTnMvE7FiaGN03Ub).
Finetune MobileFaceNet with Triplet distillation cos:
```sh
$ CUDA_VISIBLE_DEVICES='0,1' python3 -u train.py --network y1 --loss triplet_distillation_cos --dataset emore_soft --pretrained ./models/y1-arcface-emore/model
```
Performance:
| lfw | cfp-fp | agedb-30 | megaface |
| ------ | ------ | ------ | ------ |
| 99.55% | 93.30% | 95.60% | 86.52% |
* ##### Margin based with T
Download from [google drive](https://drive.google.com/open?id=18i3r7nAKfPD50Tg09qgMGsLe4m4Woyhi).
Train MobileFaceNet with Margin based distillation with T:
```sh
$ CUDA_VISIBLE_DEVICES='0,1' python3 -u train.py --network y1 --loss margin_base_with_T --dataset emore_soft
```
Performance:
| lfw | cfp-fp | agedb-30 | megaface |
| ------ | ------ | ------ | ------ |
| 99.41% | 92.40% | 96.01% | 90.77% |
* ##### MarginDistillation:
Download from [google drive](https://drive.google.com/open?id=1IWdOUsdKRC64oNIkXwSau33qy-WqdYKi).
Train MobileFaceNet with MarginDistillation:
```sh
$ CUDA_VISIBLE_DEVICES='0,1' python3 -u train.py --network y1 --loss margin_distillation --dataset emore_soft
```
Performance:
| lfw | cfp-fp | agedb-30 | megaface |
| ------ | ------ | ------ | ------ |
| 99.61% | 92.01% | 96.55% | 91.70% |
