# FastForensics: Efficient Two-Stream Design for Real-Time Image Manipulation Detection

![Powered by](https://img.shields.io/badge/Based_on-Pytorch-blue?logo=pytorch)

This repo contains an official PyTorch implementation of our paper: [FastForensics: Efficient Two-Stream Design for Real-Time Image Manipulation Detection.]()


## 🌏Overview

<img src="images/overview.png" style="zoom: 85%;" />

## 🛠️ Setup

### ⚙️ Installation

You can run the following script to configure the necessary environment:

```sh
pip install -r requirements.txt
```

###  📑Data Preparation

This paper uses [PSCC]() dataset, please put the downloaded dataset under `./data/PSCC` dataset as required. Besides, this paper does not use all the PSCC dataset, but only about 20,000 images from different PSCC tampering methods. Please extract the first 20,000 images from each tampering method of PSCC according to the `./lib/dataloader` format for training.
## 🚀 Training and Testing

In the training phase, the model is trained on PSCC dataset.

```sh
python train.py
```

In the testing phase, the training and test sets need to be swapped out according to the requirements of the dataset, 
and the results presented in this thesis are only for the test set of the dataset. The images in the test set should be placed in `./data`.After downloading the [weights](https://pan.baidu.com/s/11YdleU19DgB6bUM_hbsBUw?pwd=w8xg), please place them in the root directory of the project.

##  Citation
If you find our repo useful for your research, please consider citing our paper:
```latex
@InProceedings{li2024fastforensics,
    author    = {Zhang, YangXiang and Li, YueZun and Ao Luo and Zhou, JiaRan and Dong, JunYu},
    title     = {FastForensics: Efficient Two-Stream Design for Real-Time Image Manipulation Detection},
    booktitle = {Proceedings of the British Machine Vision Conference (BMVC)},
    year      = {2024},
}
```