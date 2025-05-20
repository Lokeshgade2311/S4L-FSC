## Spectral-Spatial Self-Supervised Learning for Few-Shot Hyperspectral Image Classification
This repository contains the official PyTorch implementation for the paper:
**Spectral-Spatial Self-Supervised Learning for Few-Shot Hyperspectral Image Classification**
Wenchen Chen, Yanmei Zhang, Zhongwei Xiao, Jianping Chu, Xingbo Wang
[[arXiv paper]](https://arxiv.org/abs/2505.12482).
This is the author's first time open-sourcing code, so please feel free to point out any errors or provide suggestions.

If possible, please give the author a star.

The work in this project is based on the excellent open-source projects DCFSL (available at https://github.com/Li-ZK/DCFSL-2021) and FSCF-SSL (available at https://github.com/Li-ZK/FSCF-SSL-2023). We express our sincere gratitude to their creators for the invaluable resources they have provided to the community and for the solid foundation upon which this work is built.

## Requirements
- CUDA = 12.4
- Python = 3.7 
- Pytorch = 1.11.0
- sklearn = 0.23.2
- numpy = 1.21.6
- torchvision = 0.12.0

## Datasets
The Chikusei dataset can be obtained from the Hyperspectral Remote Sensing Scenes website: http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes.
The mini-ImageNet dataset and the pre-trained VGG network can be downloaded from the FSCF-SSL repository: https://github.com/Li-ZK/FSCF-SSL-2023.

The directory structure for the dataset files should be as follows:
```
datasets
├── Chikusei
│   ├── HyperspecVNIR_Chikusei_20140729.mat
│   ├── HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat
├── IP
│   ├── indian_pines_corrected.mat
│   ├── indian_pines_gt.mat
└── paviaU
│   ├── paviaU.mat
│   ├── paviaU_gt.mat
└── HC
│   ├── WHU_Hi_HanChuan.mat
│   ├── WHU_Hi_HanChuan(15%)_gt.mat
└── Salinas
│   ├── Salinas_corrected.mat
│   ├── salinas_gt.mat
└── miniImagenet
│   ├── 
│   ├── 

```
## Usage:
Take FSCF-SSL method : 
1. Download the required data set and move to folder`./datasets`.
2. Download the VGG pre-training weight, put in root directory.
3. Run chikusei_imdb_128.py to generate chikusei spectrum vector data.
4. Run S4L-FSC.py,
