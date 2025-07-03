# MAFUNet: Mamba with Adaptive Fusion UNet for Medical Image Segmentation

[![Paper](https://img.shields.io/badge/Paper-Preprint-blue)](https://#) <!-- You can replace '#' with the link to your paper, e.g., an ArXiv link -->

This is the official PyTorch implementation for the paper **"MAFUNet: Mamba with Adaptive Fusion UNet for Medical Image Segmentation"**.

## 🚀 Getting Started

### 1. Installation

We recommend using Conda to create a virtual environment and install the required dependencies.

```bash
# 1. Create and activate the conda environment
conda create -n mafunet python=3.8 -y
conda activate mafunet

# 2. Install PyTorch (please choose the command that matches your CUDA version from the PyTorch website)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# 3. Install other dependencies
pip install -r requirements.txt
```


### 2. Dataset Preparation

1.  Download the datasets from the following links:
    *   [**BUSI**](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
    *   [**CVC-ClinicDB**](https://polyp.grand-challenge.org/CVCClinicDB/)
    *   [**ISIC-2018**](https://challenge.isic-archive.com/data/)

2.  Organize the datasets into the following directory structure:

    ```
    data/
    ├── BUSI/
    │   ├── images/
    │   │   ├── benign (1).png
    │   │   └── ...
    │   └── masks/
    │       ├── benign (1)_mask.png
    │       └── ...
    ├── CVC-ClinicDB/
    │   ├── Original/

    │   └── Ground Truth/
    │       ├── 1.tif
    │       └── ...
    └── ISIC2018/
        ├── ISIC2018_Task1-2_Training_Input/
        └── ISIC2018_Task1_Training_GroundTruth/
    ```

## 📜 Citation

If you find our work helpful for your research, please consider citing our paper.

## 🙏 Acknowledgements

Our work is built upon many excellent open-source projects. We would like to express our gratitude to:
*   VM-UNet
*   UltraLight-VMUNet

