# Image Annotator (IAODF)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description

Image Annotator is a Python-based tool designed for interactive annotation of objects in (historical) aerial and satellite imagery. Based on positive and negative samples (boxes or points) the model predicts bounding boxes for the desired class. Currently the model is not class agnostic, but can only predict classes available during training (see weights for the appropriate weight). The model is based on RT-DETR with a ResNet18 backbone. There are four different pretrained weights are available: CHAI, DOTAv2-Tiny, AITOD, and SARDET.

## Prerequisites

- Anaconda or Miniconda
- CUDA-compatible GPU (for PyTorch GPU acceleration)
- At least 4 GB of VRAM (GPU) (8GB recommended)
- At least 32 GB of RAM

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/mburges-cvl/WACV_IAODF
   cd WACV_IAODF
   ```

2. Create and activate a new Conda environment:

   ```
   conda create --name image_annotator python=3.11.8
   conda activate image_annotator
   ```

3. Install PyTorch and related packages:

   ```
   pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
   python -m pip install matplotlib pandas PyQT5 pillow numpy==1.* tqdm
   ```

## Usage

1. Activate the Conda environment:

   ```
   conda activate image_annotator
   ```

2. Run the main script:

   ```
   python main.py
   ```

## Features

- Interactive detection of objects in aerial and satellite imagery
- Supports positive and negative samples
- Supports four different pretrained weights: CHAI, DOTAv2-Tiny, AITOD, and SARDET
- Supports the following file formats: .jpg, .jpeg, .png, .tif, .tiff
- Supports three different modalities: Historical (Grayscale), RGB, and SAR

## Weights & Sample Data

    ```
    git lfs pull
    ```

## Limitations

- Currently only supports one class, even if the underlying model is capable of multi-class detection
- Due to the spliting of the image into patches (required for fixed memory consumption) the model might produce false positives/negatives at the border of the patches. This can be mitigated by increasing the patch size, but will increase memory consumption (not tested).

## License

This project is licensed under the MIT License.

## Acknowledgments

- Removed for double-blind review

## Citation

If you use this tool in your research, please cite the following paper:

```
- Removed for double-blind review
```
