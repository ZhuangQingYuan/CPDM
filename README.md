# CPDM

CPDM: Achieving Superior Accuracy in Semi-Supervised Remote Sensing Segmentation

# Getting Started
## Install
```python
conda create -n CPDM python=3.8
pip install -r requirements.txt
```
## Data Preparation and Pre-trained Model
Refer to the preparation of the following code repository 

https://github.com/xiaoqiang-lu/WSCL.git

## File Organization

```python
├── ./pretrained
    └── resnet101.pth
    
├── [RsDataSets/Vaihingen(DFC22/GID15/...)]
    ├── images
    └── labels
```
# Training

It is recommended to use [Visual Studio Code](https://code.visualstudio.com/) as the compiler.

```python
cd SemiExp/DATASETNAME
python train.py
```
# Acknowledgement

We thank [WSCL](https://github.com/xiaoqiang-lu/WSCL.git), [LSST](https://github.com/xiaoqiang-lu/LSST.git), and [UniMatch](https://github.com/LiheYoung/UniMatch.git), for part of their codes, processed datasets, data partitions, and pretrained models.

