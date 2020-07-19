## DTVNet &mdash; Official PyTorch Implementation

![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic) ![PyTorch 1.5.1](https://img.shields.io/badge/pytorch-1.5.1-green.svg?style=plastic) ![License MIT](https://img.shields.io/github/license/zhangzjn/APB2Face)

Official pytorch implementation of the paper "[DTVNet: Dynamic Time-lapse Video Generation via Single Still Image, ECCV'20 (Spotlight)](https://openreview.net/forum?id=49rFxGCmfm)".

For any inquiries, please contact Jiangning Zhang at [186368@zju.edu.cn](mailto:186368@zju.edu.cn)

## Using the Code

### Requirements

This code has been developed under `Python3.7`, `PyTorch 1.3.1` and `CUDA 10.1` on `Ubuntu 16.04`. 


```shell
# Install python3 packages
pip3 install -r requirements.txt
```

### Datasets in the paper
- Download [Sky Timeplase](https://drive.google.com/file/d/1xWLiU-MBGN7MrsFHQm4_yXmfHBsMbJQo/view?usp=drive_open) dataset to `data`. You can refer to [MDGAN](https://arxiv.org/pdf/1709.07592.pdf) and corresponding [code](https://github.com/weixiong-ur/mdgan) for more details about the dataset.
- Download `example datasets and checkpoints` from 
[Google Drive](https://drive.google.com/file/d/1xEnZwNLU4SmgFFh4WGV4KEOdegfFrOdp/view?usp=sharing) 
or 
[Baidu Cloud](https://pan.baidu.com/s/1gj31dZx5tp4s6OzH5K2Tnw) (Key:u6c0).

### Unsupervised Flow Estimation
1. Our another work [ARFlow (CVPR'20)](https://github.com/lliuz/ARFlow) is used as the unsupervised optical flow estimator in the paper. You can refer to `flow/ARFlow/README.md` for more details.

2. Training:

   ```shell
   > Modify `configs/sky.json` if you use another data_root or settings.
   cd flow/ARFlow
   python3 train.py
   ```
   
3. Testing:

    ```shell
    > Pre-traind model is located in `checkpoints/Sky/sky_ckpt.pth.tar`
    python3 inference.py --show  # Test and show a single pair images.
    python3 inference.py --root ../../data/sky_timelapse/ --save_path ../../data/sky_timelapse/flow/  # Generate optical flow in advance for Sky Time-lapse dataset.
    ```

### Running
1. Train `DTVNet` model.

   ```shell
   > Modify `configs/sky_timelapse.json` if you use another data_root or settings.
   python3 train.py
   ```
   
2. Test `DTVNet` model.

   ```shell
   > Pre-traind model is located in `checkpoints/DTV_Sky/200708162546`
   > Results are save in `checkpoints/DTV_Sky/200708162546/results`
   python3 Test.py
   ```
   
### Citation

TBA



