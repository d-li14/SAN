# Scale Adaptive Network
Official implementation of Scale Adaptive Network (SAN) as described in [Learning to Learn Parameterized Classification Networks for Scalable Input Images](https://arxiv.org/abs/2007.06181) (ECCV'20) by  [Duo Li](https://github.com/d-li14), [Anbang Yao](https://github.com/YaoAnbang) and [Qifeng Chen](https://github.com/CQFIO) on the [ILSVRC 2012](http://www.image-net.org) benchmark.

<p align="center"><img src="fig/schema.png" width="800" /></p>

We present a meta learning framework which dynamically parameterizes main networks conditioned on its input resolution at runtime, leading to efficient and flexible inference for arbitrarily switchable input resolutions.

<p align="center"><img src="fig/ablation.png" width="700" /></p>

## Requirements

### Dependency

* PyTorch 1.0+
* [NVIDIA-DALI](https://github.com/NVIDIA/DALI) (in development, not recommended)

### Dataset

Download the ImageNet dataset and move validation images to labeled subfolders. To do this, you can use the following script: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

## Pre-trained Models

### Baseline (individually trained on each resolution)

#### ResNet-18

| Resolution | Top-1 Acc. | Download                                                     |
| ---------- | ---------- | ------------------------------------------------------------ |
| 224x224    | 70.974     | [Google Drive](https://drive.google.com/file/d/1rpOqyLsHi7xuB3XWuXfxE_hojs3zm_BR/view?usp=sharing) |
| 192x192    | 69.754     | [Google Drive](https://drive.google.com/file/d/1fj3S-nzCzgXjIYUYAoA654HO1DNEl_4O/view?usp=sharing) |
| 160x160    | 68.482     | [Google Drive](https://drive.google.com/file/d/1tb9rUsRcw5wWEDREw0Fi1TUTyP73NXjj/view?usp=sharing) |
| 128x128    | 66.360     | [Google Drive](https://drive.google.com/file/d/1LD_s5jZixz8D3TJrjGGjBSKSWO7yb3XT/view?usp=sharing) |
| 96x96      | 62.560     | [Google Drive](https://drive.google.com/file/d/1rfz9aJDJwaadQBmwvzdKwFeQLXSbQF-A/view?usp=sharing) |

### ResNet-50

| Resolution | Top-1 Acc. | Download                                                     |
| ---------- | ---------- | ------------------------------------------------------------ |
| 224x224    | 77.150     | [Google Drive](https://drive.google.com/file/d/1ywPABwm22RRfIFAeidO3hB83WCnjhAa6/view?usp=sharing) |
| 192x192    | 76.406     | [Google Drive](https://drive.google.com/file/d/1psWXD4mkYFqrRzqq6sX54F8c1OxQe2Zo/view?usp=sharing) |
| 160x160    | 75.312     | [Google Drive](https://drive.google.com/file/d/157WAFN1ExnQKFGZSSo7Dzc0ORJ4Ccr14/view?usp=sharing) |
| 128x128    | 73.526     | [Google Drive](https://drive.google.com/file/d/1iC9XiEGKzvCdXYYOhHccdIDW9rQyBmBg/view?usp=sharing) |
| 96x96      | 70.610     | [Google Drive](https://drive.google.com/file/d/14CJg1UQuO8iYrMzKWvWcNxxiL8bZlXwE/view?usp=sharing) |

### MobileNetV2

Please visit my repository [mobilenetv2.pytorch](https://github.com/d-li14/mobilenetv2.pytorch).

### SAN

| Architecture | Download                                                     |
| ------------ | ------------------------------------------------------------ |
| ResNet-18    | [Google Drive](https://drive.google.com/file/d/1JqJSxjD6rMOlxYY44D3QEWH23Lo6XuIF/view?usp=sharing) |
| ResNet-50    | [Google Drive](https://drive.google.com/file/d/1Cci3_vAP_sXVwdUhhtZ2T07KDYUlW_B-/view?usp=sharing) |
| MobileNetV2  | [Google Drive](https://drive.google.com/file/d/1rkl_pV0_HBCVhhwxa1FK6Ec6gxOsjvPT/view?usp=sharing) |

## Training

### ResNet-18/50

```shell
python imagenet.py \
    -a meta_resnet18/50 \
    -d <path-to-ILSVRC2012-data> \
    --epochs 120 \
    --lr-decay cos \
    -c <path-to-save-checkpoints> \
    --sizes <list-of-input-resolutions> \ # default is 224, 192, 160, 128, 96
    -j <num-workers>
    --kd
```

### MobileNetV2

```shell
python imagenet.py \
    -a meta_mobilenetv2 \
    -d <path-to-ILSVRC2012-data> \
    --epochs 150 \
    --lr-decay cos \
    --lr 0.05 \
    --wd 4e-5 \
    -c <path-to-save-checkpoints> \
    --sizes <list-of-input-resolutions> \ # default is 224, 192, 160, 128, 96
    -j <num-workers>
    --kd
```

## Testing

### Proxy Inference (default)

```shell
python imagenet.py \
    -a <arch> \
    -d <path-to-ILSVRC2012-data> \
    --resume <checkpoint-file> \
    --sizes <list-of-input-resolutions> \
    -e
    -j <num-workers>
```

Arguments are:

* `checkpoint-file`: previously downloaded checkpoint file from [here](https://github.com/d-li14/SAN#san).
* `list-of-input-resolutions`: test resolutions using different privatized BNs.

which gives Table 1 in the main paper and Table 5 in the supplementary materials.

### Ideal Inference

Manually set the scale encoding [here](https://github.com/d-li14/SAN/blob/master/models/imagenet/meta_resnet.py#L60), which gives the left panel of Table 2 in the main paper.

Uncomment [this line](https://github.com/d-li14/SAN/blob/master/imagenet.py#L239) in the main script to enable post-hoc BN calibration, which gives the middle panel of Table 2 in the main paper.

### Data-Free Ideal Inference

Manually set the scale encoding [here](https://github.com/d-li14/SAN/blob/master/models/imagenet/meta_resnet.py#L60) and its corresponding shift [here](https://github.com/d-li14/SAN/blob/master/imagenet.py#L124), then uncomment [this line](https://github.com/d-li14/SAN/blob/master/imagenet.py#L209) to replace its above line, which gives Table 6 in the supplementary materials.

## Comparison to MutualNet

[MutualNet: Adaptive ConvNet via Mutual Learning from Network Width and Resolution](https://arxiv.org/abs/1909.12978) is accpepted to ECCV 2020 as oral. MutualNet and SAN are contemporary works sharing the similar motivation, regarding to switchable resolution during inference. We provide a comparison of top-1 validation accuracy on ImageNet with the same FLOPs, based on the common MobileNetV2 backbone.

Note that MutualNet training is based on the resolution range of {224, 192, 160, 128} with 4 network widths, while SAN training is based on the resolution range of {224, 192, 160, 128, 96} without tuning the network width, so the configuratuon is different in this comparison.

<table>
    <tr>
        <td width="50%">
            <table>
                <tr>
                    <td align="center">Method</td>
                    <td align="center">Config (width-resolution)</td>
                    <td align="center">MFLOPs</td>
                    <td align="center">Top-1 Acc.</td>
                </tr>
                <tr>
                    <td align="center">MutualNet<br>SAN</td>
                    <td align="center">1.0-224<br>1.0-224</td>
                    <td align="center">300<br>300</td>
                    <td align="center"><b>73.0</b><br>72.86</td>
                </tr>
                <tr>
                    <td align="center">MutualNet<br>SAN</td>
                    <td align="center">0.9-224<br>1.0-208</td>
                    <td align="center">269<br>270</td>
                    <td align="center">72.4<br><b>72.42</b></td>
                </tr>
                <tr>
                    <td align="center">MutualNet<br>SAN</td>
                    <td align="center">1.0-192<br>1.0-192</td>
                    <td align="center">221<br>221</td>
                    <td align="center">71.9<br><b>72.22</b></td>
                </tr>
                <tr>
                    <td align="center">MutualNet<br>SAN</td>
                    <td align="center">0.9-192<br>1.0-176</td>
                    <td align="center">198<br>195</td>
                    <td align="center">71.5<br><b>71.63</b></td>
                </tr>
                <tr>
                    <td align="center">MutualNet<br>SAN</td>
                    <td align="center">0.75-192<br>1.0-160</td>
                    <td align="center">154<br>154</td>
                    <td align="center">70.2<br><b>71.16</b></td>
                </tr>
                <tr>
                    <td align="center">MutualNet<br>SAN</td>
                    <td align="center">0.9-160<br>1.0-144</td>
                    <td align="center">138<br>133</td>
                    <td align="center"><b>69.9</b><br>69.80</td>
                </tr>
                <tr>
                    <td align="center">MutualNet<br>SAN</td>
                    <td align="center">1.0-128<br>1.0-128</td>
                    <td align="center">99<br>99</td>
                    <td align="center">67.8<br><b>69.14</b></td>
                </tr>
                <tr>
                    <td align="center">MutualNet<br>SAN</td>
                    <td align="center">0.85-128<br>1.0-112</td>
                    <td align="center">84<br>82</td>
                    <td align="center">66.1<br><b>66.59</b></td>
                </tr>
                <tr>
                    <td align="center">MutualNet<br>SAN</td>
                    <td align="center">0.7-128<br>1.0-96</td>
                    <td align="center">58<br>56</td>
                    <td align="center">64.3<br><b>65.07</b></td>
                </tr>
            </table>
        </td>
        <td width="50%" height="100%">
            <img src="fig/comparison.png" />
        </td>
    </tr>
</table>



## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@InProceedings{Li_2020_ECCV,
author = {Li, Duo and Yao, Anbang and Chen, Qifeng},
title = {Learning to Learn Parameterized Classification Networks for Scalable Input Images},
booktitle = {The European Conference on Computer Vision (ECCV)},
month = {August},
year = {2020}
}
```
