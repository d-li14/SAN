"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .parallel import ModuleParallel, BatchNorm2dParallel

__all__ = ['meta_mobilenetv2']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class conv_3x3_bn(nn.Module):
    def __init__(self, inp, oup, stride, num_parallel):
        super(conv_3x3_bn, self).__init__()
        self.fc = nn.Linear(1, oup * inp * 3 * 3)
        self.bn = BatchNorm2dParallel(oup, num_parallel)
        self.relu = ModuleParallel(nn.ReLU6(inplace=True))
        self.num_parallel = num_parallel
        self.inp = inp
        self.oup = oup
        self.stride = stride

    def forward(self, x):
        out = [None for _ in range(self.num_parallel)]

        for i in range(self.num_parallel):
            scale = torch.tensor([0.7 - 0.1 * i]).cuda()
            weight = self.fc(scale).resize(self.oup, self.inp, 3, 3)
            out[i] = F.conv2d(x[i], weight, stride=self.stride, padding=1)
        out = self.bn(out)
        out = self.relu(out)

        return out


class conv_1x1_bn(nn.Module):
    def __init__(self, inp, oup, num_parallel):
        super(conv_1x1_bn, self).__init__()
        self.fc = nn.Linear(1, oup * inp * 1 * 1)
        self.bn = BatchNorm2dParallel(oup, num_parallel)
        self.relu = ModuleParallel(nn.ReLU6(inplace=True))
        self.num_parallel = num_parallel
        self.inp = inp
        self.oup = oup

    def forward(self, x):
        out = [None for _ in range(self.num_parallel)]

        for i in range(self.num_parallel):
            scale = torch.tensor([0.7 - 0.1 * i]).cuda()
            weight = self.fc(scale).resize(self.oup, self.inp, 1, 1)
            out[i] = F.conv2d(x[i], weight)
        out = self.bn(out)
        out = self.relu(out)

        return out


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, num_parallel):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.fc1 = nn.Linear(1, hidden_dim * 1 * 3 * 3)
            self.bn1 = BatchNorm2dParallel(hidden_dim, num_parallel)
            self.relu = ModuleParallel(nn.ReLU6(inplace=True))
            self.fc2 = nn.Linear(1, oup * hidden_dim * 1 * 1)
            self.bn2 = BatchNorm2dParallel(oup, num_parallel)
        else:
            self.fc1 = nn.Linear(1, hidden_dim * inp * 1 * 1)
            self.bn1 = BatchNorm2dParallel(hidden_dim, num_parallel)
            self.relu = ModuleParallel(nn.ReLU6(inplace=True))
            self.fc2 = nn.Linear(1, hidden_dim * 1 * 3 * 3)
            self.bn2 = BatchNorm2dParallel(hidden_dim, num_parallel)
            self.fc3 = nn.Linear(1, oup * hidden_dim * 1 * 1)
            self.bn3 = BatchNorm2dParallel(oup, num_parallel)
        self.num_parallel = num_parallel
        self.inp = inp
        self.oup = oup
        self.hidden_dim = hidden_dim
        self.stride = stride
        self.expand_ratio = expand_ratio

    def forward(self, x):
        out = [None for _ in range(self.num_parallel)]
        scale = [torch.tensor([0.7 - 0.1 * i]).cuda() for i in range(self.num_parallel)]

        if self.expand_ratio == 1:
            for i in range(self.num_parallel):
                weight = self.fc1(scale[i]).resize(self.hidden_dim, 1, 3, 3)
                out[i] = F.conv2d(x[i], weight, stride=self.stride, padding=1, groups=self.hidden_dim)
            out = self.bn1(out)
            out = self.relu(out)

            for i in range(self.num_parallel):
                weight = self.fc2(scale[i]).resize(self.oup, self.hidden_dim, 1, 1)
                out[i] = F.conv2d(out[i], weight)
            out = self.bn2(out)
        else:
            for i in range(self.num_parallel):
                weight = self.fc1(scale[i]).resize(self.hidden_dim, self.inp, 1, 1)
                out[i] = F.conv2d(x[i], weight)
            out = self.bn1(out)
            out = self.relu(out)

            for i in range(self.num_parallel):
                weight = self.fc2(scale[i]).resize(self.hidden_dim, 1, 3, 3)
                out[i] = F.conv2d(out[i], weight, stride=self.stride, padding=1, groups=self.hidden_dim)
            out = self.bn2(out)
            out = self.relu(out)

            for i in range(self.num_parallel):
                weight = self.fc3(scale[i]).resize(self.oup, self.hidden_dim, 1, 1)
                out[i] = F.conv2d(out[i], weight)
            out = self.bn3(out)

        if self.identity:
            return [x[i] + out[i] for i in range(self.num_parallel)]
        else:
            return out


class MetaMobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1., num_parallel=5):
        super(MetaMobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2, num_parallel)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, num_parallel))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel, num_parallel)
        self.avgpool = ModuleParallel(nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = ModuleParallel(nn.Linear(output_channel, num_classes))

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = [t.view(t.size(0), -1) for t in x]
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def meta_mobilenetv2(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MetaMobileNetV2(**kwargs)

