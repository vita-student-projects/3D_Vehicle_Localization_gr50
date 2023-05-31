import os, sys
import math
import torch.nn.functional as F

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
#sys.path.append(ROOT_DIR)

import numpy as np
import torch
import torch.nn as nn


class Conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernal_szie=3, stride=1, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes,
                              out_planes,
                              kernel_size=kernal_szie,
                              stride=stride,
                              padding=kernal_szie//2,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class IDAUp(nn.Module):
    '''
    input: features map of different layers
    output: up-sampled features
    '''
    def __init__(self, in_channels_list, up_factors_list, out_channels):
        super(IDAUp, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels

        for i in range(1, len(in_channels_list)):
            in_channels = in_channels_list[i]
            up_factors = int(up_factors_list[i])

            proj = Conv2d(in_channels, out_channels, kernal_szie=3, stride=1, bias=False)
            node = Conv2d(out_channels*2, out_channels, kernal_szie=3, stride=1, bias=False)
            up = nn.ConvTranspose2d(in_channels=out_channels,
                                    out_channels=out_channels,
                                    kernel_size=up_factors * 2,
                                    stride=up_factors,
                                    padding=up_factors // 2,
                                    output_padding=0,
                                    groups=out_channels,
                                    bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)


        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, layers):
        assert len(self.in_channels_list) == len(layers), \
            '{} vs {} layers'.format(len(self.in_channels_list), len(layers))

        for i in range(1, len(layers)):
            upsample = getattr(self, 'up_' + str(i))
            project = getattr(self, 'proj_' + str(i))
            node = getattr(self, 'node_' + str(i))

            layers[i] = upsample(project(layers[i]))
            layers[i] = node(torch.cat([layers[i-1], layers[i]], 1))

        return layers


class DLAUp(nn.Module):
    def __init__(self, in_channels_list, scales_list=(1, 2, 4, 8, 16)):
        super(DLAUp, self).__init__()
        scales_list = np.array(scales_list, dtype=int)

        for i in range(len(in_channels_list) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i), IDAUp(in_channels_list=in_channels_list[j:],
                                                    up_factors_list=scales_list[j:] // scales_list[j],
                                                    out_channels=in_channels_list[j]))
            scales_list[j + 1:] = scales_list[j]
            in_channels_list[j + 1:] = [in_channels_list[j] for _ in in_channels_list[j + 1:]]

    def forward(self, layers):
        layers = list(layers)
        assert len(layers) > 1
        for i in range(len(layers) - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            layers[-i - 2:] = ida(layers[-i - 2:])
        return layers[-1]

# weight init for up-sample layers [tranposed conv2d]
def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels

        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            lateral_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)

    def forward(self, input):
        # Bottom-up pathway
        input = input[::-1]

        # Top-down pathway
        features = [self.lateral_convs[0](input[0])]
        for i in range(1, len(input)):
            lateral = self.lateral_convs[i](input[i])
            upsampled = F.interpolate(features[-1], scale_factor=2, mode='nearest')
            features.append(lateral + upsampled)

        # Output pathway
        outputs = [self.output_convs[i](features[i]) for i in range(len(features))]

        return outputs


if __name__ == '__main__':
    from dla import dla34
    backbone = dla34(return_levels=True)
    input = torch.randn(2, 3, 64, 64)
    features = backbone(input)
    print('input data shape:', input.shape)
    print('numbers of feature maps generated by DLA backbone:', len(features))
    print('feature maps generated by DLA backbone:')
    for i in range(len(features)):
        print(features[i].shape)

    channels = backbone.channels
    start_level = int(np.log2(4))
    scales = [2 ** i for i in range(len(channels[start_level:]))]
    print('channels list of DLA features:', channels)
    print('start level of features-up aggratation:', start_level)
    print('upsumapling factors of features', scales)

    dlaup = DLAUp(in_channels_list=channels[start_level:], scales_list=scales)
    features_up = dlaup(features[start_level:])
    print('shape of upsampled feature maps', features_up.shape)

    Fpn = FPN(in_channels_list=channels[start_level:-1], out_channels=256)  # Adjust the in_channels_list
    features_up2 = Fpn(features[start_level:])
    print('shape of upsampled feature maps 2', features_up2.shape)

