import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

__all__ = [
    'ResNet', 'resnet50', 'resnet101'
]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, kernel=3, downsample=None, groups=1, base_width=64, split_s_t=False,
                 dilation=1):
        super(Bottleneck, self).__init__()

        width = int(planes * (base_width / 64.)) * groups

        if split_s_t:
            self.conv1 = nn.Conv3d(inplanes, width, kernel_size=(kernel, 1, 1),
                                   padding = (kernel//2, 0, 0), bias=False)
            self.conv2 = nn.Conv3d(width, width, kernel_size=(1, 3, 3),
                                   stride=stride, padding=(0, dilation, dilation),
                                   dilation=(1, dilation, dilation),
                                   groups=groups, bias=False)
        else:
            self.conv1 = nn.Conv3d(inplanes, width, kernel_size=1, bias=False)
            self.conv2 = nn.Conv3d(width, width, kernel_size=(kernel, 3, 3),
                                   stride=stride, padding=(kernel // 2, dilation, dilation),
                                   dilation=(1, dilation, dilation),
                                   groups=groups, bias=False)
        self.bn1 = nn.BatchNorm3d(width)
        self.bn2 = nn.BatchNorm3d(width)
        self.conv3 = nn.Conv3d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 kernels,
                 groups=1,
                 plane_width=64,
                 width_per_group=64,
                 freeze_bn=False,
                 freeze_stages=-1,
                 fst_l_stride=2,
                 feature_stages=None,
                 no_temp_stride=False,
                 no_temp_pool=False,
                 split_s_t=False,
                 l4_dilation=False,
                 fuse_ratio=0.0):
        self.freeze_bn = freeze_bn
        self.freeze_stages = freeze_stages
        self.groups = groups
        self.base_width = width_per_group
        self.plane_width = plane_width
        self.inplanes = self.plane_width
        if feature_stages is None:
            feature_stages = []
        self.feature_stages = feature_stages
        self.split_s_t = split_s_t
        self.fuse_ratio = fuse_ratio

        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            self.plane_width,
            kernel_size=(kernels[0][0], 7, 7),
            stride=(1, 2, 2),
            padding=(kernels[0][0]//2, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(self.plane_width)
        self.relu = nn.ReLU(inplace=True)
        if no_temp_pool:
            self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        elif kernels[0][0] == 7:
            self.maxpool = nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)
        else:
            self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 4), stride=(1, 2, 2), padding=(0, 1, 1))

        self.layer1 = self._make_layer(block, self.plane_width, layers[0], kernels[1])
        self.layer2 = self._make_layer(
            block, self.plane_width*2, layers[1], kernels[2], stride=(1, 2, 2) if (fst_l_stride < 2 or no_temp_stride) else 2)
        self.layer3 = self._make_layer(
            block, self.plane_width*4, layers[2], kernels[3], stride=(1, 2, 2) if no_temp_stride else 2)
        if no_temp_stride:
            l4_stride = (1,1,1) if l4_dilation else (1,2,2)
        else:
            l4_stride = 2
        self.layer4 = self._make_layer(
            block, self.plane_width*8, layers[3], kernels[4], stride=l4_stride, dilation=2 if l4_dilation else 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self._freeze_stages()
        if self.freeze_bn:
            self._freeze_bn()

    def _make_layer(self, block, planes, blocks, kernel, stride=1, dilation=1):
        self.inplanes += int(self.inplanes*self.fuse_ratio)
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation>1:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, kernel[0], downsample, self.groups,
                            self.base_width, split_s_t=self.split_s_t, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel=kernel[i], groups=self.groups,
                                base_width=self.base_width, split_s_t=self.split_s_t, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x, lateral_feature=None):
        outs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if 0 in self.feature_stages:
            outs.append(x)
        if lateral_feature:
            x = torch.cat([x, lateral_feature[0]], dim=1)

        x = self.layer1(x)
        if 1 in self.feature_stages:
            outs.append(x)
        if lateral_feature:
            x = torch.cat([x, lateral_feature[1]], dim=1)
        x = self.layer2(x)
        if 2 in self.feature_stages:
            outs.append(x)
        if lateral_feature:
            x = torch.cat([x, lateral_feature[2]], dim=1)
        x = self.layer3(x)
        if 3 in self.feature_stages:
            outs.append(x)
        if lateral_feature:
            x = torch.cat([x, lateral_feature[3]], dim=1)
        x = self.layer4(x)
        if 4 in self.feature_stages:
            outs.append(x)

        if len(outs)==0:
            return x

        return tuple(outs)

    def _freeze_stages(self):
        if self.freeze_stages >= 0:
            print('Freeze Stage: 0')
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.freeze_stages + 1):
            print('Freeze Stage: ' + str(i))
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def _freeze_bn(self):
        print('Freeze BN')
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                m.eval()

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        self._freeze_stages()

        if mode and self.freeze_bn:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   groups=1,
                   width_per_group=64,
                   **kwargs)
    return model

def resnet101(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   groups=1,
                   width_per_group=64,
                   **kwargs)
    return model