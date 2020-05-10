import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self,
                 in_channels,  # [512, 1024, 2048]
                 arg,
                 ):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = arg.fpn_features_n
        self.num_ins = len(in_channels)
        self.num_outs = arg.fpn_outs_n

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.num_ins):
            l_conv = nn.Conv3d(in_channels[i], self.out_channels, kernel_size=1, stride=1)
            fpn_conv = nn.Conv3d(self.out_channels, self.out_channels,
                                 kernel_size=3, stride=1, padding=1)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = self.num_outs - self.num_ins
        if extra_levels >= 1:
            for i in range(extra_levels):
                in_channels = self.out_channels
                extra_fpn_conv = nn.Conv3d(in_channels, self.out_channels, kernel_size=3, stride=(1, 2, 2), padding=1)
                self.fpn_convs.append(extra_fpn_conv)

        # default init_weights for conv(msra) and norm in ConvModule
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # add conv layers on top of original feature maps (RetinaNet)
            outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
            for i in range(used_backbone_levels + 1, self.num_outs):
                outs.append(self.fpn_convs[i](F.relu(outs[-1])))

        return tuple(outs)