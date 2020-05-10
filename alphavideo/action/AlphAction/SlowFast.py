import torch
import torch.nn as nn
from alphavideo.base.resnet3d import resnet50, resnet101

class SlowFast(nn.Module):
    def __init__(self, slow_pathway, fast_pathway, pathway_ratio):
        super(SlowFast, self).__init__()

        self.slow = slow_pathway
        self.fast = fast_pathway
        self.pathway_ratio = pathway_ratio
        self._make_lateral_conv()

    def _make_lateral_conv(self):
        planes_width = 8
        self.lateral_conv1 = nn.Conv3d(planes_width, planes_width*2, kernel_size=(5, 1, 1),
                                       stride=(self.pathway_ratio, 1, 1), padding=(2, 0, 0),)
        self.lateral_conv2 = nn.Conv3d(planes_width*4, planes_width*8, kernel_size=(5, 1, 1),
                                       stride=(self.pathway_ratio, 1, 1), padding=(2, 0, 0),)
        self.lateral_conv3 = nn.Conv3d(planes_width*8, planes_width*16, kernel_size=(5, 1, 1),
                                       stride=(self.pathway_ratio, 1, 1), padding=(2, 0, 0), )
        self.lateral_conv4 = nn.Conv3d(planes_width*16, planes_width*32, kernel_size=(5, 1, 1),
                                       stride=(self.pathway_ratio, 1, 1), padding=(2, 0, 0), )


    def forward(self, slow_videos, fast_videos):
        fast_out = self.fast(fast_videos)
        laterals = [
            getattr(self, "lateral_conv{}".format(i+1))(fast_out[i])
            for i in range(4)
        ]
        fast_out = fast_out[-1]
        slow_out = self.slow(slow_videos, laterals)
        return slow_out, fast_out


def slowfast_res50():
    slow_kernels = [[1], [1,1,1], [1,1,1,1], [3,3,3,3,3,3], [3,3,3]]
    slow_pathway = resnet50(
        kernels=slow_kernels,
        plane_width=64,
        freeze_bn=True,
        no_temp_stride=True,
        no_temp_pool=True,
        split_s_t=True,
        l4_dilation=True,
        fuse_ratio=0.25,
    )
    fast_kernels = [[5], [3,3,3], [3,3,3,3], [3,3,3,3,3,3], [3,3,3]]
    fast_pathway = resnet50(
        kernels=fast_kernels,
        plane_width=8,
        freeze_bn=True,
        no_temp_stride=True,
        no_temp_pool=True,
        split_s_t=True,
        l4_dilation=True,
        feature_stages=[0, 1, 2, 3, 4],
    )
    return SlowFast(slow_pathway, fast_pathway, 8)

def slowfast_res101():
    slow_kernels = [[1], [1,1,1], [1,1,1,1], [3,]*23, [3,3,3]]
    slow_pathway = resnet101(
        kernels=slow_kernels,
        plane_width=64,
        freeze_bn=True,
        no_temp_stride=True,
        no_temp_pool=True,
        split_s_t=True,
        l4_dilation=True,
        fuse_ratio=0.25,
    )
    fast_kernels = [[5], [3,3,3], [3,3,3,3], [3,]*23, [3,3,3]]
    fast_pathway = resnet101(
        kernels=fast_kernels,
        plane_width=8,
        freeze_bn=True,
        no_temp_stride=True,
        no_temp_pool=True,
        split_s_t=True,
        l4_dilation=True,
        feature_stages=[0, 1, 2, 3, 4],
    )
    return SlowFast(slow_pathway, fast_pathway, 4)