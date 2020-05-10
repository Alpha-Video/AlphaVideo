import time, os
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
from alphavideo.base.resnet3d import resnet50
from alphavideo.base.fpn import FPN
from alphavideo.mot.TubeTK.head import TrackHead
from alphavideo.utils.load_url import load_url_google


class TubeTK(nn.Module):

    def __init__(self,
                 arg,
                 num_classes=1,
                 pretrained=True
                 ):
        super(TubeTK, self).__init__()
        self.arg = arg
        self.backbone = self._make_backbone()
        self.neck = FPN(in_channels=[512, 1024, 2048], arg=arg)
        self.tube_head = TrackHead(arg=arg,
                                   num_classes=num_classes,
                                   in_channels=self.neck.out_channels,
                                   strides=[[arg.model_stride[i][0]/(arg.forward_frames * 2) * arg.value_range,
                                            arg.model_stride[i][1]/arg.img_size[0] * arg.value_range,
                                            arg.model_stride[i][1]/arg.img_size[1] * arg.value_range] for i in range(5)]
                                   )

        if pretrained and arg.pretrain_model_path != '':
            if num_classes != 1:
                print("Multi-classes tubeTK has no pretrained weight. Random init will be adopted.")
            else:
                path = load_url_google(id=arg.pretrain_model_path, name='tubetk.ckpt')
                self.load_pretrain(model_path=path)

    def load_pretrain(self, model_path):
        pre_model = torch.load(model_path, map_location='cpu')['state']
        model_dict = self.state_dict()
        for key in model_dict:
            if model_dict[key].shape != pre_model['module.' + key].shape:
                p_shape = model_dict[key].shape
                pre_model['module.' + key] = pre_model['module.' + key].repeat(1, 1, p_shape[2], 1, 1) / p_shape[2]
            else:
                model_dict[key] = pre_model['module.' + key]
        self.load_state_dict(model_dict)
        del pre_model, model_dict

    def _make_backbone(self):
        kernel = [[7], [3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3, 3, 3], [3, 3, 3]]
        # kernel = [[5], [3, 3, 3], [3, 1, 3, 1], [3, 1, 3, 1, 3, 1], [3, 1, 3]]
        return resnet50(kernels=kernel, freeze_stages=self.arg.freeze_stages,
                        fst_l_stride=self.arg.model_stride[0][0], feature_stages=[2,3,4])

    def extract_feat(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_tubes,
                      gt_labels):
        x = self.extract_feat(img)
        outs = self.tube_head(x)
        loss_inputs = outs + (gt_tubes, gt_labels, img_metas)
        losses = self.tube_head.loss(*loss_inputs)
        return losses

    def forward_test(self, img, img_meta):
        x = self.extract_feat(img)
        outs = self.tube_head(x)
        tube_inputs = outs + (img_meta, self.arg)
        tube_list = self.tube_head.get_tubes(*tube_inputs)
        return tube_list

    def forward(self, img, img_meta, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta)
