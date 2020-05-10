import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict
INF = 1e8

__C = edict()
cfg = __C

# for generating the tubes
__C.min_visibility = -0.1
__C.tube_thre = 0.8
__C.forward_frames = 4
__C.frame_stride = 1
__C.value_range = 1
__C.img_size = [896, 1152]

# pretrain
__C.pretrain_model_path = '1jLgyNmiZ_c-m8Cw3NcZTEPTf6VESfIzK'

# for ResNet
__C.freeze_stages = -1
__C.backbone = 'res50'

# for FPN
__C.fpn_features_n = 256
__C.fpn_outs_n = 5

# for FCOS head
__C.tube_points = 14
__C.heads_features_n = 256
__C.heads_layers_n = 4
__C.withoutThickCenterness = False
__C.model_stride = [[2, 8],
                    [4, 16],
                    [8, 32],
                    [8, 64],
                    [8, 128]]
__C.regress_range = ([(-1, 0.25), (-1, 0.0714)],
                     [(0.25, 0.5), (0.0714, 0.1428)],
                     [(0.5, 0.75), (0.1428, 0.2857)],
                     [(0.75, INF), (0.2857, 0.5714)],
                     [(0.75, INF), (0.5714, INF)])


# for loss
__C.reg_loss = 'giou'
__C.tube_limit = 700
__C.test_nms_pre = 1000
__C.test_nms_max_per_img = 500
__C.test_nms_score_thre = 0.5
__C.test_nms_iou_thre = 0.5
__C.linking_min_iou = 0.4
__C.cos_weight = 0.2