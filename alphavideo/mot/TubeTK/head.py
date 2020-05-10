import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from alphavideo.loss.focal_loss import focal_loss
from alphavideo.mot.TubeTK.utils import distance2bbox, iou_loss, giou_loss, volume, bbox_iou_loss
from alphavideo.utils.tube_nms import multiclass_tube_nms

INF = 1e8


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


class Scale(nn.Module):
    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale


class TrackHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 strides,
                 arg):
        super(TrackHead, self).__init__()

        self.num_classes = num_classes
        self.tube_points = arg.tube_points
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = arg.heads_features_n
        self.stacked_convs = arg.heads_layers_n
        self.strides = strides
        self.regress_ranges = arg.regress_range
        self.arg = arg

        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                nn.Sequential(nn.Conv3d(chn, self.feat_channels, kernel_size=3, stride=1, padding=1, bias=False),
                              nn.GroupNorm(num_groups=32, num_channels=self.feat_channels),
                              nn.ReLU(inplace=True)))
            self.reg_convs.append(
                nn.Sequential(nn.Conv3d(chn, self.feat_channels, kernel_size=3, stride=1, padding=1, bias=False),
                              nn.GroupNorm(num_groups=32, num_channels=self.feat_channels),
                              nn.ReLU(inplace=True)))
        self.TubeTK_cls = nn.Conv3d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.TubeTK_reg = nn.Conv3d(self.feat_channels, self.tube_points, 3, padding=1)
        self.TubeTK_centerness = nn.Conv3d(self.feat_channels, 1, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.regress_ranges])

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, 0, 0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        cls_feat = x
        reg_feat = x

        # classification
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.TubeTK_cls(cls_feat)

        # centerness just using cls features
        centerness = self.TubeTK_centerness(cls_feat)

        # regression tubes
        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        # scale the bbox_pred of different level
        tube_pred = scale(self.TubeTK_reg(reg_feat)).exp()

        return cls_score, tube_pred, centerness

    def loss(self,
             cls_scores,
             tube_preds,
             centernesses,
             # every bbox extend to a tube
             gt_tubes,
             gt_labels,
             img_metas):
        assert len(cls_scores) == len(tube_preds) == len(centernesses)

        for tube_pred in tube_preds:
            tube_pred[:, [0, 1, 2, 3]].clamp_(min=0.0001)
            tube_pred[:, [4, 9]].clamp_(min=0.0001)

        featmap_sizes = [featmap.size()[-3:] for featmap in cls_scores]

        # get the center_point of the tubes
        all_level_points = self.get_points(featmap_sizes, tube_preds[0].dtype,
                                           tube_preds[0].device)

        labels, tube_targets = self.tubetk_target(all_level_points, gt_tubes, gt_labels)

        num_imgs = cls_scores[0].size(0)

        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 4, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_tube_preds = [
            bbox_pred.permute(0, 2, 3, 4, 1).reshape(-1, self.tube_points)
            for bbox_pred in tube_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 4, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_tube_preds = torch.cat(flatten_tube_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels).squeeze(1)
        flatten_tube_targets = torch.cat(tube_targets)

        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        pos_inds = flatten_labels.nonzero().reshape(-1)
        num_pos = len(pos_inds)

        # Classification Loss
        loss_cls = focal_loss(
            flatten_cls_scores, flatten_labels.long()).sum() / (num_pos + num_imgs)  # avoid num_pos is 0

        pos_tube_preds = flatten_tube_preds[pos_inds]
        pos_tube_targets = flatten_tube_targets[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        if num_pos > 0:
            pos_centerness_targets = self.centerness_target(pos_tube_targets)

            pos_points = flatten_points[pos_inds]

            pos_decoded_tube_preds = distance2bbox(pos_points, pos_tube_preds)
            pos_decoded_tube_targets = distance2bbox(pos_points, pos_tube_targets)

            # centerness weighted iou loss
            if self.arg.reg_loss == 'iou':
                reg_loss_func = iou_loss
            elif self.arg.reg_loss == 'giou':
                reg_loss_func = giou_loss
            else:
                raise RuntimeError('Wrong regression loss type. We only support iou_loss and giou_loss')

            # Tube IoU Loss
            loss_reg = reg_loss_func(
                pos_decoded_tube_preds,
                pos_decoded_tube_targets
                ) * pos_centerness_targets
            loss_reg = loss_reg.sum() / pos_centerness_targets.sum()

            # Mid Frame IoU Loss
            mid_iou_loss = bbox_iou_loss(pos_decoded_tube_preds[..., 1:5],
                                         pos_decoded_tube_targets[..., 1:5]) * pos_centerness_targets
            mid_iou_loss = mid_iou_loss.sum() / pos_centerness_targets.sum()

            # Centerness Loss
            loss_centerness = F.binary_cross_entropy_with_logits(
                pos_centerness, pos_centerness_targets, reduction='mean')

        else:
            loss_reg = pos_tube_preds.sum()
            mid_iou_loss = pos_tube_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_reg=loss_reg,
            loss_centerness=loss_centerness,
            mid_iou_loss=mid_iou_loss)

    def get_tubes(self,
                  cls_scores,
                  tube_preds,
                  centernesses,
                  img_metas,
                  cfg):
        assert len(cls_scores) == len(tube_preds)
        num_levels = len(cls_scores)

        for tube_pred in tube_preds:
            tube_pred[:, [0, 1, 2, 3]].clamp_(min=0.0001)
            tube_pred[:, [4, 9]].clamp_(min=0.0001)

        featmap_sizes = [featmap.size()[-3:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, tube_preds[0].dtype, tube_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            tube_pred_list = [
                tube_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            value_range = img_metas[img_id]['value_range']
            pad_percent = img_metas[img_id]['pad_percent']
            det_tubes = self.get_tubes_single(
                cls_score_list, tube_pred_list, centerness_pred_list,
                mlvl_points, img_shape, value_range, pad_percent, cfg)
            result_list.append(det_tubes)
        return result_list

    def get_tubes_single(self,
                         cls_scores,
                         tube_preds,
                         centernesses,
                         mlvl_points,
                         img_shape,
                         value_range,
                         pad_percent,
                         cfg):
        assert len(cls_scores) == len(tube_preds) == len(mlvl_points)
        mlvl_tubes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, tube_pred, centerness, points in zip(
                cls_scores, tube_preds, centernesses, mlvl_points):
            assert cls_score.size()[-3:] == tube_pred.size()[-3:]
            scores = cls_score.permute(1, 2, 3, 0).reshape(-1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 3, 0).reshape(-1).sigmoid()

            tube_pred = tube_pred.permute(1, 2, 3, 0).reshape(-1, self.tube_points)
            nms_pre = cfg.test_nms_pre

            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                tube_pred = tube_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            tubes = distance2bbox(points, tube_pred, max_shape=[value_range for _ in range(3)])
            mlvl_tubes.append(tubes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_tubes = torch.cat(mlvl_tubes)

        mlvl_tubes[:, [0, 5, 10]] *= img_shape[0] / value_range
        mlvl_tubes[:, [1, 3, 6, 8, 11, 13]] *= img_shape[2] / pad_percent[0] / value_range
        mlvl_tubes[:, [2, 4, 7, 9, 12, 14]] *= img_shape[1] / pad_percent[1] / value_range

        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        det_tubes, det_labels = multiclass_tube_nms(
            mlvl_tubes,
            mlvl_scores,
            cfg.test_nms_score_thre,
            cfg.test_nms_iou_thre,
            cfg.test_nms_max_per_img,
            score_factors=mlvl_centerness,
            frame_num=img_shape[0])
        return det_tubes, det_labels, (mlvl_tubes, mlvl_scores, mlvl_centerness)

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.
        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.
        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i],
                                       dtype, device))
        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        l, h, w = featmap_size
        t_range = torch.arange(
            0, l * stride[0], stride[0], dtype=dtype, device=device) + stride[0] / 2
        x_range = torch.arange(
            0, w * stride[2], stride[2], dtype=dtype, device=device) + stride[2] / 2
        y_range = torch.arange(
            0, h * stride[1], stride[1], dtype=dtype, device=device) + stride[1] / 2
        t, y, x = torch.meshgrid(t_range, y_range, x_range)
        points = torch.stack(
            (t.reshape(-1), x.reshape(-1), y.reshape(-1)), dim=-1)
        return points

    def tubetk_target(self, points, gt_tubes_list, gt_labels_list):
        """
        Args:
            points: 5 * [n_point, 3]
            gt_tubes_list: b, n_tube, 15
            gt_labels_list: b, n_tube, 1
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].repeat(
                points[i].shape[0], 1, 1) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # get labels and bbox_targets of each image
        labels_list, tube_targets_list = multi_apply(
            self.tubetk_target_single_image,
            gt_tubes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges * self.arg.value_range)

        # split to per img, per level
        num_points = [center.size(0) for center in points]
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        tube_targets_list = [
            tube_targets.split(num_points, 0)
            for tube_targets in tube_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_tube_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_tube_targets.append(
                torch.cat(
                    [tube_targets[i] for tube_targets in tube_targets_list]))
        return concat_lvl_labels, concat_lvl_tube_targets

    def tubetk_target_single_image(self, gt_tubes, gt_labels, points, regress_ranges):
        """
        Args:
            gt_tubes: n_tube, 15
            gt_labels: n_tube, 1
            points: n_points_total, 3
            regress_ranges: n_points_total, 2, 2
        """
        num_points = points.size(0)
        tube_limit = self.arg.tube_limit
        if len(gt_tubes) > tube_limit:
            random_tubes = np.random.choice(gt_tubes.shape[0], tube_limit, replace=False)
            gt_tubes = gt_tubes[random_tubes, :]
            gt_labels = gt_labels[random_tubes, :]
        num_gts = gt_tubes.size(0)
        # print(num_gts)
        middle_areas = (gt_tubes[:, 2] - gt_tubes[:, 0]) * (
                gt_tubes[:, 3] - gt_tubes[:, 1])
        front_areas = (gt_tubes[:, 2] + gt_tubes[:, 8] - gt_tubes[:, 0] - gt_tubes[:, 6]) * (
                gt_tubes[:, 3] + gt_tubes[:, 9] - gt_tubes[:, 1] - gt_tubes[:, 7])
        back_areas = (gt_tubes[:, 2] + gt_tubes[:, 13] - gt_tubes[:, 0] - gt_tubes[:, 11]) * (
                gt_tubes[:, 3] + gt_tubes[:, 14] - gt_tubes[:, 1] - gt_tubes[:, 12])
        volumes = volume(middle_areas, front_areas, gt_tubes[:, 5]) + volume(middle_areas, back_areas, gt_tubes[:, 10])

        volumes = volumes[None].repeat(num_points, 1).float()

        regress_ranges = regress_ranges[:, None, :, :].expand(
            num_points, num_gts, 2, 2)
        gt_tubes = gt_tubes[None].expand(num_points, num_gts, 15)
        ts, xs, ys = points[:, 0], points[:, 1], points[:, 2]
        ts = ts[:, None].expand(num_points, num_gts)
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        middle_left = xs - gt_tubes[..., 0]
        middle_right = gt_tubes[..., 2] - xs
        middle_top = ys - gt_tubes[..., 1]
        middle_bottom = gt_tubes[..., 3] - ys
        tube_targets = torch.cat((torch.stack((middle_left, middle_top, middle_right, middle_bottom), -1),
                                  gt_tubes[..., 5:]), -1)

        # condition0: on the specific frame
        on_frame_mask = ts == gt_tubes[..., 4]

        # condition1: inside a gt bbox
        inside_gt_tube_bm_mask = tube_targets[..., [0, 1, 2, 3]].min(-1)[0] > 0

        # condition2: limit the spatial regression range for each location
        max_regress_distance = tube_targets[..., 0:4].max(-1)[0]
        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 1, 0]) & (
                max_regress_distance <= regress_ranges[..., 1, 1])

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        volumes[on_frame_mask == 0] = INF
        volumes[inside_gt_tube_bm_mask == 0] = INF
        volumes[inside_regress_range == 0] = INF
        min_area, min_area_inds = volumes.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0
        tube_targets = tube_targets[range(num_points), min_area_inds]

        return labels, tube_targets

    def centerness_target(self, pos_tube_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_tube_targets[:, [0, 2]]
        top_bottom = pos_tube_targets[:, [1, 3]]
        front_back = pos_tube_targets[:, [4, 9]]

        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])

        if not self.arg.withoutThickCenterness:
            with_thick = front_back[:, 0] + front_back[:, 1] != 0
            if with_thick.sum() != 0:
                centerness_targets[with_thick] *= \
                    front_back[with_thick].min(dim=-1)[0] / front_back[with_thick].max(dim=-1)[0]
        centerness_targets = torch.sqrt(centerness_targets + 1e-5)
        return centerness_targets