import torch

import torch.nn as nn
import torch.nn.functional as F
from alphavideo.utils.roi_align_3d import ROIAlign3d
from .utils.misc import prepare_pooled_feature, pad_sequence, cat
from .utils.IA_helper import has_object
from .utils.structures import BoxList
from .IA_structure import make_ia_structure


class ROIActionHead(nn.Module):
    """
    Generic Action Head class.
    """

    def __init__(self, cfg, dim_in):
        super(ROIActionHead, self).__init__()
        self.feature_extractor = MLPFeatureExtractor(cfg, dim_in)
        self.predictor = FCPredictor(cfg, self.feature_extractor.dim_out)
        self.post_processor = PostProcessor(cfg.MODEL.ROI_ACTION_HEAD.NUM_PERSON_MOVEMENT_CLASSES)
        self.test_ext = cfg.TEST.EXTEND_SCALE

    def forward(self, slow_features, fast_features, boxes, objects=None, extras={}, part_forward=-1):
        # In training stage, boxes are from gt.
        # In testing stage, boxes are detected by human detector and proposals should be
        # enlarged boxes.
        if self.training:
            raise NotImplementedError("Not available yet.")

        if part_forward == 1:
            boxes = extras["current_feat_p"]
            objects = extras["current_feat_o"]

        proposals = [box.extend(self.test_ext) for box in boxes]

        x, x_pooled, x_objects = self.feature_extractor(slow_features, fast_features, proposals, objects, extras, part_forward)

        if part_forward == 0:
            pooled_feature = prepare_pooled_feature(x_pooled, boxes)
            if x_objects is None:
                object_pooled_feature = None
            else:
                object_pooled_feature = prepare_pooled_feature(x_objects, objects)
            return [pooled_feature, object_pooled_feature]

        action_logits = self.predictor(x)

        result = self.post_processor((action_logits,), boxes)
        return result

class Pooler3d(nn.Module):
    def __init__(self, output_size, scale, sampling_ratio=None):
        super(Pooler3d, self).__init__()
        assert sampling_ratio is not None, 'Sampling ratio should be specified for 3d roi align.'
        self.pooler = ROIAlign3d(
            output_size, spatial_scale=scale, sampling_ratio=sampling_ratio
        )
        self.output_size = output_size

    def convert_to_roi_format(self, boxes, dtype, device):
        bbox_list = list()
        ids_list = list()
        for i, b in enumerate(boxes):
            if not b:
                bbox_list.append(torch.zeros((0, 4), dtype=dtype, device=device))
                ids_list.append(torch.zeros((0, 1), dtype=dtype, device=device))
            else:
                bbox_list.append(b.bbox)
                ids_list.append(torch.full((len(b), 1), i, dtype=dtype, device=device))
        concat_boxes = torch.cat(bbox_list, dim=0)
        ids = torch.cat(ids_list, dim=0)
        rois = torch.cat([ids, concat_boxes], dim=1)

        return rois

    def forward(self, x, boxes):
        rois = self.convert_to_roi_format(boxes, x.dtype, x.device)
        return self.pooler(x, rois)


class MLPFeatureExtractor(nn.Module):
    def __init__(self, config, dim_in):
        super(MLPFeatureExtractor, self).__init__()
        self.config = config
        head_cfg = config.MODEL.ROI_ACTION_HEAD

        self.pooler = self._make_3d_pooler(head_cfg)

        resolution = head_cfg.POOLER_RESOLUTION

        self.max_pooler = nn.MaxPool3d((1, resolution, resolution))

        if config.IA_STRUCTURE.ACTIVE:
            self.max_feature_len_per_sec = config.IA_STRUCTURE.MAX_PER_SEC

            self.ia_structure = make_ia_structure(config, dim_in)

        representation_size = head_cfg.MLP_HEAD_DIM

        fc1_dim_in = dim_in
        if config.IA_STRUCTURE.ACTIVE and (config.IA_STRUCTURE.FUSION == "concat"):
            fc1_dim_in += config.IA_STRUCTURE.DIM_OUT

        self.fc1 = nn.Linear(fc1_dim_in, representation_size)
        self.fc2 = nn.Linear(representation_size, representation_size)

        for l in [self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)

        self.dim_out = representation_size

    def _make_3d_pooler(self, head_cfg):
        resolution = head_cfg.POOLER_RESOLUTION
        scale = head_cfg.POOLER_SCALE
        sampling_ratio = head_cfg.POOLER_SAMPLING_RATIO
        pooler = Pooler3d(
            output_size=(resolution, resolution),
            scale=scale,
            sampling_ratio=sampling_ratio,
        )
        return pooler

    def roi_pooling(self, slow_features, fast_features, proposals):
        if slow_features is not None:
            if self.config.MODEL.ROI_ACTION_HEAD.MEAN_BEFORE_POOLER:
                slow_features = slow_features.mean(dim=2, keepdim=True)
            slow_x = self.pooler(slow_features, proposals)
            if not self.config.MODEL.ROI_ACTION_HEAD.MEAN_BEFORE_POOLER:
                slow_x = slow_x.mean(dim=2, keepdim=True)
            x = slow_x
        if fast_features is not None:
            if self.config.MODEL.ROI_ACTION_HEAD.MEAN_BEFORE_POOLER:
                fast_features = fast_features.mean(dim=2, keepdim=True)
            fast_x = self.pooler(fast_features, proposals)
            if not self.config.MODEL.ROI_ACTION_HEAD.MEAN_BEFORE_POOLER:
                fast_x = fast_x.mean(dim=2, keepdim=True)
            x = fast_x

        if slow_features is not None and fast_features is not None:
            x = torch.cat([slow_x, fast_x], dim=1)
        return x

    def max_pooling_zero_safe(self, x):
        if x.size(0) == 0:
            _, c, t, h, w = x.size()
            res = self.config.MODEL.ROI_ACTION_HEAD.POOLER_RESOLUTION
            x = torch.zeros((0, c, 1, h - res + 1, w - res + 1), device=x.device)
        else:
            x = self.max_pooler(x)
        return x

    def forward(self, slow_features, fast_features, proposals, objects=None, extras={}, part_forward=-1):
        ia_active = hasattr(self, "ia_structure")
        if part_forward == 1:
            person_pooled = cat([box.get_field("pooled_feature") for box in proposals])
            object_pooled = cat([box.get_field("pooled_feature") for box in objects])
        else:
            x = self.roi_pooling(slow_features, fast_features, proposals)

            person_pooled = self.max_pooler(x)

            if has_object(self.config.IA_STRUCTURE):
                object_pooled = self.roi_pooling(slow_features, fast_features, objects)
                object_pooled = self.max_pooling_zero_safe(object_pooled)
            else:
                object_pooled = None

        if part_forward == 0:
            return None, person_pooled, object_pooled

        x_after = person_pooled

        if ia_active:
            tsfmr = self.ia_structure
            mem_len = self.config.IA_STRUCTURE.LENGTH
            mem_rate = self.config.IA_STRUCTURE.MEMORY_RATE
            memory_person, memory_person_boxes = self.get_memory_feature(extras["person_pool"], extras, mem_len, mem_rate,
                                                                       self.max_feature_len_per_sec, tsfmr.dim_others,
                                                                       person_pooled, proposals)

            ia_feature = self.ia_structure(person_pooled, proposals, object_pooled, objects, memory_person, )
            x_after = self.fusion(x_after, ia_feature, self.config.IA_STRUCTURE.FUSION)

        x_after = x_after.view(x_after.size(0), -1)

        x_after = F.relu(self.fc1(x_after))
        x_after = F.relu(self.fc2(x_after))

        return x_after, person_pooled, object_pooled

    def get_memory_feature(self, feature_pool, extras, mem_len, mem_rate, max_boxes, fixed_dim, current_x, current_box):
        before, after = mem_len
        mem_feature_list = []
        mem_pos_list = []
        device = current_x.device
        current_feat = prepare_pooled_feature(current_x, current_box, detach=True)
        for movie_id, timestamp, new_feat in zip(extras["movie_ids"], extras["timestamps"], current_feat):
            before_inds = range(timestamp - before * mem_rate, timestamp, mem_rate)
            after_inds = range(timestamp + mem_rate, timestamp + (after + 1) * mem_rate, mem_rate)
            cache_cur_mov = feature_pool[movie_id]
            mem_box_list_before = [self.check_fetch_mem_feature(cache_cur_mov, mem_ind, max_boxes)
                                   for mem_ind in before_inds]
            mem_box_list_after = [self.check_fetch_mem_feature(cache_cur_mov, mem_ind, max_boxes)
                                  for mem_ind in after_inds]
            mem_box_current = [self.sample_mem_feature(new_feat, max_boxes), ]
            mem_box_list = mem_box_list_before + mem_box_current + mem_box_list_after
            mem_feature_list += [box_list.get_field("pooled_feature")
                                 if box_list is not None
                                 else torch.zeros(0, fixed_dim, 1, 1, 1, dtype=torch.float32, device="cuda")
                                 for box_list in mem_box_list]
            mem_pos_list += [box_list.bbox
                             if box_list is not None
                             else torch.zeros(0, 4, dtype=torch.float32, device="cuda")
                             for box_list in mem_box_list]

        seq_length = sum(mem_len) + 1
        person_per_seq = seq_length * max_boxes
        mem_feature = pad_sequence(mem_feature_list, max_boxes)
        mem_feature = mem_feature.view(-1, person_per_seq, fixed_dim, 1, 1, 1)
        mem_feature = mem_feature.to(device)
        mem_pos = pad_sequence(mem_pos_list, max_boxes)
        mem_pos = mem_pos.view(-1, person_per_seq, 4)
        mem_pos = mem_pos.to(device)

        return mem_feature, mem_pos

    def check_fetch_mem_feature(self, movie_cache, mem_ind, max_num):
        if mem_ind not in movie_cache:
            return None
        box_list = movie_cache[mem_ind]
        return self.sample_mem_feature(box_list, max_num)

    def sample_mem_feature(self, box_list, max_num):
        if len(box_list) > max_num:
            idx = torch.randperm(len(box_list))[:max_num]
            return box_list[idx].to("cuda")
        else:
            return box_list.to("cuda")

    def fusion(self, x, out, type="add"):
        if type == "add":
            return x + out
        elif type == "concat":
            return torch.cat([x, out], dim=1)
        else:
            raise NotImplementedError

class FCPredictor(nn.Module):
    def __init__(self, config, dim_in):
        super(FCPredictor, self).__init__()

        num_classes = config.MODEL.ROI_ACTION_HEAD.NUM_CLASSES

        dropout_rate = config.MODEL.ROI_ACTION_HEAD.DROPOUT_RATE
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate, inplace=True)

        self.cls_score = nn.Linear(dim_in, num_classes)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        scores = self.cls_score(x)

        return scores

class PostProcessor(nn.Module):
    def __init__(self, pose_action_num):
        super(PostProcessor, self).__init__()
        self.pose_action_num = pose_action_num

    def forward(self, x, boxes):
        # boxes should be (#detections,4)
        # prob should be calculated in different way.
        class_logits, = x
        pose_action_prob = F.softmax(class_logits[:,:self.pose_action_num],-1)
        interaction_action_prob = torch.sigmoid(class_logits[:,self.pose_action_num:])

        action_prob = torch.cat((pose_action_prob,interaction_action_prob),1)

        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        box_tensors = [a.bbox for a in boxes]

        action_prob = action_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, boxes_per_image, image_shape in zip(
                action_prob, box_tensors, image_shapes
        ):
            boxlist = self.prepare_boxlist(boxes_per_image, prob, image_shape)
            results.append(boxlist)
        return results

    def prepare_boxlist(self, boxes, scores, image_shape):
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        return boxlist