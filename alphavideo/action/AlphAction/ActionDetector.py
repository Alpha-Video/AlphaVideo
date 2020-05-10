import torch
import torch.nn as nn

from .SlowFast import slowfast_res50, slowfast_res101
from .head import ROIActionHead
from .utils.structures import MemoryPool, BoxList
from alphavideo.utils.load_url import load_url_google
from .config import _C

class ActionDetector(nn.Module):
    def __init__(self, cfg, backbone_func, num_class=80, pretrain=True, name=None):
        super(ActionDetector, self).__init__()
        self.backbone = backbone_func()
        cfg.MODEL.ROI_ACTION_HEAD.NUM_CLASSES = num_class
        self.roi_heads = ROIActionHead(cfg, 2304)

        self.person_pool = MemoryPool()
        self.object_pool = MemoryPool()

        if pretrain and cfg.MODEL.WEIGHT != '':
            path = load_url_google(id=cfg.MODEL.WEIGHT, name=name)
            self.load_pretrain(model_path=path)

        # currently, only support eval
        self.eval()

    def load_pretrain(self, model_path):
        pre_model = torch.load(model_path, map_location='cpu')
        model_dict = self.state_dict()
        prefix_flag = True
        for key in pre_model:
            if not key.startswith("module."):
                prefix_flag = False
                break
        for key in model_dict:
            if prefix_flag:
                key = "module." + key
            if key not in pre_model:
                continue
            model_dict[key] = pre_model[key]
        self.load_state_dict(model_dict)
        del pre_model, model_dict

    def forward(self, video_clip, boxes, video_metas, objects=None, ):

        roi_features = self.store_feature(video_clip, boxes, video_metas, objects,)
        result = self.get_prediction(roi_features, video_metas, device=video_clip.device)

        return result

    def store_feature(self, video_clip, boxes, video_metas, objects=None, ):
        assert len(video_metas)==len(boxes)==video_clip.size(0)
        device = video_clip.device
        h, w = video_clip.shape[3:5]
        boxes = self._convert_bbox(boxes, video_metas, device, (w,h))
        objects = self._convert_bbox(objects, video_metas, device, (w,h))

        fast_video = video_clip
        slow_start = (self.backbone.pathway_ratio-1)//2
        slow_video = fast_video[:,:,slow_start::self.backbone.pathway_ratio,:,:]

        slow_features, fast_features = self.backbone(slow_video, fast_video)
        roi_features = self.roi_heads(slow_features, fast_features, boxes, objects, {}, part_forward=0)
        cpu_device = torch.device("cpu")
        person_feature = [ft.to(cpu_device) for ft in roi_features[0]]
        object_feature = [ft.to(cpu_device) for ft in roi_features[1]]
        movie_ids = [e["movie_id"] for e in video_metas]
        timestamps = [e["timestamp"] for e in video_metas]
        for movie_id, timestamp, p_ft, o_ft in zip(movie_ids, timestamps, person_feature, object_feature):
            self.person_pool[movie_id, timestamp] = p_ft
            self.object_pool[movie_id, timestamp] = o_ft
        return roi_features

    def get_prediction(self, roi_features=None, video_metas=[], device="cuda"):
        movie_ids = [e["movie_id"] for e in video_metas]
        timestamps = [e["timestamp"] for e in video_metas]

        if roi_features is None:
            person_feature = [self.person_pool[movie_id, timestamp].to(device)
                              for movie_id, timestamp in zip(movie_ids, timestamps)]
            object_feature = [self.object_pool[movie_id, timestamp].to(device)
                              for movie_id, timestamp in zip(movie_ids, timestamps)]
        else:
            person_feature, object_feature = roi_features

        extras = dict(
            person_pool=self.person_pool,
            movie_ids=movie_ids,
            timestamps=timestamps,
            current_feat_p=person_feature,
            current_feat_o=object_feature,
        )

        output = self.roi_heads(None, None, None, None, extras=extras, part_forward=1)

        return output

    def _convert_bbox(self, box_list, video_metas, device, size):
        if box_list is None:
            box_list = [[],]*len(video_metas)
        new_list = []
        for i, boxes in enumerate(box_list):
            if boxes is None:
                new_list.append(None)
            elif isinstance(boxes, BoxList):
                new_list.append(boxes.to(device))
            else:
                # height = video_metas[i]["height"]
                # width = video_metas[i]["width"]
                # assume it be a list
                box_tensor = torch.as_tensor(boxes, dtype=torch.float32)
                box_tensor = box_tensor.reshape(-1, 4)
                boxes = BoxList(box_tensor, size, mode="xyxy").to(device)
                new_list.append(boxes)
        return new_list

    def empty_pool(self):
        self.person_pool = MemoryPool()
        self.object_pool = MemoryPool()

def action_detector_res50(num_class=80, pretrain=True):
    if pretrain:
        _C.MODEL.WEIGHT = _C.MODEL.WEIGHT_50
    else:
        _C.MODEL.WEIGHT = ""
    return ActionDetector(_C, slowfast_res50, num_class=num_class,
                          pretrain=pretrain, name = "alphaction-res50.pth")

def action_detector_res101(num_class=80, pretrain=True):
    if pretrain:
        _C.MODEL.WEIGHT = _C.MODEL.WEIGHT_101
    else:
        _C.MODEL.WEIGHT = ""
    return ActionDetector(_C, slowfast_res101, num_class=num_class,
                          pretrain=pretrain, name = "alphaction-res101.pth")