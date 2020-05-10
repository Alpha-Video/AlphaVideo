import torch
from alphavideo.utils import tube_nms_cuda


def multiclass_tube_nms(multi_tubes,  # n, 15
                        multi_scores,  # n, 1 + n_cls
                        score_thr,
                        iou_thre,
                        max_num=-1,
                        score_factors=None,  # n
                        frame_num=16):
    """NMS for multi-class tubes.
    Args:
        multi_tubes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, 1+#class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        iou_thre (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS
        frame_num (int): number of frames in input
    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.shape[1]
    tubes, labels = [], []
    nms_op = tube_nms
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        # print('before: ' + str(len(cls_inds)))
        if not cls_inds.any():
            continue

        # get bboxes and scores of this class
        _tubes = multi_tubes[cls_inds, :]
        _scores = multi_scores[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
            pass

        # do nms in each frame
        for n_f in range(frame_num):
            frame_inds = torch.round(_tubes[:, 0]) == n_f
            if torch.sum(frame_inds) == 0:
                continue
            _tubes_single_frame = _tubes[frame_inds]
            # mid_frame = _bboxes_single_frame[:, 1:5]
            # cls_dets = torch.cat([mid_frame, _scores[frame_inds, None]], dim=1)  # n, 4 + 1
            cls_dets = torch.cat([_tubes_single_frame, _scores[frame_inds, None]], dim=1)  # n, 15 + 1
            _, inds = nms_op(cls_dets, iou_thre)
            # cls_dets = _bboxes_single_frame[inds]
            cls_dets = cls_dets[inds]
            cls_labels = multi_tubes.new_full(
                (cls_dets.shape[0], ), i - 1, dtype=torch.long)
            tubes.append(cls_dets)
            labels.append(cls_labels)
    if tubes:
        tubes = torch.cat(tubes)
        labels = torch.cat(labels)
        # print('middle: ' + str(len(bboxes)))

        # =====================================
        # bboxes = bboxes[bboxes[:, -1] > score_thr]
        # =====================================

        if tubes.shape[0] > max_num:
            _, inds = tubes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            tubes = tubes[inds]
            labels = labels[inds]
    else:
        tubes = multi_tubes.new_zeros((0, multi_tubes.shape[1] + 1))
        labels = multi_tubes.new_zeros((0,), dtype=torch.long)
    # print('after: ' + str(len(bboxes)))
    return tubes, labels


def tube_nms(dets, iou_thr, device_id=None):
    """Dispatch to either CPU or GPU NMS implementations.
    The input can be either a torch tensor or numpy array. GPU NMS will be used
    if the input is a gpu tensor or device_id is specified, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.
    Arguments:
        dets (torch.Tensor): bboxes with scores.
        iou_thr (float): IoU threshold for NMS.
        device_id (int, optional): when `dets` is a numpy array, if `device_id`
            is None, then cpu nms is used, otherwise gpu_nms will be used.
    Returns:
        tuple: kept bboxes and indice, which is always the same data type as
            the input.
    """
    # convert dets (tensor or numpy array) to tensor
    if isinstance(dets, torch.Tensor):
        dets_th = dets
    else:
        raise TypeError(
            'dets must be either a Tensor, but got {}'.format(
                type(dets)))

    # execute cpu or cuda nms
    if dets_th.shape[0] == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
    else:
        inds = tube_nms_cuda.nms(dets_th, iou_thr, iou_thr)

    return dets[inds, :], inds