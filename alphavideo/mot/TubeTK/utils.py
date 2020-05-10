import torch
import numpy as np


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.
    Args:
        points (Tensor): Shape (n, 3), [t, x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom, frDis, 4point, bkDis, 4point).
        max_shape (list): Shape of the image.
    Returns:
        Tensor: Decoded bboxes.
    """

    mid_t = points[:, 0]
    mid_x1 = points[:, 1] - distance[:, 0]
    mid_y1 = points[:, 2] - distance[:, 1]
    mid_x2 = points[:, 1] + distance[:, 2]
    mid_y2 = points[:, 2] + distance[:, 3]

    fr_t = points[:, 0] + distance[:, 4]
    fr_x1 = mid_x1 + distance[:, 5]
    fr_y1 = mid_y1 + distance[:, 6]
    fr_x2 = mid_x2 + distance[:, 7]
    fr_y2 = mid_y2 + distance[:, 8]

    bk_t = points[:, 0] - distance[:, 9]
    bk_x1 = mid_x1 + distance[:, 10]
    bk_y1 = mid_y1 + distance[:, 11]
    bk_x2 = mid_x2 + distance[:, 12]
    bk_y2 = mid_y2 + distance[:, 13]

    if max_shape is not None:
        mid_x1 = mid_x1.clamp(min=0, max=max_shape[2])
        mid_y1 = mid_y1.clamp(min=0, max=max_shape[1])
        mid_x2 = mid_x2.clamp(min=0, max=max_shape[2])
        mid_y2 = mid_y2.clamp(min=0, max=max_shape[1])

        fr_t = fr_t.clamp(min=0, max=max_shape[0])
        fr_x1 = fr_x1.clamp(min=0, max=max_shape[2])
        fr_y1 = fr_y1.clamp(min=0, max=max_shape[1])
        fr_x2 = fr_x2.clamp(min=0, max=max_shape[2])
        fr_y2 = fr_y2.clamp(min=0, max=max_shape[1])

        bk_t = bk_t.clamp(min=0, max=max_shape[0])
        bk_x1 = bk_x1.clamp(min=0, max=max_shape[2])
        bk_y1 = bk_y1.clamp(min=0, max=max_shape[1])
        bk_x2 = bk_x2.clamp(min=0, max=max_shape[2])
        bk_y2 = bk_y2.clamp(min=0, max=max_shape[1])

    return torch.stack([mid_t, mid_x1, mid_y1, mid_x2, mid_y2,
                        fr_t, fr_x1, fr_y1, fr_x2, fr_y2,
                        bk_t, bk_x1, bk_y1, bk_x2, bk_y2], -1)


def iou_loss(pred_tubes, target_tubes):
    ious = tube_iou(pred_tubes, target_tubes)
    loss = 1 - ious
    return loss


def giou_loss(pred_tubes, target_tubes):
    gious = tube_giou(pred_tubes, target_tubes)
    loss = 1 - gious
    loss = loss.clamp(min=0, max=2)
    return loss


def tube_giou(pred_tubes, target_tubes):
    mid_t_pred, mid_bboxes_pred, fr_t_pred, fr_bboxes_pred, bk_t_pred, bk_bboxes_pred = get3bboxes_from_tube(pred_tubes)
    mid_t_gt, mid_bboxes_gt, fr_t_gt, fr_bboxes_gt, bk_t_gt, bk_bboxes_gt = get3bboxes_from_tube(target_tubes)

    # get giou of mid_frame
    tube_vol_pred = volume(area(mid_bboxes_pred), area(fr_bboxes_pred), fr_t_pred - mid_t_pred) + \
                    volume(area(mid_bboxes_pred), area(bk_bboxes_pred), mid_t_pred - bk_t_pred)
    tube_vol_gt = volume(area(mid_bboxes_gt), area(fr_bboxes_gt), fr_t_gt - mid_t_gt) + \
                  volume(area(mid_bboxes_gt), area(bk_bboxes_gt), mid_t_gt - bk_t_gt)

    mid_intersect = bbox_overlaps(mid_bboxes_pred, mid_bboxes_gt)
    mid_enclose = bbox_enclose(mid_bboxes_pred, mid_bboxes_gt)

    iou = mid_intersect / (area(mid_bboxes_gt) + area(mid_bboxes_pred) - mid_intersect)
    giou = iou - (mid_enclose - (area(mid_bboxes_gt) + area(mid_bboxes_pred) - mid_intersect)) / mid_enclose

    # get intersect of front and back frame
    dis_fr_min, fr_bboxes_pred_align_min, fr_bboxes_gt_align_min = \
        align_bbox_on_frame(mid_bboxes_pred, fr_bboxes_pred, fr_t_pred - mid_t_pred,
                            mid_bboxes_gt, fr_bboxes_gt, fr_t_gt - mid_t_gt)
    fr_intersect = bbox_overlaps(fr_bboxes_pred_align_min, fr_bboxes_gt_align_min)

    dis_bk_min, bk_bboxes_pred_align_min, bk_bboxes_gt_align_min = \
        align_bbox_on_frame(mid_bboxes_pred, bk_bboxes_pred, mid_t_pred - bk_t_pred,
                            mid_bboxes_gt, bk_bboxes_gt, mid_t_gt - bk_t_gt)
    bk_intersect = bbox_overlaps(bk_bboxes_pred_align_min, bk_bboxes_gt_align_min)

    #  get enclose of front and back frame
    dis_fr_max, fr_bboxes_pred_align_max, fr_bboxes_gt_align_max = \
        align_bbox_on_frame(mid_bboxes_pred, fr_bboxes_pred, fr_t_pred - mid_t_pred,
                            mid_bboxes_gt, fr_bboxes_gt, fr_t_gt - mid_t_gt, mode='max')
    fr_enclose = bbox_enclose(fr_bboxes_pred_align_max, fr_bboxes_gt_align_max)

    dis_bk_max, bk_bboxes_pred_align_max, bk_bboxes_gt_align_max = \
        align_bbox_on_frame(mid_bboxes_pred, bk_bboxes_pred, mid_t_pred - bk_t_pred,
                            mid_bboxes_gt, bk_bboxes_gt, mid_t_gt - bk_t_gt, mode='max')
    bk_enclose = bbox_enclose(bk_bboxes_pred_align_max, bk_bboxes_gt_align_max)

    isTube = dis_fr_min + dis_bk_min != 0
    intersect = volume(mid_intersect[isTube], fr_intersect[isTube], dis_fr_min[isTube]) + \
                volume(mid_intersect[isTube], bk_intersect[isTube], dis_bk_min[isTube])
    iou[isTube] = intersect / (tube_vol_pred[isTube] + tube_vol_gt[isTube] - intersect)

    enclose = volume(mid_enclose[isTube], fr_enclose[isTube], dis_fr_max[isTube]) + \
              volume(mid_enclose[isTube], bk_enclose[isTube], dis_bk_max[isTube])

    giou[isTube] = iou[isTube] - (enclose - (tube_vol_pred[isTube] + tube_vol_gt[isTube] - intersect)) / enclose

    return giou


def tube_iou(pred_tubes, target_tubes):
    mid_t_pred, mid_bboxes_pred, fr_t_pred, fr_bboxes_pred, bk_t_pred, bk_bboxes_pred = get3bboxes_from_tube(pred_tubes)
    mid_t_gt, mid_bboxes_gt, fr_t_gt, fr_bboxes_gt, bk_t_gt, bk_bboxes_gt = get3bboxes_from_tube(target_tubes)

    # get the tubes volume
    tube_vol_pred = volume(area(mid_bboxes_pred), area(fr_bboxes_pred), fr_t_pred - mid_t_pred) + \
                    volume(area(mid_bboxes_pred), area(bk_bboxes_pred), mid_t_pred - bk_t_pred)
    tube_vol_gt = volume(area(mid_bboxes_gt), area(fr_bboxes_gt), fr_t_gt - mid_t_gt) + \
                  volume(area(mid_bboxes_gt), area(bk_bboxes_gt), mid_t_gt - bk_t_gt)

    # overlap area on mid bbox
    mid_overlap = bbox_overlaps(mid_bboxes_pred, mid_bboxes_gt)

    # overlap area on front bbox
    dis_fr, fr_bboxes_pred_align, fr_bboxes_gt_align = \
        align_bbox_on_frame(mid_bboxes_pred, fr_bboxes_pred, fr_t_pred - mid_t_pred,
                            mid_bboxes_gt, fr_bboxes_gt, fr_t_gt - mid_t_gt)
    fr_overlap = bbox_overlaps(fr_bboxes_pred_align, fr_bboxes_gt_align)

    # overlap area on back bbox
    dis_bk, bk_bboxes_pred_align, bk_bboxes_gt_align = \
        align_bbox_on_frame(mid_bboxes_pred, bk_bboxes_pred, mid_t_pred - bk_t_pred,
                            mid_bboxes_gt, bk_bboxes_gt, mid_t_gt - bk_t_gt)
    bk_overlap = bbox_overlaps(bk_bboxes_pred_align, bk_bboxes_gt_align)

    # overlap volume
    res = mid_overlap / (area(mid_bboxes_gt) + area(mid_bboxes_pred) - mid_overlap)
    isTube = dis_fr + dis_bk != 0
    overlap = volume(mid_overlap[isTube], fr_overlap[isTube], dis_fr[isTube]) + \
              volume(mid_overlap[isTube], bk_overlap[isTube], dis_bk[isTube])
    res[isTube] = overlap / (tube_vol_pred[isTube] + tube_vol_gt[isTube] - overlap)

    res = res.clamp(min=1e-5, max=1)
    return res


def get3bboxes_from_tube(tubes):
    mid_t = tubes[:, 0]
    mid_bboxes = tubes[:, 1:5]
    fr_t = tubes[:, 5]
    fr_bboxes = tubes[:, 6:10]
    bk_t = tubes[:, 10]
    bk_bboxes = tubes[:, 11:15]
    return mid_t, mid_bboxes, fr_t, fr_bboxes, bk_t, bk_bboxes


def area(bboxes):
    a = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    if isinstance(a, np.ndarray):
        return np.abs(a)
    else:
        return torch.abs(a)


def volume(bbox1_area, bbox2_area, dis):
    return (bbox1_area + bbox2_area + torch.sqrt(bbox1_area + 1e-5) * torch.sqrt(bbox2_area + 1e-5)) * dis


def align_bbox_on_frame(mid1, bbox1, t1, mid2, bbox2, t2, mode='min'):
    if mode == 'min':
        t = torch.min(t1, t2)
    else:
        t = torch.max(t1, t2)

    t1_zero_ind = t1 == 0
    t1_notzero_ind = t1 != 0
    bbox1_aligned = torch.zeros(mid1.shape, device=mid1.device)
    bbox1_aligned[t1_zero_ind] = mid1[t1_zero_ind]
    bbox1_aligned[t1_notzero_ind] = mid1[t1_notzero_ind] * ((t1[t1_notzero_ind]-t[t1_notzero_ind])/(t1[t1_notzero_ind]+1e-4)).unsqueeze(1).repeat(1, 4) + \
                                    bbox1[t1_notzero_ind] * (t[t1_notzero_ind]/(t1[t1_notzero_ind]+1e-4)).unsqueeze(1).repeat(1, 4)

    t2_zero_ind = t2 == 0
    t2_notzero_ind = t2 != 0
    bbox2_aligned = torch.zeros(mid2.shape, device=mid2.device)
    bbox2_aligned[t2_zero_ind] = mid2[t2_zero_ind]
    bbox2_aligned[t2_notzero_ind] = mid2[t2_notzero_ind] * ((t2[t2_notzero_ind]-t[t2_notzero_ind])/(t2[t2_notzero_ind]+1e-4)).unsqueeze(1).repeat(1, 4) + \
                                    bbox2[t2_notzero_ind] * (t[t2_notzero_ind]/(t2[t2_notzero_ind]+1e-4)).unsqueeze(1).repeat(1, 4)

    return t, bbox1_aligned, bbox2_aligned


def bbox_overlaps(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]

    if rows * cols == 0:
        return bboxes1.new(rows, 1)

    if isinstance(bboxes1, np.ndarray):
        # To avoid wrong pred bbox which is not left top cord and right bottom cord
        lt = np.maximum(np.minimum(bboxes1[:, :2], bboxes1[:, 2:]), np.minimum(bboxes2[:, :2], bboxes2[:, 2:]))
        rb = np.minimum(np.maximum(bboxes1[:, 2:], bboxes1[:, :2]), np.maximum(bboxes2[:, 2:], bboxes2[:, :2]))
        wh = np.clip(rb - lt, 0, None)
    else:
        lt = torch.max(torch.min(bboxes1[:, :2], bboxes1[:, 2:]), torch.min(bboxes2[:, :2], bboxes2[:, 2:]))
        rb = torch.min(torch.max(bboxes1[:, 2:], bboxes1[:, :2]), torch.max(bboxes2[:, 2:], bboxes2[:, :2]))
        wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    return overlap


def bbox_enclose(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]

    if rows * cols == 0:
        return bboxes1.new(rows, 1)

    if isinstance(bboxes1, np.ndarray):
        # To avoid wrong pred bbox which is not left top cord and right bottom cord
        lt = np.minimum(np.minimum(bboxes1[:, :2], bboxes1[:, 2:]),
                        np.minimum(bboxes2[:, :2], bboxes2[:, 2:]))
        rb = np.maximum(np.maximum(bboxes1[:, 2:], bboxes1[:, :2]),
                        np.maximum(bboxes2[:, 2:], bboxes2[:, :2]))
        wh = np.clip(rb - lt, 0, None)
    else:
        lt = torch.min(torch.min(bboxes1[:, :2], bboxes1[:, 2:]),
                       torch.min(bboxes2[:, :2], bboxes2[:, 2:]))
        rb = torch.max(torch.max(bboxes1[:, 2:], bboxes1[:, :2]),
                       torch.max(bboxes2[:, 2:], bboxes2[:, :2]))
        wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    return overlap


def bbox_iou_loss(bboxes1, bboxes2):
    iou = bbox_iou(bboxes1, bboxes2)
    return 1 - iou


def bbox_iou(bboxes1, bboxes2):

    overlap = bbox_overlaps(bboxes1, bboxes2)

    area1 = area(bboxes1)
    area2 = area(bboxes2)

    ious = overlap / (area1 + area2 - overlap)

    return ious