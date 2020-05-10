import torch
from torch.autograd import Variable
import torch.nn.functional as F


def one_hot_embedding(labels, num_classes):
    '''
    Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)
    return y[labels]


def focal_loss(x, y):
    '''
    Focal loss.
    Args:
      x: (tensor) sized [N,D].
      y: (tensor) sized [N,].
    Return:
      (tensor) focal loss.
    '''
    alpha = 0.25
    gamma = 2

    t = one_hot_embedding(y, x.shape[1] + 1)

    # exclude background
    t = t[:, 1:]

    t = Variable(t).cuda()
    p = x.sigmoid().float()

    # pt = p if t > 0 else 1-p
    pt = p * t + (1 - p) * (1 - t)

    # w = alpha if t > 0 else 1-alpha
    w = alpha * t + (1 - alpha) * (1 - t)

    w = w * (1 - pt).pow(gamma)
    return F.binary_cross_entropy_with_logits(x.float(), t, w.detach(), size_average=True)