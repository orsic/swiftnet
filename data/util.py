import random
from torch.utils.data import Dataset
import torch
import numpy as np
import pickle
from collections import defaultdict
from PIL import Image as pimg


def disparity_distribution_uniform(max_disp, num_bins):
    return np.linspace(0, max_disp, num_bins - 1)


def disparity_distribution_log(num_bins):
    return np.power(np.sqrt(2), np.arange(num_bins - 1))


def downsample_distribution(labels, factor, num_classes):
    h, w = labels.shape
    assert h % factor == 0 and w % factor == 0
    new_h = h // factor
    new_w = w // factor
    labels_4d = np.ascontiguousarray(labels.reshape(new_h, factor, new_w, factor), labels.dtype)
    labels_oh = np.eye(num_classes, dtype=np.float32)[labels_4d]
    target_dist = labels_oh.sum((1, 3)) / factor ** 2
    return target_dist


def downsample_distribution_th(labels, factor, num_classes, ignore_id=None):
    n, h, w = labels.shape
    assert h % factor == 0 and w % factor == 0
    new_h = h // factor
    new_w = w // factor
    labels_4d = labels.view(n, new_h, factor, new_w, factor)
    labels_oh = torch.eye(num_classes).to(labels_4d.device)[labels_4d]
    target_dist = labels_oh.sum(2).sum(3) / factor ** 2
    return target_dist


def downsample_labels_th(labels, factor, num_classes):
    '''
    :param labels: Tensor(N, H, W)
    :param factor: int
    :param num_classes:  int
    :return: FloatTensor(-1, num_classes), ByteTensor(-1, 1)
    '''
    n, h, w = labels.shape
    assert h % factor == 0 and w % factor == 0
    new_h = h // factor
    new_w = w // factor
    labels_4d = labels.view(n, new_h, factor, new_w, factor)
    # +1 class here because ignore id = num_classes
    labels_oh = torch.eye(num_classes + 1).to(labels_4d.device)[labels_4d]
    target_dist = labels_oh.sum(2).sum(3) / factor ** 2
    C = target_dist.shape[-1]
    target_dist = target_dist.view(-1, C)
    # keep only boxes which have p(ignore) < 0.5
    valid_mask = target_dist[:, -1] < 0.5
    target_dist = target_dist[:, :-1].contiguous()
    dist_sum = target_dist.sum(1, keepdim=True)
    # avoid division by zero
    dist_sum[dist_sum == 0] = 1
    # renormalize distribution after removing p(ignore)
    target_dist /= dist_sum
    return target_dist, valid_mask


def equalize_hist_disparity_distribution(d, L):
    cd = np.cumsum(d / d.sum())
    Y = np.round((L - 1) * cd).astype(np.uint8)
    return np.array([np.argmax(Y == i) for i in range(L - 1)])


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def one_hot_encoding(labels, C):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).to(labels.device).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    return target


def crop_and_scale_img(img: pimg, crop_box, target_size, pad_size, resample, blank_value):
    target = pimg.new(img.mode, pad_size, color=blank_value)
    target.paste(img)
    res = target.crop(crop_box).resize(target_size, resample=resample)
    return res
