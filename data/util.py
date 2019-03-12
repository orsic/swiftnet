import random
from torch.utils.data import Dataset
import torch
import numpy as np
from time import perf_counter


class SplitDataset(Dataset):
    def __init__(self, dataset, transforms=lambda x: x, indices=()):
        self.dataset = dataset
        self.transforms = transforms
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = self.indices[item]
        return self.transforms(self.dataset[idx])

    def __getattr__(self, item):
        return getattr(self.dataset, item)


def random_indices(N, seed=5, ratio=.8):
    indices = list(range(N))
    random.seed(seed)
    random.shuffle(indices)
    split = int(ratio * N)
    return indices[:split], indices[split:]


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
