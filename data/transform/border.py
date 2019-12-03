import cv2
import numpy as np

__all__ = ['LabelDistanceTransform', 'NeighborhoodLabels', 'InstanceBorders']


class LabelDistanceTransform:
    def __init__(self, num_classes, bins=(4, 16, 64, 128), alphas=(8., 6., 4., 2., 1.), reduce=False,
                 ignore_id=19):
        self.num_classes = num_classes
        self.reduce = reduce
        self.bins = bins
        self.alphas = alphas
        self.ignore_id = ignore_id

    def __call__(self, example):
        labels = np.array(example['labels'])
        present_classes = np.unique(labels)
        distances = np.zeros([self.num_classes] + list(labels.shape), dtype=np.float32) - 1.
        for i in range(self.num_classes):
            if i not in present_classes:
                continue
            class_mask = labels == i
            distances[i][class_mask] = cv2.distanceTransform(np.uint8(class_mask), cv2.DIST_L2, maskSize=5)[class_mask]
        if self.reduce:
            ignore_mask = labels == self.ignore_id
            distances[distances < 0] = 0
            distances = distances.sum(axis=0)
            label_distance_bins = np.digitize(distances, self.bins)
            label_distance_alphas = np.zeros(label_distance_bins.shape, dtype=np.float32)
            for idx, alpha in enumerate(self.alphas):
                label_distance_alphas[label_distance_bins == idx] = alpha
            label_distance_alphas[ignore_mask] = 0
            example['label_distance_alphas'] = label_distance_alphas
        else:
            example['label_distance_transform'] = distances
        return example


class InstanceBorders:
    def __init__(self, instance_classes=8, thresh=.3):
        self.instance_classes = instance_classes
        self.thresh = thresh

    def __call__(self, example):
        shape = [self.instance_classes] + list(example['labels'].size)[::-1]
        instance_borders = np.zeros(shape, dtype=np.float32)
        instances = example['instances']
        for k in instances:
            for instance in instances[k]:
                dist_trans = cv2.distanceTransform(instance.astype(np.uint8), cv2.DIST_L2, maskSize=5)
                dist_trans[instance] = 1. / dist_trans[instance]
                dist_trans[dist_trans < self.thresh] = .0
                instance_borders[k] += dist_trans
        example['instance_borders'] = instance_borders
        return example


class NeighborhoodLabels:
    def __init__(self, num_classes, k=3, stride=1, discrete=False):
        self.num_classes = num_classes
        self.k = k
        self.pad = k // 2
        self.stride = stride
        self.discrete = discrete

    def __call__(self, example):
        labels = np.array(example['labels'])
        p = self.pad
        labels_padded = self.num_classes * np.ones([1, 1] + [sh + 2 * p for sh in labels.shape], dtype=labels.dtype)
        labels_padded[..., p:-p, p:-p] = labels.copy()
        label_col = im2col_cython.im2col_cython(labels_padded, self.k, self.k, padding=0, stride=self.stride)
        label_col_hist = im2col_cython.hist_from_cols(label_col, self.num_classes).reshape(
            [self.num_classes + 1] + list(labels.shape))
        label_neighborhood_hist = label_col_hist / np.float32(self.k ** 2)
        if self.discrete:
            example['label_neighborhood_hist'] = (label_neighborhood_hist[:self.num_classes] > 0.).astype(np.float32)
        else:
            example['label_neighborhood_hist'] = label_neighborhood_hist
        return example
