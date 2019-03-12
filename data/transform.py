import numpy as np
from PIL import Image as pimg
import torch

from collections import defaultdict
from torch.utils.data.dataloader import default_collate

__all__ = ['Open', 'RemapLabels', 'Normalize', 'Denormalize', 'DenormalizeTh', 'Downsample', 'Resize', 'Tensor',
           'Numpy', 'ColorizeLabels', 'detection_collate', 'custom_collate']

RESAMPLE = pimg.BICUBIC
RESAMPLE_D = pimg.BILINEAR


class Open:
    def __init__(self, palette=None):
        self.palette = palette

    def __call__(self, example: dict):
        try:
            ret_dict = {
                'image': pimg.open(example['image']).convert('RGB'),
            }
            ret_dict['target_size'] = ret_dict['image'].size
            if 'labels' in example:
                ret_dict['labels'] = pimg.open(example['labels'])
                if self.palette is not None:
                    ret_dict['labels'].putpalette(self.palette)
                ret_dict['original_labels'] = ret_dict['labels'].copy()
        except OSError:
            print(example)
            raise
        return {**example, **ret_dict}


class RemapLabels:
    def __init__(self, mapping, ignore_id, total=35):
        self.mapping = np.ones((total + 1,), dtype=np.uint8) * ignore_id
        self.ignore_id = ignore_id
        for i in range(len(self.mapping)):
            self.mapping[i] = mapping[i] if i in mapping else ignore_id

    def _trans(self, labels):
        labels = self.mapping[labels].astype(labels.dtype)
        return labels

    def __call__(self, example):
        if not isinstance(example, dict):
            return self._trans(example)
        if 'labels' not in example:
            return example
        ret_dict = {
            'labels': pimg.fromarray(self._trans(np.array(example['labels']))),
            'original_labels': pimg.fromarray(self._trans(np.array(example['original_labels']))),
        }
        return {**example, **ret_dict}


class Norm:
    def __init__(self, scale, mean, std):
        self.scale = scale
        self.mean = mean
        self.std = std

    def _trans(self, img):
        raise NotImplementedError

    def __call__(self, example):
        ret_dict = {
            'image': self._trans(example['image'])
        }
        if 'pyramid' in example:
            ret_dict['pyramid'] = [self._trans(p) for p in example['pyramid']]
        if 'pyramid_ms' in example:
            ret_dict['pyramid_ms'] = [[self._trans(p) for p in pyramid] for pyramid in example['pyramid_ms']]
        return {**example, **ret_dict}


class Normalize(Norm):
    def _trans(self, img):
        img = np.array(img).astype(np.float32)
        if self.scale != 1:
            img /= self.scale
        img -= self.mean
        img /= self.std
        return img


class Denormalize(Norm):
    def _trans(self, img):
        img = np.array(img)
        img *= self.std
        img += self.mean
        if self.scale != 1:
            img *= self.scale
        return img


class DenormalizeTh(Norm):
    def __init__(self, scale, mean, std):
        super(DenormalizeTh, self).__init__(scale, mean, std)
        self.mean = torch.FloatTensor(mean).view(1, 3, 1, 1)
        self.std = torch.FloatTensor(std).view(1, 3, 1, 1)

    def _trans(self, img):
        img *= self.std
        img += self.mean
        if self.scale != 1:
            img *= self.scale
        return img


class Downsample:
    def __init__(self, factor=2):
        self.factor = factor

    def __call__(self, example):
        if self.factor <= 1:
            return example
        W, H = example['image'].size
        w, h = W // self.factor, H // self.factor
        size = (w, h)
        ret_dict = {
            'image': example['image'].resize(size, resample=RESAMPLE),
            'labels': example['labels'].resize(size, resample=pimg.NEAREST),
        }
        return {**example, **ret_dict}


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, example):
        # raise NotImplementedError()
        ret_dict = {'image': example['image'].resize(self.size, resample=RESAMPLE)}
        if 'labels' in example:
            ret_dict['labels'] = example['labels'].resize(self.size, resample=pimg.NEAREST)
        return {**example, **ret_dict}


class Tensor:
    def _trans(self, img, dtype):
        img = np.array(img, dtype=dtype)
        if len(img.shape) == 3:
            img = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))
        return torch.from_numpy(img)

    def __call__(self, example):
        ret_dict = {
            'image': self._trans(example['image'], np.float32),
        }
        if 'labels' in example:
            ret_dict['labels'] = self._trans(example['labels'], np.int64)
        if 'original_labels' in example:
            ret_dict['original_labels'] = self._trans(example['original_labels'], np.int64)
        if 'pyramid' in example:
            ret_dict['pyramid'] = [self._trans(p, np.float32) for p in example['pyramid']]
        if 'pyramid_ms' in example:
            ret_dict['pyramid_ms'] = [[self._trans(p, np.float32) for p in pyramids] for pyramids in
                                      example['pyramid_ms']]
        return {**example, **ret_dict}


class Numpy:
    def __call__(self, example):
        image = example['image']
        axes = [0, 2, 3, 1] if len(image.shape) == 4 else [1, 2, 0]
        ret_dict = {
            'image': image.numpy().transpose(axes)
        }
        if 'labels' in example:
            ret_dict['labels'] = example['labels'].numpy()
        return {**example, **ret_dict}


class ColorizeLabels:
    def __init__(self, color_info):
        self.color_info = np.array(color_info)

    def _trans(self, lab):
        R, G, B = [np.zeros_like(lab) for _ in range(3)]
        for l in np.unique(lab):
            mask = lab == l
            R[mask] = self.color_info[l][0]
            G[mask] = self.color_info[l][1]
            B[mask] = self.color_info[l][2]
        return np.stack((R, G, B), axis=-1).astype(np.uint8)

    def __call__(self, example):
        if not isinstance(example, dict):
            return self._trans(example)
        assert 'labels' in example
        return {**example, **{'labels': self._trans(example['labels']),
                              'original_labels': self._trans(example['original_labels'])}}


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    custom = defaultdict(list)
    custom_keys = ['target_size', ]
    for sample in batch:
        for k in custom_keys:
            custom[k] += [sample[k]]
    other = {k: default_collate([b[k] for b in batch]) for k in
             filter(lambda x: x not in custom, batch[0].keys())}
    return {**other, **custom}


def custom_collate(batch):
    target_size = batch[0]['target_size']
    target_size_feats = batch[0]['target_size_feats']
    alphas = batch[0]['alphas']
    target_level = batch[0]['target_level']
    for b in batch:
        del b['target_size']
        del b['target_size_feats']
        del b['alphas']
        del b['target_level']
    batch = default_collate(batch)
    batch['target_size'] = target_size
    batch['target_size_feats'] = target_size_feats
    batch['alphas'] = alphas
    batch['target_level'] = target_level
    return batch
