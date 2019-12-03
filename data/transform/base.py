from collections import defaultdict
from torch.utils.data.dataloader import default_collate
import numpy as np
import torch
from PIL import Image as pimg

from data.transform.flow_utils import readFlow

RESAMPLE = pimg.BICUBIC
RESAMPLE_D = pimg.BILINEAR

__all__ = ['Open', 'SetTargetSize', 'Numpy', 'Tensor', 'detection_collate', 'custom_collate', 'RESAMPLE', 'RESAMPLE_D']


class Open:
    def __init__(self, palette=None, copy_labels=True):
        self.palette = palette
        self.copy_labels = copy_labels

    def __call__(self, example: dict):
        try:
            ret_dict = {}
            for k in ['image', 'image_next', 'image_prev']:
                if k in example:
                    ret_dict[k] = pimg.open(example[k]).convert('RGB')
                    if k == 'image':
                        ret_dict['target_size'] = ret_dict['image'].size
            if 'depth' in example:
                example['depth'] = pimg.open(example['depth'])
            if 'labels' in example:
                ret_dict['labels'] = pimg.open(example['labels'])
                if self.palette is not None:
                    ret_dict['labels'].putpalette(self.palette)
                if self.copy_labels:
                    ret_dict['original_labels'] = ret_dict['labels'].copy()
            if 'flow' in example:
                ret_dict['flow'] = readFlow(example['flow'])
        except OSError:
            print(example)
            raise
        return {**example, **ret_dict}


class SetTargetSize:
    def __init__(self, target_size, target_size_feats, stride=4):
        self.target_size = target_size
        self.target_size_feats = target_size_feats
        self.stride = stride

    def __call__(self, example):
        if all([self.target_size, self.target_size_feats]):
            example['target_size'] = self.target_size[::-1]
            example['target_size_feats'] = self.target_size_feats[::-1]
        else:
            k = 'original_labels' if 'original_labels' in example else 'image'
            example['target_size'] = example[k].shape[-2:]
            example['target_size_feats'] = tuple([s // self.stride for s in example[k].shape[-2:]])
        example['alphas'] = [-1]
        example['target_level'] = 0
        return example


class Tensor:
    def _trans(self, img, dtype):
        img = np.array(img, dtype=dtype)
        if len(img.shape) == 3:
            img = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))
        return torch.from_numpy(img)

    def __call__(self, example):
        ret_dict = {}
        for k in ['image', 'image_next', 'image_prev']:
            if k in example:
                ret_dict[k] = self._trans(example[k], np.float32)
        if 'depth' in example:
            ret_dict['depth'] = self._trans(example['depth'], np.uint8)
        if 'labels' in example:
            ret_dict['labels'] = self._trans(example['labels'], np.int64)
        if 'original_labels' in example:
            ret_dict['original_labels'] = self._trans(example['original_labels'], np.int64)
        if 'depth_hist' in example:
            ret_dict['depth_hist'] = [self._trans(d, np.float32) for d in example['depth_hist']] if isinstance(
                example['depth_hist'], list) else self._trans(example['depth_hist'], np.float32)
        if 'pyramid' in example:
            ret_dict['pyramid'] = [self._trans(p, np.float32) for p in example['pyramid']]
        if 'pyramid_ms' in example:
            ret_dict['pyramid_ms'] = [[self._trans(p, np.float32) for p in pyramids] for pyramids in
                                      example['pyramid_ms']]
        if 'mux_indices' in example:
            ret_dict['mux_indices'] = torch.stack([torch.from_numpy(midx.flatten()) for midx in example['mux_indices']])
        if 'mux_masks' in example:
            ret_dict['mux_masks'] = [torch.from_numpy(np.uint8(mi)).unsqueeze(0) for mi in example['mux_masks']]
        if 'depth_bins' in example:
            ret_dict['depth_bins'] = torch.stack([torch.from_numpy(b) for b in example['depth_bins']])
        if 'flow' in example:
            # ret_dict['flow'] = torch.from_numpy(example['flow']).permute(2, 0, 1).contiguous()
            ret_dict['flow'] = torch.from_numpy(np.ascontiguousarray(example['flow']))
        # if 'flow_next' in example:
        #     ret_dict['flow_next'] = torch.from_numpy(example['flow_next']).permute(2, 0, 1 ).contiguous()
        if 'flow_sub' in example:
            # ret_dict['flow_sub'] = torch.from_numpy(example['flow_sub']).permute(2, 0, 1).contiguous()
            ret_dict['flow_sub'] = torch.from_numpy(np.ascontiguousarray(example['flow_sub']))
        if 'flipped' in example:
            del example['flipped']
        return {**example, **ret_dict}


class Numpy:
    def __call__(self, example):
        image = example['image']
        axes = [0, 2, 3, 1] if len(image.shape) == 4 else [1, 2, 0]
        ret_dict = {
            'image': image.numpy().transpose(axes)
        }
        for k in ['labels', 'original_labels']:
            if k in example and isinstance(example[k], torch.Tensor):
                ret_dict[k] = example[k].numpy()
        return {**example, **ret_dict}


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


def custom_collate(batch, del_orig_labels=False):
    keys = ['target_size', 'target_size_feats', 'alphas', 'target_level']
    values = {}
    for k in keys:
        if k in batch[0]:
            values[k] = batch[0][k]
    for b in batch:
        if del_orig_labels: del b['original_labels']
        for k in values.keys():
            del b[k]
        if 'mux_indices' in b:
            b['mux_indices'] = b['mux_indices'].view(-1)
    batch = default_collate(batch)
    # if 'image_next' in batch:
    #     batch['image'] = torch.cat([batch['image'], batch['image_next']], dim=0).contiguous()
    #     del batch['image_next']
    for k, v in values.items():
        batch[k] = v
    return batch
