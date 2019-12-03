import pickle
import random
from math import ceil

import numpy as np
import torch
from PIL import Image as pimg

from data.transform import RESAMPLE, RESAMPLE_D
from data.transform.flow_utils import pad_flow, crop_and_scale_flow, flip_flow_horizontal
from data.util import bb_intersection_over_union, crop_and_scale_img

__all__ = ['Pad', 'PadToFactor', 'Normalize', 'Denormalize', 'DenormalizeTh', 'Resize', 'RandomFlip',
           'RandomSquareCropAndScale', 'ResizeLongerSide', 'Downsample']


class Pad:
    def __init__(self, size, ignore_id, mean):
        self.size = size
        self.ignore_id = ignore_id
        self.mean = mean

    def _do(self, data, color):
        blank = pimg.new(mode=data.mode, size=self.size, color=color)
        blank.paste(data)
        return blank

    def __call__(self, example):
        ret_dict = {}
        for k, c in zip(['image', 'labels', 'original_labels', 'image_next', 'image_prev'],
                        [self.mean, self.ignore_id, self.ignore_id, self.mean, self.mean]):
            if k in example:
                ret_dict[k] = self._do(example[k], c)
        if 'flow' in example:
            ret_dict['flow'] = pad_flow(example['flow'], self.size)
        return {**example, **ret_dict}


class PadToFactor:
    def __init__(self, factor, ignore_id, mean):
        self.factor = factor
        self.ignore_id = ignore_id
        self.mean = mean

    def _do(self, data, color, size):
        blank = pimg.new(mode=data.mode, size=size, color=color)
        blank.paste(data)
        return blank

    def __call__(self, example):
        ret_dict = {}
        size = tuple(map(lambda x: ceil(x / self.factor) * self.factor, example['image'].size))
        for k, c in zip(['image', 'labels', 'original_labels', 'image_next', 'image_prev'],
                        [self.mean, self.ignore_id, self.ignore_id, self.mean]):
            if k in example:
                ret_dict[k] = self._do(example[k], c, size)
        if 'flow' in example:
            ret_dict['flow'] = pad_flow(example['flow'], size)
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
        for k in ['image_prev', 'image_next']:
            if k in example:
                ret_dict[k] = self._trans(example[k])
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
        if 'depth' in example:
            ret_dict['depth'] = example['depth'].resize(size, resample=RESAMPLE)
        return {**example, **ret_dict}


class RandomSquareCropAndScale:
    def __init__(self, wh, mean, ignore_id, min=.5, max=2., class_incidence=None, class_instances=None,
                 inst_classes=(3, 12, 14, 15, 16, 17, 18), scale_method=lambda scale, wh, size: int(scale * wh)):
        self.wh = wh
        self.min = min
        self.max = max
        self.mean = mean
        self.ignore_id = ignore_id
        self.random_gens = [self._rand_location]
        self.scale_method = scale_method

        if class_incidence is not None and class_instances is not None:
            self.true_random = False
            class_incidence_obj = np.load(class_incidence)
            with open(class_instances, 'rb') as f:
                self.class_instances = pickle.load(f)
            inst_classes = np.array(inst_classes)
            class_freq = class_incidence_obj[inst_classes].astype(np.float32)
            class_prob = 1. / (class_freq / class_freq.sum())
            class_prob /= class_prob.sum()
            self.p_class = {k.item(): v.item() for k, v in zip(inst_classes, class_prob)}
            self.random_gens += [self._gen_instance_box]
            print(f'Instance based random cropping:\n\t{self.p_class}')

    def _random_instance(self, name, W, H):
        def weighted_random_choice(choices):
            max = sum(choices)
            pick = random.uniform(0, max)
            key, current = 0, 0.
            for key, value in enumerate(choices):
                current += value
                if current > pick:
                    return key
                key += 1
            return key

        instances = self.class_instances[name]
        possible_classes = list(set(self.p_class.keys()).intersection(instances.keys()))
        roulette = []
        flat_instances = []
        for c in possible_classes:
            flat_instances += instances[c]
            roulette += [self.p_class[c]] * len(instances[c])
        if len(flat_instances) == 0:
            return [0, W - 1, 0, H - 1]
        index = weighted_random_choice(roulette)
        return flat_instances[index]

    def _gen_instance_box(self, W, H, target_wh, name, flipped):
        wmin, wmax, hmin, hmax = self._random_instance(name, W, H)
        if flipped:
            wmin, wmax = W - 1 - wmax, W - 1 - wmin
        inst_box = [wmin, hmin, wmax, hmax]
        for _ in range(50):
            box = self._rand_location(W, H, target_wh)
            if bb_intersection_over_union(box, inst_box) > 0.:
                break
        return box

    def _rand_location(self, W, H, target_wh, *args, **kwargs):
        try:
            w = np.random.randint(0, W - target_wh + 1)
            h = np.random.randint(0, H - target_wh + 1)
        except ValueError:
            print(f'Exception in RandomSquareCropAndScale: {target_wh}')
            w = h = 0
        # left, upper, right, lower)
        return w, h, w + target_wh, h + target_wh

    def _trans(self, img: pimg, crop_box, target_size, pad_size, resample, blank_value):
        return crop_and_scale_img(img, crop_box, target_size, pad_size, resample, blank_value)

    def __call__(self, example):
        image = example['image']
        scale = np.random.uniform(self.min, self.max)
        W, H = image.size
        box_size = self.scale_method(scale, self.wh, image.size)
        pad_size = (max(box_size, W), max(box_size, H))
        target_size = (self.wh, self.wh)
        crop_fn = random.choice(self.random_gens)
        flipped = example['flipped'] if 'flipped' in example else False
        crop_box = crop_fn(pad_size[0], pad_size[1], box_size, example.get('name'), flipped)
        ret_dict = {
            'image': self._trans(image, crop_box, target_size, pad_size, RESAMPLE, self.mean),
        }
        if 'labels' in example:
            ret_dict['labels'] = self._trans(example['labels'], crop_box, target_size, pad_size, pimg.NEAREST, self.ignore_id)
        for k in ['image_prev', 'image_next']:
            if k in example:
                ret_dict[k] = self._trans(example[k], crop_box, target_size, pad_size, RESAMPLE,
                                          self.mean)
        if 'depth' in example:
            ret_dict['depth'] = self._trans(example['depth'], crop_box, target_size, pad_size, RESAMPLE_D, 0)
        if 'flow' in example:
            ret_dict['flow'] = crop_and_scale_flow(example['flow'], crop_box, target_size, pad_size, scale)
        return {**example, **ret_dict}


class RandomFlip:
    def _trans(self, img: pimg, flip: bool):
        return img.transpose(pimg.FLIP_LEFT_RIGHT) if flip else img

    def __call__(self, example):
        flip = np.random.choice([False, True])
        ret_dict = {}
        for k in ['image', 'image_next', 'image_prev', 'labels', 'depth']:
            if k in example:
                ret_dict[k] = self._trans(example[k], flip)
        if ('flow' in example) and flip:
            ret_dict['flow'] = flip_flow_horizontal(example['flow'])
        return {**example, **ret_dict}


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, example):
        # raise NotImplementedError()
        ret_dict = {'image': example['image'].resize(self.size, resample=RESAMPLE)}
        if 'labels' in example:
            ret_dict['labels'] = example['labels'].resize(self.size, resample=pimg.NEAREST)
        if 'depth' in example:
            ret_dict['depth'] = example['depth'].resize(self.size, resample=RESAMPLE_D)
        return {**example, **ret_dict}


class ResizeLongerSide:
    def __init__(self, size):
        self.size = size

    def __call__(self, example):
        ret_dict = {}
        k = 'image' if 'image' in example else 'labels'
        scale = self.size / max(example[k].size)
        size = tuple([int(wh * scale) for wh in example[k].size])
        if 'image' in example:
            ret_dict['image'] = example['image'].resize(size, resample=RESAMPLE)
        if 'labels' in example:
            ret_dict['labels'] = example['labels'].resize(size, resample=pimg.NEAREST)
        # if 'original_labels' in example:
        #     ret_dict['original_labels'] = example['original_labels'].resize(size, resample=pimg.NEAREST)
        if 'depth' in example:
            ret_dict['depth'] = example['depth'].resize(size, resample=RESAMPLE_D)
        return {**example, **ret_dict}
