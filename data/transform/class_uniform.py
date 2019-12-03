import pickle
import numpy as np
from tqdm import tqdm
import random
from PIL import Image as pimg
from collections import defaultdict
import warnings

from data.transform import RESAMPLE, RESAMPLE_D
from data.util import bb_intersection_over_union, crop_and_scale_img
from data.transform.flow_utils import crop_and_scale_flow

__all__ = ['create_class_uniform_strategy', 'ClassUniformSquareCropAndScale']


def create_class_uniform_strategy(instances, incidences, epochs=1):
    incidences = incidences[:-1]  # remove ignore id
    num_images = len(instances)
    num_classes = incidences.shape[0]
    present_in_image = np.zeros((num_images, num_classes), dtype=np.uint32)
    image_names = np.array(list(instances.keys()))

    for i, (k, v) in enumerate(tqdm(instances.items(), total=len(instances))):
        for idx in v.keys():
            if idx >= num_classes:
                continue
            present_in_image[i, idx] += len(v[idx])

    class_incidence_histogram = incidences / incidences.sum()
    indices_by_occurence = np.argsort(class_incidence_histogram)
    p_r = class_incidence_histogram.sum() / class_incidence_histogram
    p_r[np.logical_or(np.isnan(p_r), np.isinf(p_r))] = 0.
    p_r /= p_r.sum()
    images_to_sample = np.round(num_images * p_r).astype(np.uint32)

    # weights = ((present_in_image > 0) * p_r.reshape(1, -1)).sum(-1)
    weights = (present_in_image * p_r.reshape(1, -1)).sum(-1)

    strategy = []
    for e in range(epochs):
        chosen_classes = {}
        chosen_class = num_classes * np.ones(num_images, dtype=np.uint32)
        is_image_chosen = np.zeros(num_images, dtype=np.bool)
        for idx in indices_by_occurence:
            possibilities = np.where(present_in_image[:, idx] > 0 & ~is_image_chosen)[0]
            to_sample = min(images_to_sample[idx], len(possibilities))
            chosen = np.random.choice(possibilities, to_sample)
            is_image_chosen[chosen] = 1
            chosen_class[chosen] = idx
        for n, c in zip(image_names, chosen_class):
            chosen_classes[n] = c
        strategy += [chosen_classes]
        statistics = defaultdict(int)
        for v in chosen_classes.values():
            statistics[v] += 1
    return strategy, weights


class ClassUniformSquareCropAndScale:
    def __init__(self, wh, mean, ignore_id, strategy, class_instances, min=.5, max=2.,
                 scale_method=lambda scale, wh, size: int(scale * wh), p_true_random_crop=.5):
        self.wh = wh
        self.min = min
        self.max = max
        self.mean = mean
        self.ignore_id = ignore_id
        self.random_gens = [self._rand_location, self._gen_instance_box]
        self.scale_method = scale_method
        self.strategy = strategy
        self.class_instances = class_instances
        self.p_true_random_crop = p_true_random_crop

    def _random_instance(self, name, epoch):
        instances = self.class_instances[name]
        chosen_class = self.strategy[epoch][name]
        if chosen_class == self.ignore_id:
            return None
        try:
            return random.choice(instances[chosen_class])
        except IndexError:
            return None

    def _gen_instance_box(self, W, H, target_wh, name, flipped, epoch):
        # warnings.warn(f'ClassUniformSquareCropAndScale, epoch {epoch}')
        bbox = self._random_instance(name, epoch)
        if bbox is not None:
            if not (random.uniform(0, 1) < self.p_true_random_crop):
                wmin, wmax, hmin, hmax = bbox
                if flipped:
                    wmin, wmax = W - 1 - wmax, W - 1 - wmin
                inst_box = [wmin, hmin, wmax, hmax]
                for _ in range(50):
                    box = self._rand_location(W, H, target_wh)
                    if bb_intersection_over_union(box, inst_box) > 0.:
                        break
                return box
        return self._rand_location(W, H, target_wh)

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
        flipped = example['flipped'] if 'flipped' in example else False
        crop_box = self._gen_instance_box(pad_size[0], pad_size[1], box_size, example.get('name'), flipped,
                                          example.get('epoch', 0))
        ret_dict = {
            'image': self._trans(image, crop_box, target_size, pad_size, RESAMPLE, self.mean),
        }
        if 'labels' in example:
            ret_dict['labels'] = self._trans(example['labels'], crop_box, target_size, pad_size, pimg.NEAREST,
                                             self.ignore_id)
        for k in ['image_prev', 'image_next']:
            if k in example:
                ret_dict[k] = self._trans(example[k], crop_box, target_size, pad_size, RESAMPLE,
                                          self.mean)
        if 'depth' in example:
            ret_dict['depth'] = self._trans(example['depth'], crop_box, target_size, pad_size, RESAMPLE_D, 0)
        if 'flow' in example:
            ret_dict['flow'] = crop_and_scale_flow(example['flow'], crop_box, target_size, pad_size, scale)
        return {**example, **ret_dict}
