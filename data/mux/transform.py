from PIL import Image as pimg
from math import ceil

RESAMPLE = pimg.BICUBIC
RESAMPLE_D = pimg.BILINEAR

__all__ = ['Pyramid', 'SetTargetSize']


def pyramid_sizes(size, alphas, scale=1.0):
    w, h = size[0], size[1]
    th_sc = lambda wh, alpha: int(ceil(wh / (alpha * scale)))
    return [(th_sc(w, a), th_sc(h, a)) for a in alphas]


class Pyramid(object):
    def __init__(self, alphas):
        self.alphas = alphas
        self.pyramid_sizes = None

    def __call__(self, example):
        img = example['image']
        if self.pyramid_sizes is None:
            self.pyramid_sizes = pyramid_sizes(img.size, self.alphas)
        ret_dict = {'pyramid': [img.resize(size, resample=RESAMPLE) for size in self.pyramid_sizes]}
        return {**example, **ret_dict}


class SetTargetSize:
    def __init__(self, target_size, target_size_feats):
        self.target_size = target_size
        self.target_size_feats = target_size_feats

    def __call__(self, example):
        example['target_size'] = self.target_size[::-1]
        example['target_size_feats'] = self.target_size_feats[::-1]
        example['alphas'] = [-1]
        example['target_level'] = 0
        return example
