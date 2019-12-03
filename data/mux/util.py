from math import ceil


def pyramid_sizes(size, alphas, scale=1.0):
    w, h = size[0], size[1]
    th_sc = lambda wh, alpha: int(ceil(wh / (alpha * scale)))
    return [(th_sc(w, a), th_sc(h, a)) for a in alphas]
