import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image as pimg

from data.util import crop_and_scale_img

'''
Adapted from https://github.com/NVIDIA/flownet2-pytorch
'''


def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def flow2rgb(flow):
    hsv = np.zeros(list(flow.shape[:-1]) + [3], dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def offset_flow(img, flow):
    '''
    :param img: torch.FloatTensor of shape NxCxHxW
    :param flow: torch.FloatTensor of shape NxHxWx2
    :return: torch.FloatTensor of shape NxCxHxW
    '''
    N, C, H, W = img.shape
    # generate identity sampling grid
    gx, gy = torch.meshgrid(torch.arange(H), torch.arange(W))
    gx = gx.float().div(gx.max() - 1).view(1, H, W, 1)
    gy = gy.float().div(gy.max() - 1).view(1, H, W, 1)
    grid = torch.cat([gy, gx], dim=-1).mul(2.).sub(1)
    # generate normalized flow field
    flown = flow.clone()
    flown[..., 0] /= W
    flown[..., 1] /= H
    # calculate offset field
    grid += flown
    return F.grid_sample(img, grid), grid


def backward_warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).to(x.device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).to(x.device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid)

    mask = torch.ones_like(x)
    mask = F.grid_sample(mask, vgrid)

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask, mask > 0.


def pad_flow(flow, size):
    h, w, _ = flow.shape
    shape = list(size) + [2]
    new_flow = np.zeros(shape, dtype=flow.dtype)
    new_flow[:h, :w] = flow


def flip_flow_horizontal(flow):
    flow = np.flip(flow, axis=1)
    flow[..., 0] *= -1
    return flow


def crop_and_scale_flow(flow, crop_box, target_size, pad_size, scale):
    def _trans(uv):
        return crop_and_scale_img(uv, crop_box, target_size, pad_size, resample=pimg.NEAREST, blank_value=0)

    u, v = [pimg.fromarray(uv.squeeze()) for uv in np.split(flow * scale, 2, axis=-1)]
    dtype = flow.dtype
    return np.stack([np.array(_trans(u), dtype=dtype), np.array(_trans(v), dtype=dtype)], axis=-1)


def subsample_flow(flow, subsampling):
    dtype = flow.dtype
    u, v = [pimg.fromarray(uv.squeeze()) for uv in np.split(flow / subsampling, 2, axis=-1)]
    size = tuple([int(round(wh / subsampling)) for wh in u.size])
    u, v = u.resize(size), v.resize(size)
    return np.stack([np.array(u, dtype=dtype), np.array(v, dtype=dtype)], axis=-1)
