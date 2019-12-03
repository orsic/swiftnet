import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from itertools import chain
import torch.utils.checkpoint as cp
from collections import defaultdict
from math import log2

from ..util import _UpsampleBlend

__all__ = ['ResNet', 'resnet18', 'resnet34', 'BasicBlock']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def convkxk(in_planes, out_planes, stride=1, k=3):
    """kxk convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=stride, padding=k // 2, bias=False)


def _bn_function_factory(conv, norm, relu=None):
    def bn_function(x):
        x = norm(conv(x))
        if relu is not None:
            x = relu(x)
        return x

    return bn_function


def do_efficient_fwd(block, x, efficient):
    # return block(x)
    if efficient and x.requires_grad:
        return cp.checkpoint(block, x)
    else:
        return block(x)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, efficient=True, bn_class=nn.BatchNorm2d, levels=3):
        super(BasicBlock, self).__init__()
        self.conv1 = convkxk(inplanes, planes, stride)
        self.bn1 = nn.ModuleList([bn_class(planes) for _ in range(levels)])
        self.relu_inp = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = convkxk(planes, planes)
        self.bn2 = nn.ModuleList([bn_class(planes) for _ in range(levels)])
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient
        self.num_levels = levels

    def forward(self, x, level):
        residual = x

        bn_1 = _bn_function_factory(self.conv1, self.bn1[level], self.relu_inp)
        bn_2 = _bn_function_factory(self.conv2, self.bn2[level])

        out = do_efficient_fwd(bn_1, x, self.efficient)
        out = do_efficient_fwd(bn_2, out, self.efficient)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        relu = self.relu(out)

        return relu, out

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(BasicBlock, self)._load_from_state_dict(state_dict, prefix, local_metadata, False, missing_keys,
                                                      unexpected_keys, error_msgs)
        missing_keys = []
        unexpected_keys = []
        for bn in self.bn1:
            bn._load_from_state_dict(state_dict, prefix + 'bn1.', local_metadata, strict, missing_keys, unexpected_keys,
                                     error_msgs)
        for bn in self.bn2:
            bn._load_from_state_dict(state_dict, prefix + 'bn2.', local_metadata, strict, missing_keys, unexpected_keys,
                                     error_msgs)


class ResNet(nn.Module):
    def _make_layer(self, block, planes, blocks, stride=1, bn_class=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                bn_class(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.efficient, bn_class=bn_class,
                            levels=self.pyramid_levels))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_class=bn_class, levels=self.pyramid_levels, efficient=self.efficient))

        return nn.Sequential(*layers)

    def __init__(self, block, layers, *, num_features=128, pyramid_levels=3, use_bn=True, k_bneck=1, k_upsample=3,
                 efficient=False, upsample_skip=True, mean=(73.1584, 82.9090, 72.3924),
                 std=(44.9149, 46.1529, 45.3192), scale=1, detach_upsample_skips=(), detach_upsample_in=False,
                 align_corners=None, pyramid_subsample='bicubic', target_size=None,
                 output_stride=4, **kwargs):
        self.inplanes = 64
        self.efficient = efficient
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        bn_class = nn.BatchNorm2d if use_bn else Identity
        self.register_buffer('img_mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer('img_std', torch.tensor(std).view(1, -1, 1, 1))
        if scale != 1:
            self.register_buffer('img_scale', torch.tensor(scale).view(1, -1, 1, 1).float())

        self.pyramid_levels = pyramid_levels
        self.num_features = num_features
        self.replicated = False

        self.align_corners = align_corners
        self.pyramid_subsample = pyramid_subsample

        self.bn1 = nn.ModuleList([bn_class(64) for _ in range(pyramid_levels)])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        bottlenecks = []
        self.layer1 = self._make_layer(block, 64, layers[0], bn_class=bn_class)
        bottlenecks += [convkxk(self.inplanes, num_features, k=k_bneck)]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, bn_class=bn_class)
        bottlenecks += [convkxk(self.inplanes, num_features, k=k_bneck)]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, bn_class=bn_class)
        bottlenecks += [convkxk(self.inplanes, num_features, k=k_bneck)]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, bn_class=bn_class)
        bottlenecks += [convkxk(self.inplanes, num_features, k=k_bneck)]

        num_bn_remove = max(0, int(log2(output_stride) - 2))
        self.num_skip_levels = self.pyramid_levels + 3 - num_bn_remove
        bottlenecks = bottlenecks[num_bn_remove:]

        self.fine_tune = [self.conv1, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4, self.bn1]

        self.upsample_bottlenecks = nn.ModuleList(bottlenecks[::-1])
        num_pyr_modules = 2 + pyramid_levels - num_bn_remove
        self.target_size = target_size
        if self.target_size is not None:
            h, w = target_size
            target_sizes = [(h // 2 ** i, w // 2 ** i) for i in range(2, 2 + num_pyr_modules)][::-1]
        else:
            target_sizes = [None] * num_pyr_modules
        self.upsample_blends = nn.ModuleList(
            [_UpsampleBlend(num_features,
                            use_bn=use_bn,
                            use_skip=upsample_skip,
                            detach_skip=i in detach_upsample_skips,
                            fixed_size=ts,
                            k=k_upsample)
             for i, ts in enumerate(target_sizes)])
        self.detach_upsample_in = detach_upsample_in

        self.random_init = [self.upsample_bottlenecks, self.upsample_blends]

        self.features = num_features

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def random_init_params(self):
        return chain(*[f.parameters() for f in self.random_init])

    def fine_tune_params(self):
        return chain(*[f.parameters() for f in self.fine_tune])

    def forward_resblock(self, x, layers, idx):
        skip = None
        for l in layers:
            x = l(x) if not isinstance(l, BasicBlock) else l(x, idx)
            if isinstance(x, tuple):
                x, skip = x
        return x, skip

    def forward_down(self, image, skips, idx=-1):
        x = self.conv1(image)
        x = self.bn1[idx](x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = []
        x, skip = self.forward_resblock(x, self.layer1, idx)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer2, idx)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer3, idx)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer4, idx)
        features += [skip]

        skip_feats = [b(f) for b, f in zip(self.upsample_bottlenecks, reversed(features))]

        for i, s in enumerate(reversed(skip_feats)):
            skips[idx + i] += [s]

        return skips

    def forward(self, image):
        if isinstance(self.bn1[0], nn.BatchNorm2d):
            if hasattr(self, 'img_scale'):
                image /= self.img_scale
            image -= self.img_mean
            image /= self.img_std
        pyramid = [image]
        for l in range(1, self.pyramid_levels):
            if self.target_size is not None:
                ts = list([si // 2 ** l for si in self.target_size])
                pyramid += [
                    F.interpolate(image, size=ts, mode=self.pyramid_subsample, align_corners=self.align_corners)]
            else:
                pyramid += [F.interpolate(image, scale_factor=1 / 2 ** l, mode=self.pyramid_subsample,
                                          align_corners=self.align_corners)]
        skips = [[] for _ in range(self.num_skip_levels)]
        additional = {'pyramid': pyramid}
        for idx, p in enumerate(pyramid):
            skips = self.forward_down(p, skips, idx=idx)
        skips = skips[::-1]
        x = skips[0][0]
        if self.detach_upsample_in:
            x = x.detach()
        for i, (sk, blend) in enumerate(zip(skips[1:], self.upsample_blends)):
            x = blend(x, sum(sk))
        return x, additional

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(ResNet, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                                  unexpected_keys, error_msgs)
        for bn in self.bn1:
            bn._load_from_state_dict(state_dict, prefix + 'bn1.', local_metadata, strict, missing_keys, unexpected_keys,
                                     error_msgs)


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model
