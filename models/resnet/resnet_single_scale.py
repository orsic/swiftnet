import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from itertools import chain
import torch.utils.checkpoint as cp
from math import log2

from ..util import _Upsample, SpatialPyramidPooling, SeparableConv2d
from evaluation.evaluate import mt

__all__ = ['ResNet', 'resnet18', 'resnet18dws', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'BasicBlock']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet18dws': '/home/morsic/saves/imagenet/resnet18dws/model_best.pth.tar',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, separable=False):
    """3x3 convolution with padding"""
    conv_class = SeparableConv2d if separable else nn.Conv2d
    return conv_class(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def _bn_function_factory(conv, norm, relu=None):
    def bn_function(x):
        x = conv(x)
        if norm is not None:
            x = norm(x)
        if relu is not None:
            x = relu(x)
        return x

    return bn_function

def do_efficient_fwd(block, x, efficient):
    if efficient and x.requires_grad:
        return cp.checkpoint(block, x)
    else:
        return block(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, efficient=True, use_bn=True, deleting=False,
                 separable=False):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride, separable=separable)
        self.bn1 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, separable=separable)
        self.bn2 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient
        self.deleting = deleting

    def forward(self, x):
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.deleting is False:
            bn_1 = _bn_function_factory(self.conv1, self.bn1, self.relu)
            bn_2 = _bn_function_factory(self.conv2, self.bn2)

            out = do_efficient_fwd(bn_1, x, self.efficient)
            out = do_efficient_fwd(bn_2, out, self.efficient)
        else:
            out = torch.zeros_like(residual)

        out = out + residual
        relu = self.relu(out)
        # print(f'Basic Block memory: {torch.cuda.memory_allocated() // 2**20}')

        return relu, out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, efficient=True, use_bn=True, separable=False):
        super(Bottleneck, self).__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) if self.use_bn else None
        conv_class = SeparableConv2d if separable else nn.Conv2d
        self.conv2 = conv_class(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion) if self.use_bn else None
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient

    def forward(self, x):
        residual = x

        bn_1 = _bn_function_factory(self.conv1, self.bn1, self.relu)
        bn_2 = _bn_function_factory(self.conv2, self.bn2, self.relu)
        bn_3 = _bn_function_factory(self.conv3, self.bn3, self.relu)

        out = do_efficient_fwd(bn_1, x, self.efficient)
        out = do_efficient_fwd(bn_2, out, self.efficient)
        out = do_efficient_fwd(bn_3, out, self.efficient)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        relu = self.relu(out)

        return relu, out


class ResNet(nn.Module):
    def __init__(self, block, layers, *, num_features=128, k_up=3, efficient=False, use_bn=True,
                 spp_grids=(8, 4, 2, 1), spp_square_grid=False, spp_drop_rate=0.0,
                 upsample_skip=True, upsample_only_skip=False,
                 detach_upsample_skips=(), detach_upsample_in=False,
                 target_size=None, output_stride=4, mean=(73.1584, 82.9090, 72.3924),
                 std=(44.9149, 46.1529, 45.3192), scale=1, separable=False,
                 upsample_separable=False, **kwargs):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.efficient = efficient
        self.use_bn = use_bn
        self.separable = separable
        self.register_buffer('img_mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer('img_std', torch.tensor(std).view(1, -1, 1, 1))
        if scale != 1:
            self.register_buffer('img_scale', torch.tensor(scale).view(1, -1, 1, 1).float())

        self.detach_upsample_in = detach_upsample_in
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64) if self.use_bn else lambda x: x
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.target_size = target_size
        if self.target_size is not None:
            h, w = target_size
            target_sizes = [(h // 2 ** i, w // 2 ** i) for i in range(2, 6)]
        else:
            target_sizes = [None] * 4
        upsamples = []
        self.layer1 = self._make_layer(block, 64, layers[0])
        upsamples += [
            _Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up, use_skip=upsample_skip,
                      only_skip=upsample_only_skip, detach_skip=2 in detach_upsample_skips, fixed_size=target_sizes[0],
                      separable=upsample_separable)]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        upsamples += [
            _Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up, use_skip=upsample_skip,
                      only_skip=upsample_only_skip, detach_skip=1 in detach_upsample_skips, fixed_size=target_sizes[1],
                      separable=upsample_separable)]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        upsamples += [
            _Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up, use_skip=upsample_skip,
                      only_skip=upsample_only_skip, detach_skip=0 in detach_upsample_skips, fixed_size=target_sizes[2],
                      separable=upsample_separable)]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.fine_tune = [self.conv1, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4]
        if self.use_bn:
            self.fine_tune += [self.bn1]

        num_levels = 3
        self.spp_size = kwargs.get('spp_size', num_features)
        bt_size = self.spp_size

        level_size = self.spp_size // num_levels

        self.spp = SpatialPyramidPooling(self.inplanes, num_levels, bt_size=bt_size, level_size=level_size,
                                         out_size=num_features, grids=spp_grids, square_grid=spp_square_grid,
                                         bn_momentum=0.01 / 2, use_bn=self.use_bn, drop_rate=spp_drop_rate
                                         , fixed_size=target_sizes[3])
        num_up_remove = max(0, int(log2(output_stride) - 2))
        self.upsample = nn.ModuleList(list(reversed(upsamples[num_up_remove:])))

        self.random_init = [self.spp, self.upsample]

        self.num_features = num_features

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = [nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)]
            if self.use_bn:
                layers += [nn.BatchNorm2d(planes * block.expansion)]
            downsample = nn.Sequential(*layers)
        layers = [block(self.inplanes, planes, stride, downsample, efficient=self.efficient, use_bn=self.use_bn,
                        separable=self.separable)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers += [block(self.inplanes, planes, efficient=self.efficient, use_bn=self.use_bn,
                             separable=self.separable)]

        return nn.Sequential(*layers)

    def random_init_params(self):
        return chain(*[f.parameters() for f in self.random_init])

    def fine_tune_params(self):
        return chain(*[f.parameters() for f in self.fine_tune])

    def forward_resblock(self, x, layers):
        skip = None
        for l in layers:
            x = l(x)
            if isinstance(x, tuple):
                x, skip = x
        return x, skip

    def forward_down(self, image):
        if hasattr(self, 'img_scale'):
            image /= self.img_scale
        image -= self.img_mean
        image /= self.img_std

        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = []
        x, skip = self.forward_resblock(x, self.layer1)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer2)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer3)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer4)
        features += [self.spp.forward(skip)]
        return features

    def forward_up(self, features):
        features = features[::-1]

        x = features[0]
        if self.detach_upsample_in:
            x = x.detach()

        upsamples = []
        for skip, up in zip(features[1:], self.upsample):
            x = up(x, skip)
            upsamples += [x]
        return x, {'features': features, 'upsamples': upsamples}

    def forward(self, image):
        return self.forward_up(self.forward_down(image))


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def resnet18dws(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], separable=True, **kwargs)
    if pretrained:
        try:
            model.load_state_dict(torch.load(model_urls['resnet18dws'])['state_dict'], strict=True)
        except Exception as e:
            print(e)
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


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


def resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model


def resnet152(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=False)
    return model
