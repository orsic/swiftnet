import torch
import torch.nn as nn
from itertools import chain

from .util import _BNReluConv, upsample


class SemsegModel(nn.Module):
    def __init__(self, backbone, num_classes, use_bn=True):
        super(SemsegModel, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.logits = _BNReluConv(self.backbone.num_features, self.num_classes, batch_norm=use_bn)

    def forward(self, pyramid, target_size, image_size):
        feats, additional = zip(*[self.backbone(p) for p in pyramid])
        feature_pyramid = [upsample(f, target_size) for f in feats]
        features = feature_pyramid[0] if len(feature_pyramid) == 1 else None
        logits = self.logits.forward(features)
        return upsample(logits, image_size), additional[0]

    def prepare_data(self, batch, image_size, device=torch.device('cuda')):
        if image_size is None:
            image_size = batch['target_size']
        pyramid = [p.clone().detach().requires_grad_(False).to(device) for p in batch['pyramid']]
        return {
            'pyramid': pyramid,
            'image_size': image_size,
            'target_size': batch['target_size_feats']
        }

    def do_forward(self, batch, image_size=None):
        data = self.prepare_data(batch, image_size)
        return self.forward(**data)

    def random_init_params(self):
        return chain(*([self.logits.parameters(), self.backbone.random_init_params()]))

    def fine_tune_params(self):
        return self.backbone.fine_tune_params()


class SemsegPyramidModel(nn.Module):
    def __init__(self, backbone, num_classes, use_bn=True, aux_logits=False):
        super(SemsegPyramidModel, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.num_features = self.backbone.features if isinstance(self.backbone.features,
                                                                 int) else self.backbone.num_features
        self.logits = _BNReluConv(self.num_features, self.num_classes, batch_norm=use_bn)
        self.has_aux_logits = aux_logits
        if aux_logits:
            self.num_features_aux = self.backbone.num_features_aux
            self.add_module('aux_logits', _BNReluConv(self.num_features_aux, self.num_classes, batch_norm=use_bn))

    def forward(self, pyramid, image_size):
        features, additional = self.backbone(pyramid)
        if self.has_aux_logits:
            additional['aux_logits'] = self.aux_logits.forward(additional['upsamples'][0])
        logits = self.logits.forward(features)
        return upsample(logits, image_size), additional

    def prepare_data(self, batch, image_size, device=torch.device('cuda')):
        if image_size is None:
            image_size = batch['target_size']
        pyr = [p.clone().detach().requires_grad_(False).to(device) for p in batch['pyramid']]
        return {'image_size': image_size, 'pyramid': pyr}

    def do_forward(self, batch, image_size=None):
        data = self.prepare_data(batch, image_size)
        return self.forward(**data)

    def random_init_params(self):
        params = [self.logits.parameters(), self.backbone.random_init_params()]
        if self.has_aux_logits:
            params += [self.aux_logits.parameters()]
        return chain(*params)

    def fine_tune_params(self):
        return self.backbone.fine_tune_params()
