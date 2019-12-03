import torch
from torch import nn as nn
from torch.nn import functional as F

from models.util import upsample


class BoundaryAwareFocalLoss(nn.Module):
    def __init__(self, gamma=0, num_classes=19, ignore_id=19, print_each=20):
        super(BoundaryAwareFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_id = ignore_id
        self.print_each = print_each
        self.step_counter = 0
        self.gamma = gamma

    def forward(self, input, target, batch, **kwargs):
        if input.shape[-2:] != target.shape[-2:]:
            input = upsample(input, target.shape[-2:])
        target[target == self.ignore_id] = 0  # we can do this because alphas are zero in ignore_id places
        label_distance_alphas = batch['label_distance_alphas'].to(input.device)
        N = (label_distance_alphas.data > 0.).sum()
        if N.le(0):
            return torch.zeros(size=(0,), device=label_distance_alphas.device, requires_grad=True).sum()
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        alphas = label_distance_alphas.view(-1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.detach().exp()

        loss = -1 * alphas * torch.exp(self.gamma * (1 - pt)) * logpt
        loss = loss.sum() / N

        if (self.step_counter % self.print_each) == 0:
            print(f'Step: {self.step_counter} Loss: {loss.data.cpu().item():.4f}')
        self.step_counter += 1

        return loss
